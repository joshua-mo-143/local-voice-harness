from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from local_voice_harness.agents import (
    HarnessSession,
    ReconciliationState,
    SessionReconciliation,
)
from local_voice_harness.cursor import delivery, provisioning, worker_lifecycle
from local_voice_harness.cursor.agent_outbox import TASK_SUBMIT
from local_voice_harness.cursor.coordinator import (
    CoordinatorCommand,
    CoordinatorDecision,
    DurableEffect,
    EffectObservation,
)
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
    WorkflowParticipant,
    transition,
)
from local_voice_harness.cursor.operations import (
    AgentSessionOperation,
    AgentSessionSpec,
    AgentSessionState,
    WorkerOwnership,
)
from local_voice_harness.cursor.recovery import (
    acknowledge_worktree_quarantine,
    cancel_target_and_release,
    reconcile_prompt_and_pane_operations,
    reconcile_uncertain_agent,
    reconcile_uncertain_clone,
    reconcile_uncertain_fork,
    reconcile_uncertain_issue_creation,
    reconcile_uncertain_linear_ticket_creation,
    reconcile_uncertain_pr_creation,
    reconcile_uncertain_pr_merge,
    reconcile_uncertain_repo_creation,
    recover_jobs,
    resolve_manual_reconciliation,
    stage_terminal_intent,
)
from local_voice_harness.cursor.store import (
    JobQuarantineWarning,
    JobStore,
    MaintenanceLease,
)
from local_voice_harness.integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubIssue,
    GitHubIssueCreationResult,
    GitHubPullRequest,
    GitHubPullRequestCreationResult,
    GitHubPullRequestMergeResult,
    GitHubRepoCreationResult,
    GitHubRepository,
)
from local_voice_harness.integrations.herdr import HerdrError
from local_voice_harness.integrations.linear import (
    LinearError,
    LinearIntegration,
    LinearIssue,
    LinearTicketCreationResult,
)
from local_voice_harness.prompt_operations import (
    AmbiguousPrompt,
    PromptIdentity,
    SubmittingPrompt,
)

WORKER = WorkerOwnership("worker", 42, "boot", "start", "test", 1)


class CursorRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = JobStore(self.root / "jobs", self.root / "legacy")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self, values: dict[str, object]) -> CursorJob:
        values = dict(values)
        agent_state = values.get("agent_dispatch_state")
        if agent_state is not None and values.get("herdr_target"):
            values.setdefault("agent_operation_target", values["herdr_target"])
            values.setdefault(
                "agent_operation_checkout",
                values.get("worktree_path") or values.get("repository") or "/worktree",
            )
            values.setdefault("herdr_workspace_id", "workspace")
            values.setdefault("herdr_pane_id", "pane")
            values.setdefault(
                "agent_operation_workspace_id", values["herdr_workspace_id"]
            )
            values.setdefault("agent_operation_pane_id", values["herdr_pane_id"])
            if agent_state in {"ready", "retained"}:
                values.setdefault("agent_provider", "cursor/herdr")
                values.setdefault("agent_provider_session_id", "session")
                values.setdefault("agent_state_sequence", 1)
        if values.get("fork_operation_state") is not None and values.get(
            "fork_operation_target"
        ):
            values.setdefault("fork_operation_source", "source/project")
            values.setdefault(
                "fork_operation_source_url",
                "https://github.com/source/project",
            )
            values.setdefault("fork_operation_source_default_branch", "main")
            values.setdefault("fork_operation_source_private", False)
        if values.get("participant_creation_state") not in {None, "none"}:
            values.setdefault(
                "participant_creation_checkout",
                values.get("worktree_path") or values.get("repository") or "/worktree",
            )
        if values.get("github_pr_create_operation_state") is not None:
            values.setdefault("github_pr_create_base", "main")
            values.setdefault("github_pr_create_head_oid", "b" * 40)
            values.setdefault("github_pr_create_published_head_oid", "c" * 40)
            values.setdefault(
                "github_pr_create_head_repository",
                values.get("github_repository") or "example/project",
            )
            values.setdefault(
                "github_pr_create_checkout_origin",
                "github.com/example/project",
            )
            values.setdefault("github_pr_create_status_digest", "d" * 64)
        return self.store.create(CursorJob.from_dict(values))

    def test_ambiguous_clone_recovery_only_observes_expected_destination(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "request": "clone example/project",
                "clone_source": "example/project",
                "clone_confirmed": True,
                "clone_operation_state": "ambiguous",
                "status": "queued",
                "reconcile": True,
                "created_at": 1,
                "delivered": False,
            }
        )
        source = GitHubRepository(
            "example/project",
            "https://github.com/example/project",
            False,
            "main",
        )
        checkout = self.root / "src" / "example" / "project"
        client = mock.Mock(spec=GitHubClient)
        client.local_git = mock.Mock()
        client.inspect_repository.return_value = source
        client.local_git.observe_materialized.return_value = None

        reconcile_uncertain_clone(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )
        pending = self.store.get(job.id)
        self.assertEqual(pending.clone_operation_state, "ambiguous")
        self.assertIsNone(pending.repository)
        client.ensure_repository_clone.assert_not_called()

        client.local_git.observe_materialized.return_value = checkout
        reconcile_uncertain_clone(
            self.store,
            pending,
            now=101,
            github_factory=lambda: client,
        )
        recovered = self.store.get(job.id)
        self.assertEqual(recovered.clone_operation_state, "cloned")
        self.assertEqual(recovered.repository, str(checkout))
        self.assertFalse(recovered.reconcile)
        client.ensure_repository_clone.assert_not_called()

    def admit_unknown_prompt_effect(self, job: CursorJob) -> CursorJob:
        operation = job.prompt_operation
        assert isinstance(operation, SubmittingPrompt | AmbiguousPrompt)
        identity = operation.identity
        key = f"{TASK_SUBMIT}:{job.id}:{identity.phase}:{identity.turn_token}"
        admitted = self.store.apply(
            CoordinatorCommand(
                job_id=job.id,
                expected_revision=job.revision,
                command_id=f"admit:{key}",
                kind=f"{TASK_SUBMIT}.admit",
            ),
            lambda current: CoordinatorDecision(
                job=current.evolve(),
                effects=(
                    DurableEffect(
                        kind=TASK_SUBMIT,
                        idempotency_key=key,
                        concurrency_key=f"cursor/herdr:{identity.target}",
                        payload={
                            "expected_revision": job.revision + 1,
                            "prompt_job_id": identity.job_id,
                            "phase": identity.phase,
                            "turn": identity.turn,
                            "turn_token": identity.turn_token,
                            "target": identity.target,
                            "session_id": identity.agent_session,
                            "baseline_sequence": identity.baseline_sequence,
                        },
                    ),
                ),
            ),
        )
        assert admitted is not None
        lease = self.store.claim_outbox((TASK_SUBMIT,), now=1)
        assert lease is not None
        assert self.store.mark_outbox_dispatched(lease)
        assert (
            self.store.observe_outbox(
                lease,
                EffectObservation(outcome="OutcomeUnknown"),
                now=2,
            )
            == "applied"
        )
        return self.store.get(job.id)

    def observe_owned_sessions(
        self, client: mock.Mock, job_id: str = "123456789abc"
    ) -> None:
        def get_agent(target: str) -> dict[str, object]:
            current = self.store.get(job_id)
            if target == current.herdr_target:
                return {
                    "pane_id": current.herdr_pane_id,
                    "workspace_id": current.herdr_workspace_id,
                    "cwd": current.worktree_path or current.repository,
                }
            if target == current.participant_creation_target:
                return {
                    "pane_id": current.participant_creation_pane_id,
                    "workspace_id": current.participant_creation_workspace_id,
                    "cwd": current.participant_creation_checkout,
                }
            owner = next(
                item
                for item in current.participant_session_owners
                if item["target"] == target
            )
            return {
                "pane_id": owner["pane_id"],
                "workspace_id": owner["workspace_id"],
                "cwd": owner["checkout"],
            }

        def reconcile(
            target: str, *, expected_session_id: str
        ) -> SessionReconciliation:
            current = self.store.get(job_id)
            operation = current.agent_session_operation
            if operation is not None and operation.spec.target == target:
                provider = (
                    operation.session.provider if operation.session else "cursor/herdr"
                )
                sequence = (
                    operation.session.state_sequence + 1 if operation.session else 1
                )
                pane_id = operation.spec.pane_id
                workspace_id = operation.spec.workspace_id
                checkout = operation.spec.checkout
            else:
                owner = next(
                    item
                    for item in current.participant_session_owners
                    if item["target"] == target
                )
                provider = str(owner["provider"])
                state_sequence = owner["state_sequence"]
                assert isinstance(state_sequence, int) and not isinstance(
                    state_sequence, bool
                )
                sequence = state_sequence + 1
                pane_id = str(owner["pane_id"])
                workspace_id = str(owner["workspace_id"])
                checkout = str(owner["checkout"])
            return SessionReconciliation(
                ReconciliationState.ACTIVE,
                HarnessSession(
                    provider,
                    expected_session_id,
                    target,
                    sequence,
                    {
                        "pane_id": pane_id,
                        "workspace_id": workspace_id,
                        "cwd": checkout,
                    },
                ),
                "active",
                True,
            )

        client.get_agent.side_effect = get_agent
        client.reconcile_session.side_effect = reconcile

    def test_reconciles_ambiguous_issue_creation_without_resubmitting(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "create an issue",
                "created_at": 1,
                "delivered": False,
                "issue_provider": "github",
                "github_repository": "example/project",
                "github_issue_create_requested": True,
                "github_issue_create_confirmed": True,
                "github_issue_create_title": "Fix startup",
                "github_issue_create_body": "Startup fails.",
                "github_issue_create_marker": "a" * 32,
                "github_issue_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_issue.return_value = GitHubIssueCreationResult(
            GitHubIssue("example", "project", 42),
            "https://github.com/example/project/issues/42",
            "a" * 32,
        )

        reconcile_uncertain_issue_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.COMPLETED)
        self.assertEqual(updated.github_issue, 42)
        self.assertEqual(
            updated.github_issue_url,
            "https://github.com/example/project/issues/42",
        )
        self.assertEqual(updated.github_issue_created_number, 42)
        self.assertEqual(
            updated.github_issue_created_url,
            "https://github.com/example/project/issues/42",
        )
        self.assertNotIn("github_issue_created_number", updated.to_dict())
        self.assertNotIn("github_issue_created_url", updated.to_dict())
        client.submit_issue.assert_not_called()

    def test_reconciles_continue_workflow_issue_creation_without_completing(
        self,
    ) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "create me a SaaS called widgets",
                "created_at": 1,
                "delivered": False,
                "issue_provider": "github",
                "github_repository": "alice/widgets",
                "github_repo_create_requested": True,
                "github_repo_create_continue_workflow": True,
                "github_repo_create_visibility": "private",
                "github_repo_create_marker": "a" * 32,
                "github_repo_create_operation_state": "created",
                "github_issue_create_requested": True,
                "github_issue_create_confirmed": True,
                "github_issue_create_title": "create me a SaaS called widgets",
                "github_issue_create_body": "create me a SaaS called widgets",
                "github_issue_create_marker": "b" * 32,
                "github_issue_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_issue.return_value = GitHubIssueCreationResult(
            GitHubIssue("alice", "widgets", 1),
            "https://github.com/alice/widgets/issues/1",
            "b" * 32,
        )

        reconcile_uncertain_issue_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.QUEUED)
        self.assertEqual(updated.issue_provider, "github")
        self.assertEqual(updated.github_issue, 1)
        self.assertEqual(
            updated.github_issue_url,
            "https://github.com/alice/widgets/issues/1",
        )
        self.assertEqual(updated.github_issue_created_number, 1)
        self.assertEqual(updated.github_issue_create_operation_state, "created")
        self.assertNotIn("github_issue_created_number", updated.to_dict())
        self.assertNotIn("github_issue_created_url", updated.to_dict())
        client.submit_issue.assert_not_called()

    def test_unobserved_issue_creation_requires_manual_check(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "create an issue",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_provider": "github",
                "github_repository": "example/project",
                "github_issue_create_requested": True,
                "github_issue_create_confirmed": True,
                "github_issue_create_title": "Fix startup",
                "github_issue_create_body": "Startup fails.",
                "github_issue_create_marker": "a" * 32,
                "github_issue_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_issue.return_value = None

        reconcile_uncertain_issue_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.BLOCKED)
        self.assertEqual(
            updated.github_issue_create_operation_state,
            "manual_required",
        )
        client.submit_issue.assert_not_called()

    def test_reconciles_ambiguous_pr_creation_without_resubmitting(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "open a pull request",
                "created_at": 1,
                "delivered": False,
                "github_repository": "example/project",
                "worktree_branch": "voice/job",
                "github_pr_create_requested": True,
                "github_pr_create_confirmed": True,
                "github_pr_create_title": "Open the change",
                "github_pr_create_body": "Detailed body",
                "github_pr_create_marker": "a" * 32,
                "github_pr_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_pull_request_creation.return_value = (
            GitHubPullRequestCreationResult(
                GitHubPullRequest("example", "project", 7),
                "https://github.com/example/project/pull/7",
                "a" * 32,
            )
        )

        reconcile_uncertain_pr_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.COMPLETED)
        self.assertEqual(updated.github_pr_created_number, 7)
        self.assertEqual(
            updated.github_pr_created_url,
            "https://github.com/example/project/pull/7",
        )
        self.assertIn("https://github.com/example/project/pull/7", updated.result or "")
        client.submit_pull_request_creation.assert_not_called()

    def test_unobserved_pr_creation_requires_manual_check(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "open a pull request",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "github_repository": "example/project",
                "worktree_branch": "voice/job",
                "github_pr_create_requested": True,
                "github_pr_create_confirmed": True,
                "github_pr_create_title": "Open the change",
                "github_pr_create_body": "Detailed body",
                "github_pr_create_marker": "a" * 32,
                "github_pr_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_pull_request_creation.return_value = None

        reconcile_uncertain_pr_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.BLOCKED)
        self.assertEqual(
            updated.github_pr_create_operation_state,
            "manual_required",
        )
        client.submit_pull_request_creation.assert_not_called()

    def test_reconciles_ambiguous_pr_merge_without_resubmitting(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "merge the pull request",
                "created_at": 1,
                "delivered": False,
                "github_repository": "example/project",
                "github_pr_merge_requested": True,
                "github_pr_merge_confirmed": True,
                "github_pr_merge_number": 7,
                "github_pr_merge_url": "https://github.com/example/project/pull/7",
                "github_pr_merge_marker": "a" * 32,
                "github_pr_merge_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_pull_request_merge.return_value = GitHubPullRequestMergeResult(
            GitHubPullRequest("example", "project", 7),
            "https://github.com/example/project/pull/7",
            "a" * 32,
        )

        reconcile_uncertain_pr_merge(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.COMPLETED)
        self.assertEqual(updated.github_pr_merge_operation_state, "merged")
        self.assertIn("https://github.com/example/project/pull/7", updated.result or "")
        client.submit_pull_request_merge.assert_not_called()

    def test_unobserved_pr_merge_requires_manual_check(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "merge the pull request",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "github_repository": "example/project",
                "github_pr_merge_requested": True,
                "github_pr_merge_confirmed": True,
                "github_pr_merge_number": 7,
                "github_pr_merge_url": "https://github.com/example/project/pull/7",
                "github_pr_merge_marker": "a" * 32,
                "github_pr_merge_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_pull_request_merge.return_value = None

        reconcile_uncertain_pr_merge(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.BLOCKED)
        self.assertEqual(
            updated.github_pr_merge_operation_state,
            "manual_required",
        )
        client.submit_pull_request_merge.assert_not_called()

    def test_reconciles_ambiguous_repo_creation_without_resubmitting(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "create a GitHub repository called payments",
                "created_at": 1,
                "delivered": False,
                "issue_provider": "github",
                "github_repository": "alice/payments",
                "github_repo_create_requested": True,
                "github_repo_create_confirmed": True,
                "github_repo_create_visibility": "private",
                "github_repo_create_marker": "a" * 32,
                "github_repo_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_repository_creation.return_value = GitHubRepoCreationResult(
            GitHubRepository(
                "alice/payments",
                "https://github.com/alice/payments",
                True,
                "main",
            ),
            "https://github.com/alice/payments",
            "a" * 32,
        )
        checkout = self.root / "src" / "alice" / "payments"
        client.ensure_repository_clone.return_value = checkout

        reconcile_uncertain_repo_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.COMPLETED)
        self.assertEqual(updated.repository, str(checkout))
        self.assertEqual(
            updated.github_repo_create_operation_state,
            "clone_verified",
        )
        self.assertEqual(
            updated.github_repo_created_url,
            "https://github.com/alice/payments",
        )
        client.submit_repository_creation.assert_not_called()

    def test_repo_creation_recovery_retains_remote_phase_until_clone_verified(
        self,
    ) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "create a GitHub repository called payments",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_provider": "github",
                "github_repository": "alice/payments",
                "github_repo_create_requested": True,
                "github_repo_create_confirmed": True,
                "github_repo_create_visibility": "private",
                "github_repo_create_marker": "a" * 32,
                "github_repo_create_operation_state": "remote_created",
                "github_repo_created_url": "https://github.com/alice/payments",
                "reconcile": True,
            }
        )
        result = GitHubRepoCreationResult(
            GitHubRepository(
                "alice/payments",
                "https://github.com/alice/payments",
                True,
                "main",
            ),
            "https://github.com/alice/payments",
            "a" * 32,
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_repository_creation.return_value = result
        client.ensure_repository_clone.side_effect = GitHubError("clone unavailable")

        reconcile_uncertain_repo_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        pending = self.store.get(job.id)
        self.assertEqual(pending.status, JobStatus.QUEUED)
        self.assertEqual(
            pending.github_repo_create_operation_state,
            "remote_created",
        )
        self.assertIsNone(pending.repository)
        client.submit_repository_creation.assert_not_called()

    def test_reconciles_ambiguous_org_repo_creation_without_resubmitting(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "create a GitHub repository in the acme org called payments",
                "created_at": 1,
                "delivered": False,
                "issue_provider": "github",
                "github_repository": "acme/payments",
                "github_repo_create_requested": True,
                "github_repo_create_org_requested": True,
                "github_repo_create_owner": "acme",
                "github_repo_create_confirmed": True,
                "github_repo_create_visibility": "private",
                "github_repo_create_marker": "a" * 32,
                "github_repo_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_repository_creation.return_value = GitHubRepoCreationResult(
            GitHubRepository(
                "acme/payments",
                "https://github.com/acme/payments",
                True,
                "main",
            ),
            "https://github.com/acme/payments",
            "a" * 32,
        )
        checkout = self.root / "src" / "acme" / "payments"
        client.ensure_repository_clone.return_value = checkout

        reconcile_uncertain_repo_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.COMPLETED)
        self.assertEqual(updated.repository, str(checkout))
        self.assertEqual(
            updated.github_repo_create_operation_state,
            "clone_verified",
        )
        self.assertEqual(
            updated.github_repo_created_url,
            "https://github.com/acme/payments",
        )
        client.submit_repository_creation.assert_not_called()

    def test_reconciles_continue_workflow_repo_creation_without_completing(
        self,
    ) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "create me a SaaS called widgets",
                "created_at": 1,
                "delivered": False,
                "github_repository": "alice/widgets",
                "github_repo_create_requested": True,
                "github_repo_create_continue_workflow": True,
                "github_repo_create_confirmed": True,
                "github_repo_create_visibility": "private",
                "github_repo_create_marker": "a" * 32,
                "github_repo_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_repository_creation.return_value = GitHubRepoCreationResult(
            GitHubRepository(
                "alice/widgets",
                "https://github.com/alice/widgets",
                True,
                "main",
            ),
            "https://github.com/alice/widgets",
            "a" * 32,
        )
        client.ensure_repository_clone.return_value = Path("/home/test/src/widgets")

        reconcile_uncertain_repo_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.QUEUED)
        self.assertEqual(
            updated.github_repo_create_operation_state,
            "clone_verified",
        )
        self.assertTrue(updated.github_repo_create_continue_workflow)
        client.submit_repository_creation.assert_not_called()
        client.ensure_repository_clone.assert_called_once()

    def test_unobserved_repo_creation_requires_manual_check(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "create a GitHub repository called payments",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_provider": "github",
                "github_repository": "alice/payments",
                "github_repo_create_requested": True,
                "github_repo_create_confirmed": True,
                "github_repo_create_visibility": "private",
                "github_repo_create_marker": "a" * 32,
                "github_repo_create_operation_state": "ambiguous",
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_repository_creation.return_value = None

        reconcile_uncertain_repo_creation(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.BLOCKED)
        self.assertEqual(
            updated.github_repo_create_operation_state,
            "manual_required",
        )
        client.submit_repository_creation.assert_not_called()

    def test_reconciles_ambiguous_clone_without_retrying(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "Use Cursor to fix the bug",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "clone_source": "example/project",
                "clone_confirmed": True,
                "clone_operation_state": "ambiguous",
                "reconcile": True,
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_clone.return_value = Path("/home/test/src/example/project")

        reconcile_uncertain_clone(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.QUEUED)
        self.assertEqual(updated.clone_operation_state, "cloned")
        self.assertEqual(updated.repository, "/home/test/src/example/project")
        self.assertFalse(updated.reconcile)
        client.ensure_repository_clone.assert_not_called()

    def test_unobserved_clone_requires_manual_check(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "Use Cursor to fix the bug",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "clone_source": "example/project",
                "clone_confirmed": True,
                "clone_operation_state": "ambiguous",
                "reconcile": True,
            }
        )
        client = mock.Mock(spec=GitHubClient)
        client.observe_clone.return_value = None

        reconcile_uncertain_clone(
            self.store,
            job,
            now=100,
            github_factory=lambda: client,
        )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.BLOCKED)
        self.assertEqual(updated.clone_operation_state, "manual_required")
        client.ensure_repository_clone.assert_not_called()

    def test_reconciles_ambiguous_linear_creation_without_resubmitting(
        self,
    ) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "create a Linear ticket",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_provider": "linear",
                "linear_ticket_create_requested": True,
                "linear_ticket_create_confirmed": True,
                "linear_ticket_create_team": "API",
                "linear_ticket_create_team_id": "team-id-api",
                "linear_ticket_create_title": "Fix startup",
                "linear_ticket_create_description": "Startup fails.",
                "linear_ticket_create_marker": "a" * 32,
                "linear_ticket_create_operation_state": "ambiguous",
            }
        )
        provider = LinearIntegration()
        result = LinearTicketCreationResult(
            LinearIssue("API-42"),
            "https://linear.app/acme/issue/API-42/fix-startup",
            "a" * 32,
        )
        with mock.patch.object(
            provider,
            "observe_ticket_creation",
            return_value=result,
        ) as observe:
            reconcile_uncertain_linear_ticket_creation(
                self.store,
                job,
                now=100,
                herdr_factory=mock.Mock,
                linear_factory=lambda: provider,
            )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.COMPLETED)
        self.assertEqual(updated.linear_ticket_created_identifier, "API-42")
        observe.assert_called_once()

    def test_unobserved_linear_creation_requires_manual_check(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "create a Linear ticket",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_provider": "linear",
                "linear_ticket_create_requested": True,
                "linear_ticket_create_confirmed": True,
                "linear_ticket_create_team": "API",
                "linear_ticket_create_team_id": "team-id-api",
                "linear_ticket_create_title": "Fix startup",
                "linear_ticket_create_description": "Startup fails.",
                "linear_ticket_create_marker": "a" * 32,
                "linear_ticket_create_operation_state": "ambiguous",
            }
        )
        provider = LinearIntegration()
        with mock.patch.object(
            provider,
            "observe_ticket_creation",
            return_value=None,
        ):
            reconcile_uncertain_linear_ticket_creation(
                self.store,
                job,
                now=100,
                herdr_factory=mock.Mock,
                linear_factory=lambda: provider,
            )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.BLOCKED)
        self.assertEqual(
            updated.linear_ticket_create_operation_state,
            "manual_required",
        )

    def test_incomplete_linear_observation_stays_ambiguous(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "create a Linear ticket",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_provider": "linear",
                "linear_ticket_create_requested": True,
                "linear_ticket_create_confirmed": True,
                "linear_ticket_create_team": "API",
                "linear_ticket_create_team_id": "team-id-api",
                "linear_ticket_create_title": "Fix startup",
                "linear_ticket_create_description": "Startup fails.",
                "linear_ticket_create_marker": "a" * 32,
                "linear_ticket_create_operation_state": "ambiguous",
            }
        )
        provider = LinearIntegration()
        with mock.patch.object(
            provider,
            "observe_ticket_creation",
            side_effect=LinearError("Linear ticket creation could not be observed"),
        ):
            reconcile_uncertain_linear_ticket_creation(
                self.store,
                job,
                now=100,
                herdr_factory=mock.Mock,
                linear_factory=lambda: provider,
            )

        updated = self.store.get(job.id)
        self.assertEqual(updated.status, JobStatus.QUEUED)
        self.assertEqual(updated.linear_ticket_create_operation_state, "ambiguous")

    def test_migration_and_pruning_precede_recovery_scans(self) -> None:
        store = mock.Mock(spec=JobStore)
        calls: list[str] = []
        store.migrate_legacy.side_effect = lambda **_kwargs: (
            calls.append("migrate") or set()
        )
        store.prune.side_effect = lambda **_kwargs: calls.append("prune") or []
        store.list.side_effect = lambda: calls.append("scan") or []
        store.claim_outbox.return_value = None
        store.unconsumed_outbox_results.return_value = ()

        recover_jobs(store, launch_worker=mock.Mock(), now=100)

        self.assertEqual(
            calls,
            ["migrate", "prune", "scan", "scan", "scan", "scan"],
        )

    def test_recovery_does_not_scan_or_launch_during_maintenance(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
            }
        )
        lease = MaintenanceLease("maintenance", 1, 42, "boot", "start")
        self.store.begin_maintenance(
            lease,
            lambda _job: None,
            owner_alive=lambda _lease: False,
        )
        launch = mock.Mock()

        recover_jobs(self.store, launch_worker=launch, now=100)

        launch.assert_not_called()
        self.assertEqual(self.store.get("123456789abc").revision, 0)
        self.assertTrue(self.store.abort_maintenance(lease.token))

    def test_abandoned_queued_job_is_launched_once_across_recovery_passes(
        self,
    ) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
            }
        )
        launches: list[str] = []

        def launch(job_id: str) -> None:
            launches.append(job_id)
            self.store.update(
                job_id,
                lambda job: transition(
                    job,
                    JobStatus.QUEUED,
                    worker_token="claim",
                    worker_pid=42,
                    worker_boot_id="boot",
                    worker_process_start="start",
                    worker_claim_operation="test",
                    worker_claimed_at=1,
                ),
            )

        def is_alive(job: CursorJob) -> bool:
            return bool(job.worker_token)

        for now in (100.0, 101.0):
            recover_jobs(
                self.store,
                launch_worker=launch,
                is_worker_alive=is_alive,
                get_boot_identity=lambda: "boot",
                get_process_identity=lambda _pid: "start",
                now=now,
            )

        self.assertEqual(launches, ["123456789abc"])

    def test_restart_launch_retains_admitted_provider_and_harness(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_key": "ENG-42",
                "issue_provider": "linear",
            }
        )
        launched: list[tuple[str | None, str]] = []

        def launch(job_id: str) -> None:
            job = self.store.get(job_id)
            launched.append((job.issue_provider, job.harness_kind.value))

        recover_jobs(
            self.store,
            launch_worker=launch,
            require_issue_provider=lambda name: self.assertEqual(name, "linear"),
            now=100,
        )

        self.assertEqual(launched, [("linear", "cursor")])
        current = self.store.get("123456789abc")
        self.assertEqual(current.issue_provider, "linear")
        self.assertEqual(current.harness_kind.value, "cursor")

    def test_recovery_durably_fails_when_selected_provider_is_unavailable(
        self,
    ) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "issue_key": "ENG-42",
                "issue_provider": "linear",
            }
        )
        launch = mock.Mock()

        def unavailable(_name: str | None) -> None:
            raise ValueError("selected issue provider 'linear' is unavailable")

        recover_jobs(
            self.store,
            launch_worker=launch,
            require_issue_provider=unavailable,
            now=100,
        )

        launch.assert_not_called()
        failed = self.store.get("123456789abc")
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertFalse(failed.delivered)
        self.assertEqual(failed.issue_provider, "linear")
        self.assertIn(
            "Selected issue provider 'linear' is unavailable", failed.error or ""
        )

    def test_unsafe_live_legacy_worker_blocks_duplicate_launch(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
            }
        )
        self.store.legacy_dir.mkdir(exist_ok=True)
        source = self.store.legacy_dir / "123456789abc.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "test",
                    "status": "running",
                    "created_at": 1,
                    "delivered": False,
                    "worker_pid": 42,
                    "worker_process_start": "start",
                    "worker_claim_operation": "test",
                    "worker_claimed_at": 1,
                }
            )
        )
        launch = mock.Mock()

        with self.assertWarns(JobQuarantineWarning):
            recover_jobs(
                self.store,
                launch_worker=launch,
                inspect_legacy_worker=lambda _job: "unsafe",
                now=100,
            )

        launch.assert_not_called()
        self.assertTrue(source.exists())

    def test_unsafe_durable_legacy_owner_is_preserved_and_never_relaunched(
        self,
    ) -> None:
        self.store.durable_dir.mkdir()
        path = self.store.durable_dir / "123456789abc.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "legacy",
                    "status": "running",
                    "created_at": 1,
                    "delivered": False,
                    "worker_token": "legacy-claim",
                    "worker_pid": 42,
                    "worker_process_start": "start",
                    "worker_claim_operation": "test",
                    "worker_claimed_at": 1,
                }
            )
        )
        before = path.read_bytes()
        launch = mock.Mock()

        recover_jobs(
            self.store,
            launch_worker=launch,
            inspect_legacy_worker=lambda _job: "unsafe",
            now=100,
        )

        launch.assert_not_called()
        self.assertEqual(path.with_suffix(".json.imported").read_bytes(), before)
        recovered = self.store.get("123456789abc")
        self.assertEqual(
            recovered.loaded_schema_version,
            CURRENT_SCHEMA_VERSION,
        )
        self.assertEqual(recovered.status, JobStatus.RUNNING)
        self.assertEqual(recovered.worker_boot_id, "legacy-unknown")

    def test_clearing_dead_legacy_owner_preserves_terminal_reconciliation(
        self,
    ) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "reconciling",
                "request": "test",
                "created_at": 1,
                "delivered": True,
                "delivered_at": 3,
                "terminal_intent_status": "failed",
                "terminal_intent_completed_at": 2,
                "terminal_intent_error": "target cleanup failed",
                "terminal_intent_result": "target cleanup failed",
                "target_release_pending": True,
                "target_release_token": "release",
                "cancellation_reconciliation_pending": False,
                "worker_token": "legacy-claim",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "legacy:target_cleanup",
                "worker_claimed_at": 1,
                "worker_operation": "target_cleanup",
            }
        )
        launch = mock.Mock()

        recover_jobs(
            self.store,
            launch_worker=launch,
            inspect_legacy_worker=lambda _job: "stopped",
            is_worker_alive=lambda _job: False,
            get_boot_identity=lambda: None,
            get_process_identity=lambda _pid: None,
            now=100,
        )

        recovered = self.store.get("123456789abc")
        self.assertEqual(recovered.status, JobStatus.RECONCILING)
        self.assertEqual(recovered.terminal_intent_status, JobStatus.FAILED)
        self.assertEqual(recovered.terminal_intent_completed_at, 2)
        self.assertEqual(recovered.terminal_intent_error, "target cleanup failed")
        self.assertEqual(recovered.terminal_intent_result, "target cleanup failed")
        self.assertTrue(recovered.target_release_pending)
        self.assertEqual(recovered.target_release_token, "release")
        self.assertFalse(recovered.cancellation_reconciliation_pending)
        self.assertEqual(recovered.worker_operation, "target_cleanup")
        self.assertTrue(recovered.delivered)
        self.assertEqual(recovered.delivered_at, 3)
        self.assertIsNone(recovered.worker_token)
        self.assertIsNone(recovered.worker_pid)
        self.assertIsNone(recovered.worker_boot_id)
        self.assertIsNone(recovered.worker_process_start)
        launch.assert_not_called()

    def test_clearing_dead_legacy_owner_requeues_active_job_without_terminal_intent(
        self,
    ) -> None:
        self.create(
            {
                "id": "aaaaaaaaaaaa",
                "status": "running",
                "request": "test",
                "created_at": 1,
                "delivered": False,
                "worker_token": "legacy-claim",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "legacy:running",
                "worker_claimed_at": 1,
            }
        )
        launch = mock.Mock()

        recover_jobs(
            self.store,
            launch_worker=launch,
            inspect_legacy_worker=lambda _job: "stopped",
            is_worker_alive=lambda _job: False,
            now=100,
        )

        recovered = self.store.get("aaaaaaaaaaaa")
        self.assertEqual(recovered.status, JobStatus.QUEUED)
        self.assertIsNone(recovered.terminal_intent_status)
        self.assertIsNone(recovered.worker_token)
        self.assertIsNone(recovered.worker_pid)
        self.assertIsNone(recovered.worker_boot_id)
        self.assertIsNone(recovered.worker_process_start)
        launch.assert_called_once_with("aaaaaaaaaaaa")

    def test_all_uncertain_operations_reconcile_before_any_launch(self) -> None:
        base = {
            "status": "queued",
            "request": "test",
            "created_at": 1,
            "queued_at": 1,
            "delivered": False,
        }
        self.create(
            {
                **base,
                "id": "aaaaaaaaaaaa",
                "herdr_target": "agent",
                "agent_dispatch_state": "ambiguous",
            }
        )
        self.create(
            {
                **base,
                "id": "bbbbbbbbbbbb",
                "fork_operation_state": "submitted",
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_target": "me/project",
            }
        )
        checkout = self.root / "worktree"
        self.create(
            {
                **base,
                "id": "cccccccccccc",
                "repository": str(self.root / "repository"),
                "worktree_branch": "voice/task",
                "worktree_path": str(checkout),
                "worktree_provision_state": "dispatching",
            }
        )
        herdr = mock.Mock()
        herdr.get_agent.side_effect = HerdrError("observe", code="operation_timeout")
        herdr.run_json.side_effect = HerdrError("observe", code="operation_timeout")
        github = mock.Mock()
        github.inspect_public_repository.return_value = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github.reconcile_fork.side_effect = GitHubError("observe")
        launch = mock.Mock()

        recover_jobs(
            self.store,
            launch_worker=launch,
            herdr_factory=lambda: herdr,
            github_factory=lambda: github,
            now=100,
        )

        launch.assert_not_called()
        herdr.get_agent.assert_not_called()
        self.assertEqual(
            self.store.get("aaaaaaaaaaaa").agent_dispatch_state,
            "manual_required",
        )
        github.reconcile_fork.assert_called_once()
        herdr.run_json.assert_called_once()
        herdr.cancel_agent.assert_not_called()

    def test_committed_fork_remains_fenced_until_late_visibility(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "status": "reconciling",
                "request": "test",
                "created_at": 1,
                "terminal_intent_status": "failed",
                "terminal_intent_completed_at": 2,
                "terminal_intent_error": "fork visibility pending",
                "terminal_intent_result": "fork visibility pending",
                "delivered": True,
                "fork_committed": True,
                "fork_operation_state": "failed_observing",
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_target": "me/project",
                "target_release_pending": True,
                "target_release_token": "release",
            }
        )
        visible = GitHubRepository(
            "me/project",
            "https://github.com/me/project",
            False,
            "main",
            "source/project",
        )
        github = mock.Mock()
        github.inspect_public_repository.return_value = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github.reconcile_fork.side_effect = [None, None, None, None, None, visible]

        for observed_at in (100.0, 105.0, 115.0, 135.0, 175.0):
            reconcile_uncertain_fork(
                self.store,
                self.store.get("123456789abc"),
                now=observed_at,
                github_factory=lambda: github,
            )
            pending = self.store.get("123456789abc")
            self.assertNotEqual(
                pending.fork_operation_state,
                "confirmed_absent",
            )
            self.assertTrue(pending.target_release_pending)

        reconcile_uncertain_fork(
            self.store,
            self.store.get("123456789abc"),
            now=235,
            github_factory=lambda: github,
        )

        recovered = self.store.get("123456789abc")
        self.assertEqual(recovered.fork_operation_state, "exists")
        self.assertEqual(recovered.fork_repository, "me/project")
        github.ensure_fork.assert_not_called()

    def test_committed_fork_exhaustion_requires_manual_review_not_absence(
        self,
    ) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "status": "reconciling",
                "request": "test",
                "created_at": 1,
                "terminal_intent_status": "cancelled",
                "terminal_intent_completed_at": 2,
                "terminal_intent_result": "cancelled",
                "delivered": True,
                "fork_committed": True,
                "fork_operation_state": "failed_observing",
                "fork_operation_target": "me/project",
                "fork_reconcile_attempts": 5,
                "fork_absent_observations": 5,
                "target_release_pending": True,
                "target_release_token": "release",
            }
        )

        updated = self.store.update(
            job.id,
            lambda current: current.record_operation_observation(
                "fork",
                "fork_operation_state",
                frozenset({"failed_observing"}),
                now=100,
                observed_absent=True,
                failed_max_attempts=3,
                uncertain_max_attempts=6,
                base_seconds=5,
                max_seconds=60,
            ),
        )

        assert updated is not None
        self.assertEqual(updated.fork_operation_state, "manual_required")
        self.assertNotEqual(updated.fork_operation_state, "confirmed_absent")
        self.assertTrue(updated.target_release_pending)
        self.assertTrue(updated.delivered)

    def test_dead_cleanup_owner_is_fenced_and_taken_over_for_release(self) -> None:
        self.create(
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "id": "123456789abc",
                "revision": 0,
                "status": "reconciling",
                "request": "test",
                "created_at": 1,
                "delivered": False,
                "terminal_intent_status": "cancelled",
                "terminal_intent_result": "cancelled",
                "terminal_intent_completed_at": 2,
                "target_release_pending": True,
                "target_release_token": "stale-release",
                "target_release_owner_pid": 999,
                "target_release_owner_boot_id": "dead-boot",
                "target_release_owner_start": "dead-start",
                "herdr_target": "agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_dispatch_state": "ready",
            }
        )
        client = mock.Mock()
        launch = mock.Mock()

        recover_jobs(
            self.store,
            launch_worker=launch,
            herdr_factory=lambda: client,
            is_worker_alive=lambda _job: False,
            get_boot_identity=lambda: "live-boot",
            get_process_identity=lambda pid: (
                "live-start" if pid == os.getpid() else None
            ),
            now=100,
        )

        recovered = self.store.get("123456789abc")
        self.assertEqual(recovered.status, JobStatus.RECONCILING)
        self.assertTrue(recovered.target_release_pending)
        self.assertTrue(recovered.target_release_manual_required)
        client.close_owned_pane.assert_not_called()
        launch.assert_not_called()

    def test_wrong_checkout_stages_failure_until_cleanup_settles(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "parent_job_id": "aaaaaaaaaaaa",
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "repository": "/repo",
                "worktree_branch": "voice/task",
                "worktree_path": "/repo-worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "root-pane",
                "worktree_provision_state": "ready",
                "herdr_target": "agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_dispatch_state": "ambiguous",
                "agent_provider": "cursor/herdr",
                "agent_provider_session_id": "session",
                "agent_state_sequence": 1,
            }
        )
        client = mock.Mock()
        client.reconcile_session.return_value = SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession(
                "cursor/herdr",
                "session",
                "agent",
                2,
                {
                    "pane_id": "wrong-pane",
                    "workspace_id": "wrong-workspace",
                    "cwd": "/wrong-checkout",
                },
            ),
            "active",
            True,
        )

        reconcile_uncertain_agent(
            self.store,
            self.store.get("123456789abc"),
            now=100,
            herdr_factory=lambda: client,
        )

        staged = self.store.get("123456789abc")
        self.assertEqual(staged.status, JobStatus.RECONCILING)
        self.assertEqual(staged.terminal_intent_status, JobStatus.FAILED)
        self.assertTrue(staged.target_release_pending)
        self.assertIsNone(staged.completed_at)

    def test_materialized_failure_reconciliation_preserves_outcome(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "failed",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "error": "original failure",
                "result": "original failure",
                "delivered": True,
                "delivery_generation": 3,
                "agent_dispatch_state": "ambiguous",
                "herdr_target": "agent",
                "agent_reconcile_attempts": 5,
            }
        )

        updated = self.store.update(
            job.id,
            lambda current: current.record_operation_observation(
                "agent",
                "agent_dispatch_state",
                frozenset({"ambiguous"}),
                now=100,
                observed_absent=False,
                failed_max_attempts=3,
                uncertain_max_attempts=6,
                base_seconds=5,
                max_seconds=60,
            ),
        )

        assert updated is not None
        self.assertEqual(updated.status, JobStatus.FAILED)
        self.assertEqual(updated.result, "original failure")
        self.assertEqual(updated.error, "original failure")
        self.assertEqual(updated.completed_at, 2)
        self.assertEqual(updated.agent_dispatch_state, "manual_required")
        self.assertFalse(updated.delivered)
        self.assertEqual(updated.delivery_generation, 4)

    def test_terminal_manual_escalations_remain_preterminal(self) -> None:
        operations = {
            "agent": {
                "agent_dispatch_state": "ambiguous",
                "herdr_target": "agent",
            },
            "fork": {
                "fork_operation_state": "ambiguous",
                "fork_operation_target": "me/project",
            },
            "worktree": {
                "worktree_provision_state": "ambiguous",
                "repository": str(self.root / "repository"),
                "worktree_branch": "voice/task",
                "worktree_path": str(self.root / "worktree"),
            },
        }
        state_keys = {
            "agent": "agent_dispatch_state",
            "fork": "fork_operation_state",
            "worktree": "worktree_provision_state",
        }
        for index, (operation, fields) in enumerate(operations.items(), start=1):
            job_id = f"{index:012x}"
            self.create(
                {
                    "id": job_id,
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "revision": 0,
                    "status": "reconciling",
                    "request": "test",
                    "created_at": 1,
                    "terminal_intent_status": "cancelled",
                    "terminal_intent_completed_at": 2,
                    "terminal_intent_result": "cancelled",
                    "target_release_pending": True,
                    "target_release_token": "release",
                    "delivered": True,
                    "delivery_generation": 3,
                    f"{operation}_reconcile_attempts": 5,
                    **fields,
                }
            )

            updated = self.store.update(
                job_id,
                lambda job, operation=operation: job.record_operation_observation(
                    operation,
                    state_keys[operation],
                    frozenset({"ambiguous"}),
                    now=100,
                    observed_absent=False,
                    failed_max_attempts=3,
                    uncertain_max_attempts=6,
                    base_seconds=5,
                    max_seconds=60,
                ),
            )

            assert updated is not None
            self.assertEqual(updated.operation_state(operation), "manual_required")
            self.assertTrue(updated.delivered)
            self.assertEqual(updated.delivery_generation, 3)
            self.assertIsNone(
                delivery.claim_delivery(self.store, job_id, foreground=True, now=101)
            )

    def test_cancellation_release_is_idempotent_without_duplicate_cancel(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "result": "cancelled",
                "delivered": False,
                "herdr_target": "agent",
                "herdr_pane_id": "pane-agent",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "pane-anchor",
                "target_release_pending": True,
                "target_release_token": "release",
            }
        )
        client = mock.Mock()
        self.observe_owned_sessions(client)
        factory = mock.Mock(return_value=client)

        for _ in range(2):
            cancel_target_and_release(
                self.store,
                "123456789abc",
                "agent",
                "release",
                herdr_factory=factory,
            )

        client.close_owned_pane.assert_called_once_with(
            "agent", "pane-agent", "workspace"
        )
        self.assertFalse(self.store.get("123456789abc").target_release_pending)

    def test_cancellation_releases_every_workflow_participant(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "result": "cancelled",
                "delivered": False,
                "herdr_target": "implementer",
                "herdr_pane_id": "pane-implementer",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "pane-anchor",
                "planner_target": "planner",
                "reviewer_target": "reviewer",
                "implementer_target": "implementer",
                "agent_dispatch_state": "ready",
                "participant_session_owners": [
                    {
                        "provider": "cursor/herdr",
                        "session_id": "planner-session",
                        "target": "planner",
                        "state_sequence": 1,
                        "checkout": "/worktree",
                        "workspace_id": "workspace",
                        "pane_id": "pane-planner",
                    },
                    {
                        "provider": "cursor/herdr",
                        "session_id": "reviewer-session",
                        "target": "reviewer",
                        "state_sequence": 1,
                        "checkout": "/worktree",
                        "workspace_id": "workspace",
                        "pane_id": "pane-reviewer",
                    },
                ],
                "target_release_pending": True,
                "target_release_token": "release",
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = lambda target: {
            "name": target,
            "pane_id": f"pane-{target}",
            "workspace_id": "workspace",
            "cwd": "/worktree",
        }
        self.observe_owned_sessions(client)

        cancel_target_and_release(
            self.store,
            "123456789abc",
            "implementer",
            "release",
            herdr_factory=mock.Mock(return_value=client),
        )

        self.assertEqual(
            client.close_owned_pane.call_args_list,
            [
                mock.call("implementer", "pane-implementer", "workspace"),
                mock.call("planner", "pane-planner", "workspace"),
                mock.call("reviewer", "pane-reviewer", "workspace"),
            ],
        )
        self.assertFalse(self.store.get("123456789abc").target_release_pending)

    def test_uncertain_agent_absence_uses_backoff_then_releases_fence(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "result": "cancelled; reconciliation pending",
                "delivered": False,
                "herdr_target": "missing-agent",
                "agent_dispatch_state": "failed_observing",
                "agent_dispatch_exited": True,
                "target_release_pending": True,
                "cancellation_reconciliation_pending": True,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("not found", code="agent_not_found")

        for now in (100.0, 105.0, 115.0):
            reconcile_uncertain_agent(
                self.store,
                self.store.get("123456789abc"),
                now=now,
                herdr_factory=lambda: client,
            )

        recovered = self.store.get("123456789abc")
        self.assertEqual(recovered.agent_dispatch_state, "confirmed_absent")
        self.assertEqual(recovered.to_dict()["agent_reconcile_attempts"], 3)
        self.assertFalse(recovered.cancellation_reconciliation_pending)

    def test_manual_agent_absence_clears_selection_for_fresh_dispatch(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "failed",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "error": "manual reconciliation required",
                "result": "manual reconciliation required",
                "delivered": False,
                "herdr_target": "stale-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_name": "stale-name",
                "agent_dispatch_state": "manual_required",
                "agent_dispatch_exited": True,
                "agent_next_reconcile_at": 200,
                "manual_reconcile_operation": "agent",
                "manual_reconcile_token": "manual-token",
                "target_release_pending": True,
            }
        )

        resolved = resolve_manual_reconciliation(
            self.store,
            "123456789abc",
            "agent",
            "manual-token",
            "confirmed_absent",
            now=100,
        )

        self.assertEqual(resolved.agent_dispatch_state, "confirmed_absent")
        self.assertIsNone(resolved.herdr_target)
        self.assertIsNone(resolved.herdr_pane_id)
        self.assertIsNone(resolved.herdr_workspace_id)
        self.assertIsNone(resolved.agent_name)
        self.assertIsNone(resolved.to_dict().get("agent_dispatch_exited"))
        self.assertIsNone(resolved.to_dict().get("agent_next_reconcile_at"))

    def test_quarantine_acknowledgement_releases_worktree_path(self) -> None:
        checkout = self.root / "worktree"
        self.create(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "result": "cancelled",
                "delivered": True,
                "repository": str(self.root / "repository"),
                "worktree_branch": "voice/task",
                "worktree_path": str(checkout),
                "worktree_provision_state": "quarantined",
                "worktree_manual_inspection_required": True,
            }
        )

        acknowledge_worktree_quarantine(self.store, "123456789abc", now=100)

        acknowledged = self.store.get("123456789abc")
        self.assertEqual(acknowledged.worktree_provision_state, "ambiguous")
        self.assertFalse(acknowledged.to_dict()["worktree_manual_inspection_required"])

    def test_prompt_submit_recovery_requires_positive_acceptance_evidence(self) -> None:
        created = self.create(
            {
                "id": "123456789abc",
                "status": "running",
                "request": "test",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_phase": "classifying",
                "turn": 1,
                "workflow_turn_phase": "classifying",
                "herdr_target": "planner",
                "herdr_pane_id": "pane-planner",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "pane-anchor",
                "planner_target": "planner",
                "active_participant": "planner",
                "prompt_operation_state": "submitting",
                "prompt_operation_phase": "classifying",
                "prompt_operation_turn": 1,
                "prompt_operation_target": "planner",
                "prompt_operation_agent_session": "planner-session",
                "prompt_baseline_sequence": 7,
            }
        )
        self.admit_unknown_prompt_effect(created)
        unavailable = mock.Mock()
        unavailable.ensure_server.side_effect = HerdrError("offline")

        reconcile_prompt_and_pane_operations(
            self.store,
            self.store.get("123456789abc"),
            now=10,
            herdr_factory=lambda: unavailable,
        )
        self.assertEqual(
            self.store.get("123456789abc").prompt_operation_state,
            "submitting",
        )

        visible = mock.Mock()
        visible.reconcile_session.return_value = SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession(
                "cursor/herdr",
                "planner-session",
                "planner",
                8,
            ),
            "working",
            True,
        )
        visible.harness.reconcile.return_value = visible.reconcile_session.return_value
        reconcile_prompt_and_pane_operations(
            self.store,
            self.store.get("123456789abc"),
            now=11,
            herdr_factory=lambda: visible,
        )
        recovered = self.store.get("123456789abc")
        self.assertEqual(recovered.prompt_operation_state, "ambiguous")
        self.assertEqual(recovered.manual_reconcile_operation, "prompt")
        self.assertIsNotNone(recovered.manual_reconcile_token)

    def test_incomplete_native_prompt_identity_is_rejected_at_durable_write(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            JobValidationError, "prompt identity fields must not be empty"
        ):
            self.create(
                {
                    "id": "123456789abc",
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "revision": 0,
                    "status": "running",
                    "request": "test",
                    "created_at": 1,
                    "delivered": False,
                    "worker_token": "worker",
                    "worker_pid": 42,
                    "worker_boot_id": "boot",
                    "worker_process_start": "start",
                    "worker_claim_operation": "test",
                    "worker_claimed_at": 1,
                    "workflow_phase": "classifying",
                    "turn": 1,
                    "workflow_turn_phase": "classifying",
                    "herdr_target": "planner",
                    "planner_target": "planner",
                    "active_participant": "planner",
                    "prompt_operation_state": "submitting",
                    "prompt_operation_phase": "classifying",
                    "prompt_operation_turn": 1,
                    "prompt_operation_target": "planner",
                    "prompt_baseline_sequence": -1,
                }
            )

    def test_plan_approval_recovery_rejects_replaced_agent_session(self) -> None:
        created = self.create(
            {
                "id": "123456789abc",
                "status": "running",
                "request": "implement reviewed behavior",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "medium",
                "workflow_classification_reason": "cross-component",
                "workflow_phase": "reviewing",
                "turn": 4,
                "turn_token": "123456789abc-4",
                "workflow_turn_phase": "reviewing",
                "herdr_target": "planner",
                "herdr_pane_id": "pane-planner",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "pane-anchor",
                "planner_target": "planner",
                "active_participant": "planner",
                "plan_approval_state": "boundary",
                "plan_approval_id": "gate-id",
                "plan_approval_agent_session": "original-session",
                "plan_approval_state_change_sequence": 7,
                "plan_approval_revision": 3,
            }
        )
        plan = "Implement reviewed behavior."
        plan_reference = self.store.write_artifact(created.id, "plan", 0, plan)
        review_reference = self.store.write_artifact(
            created.id,
            "review",
            0,
            "The plan is safe.",
            source_text=plan,
        )
        submitting = self.store.update(
            created.id,
            lambda current: current.evolve(
                workflow_phase="implementing",
                workflow_turn_phase="implementing",
                plan_artifact=plan_reference,
                review_artifact=review_reference,
                review_approved=True,
                review_decision="approve",
                review_approval_source="reviewer",
                plan_approval_state="approved",
                plan_approval_source="explicit",
                plan_approval_plan_artifact=plan_reference,
                plan_approval_review_artifact=review_reference,
                prompt_operation_state="submitting",
                prompt_operation_phase="implementing",
                prompt_operation_turn=4,
                prompt_operation_target="planner",
                prompt_operation_agent_session="original-session",
                prompt_baseline_sequence=7,
            ),
        )
        assert submitting is not None
        submitting = self.admit_unknown_prompt_effect(submitting)
        replacement = mock.Mock()
        replacement.get_agent.return_value = {
            "state_change_seq": 8,
            "agent_session": "replacement-session",
        }
        replacement.harness.reconcile.return_value = SessionReconciliation(
            ReconciliationState.CHANGED,
            HarnessSession(
                "cursor/herdr",
                "replacement-session",
                "planner",
                8,
            ),
            "working",
            False,
        )

        reconcile_prompt_and_pane_operations(
            self.store,
            submitting,
            now=10,
            herdr_factory=lambda: replacement,
        )

        recovered = self.store.get(created.id)
        self.assertEqual(recovered.prompt_operation_state, "ambiguous")
        self.assertFalse(recovered.plan_approval_counted)

    def test_uncertain_pane_creation_becomes_manual_without_retry(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "routing",
                "request": "test",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "participant_creation_state": "submitting",
                "participant_creation_participant": "reviewer",
                "participant_creation_target": "reviewer",
                "participant_creation_label": "task-reviewer",
                "participant_creation_workspace_id": "workspace",
            }
        )

        reconcile_prompt_and_pane_operations(
            self.store,
            self.store.get("123456789abc"),
            now=10,
        )

        updated = self.store.get("123456789abc")
        self.assertEqual(updated.participant_creation_state, "manual_required")
        self.assertEqual(updated.manual_reconcile_operation, "pane")
        self.assertTrue(updated.manual_reconcile_token)

    def test_reviewer_pane_fence_survives_terminal_intent_and_cleanup(self) -> None:
        created = self.create(
            {
                "id": "123456789abc",
                "status": "running",
                "request": "review a plan",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "herdr_target": "planner",
                "herdr_pane_id": "pane-planner",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "pane-anchor",
                "planner_target": "planner",
                "active_participant": "planner",
                "agent_dispatch_state": "ready",
                "participant_creation_state": "manual_required",
                "participant_creation_participant": "reviewer",
                "participant_creation_target": "reviewer-pending",
                "participant_creation_label": "task-reviewer",
                "participant_creation_workspace_id": "workspace",
                "manual_reconcile_operation": "pane",
                "manual_reconcile_token": "pane-fence",
                "manual_reconcile_required_at": 9,
            }
        )
        staged = self.store.update(
            created.id,
            lambda job: stage_terminal_intent(
                job,
                JobStatus.FAILED,
                now=10,
                result="Reviewer startup failed.",
                error="pane request timed out",
            ),
        )
        assert staged is not None and staged.target_release_token
        client = mock.Mock()
        self.observe_owned_sessions(client)

        cancel_target_and_release(
            self.store,
            created.id,
            "planner",
            staged.target_release_token,
            herdr_factory=lambda: client,
        )

        current = self.store.get(created.id)
        self.assertEqual(current.status, JobStatus.RECONCILING)
        self.assertEqual(current.terminal_intent_status, JobStatus.FAILED)
        self.assertEqual(current.participant_creation_state, "manual_required")
        self.assertEqual(current.manual_reconcile_operation, "pane")
        self.assertEqual(current.manual_reconcile_token, "pane-fence")
        self.assertEqual(current.participant_creation_target, "reviewer-pending")
        self.assertIn("confirmed absent", current.terminal_intent_result or "")
        client.close_owned_pane.assert_called_once_with(
            "planner", "pane-planner", "workspace"
        )

    def test_materialized_reviewer_pane_identity_allows_terminal_cleanup(self) -> None:
        created = self.create(
            {
                "id": "123456789abc",
                "status": "running",
                "request": "review a plan",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "herdr_target": "planner",
                "herdr_pane_id": "pane-planner",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "pane-anchor",
                "planner_target": "planner",
                "active_participant": "planner",
                "agent_dispatch_state": "ready",
                "participant_creation_state": "manual_required",
                "participant_creation_participant": "reviewer",
                "participant_creation_target": "reviewer-pending",
                "participant_creation_label": "task-reviewer",
                "participant_creation_workspace_id": "workspace",
                "manual_reconcile_operation": "pane",
                "manual_reconcile_token": "pane-fence",
            }
        )
        staged = self.store.update(
            created.id,
            lambda job: stage_terminal_intent(
                job,
                JobStatus.FAILED,
                now=10,
                result="Reviewer startup failed.",
                error="pane request timed out",
            ),
        )
        assert staged is not None and staged.target_release_token

        resolved = resolve_manual_reconciliation(
            self.store,
            created.id,
            "pane",
            "pane-fence",
            "materialized",
            now=11,
            pane_id="pane-reviewer",
            workspace_id="workspace",
        )

        self.assertEqual(resolved.participant_creation_state, "created")
        self.assertEqual(resolved.participant_creation_pane_id, "pane-reviewer")
        self.assertEqual(resolved.herdr_pane_id, "pane-reviewer")
        client = mock.Mock()
        client.get_agent.return_value = {
            "name": "planner",
            "pane_id": "pane-planner",
            "workspace_id": "workspace",
            "cwd": "/worktree",
        }
        self.observe_owned_sessions(client)
        cancel_target_and_release(
            self.store,
            created.id,
            "planner",
            resolved.target_release_token or "",
            herdr_factory=lambda: client,
        )
        finished = self.store.get(created.id)
        self.assertEqual(finished.status, JobStatus.FAILED)
        self.assertIsNone(finished.manual_reconcile_operation)
        self.assertEqual(
            {call.args for call in client.close_owned_pane.call_args_list},
            {
                ("planner", "pane-planner", "workspace"),
                ("reviewer-pending", "pane-reviewer", "workspace"),
            },
        )

    def test_materialized_worktree_records_required_workspace_identity(self) -> None:
        created = self.create(
            {
                "id": "123456789abc",
                "status": "routing",
                "request": "recover a worktree",
                "created_at": 1,
                "delivered": False,
                "repository": "/repo",
                "worktree_branch": "voice/recovery",
                "worktree_path": "/worktree",
                "worktree_provision_state": "manual_required",
                "manual_reconcile_operation": "worktree",
                "manual_reconcile_token": "worktree-fence",
                "manual_reconcile_required_at": 9,
            }
        )

        resolved = resolve_manual_reconciliation(
            self.store,
            created.id,
            "worktree",
            "worktree-fence",
            "materialized",
            now=11,
            pane_id="root-pane",
            workspace_id="workspace",
        )

        self.assertEqual(resolved.worktree_provision_state, "retained")
        self.assertEqual(resolved.worktree_root_pane_id, "root-pane")
        self.assertEqual(resolved.worktree_workspace_id, "workspace")
        self.assertIsNone(resolved.manual_reconcile_operation)

    def test_concurrent_terminal_intent_fences_reviewer_pane_acceptance(self) -> None:
        created = self.create(
            {
                "id": "123456789abc",
                "status": "running",
                "request": "review a plan",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "herdr_target": "planner",
                "planner_target": "planner",
                "active_participant": "planner",
                "agent_dispatch_state": "ready",
            }
        )
        planned = provisioning._plan_participant_creation(
            self.store,
            created,
            WORKER,
            WorkflowParticipant.REVIEWER,
            target="reviewer-pending",
            label="task-reviewer",
            workspace_id="workspace",
        )
        before_submit, accepted, _revision = provisioning._participant_pane_callbacks(
            self.store,
            planned.id,
            WORKER,
            "reviewer-pending",
            revision_state=[planned.revision],
        )
        before_submit()
        staged = self.store.update(
            created.id,
            lambda job: stage_terminal_intent(
                job,
                JobStatus.FAILED,
                now=10,
                result="Reviewer startup failed.",
                error="worker exited",
            ),
        )
        assert staged is not None

        with self.assertRaises(provisioning.WorkerCancelled):
            accepted("pane-reviewer", "workspace")
        recover_jobs(
            self.store,
            launch_worker=mock.Mock(),
            is_worker_alive=lambda _job: False,
            now=11,
        )

        recovered = self.store.get(created.id)
        self.assertEqual(recovered.participant_creation_state, "manual_required")
        self.assertEqual(recovered.manual_reconcile_operation, "pane")
        self.assertEqual(recovered.participant_creation_target, "reviewer-pending")
        self.assertEqual(recovered.participant_creation_label, "task-reviewer")
        self.assertIsNone(recovered.participant_creation_pane_id)
        self.assertIsNone(recovered.worker_token)
        self.assertIn("Inspect Herdr", recovered.terminal_intent_error or "")

    def test_worker_logs_terminal_cleanup_failure(self) -> None:
        created = self.create(
            {
                "id": "123456789abc",
                "status": "running",
                "request": "review a plan",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
            }
        )
        staged = self.store.update(
            created.id,
            lambda job: stage_terminal_intent(
                job,
                JobStatus.FAILED,
                now=10,
                result="failed",
                error="original failure",
            ),
        )
        assert staged is not None
        context = worker_lifecycle.WorkerContext(
            self.store,
            staged,
            "worker",
            threading.Event(),
        )
        factories = provisioning.ClientFactories(
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
        )
        diagnostic = io.StringIO()

        with (
            mock.patch.object(
                provisioning,
                "require_issue_provider",
                side_effect=provisioning.WorkerCancelled,
            ),
            mock.patch.object(
                provisioning.recovery,
                "cancel_target_and_release",
                side_effect=RuntimeError("cleanup exploded"),
            ),
            redirect_stderr(diagnostic),
        ):
            provisioning.run_claimed_worker(context, factories)

        logged = diagnostic.getvalue()
        self.assertIn("terminal cleanup failed", logged)
        self.assertIn("RuntimeError: cleanup exploded", logged)

    def test_terminal_intent_cancels_every_target_before_publication(self) -> None:
        for job_id, terminal in (
            ("123456789abc", JobStatus.COMPLETED),
            ("bbbbbbbbbbbb", JobStatus.FAILED),
            ("cccccccccccc", JobStatus.CANCELLED),
        ):
            with self.subTest(terminal=terminal):
                job = self.create(
                    {
                        "id": job_id,
                        "status": "running",
                        "request": "test",
                        "created_at": 1,
                        "delivered": False,
                        "worker_token": f"worker-{job_id}",
                        "worker_pid": 42,
                        "worker_boot_id": "boot",
                        "worker_process_start": f"start-{job_id}",
                        "worker_claim_operation": "test",
                        "worker_claimed_at": 1,
                        "workflow_phase": "implementing",
                        "workflow_tier": "simple",
                        "workflow_classification_reason": "localized",
                        "herdr_target": f"implementer-{job_id}",
                        "herdr_pane_id": f"pane-implementer-{job_id}",
                        "herdr_workspace_id": "workspace",
                        "worktree_path": "/worktree",
                        "worktree_workspace_id": "workspace",
                        "worktree_root_pane_id": "pane-anchor",
                        "active_participant": "implementer",
                        "planner_target": f"planner-{job_id}",
                        "reviewer_target": f"reviewer-{job_id}",
                        "implementer_target": f"implementer-{job_id}",
                        "agent_dispatch_state": "ready",
                        "participant_session_owners": [
                            {
                                "provider": "cursor/herdr",
                                "session_id": f"planner-session-{job_id}",
                                "target": f"planner-{job_id}",
                                "state_sequence": 1,
                                "checkout": "/worktree",
                                "workspace_id": "workspace",
                                "pane_id": f"pane-planner-{job_id}",
                            },
                            {
                                "provider": "cursor/herdr",
                                "session_id": f"reviewer-session-{job_id}",
                                "target": f"reviewer-{job_id}",
                                "state_sequence": 1,
                                "checkout": "/worktree",
                                "workspace_id": "workspace",
                                "pane_id": f"pane-reviewer-{job_id}",
                            },
                        ],
                    }
                )

                def terminalize(
                    current: CursorJob,
                    intended: JobStatus = terminal,
                ) -> CursorJob:
                    return stage_terminal_intent(
                        current,
                        intended,
                        now=10,
                        result=intended.value,
                        error=("failed" if intended == JobStatus.FAILED else None),
                    )

                staged = self.store.update(
                    job.id,
                    terminalize,
                )
                assert staged is not None and staged.target_release_token
                client = mock.Mock()
                client.get_agent.side_effect = lambda target: {
                    "name": target,
                    "pane_id": f"pane-{target}",
                    "workspace_id": "workspace",
                    "cwd": "/worktree",
                }
                self.observe_owned_sessions(client, job.id)

                cancel_target_and_release(
                    self.store,
                    job.id,
                    f"implementer-{job_id}",
                    staged.target_release_token,
                    herdr_factory=lambda current_client=client: current_client,
                )

                self.assertEqual(
                    {call.args[0] for call in client.close_owned_pane.call_args_list},
                    {
                        f"planner-{job_id}",
                        f"reviewer-{job_id}",
                        f"implementer-{job_id}",
                    },
                )
                completed = self.store.get(job.id)
                self.assertEqual(completed.status, terminal)
                self.assertIsNone(completed.herdr_target)
                for participant in WorkflowParticipant:
                    self.assertIsNone(completed.participant_target(participant))

    def test_terminal_intent_with_ambiguous_prompt_releases_ticket_fences(
        self,
    ) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "status": "running",
                "request": "work on issue 56",
                "created_at": 1,
                "delivered": False,
                "github_repository": "Example/Project",
                "github_issue": 56,
                "herdr_target": "planner-agent",
                "planner_target": "planner-agent",
                "active_participant": "planner",
                "agent_dispatch_state": "ready",
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "root-pane",
                "worker_token": "worker-token",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
            }
        )

        def terminalize(current: CursorJob) -> CursorJob:
            return stage_terminal_intent(
                current,
                JobStatus.CANCELLED,
                now=10,
                result="Cursor job 123456789abc was cancelled.",
                clear_worker=True,
            )

        staged = self.store.update(job.id, terminalize)
        assert staged is not None and staged.target_release_token

        def mark_prompt_ambiguous(current: CursorJob) -> CursorJob:
            return current.evolve(
                prompt_operation=AmbiguousPrompt(
                    PromptIdentity(
                        current.id,
                        "planning",
                        1,
                        f"{current.id}-1",
                        "planner-agent",
                        "session-planner",
                        0,
                    )
                ),
                manual_reconcile_operation="prompt",
                manual_reconcile_token="manual-token",
                manual_reconcile_required_at=10,
            )

        staged = self.store.update(job.id, mark_prompt_ambiguous)
        assert staged is not None and staged.target_release_token

        client = mock.Mock()
        self.observe_owned_sessions(client)
        cancel_target_and_release(
            self.store,
            job.id,
            "planner-agent",
            staged.target_release_token,
            herdr_factory=lambda: client,
        )

        released = self.store.get(job.id)
        self.assertEqual(released.status, JobStatus.CANCELLED)
        self.assertFalse(released.target_release_pending)
        self.assertFalse(released.cancellation_reconciliation_pending)
        self.assertIsNone(released.manual_reconcile_operation)
        self.assertEqual(released.prompt_operation_state, "none")

    def test_typed_session_reconciliation_rejects_stale_sequence(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "worktree_path": "/worktree",
                "herdr_target": "agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_dispatch_state": "ambiguous",
                "agent_provider": "cursor/herdr",
                "agent_provider_session_id": "session",
                "agent_state_sequence": 7,
            }
        )
        client = mock.Mock()
        client.reconcile_session.return_value = SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession("cursor/herdr", "session", "agent", 6),
            "working",
            True,
        )

        reconcile_uncertain_agent(
            self.store,
            job,
            now=10,
            herdr_factory=lambda: client,
        )

        self.assertEqual(
            self.store.get(job.id).agent_dispatch_state,
            "ambiguous",
        )
        client.reconcile_session.assert_called_once_with(
            "agent",
            expected_session_id="session",
        )

    def test_session_reconciliation_cannot_advance_same_target_replacement(
        self,
    ) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "worktree_path": "/worktree-a",
                "herdr_target": "agent",
                "herdr_pane_id": "pane-a",
                "herdr_workspace_id": "workspace-a",
                "agent_dispatch_state": "ambiguous",
                "agent_provider": "cursor/herdr",
                "agent_provider_session_id": "session-a",
                "agent_state_sequence": 7,
            }
        )
        client = mock.Mock()

        def replace_before_observation(*_args: object, **_kwargs: object) -> object:
            current = self.store.get(job.id)
            operation = current.agent_session_operation
            assert operation is not None
            absent = operation.transition(AgentSessionState.CONFIRMED_ABSENT)
            confirmed = self.store.update(
                current.id,
                lambda latest: latest.evolve(agent_session_operation=absent),
            )
            assert confirmed is not None
            replacement = AgentSessionOperation(
                AgentSessionState.DISPATCHING,
                AgentSessionSpec("agent", "/worktree-b", "workspace-b", "pane-b"),
            )
            self.store.update(
                current.id,
                lambda latest: latest.evolve(
                    agent_session_operation=replacement,
                    worktree_path="/worktree-b",
                    herdr_workspace_id="workspace-b",
                    herdr_pane_id="pane-b",
                    agent_provider=None,
                    agent_provider_session_id=None,
                    agent_state_sequence=None,
                ),
            )
            return SessionReconciliation(
                ReconciliationState.ACTIVE,
                HarnessSession(
                    "cursor/herdr",
                    "session-a",
                    "agent",
                    8,
                    metadata={
                        "cwd": "/worktree-a",
                        "workspace_id": "workspace-a",
                        "pane_id": "pane-a",
                    },
                ),
                "working",
                True,
            )

        client.reconcile_session.side_effect = replace_before_observation

        reconcile_uncertain_agent(
            self.store,
            job,
            now=10,
            herdr_factory=lambda: client,
        )

        current = self.store.get(job.id)
        self.assertEqual(current.agent_dispatch_state, "dispatching")
        self.assertEqual(current.herdr_workspace_id, "workspace-b")
        current_operation = current.agent_session_operation
        self.assertIsNotNone(current_operation)
        assert current_operation is not None
        self.assertIsNone(current_operation.session)

    def test_typed_cleanup_never_closes_mismatched_checkout(self) -> None:
        job = self.create(
            {
                "id": "123456789abc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "status": "cancelled",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "result": "cancelled",
                "delivered": False,
                "worktree_path": "/worktree",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "root-pane",
                "herdr_target": "agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_dispatch_state": "ready",
                "agent_provider": "cursor/herdr",
                "agent_provider_session_id": "session",
                "agent_state_sequence": 7,
                "target_release_pending": True,
                "target_release_token": "release",
            }
        )
        client = mock.Mock()
        client.get_agent.return_value = {
            "name": "agent",
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/different-worktree",
        }

        cancel_target_and_release(
            self.store,
            job.id,
            "agent",
            "release",
            herdr_factory=lambda: client,
        )

        client.close_owned_pane.assert_not_called()
        self.assertTrue(self.store.get(job.id).target_release_pending)


if __name__ == "__main__":
    unittest.main()
