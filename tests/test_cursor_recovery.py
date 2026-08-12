from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor import delivery, provisioning, worker_lifecycle
from local_voice_harness.cursor.model import (
    CursorJob,
    JobStatus,
    WorkflowParticipant,
    transition,
)
from local_voice_harness.cursor.recovery import (
    acknowledge_worktree_quarantine,
    cancel_target_and_release,
    reconcile_prompt_and_pane_operations,
    reconcile_uncertain_agent,
    reconcile_uncertain_fork,
    recover_jobs,
    resolve_manual_reconciliation,
    stage_terminal_intent,
)
from local_voice_harness.cursor.store import (
    JobQuarantineWarning,
    JobStore,
    MaintenanceLease,
)
from local_voice_harness.integrations.github import GitHubError, GitHubRepository
from local_voice_harness.integrations.herdr import HerdrError


class CursorRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = JobStore(self.root / "jobs", self.root / "legacy")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self, values: dict[str, object]) -> CursorJob:
        return self.store.create(CursorJob.from_dict(values))

    def test_migration_and_pruning_precede_recovery_scans(self) -> None:
        store = mock.Mock(spec=JobStore)
        calls: list[str] = []
        store.migrate_legacy.side_effect = lambda **_kwargs: (
            calls.append("migrate") or set()
        )
        store.prune.side_effect = lambda **_kwargs: calls.append("prune") or []
        store.list.side_effect = lambda: calls.append("scan") or []

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
        self.assertEqual(self.store.get("123456789abc").loaded_schema_version, 5)

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
        herdr.get_agent.assert_called_once_with("agent")
        github.reconcile_fork.assert_called_once()
        herdr.run_json.assert_called_once()
        herdr.cancel_agent.assert_not_called()

    def test_committed_fork_remains_fenced_until_late_visibility(self) -> None:
        self.create(
            {
                "id": "123456789abc",
                "status": "failed",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "error": "fork visibility pending",
                "result": "fork visibility pending",
                "delivered": True,
                "fork_committed": True,
                "fork_operation_state": "failed_observing",
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_target": "me/project",
                "target_release_pending": True,
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
                "status": "cancelled",
                "request": "test",
                "created_at": 1,
                "completed_at": 2,
                "result": "cancelled",
                "delivered": True,
                "fork_committed": True,
                "fork_operation_state": "failed_observing",
                "fork_operation_target": "me/project",
                "fork_reconcile_attempts": 5,
                "fork_absent_observations": 5,
                "target_release_pending": True,
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
        self.assertFalse(updated.delivered)

    def test_terminal_manual_escalations_create_at_least_once_delivery(self) -> None:
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
                    "status": "cancelled",
                    "request": "test",
                    "created_at": 1,
                    "completed_at": 2,
                    "result": "cancelled",
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
            self.assertFalse(updated.delivered)
            self.assertEqual(updated.delivery_generation, 4)
            self.assertIn("manual reconciliation required", updated.result or "")
            first = delivery.claim_delivery(
                self.store, job_id, foreground=True, now=101
            )
            assert first is not None
            self.assertTrue(
                delivery.release_delivery(
                    self.store,
                    job_id,
                    first.token,
                    retry=False,
                    now=101,
                )
            )
            second = delivery.claim_delivery(
                self.store, job_id, foreground=True, now=102
            )
            assert second is not None
            self.assertNotEqual(first.token, second.token)

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
                "target_release_pending": True,
                "target_release_token": "release",
            }
        )
        client = mock.Mock()
        factory = mock.Mock(return_value=client)

        for _ in range(2):
            cancel_target_and_release(
                self.store,
                "123456789abc",
                "agent",
                "release",
                herdr_factory=factory,
            )

        client.cancel_agent.assert_called_once_with("agent")
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
                "planner_target": "planner",
                "reviewer_target": "reviewer",
                "implementer_target": "implementer",
                "target_release_pending": True,
                "target_release_token": "release",
            }
        )
        client = mock.Mock()

        cancel_target_and_release(
            self.store,
            "123456789abc",
            "implementer",
            "release",
            herdr_factory=mock.Mock(return_value=client),
        )

        self.assertEqual(
            client.cancel_agent.call_args_list,
            [mock.call("implementer"), mock.call("planner"), mock.call("reviewer")],
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
        self.assertEqual(acknowledged.worktree_provision_state, "retained")
        self.assertFalse(acknowledged.to_dict()["worktree_manual_inspection_required"])

    def test_prompt_submit_recovery_requires_positive_acceptance_evidence(self) -> None:
        self.create(
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
                "prompt_operation_agent_session": "planner-session",
                "prompt_baseline_sequence": 7,
            }
        )
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
        visible.get_agent.return_value = {
            "state_change_seq": 8,
            "agent_session": "planner-session",
        }
        reconcile_prompt_and_pane_operations(
            self.store,
            self.store.get("123456789abc"),
            now=11,
            herdr_factory=lambda: visible,
        )
        self.assertEqual(
            self.store.get("123456789abc").prompt_operation_state,
            "submitted",
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
                "workflow_tier": "medium",
                "workflow_classification_reason": "cross-component",
                "workflow_phase": "reviewing",
                "turn": 4,
                "turn_token": "123456789abc-4",
                "workflow_turn_phase": "reviewing",
                "herdr_target": "planner",
                "planner_target": "planner",
                "active_participant": "planner",
                "plan_approval_state": "boundary",
                "plan_approval_id": "gate-id",
                "plan_approval_agent_session": "original-session",
                "plan_approval_state_change_sequence": 7,
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
                prompt_operation_state="submitting",
                prompt_operation_phase="implementing",
                prompt_operation_turn=4,
                prompt_operation_target="planner",
                prompt_operation_agent_session="original-session",
                prompt_baseline_sequence=7,
            ),
        )
        assert submitting is not None
        replacement = mock.Mock()
        replacement.get_agent.return_value = {
            "state_change_seq": 8,
            "agent_session": "replacement-session",
        }

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
                "participant_creation_state": "submitting",
                "participant_creation_participant": "reviewer",
                "participant_creation_target": "reviewer",
                "participant_creation_label": "task-reviewer",
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
                "herdr_target": "planner",
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
        client.cancel_agent.assert_called_once_with("planner")

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
                "herdr_target": "planner",
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
            {call.args[0] for call in client.cancel_agent.call_args_list},
            {"planner", "reviewer-pending"},
        )

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
                "herdr_target": "planner",
                "planner_target": "planner",
                "active_participant": "planner",
                "agent_dispatch_state": "ready",
            }
        )
        planned = provisioning._plan_participant_creation(
            self.store,
            created,
            "worker",
            WorkflowParticipant.REVIEWER,
            target="reviewer-pending",
            label="task-reviewer",
            workspace_id="workspace",
        )
        before_submit, accepted = provisioning._participant_pane_callbacks(
            self.store,
            planned.id,
            "worker",
            "reviewer-pending",
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
                        "workflow_phase": "implementing",
                        "workflow_tier": "simple",
                        "workflow_classification_reason": "localized",
                        "herdr_target": f"implementer-{job_id}",
                        "active_participant": "implementer",
                        "planner_target": f"planner-{job_id}",
                        "reviewer_target": f"reviewer-{job_id}",
                        "implementer_target": f"implementer-{job_id}",
                        "agent_dispatch_state": "ready",
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

                cancel_target_and_release(
                    self.store,
                    job.id,
                    f"implementer-{job_id}",
                    staged.target_release_token,
                    herdr_factory=lambda current_client=client: current_client,
                )

                self.assertEqual(
                    {call.args[0] for call in client.cancel_agent.call_args_list},
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
                "worker_token": "worker-token",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
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
                prompt_operation_state="ambiguous",
                prompt_operation_phase="planning",
                prompt_operation_turn=1,
                prompt_operation_target="planner-agent",
                prompt_baseline_sequence=0,
                manual_reconcile_operation="prompt",
                manual_reconcile_token="manual-token",
                manual_reconcile_required_at=10,
            )

        staged = self.store.update(job.id, mark_prompt_ambiguous)
        assert staged is not None and staged.target_release_token

        cancel_target_and_release(
            self.store,
            job.id,
            "planner-agent",
            staged.target_release_token,
            herdr_factory=mock.Mock(),
        )

        released = self.store.get(job.id)
        self.assertEqual(released.status, JobStatus.CANCELLED)
        self.assertFalse(released.target_release_pending)
        self.assertFalse(released.cancellation_reconciliation_pending)
        self.assertIsNone(released.manual_reconcile_operation)
        self.assertEqual(released.prompt_operation_state, "none")


if __name__ == "__main__":
    unittest.main()
