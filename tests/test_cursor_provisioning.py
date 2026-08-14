from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest import mock

from local_voice_harness import user_config
from local_voice_harness.agents.harness import (
    HarnessEvent,
    HarnessEventKind,
    HarnessSession,
    HarnessTask,
    ReconciliationState,
    SessionReconciliation,
    SessionRequest,
    TaskSubmission,
)
from local_voice_harness.cursor import provisioning as production_jobs
from local_voice_harness.cursor import service, worker_lifecycle
from local_voice_harness.cursor.delivery import DeliveryClaim, DeliveryClaims
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    WorkflowParticipant,
    WorkflowPhase,
)
from local_voice_harness.cursor.operations import WorkerOwnership
from local_voice_harness.cursor.recovery import stage_terminal_intent
from local_voice_harness.cursor.store import JobQuarantineWarning, JobStore
from local_voice_harness.github_issue_creation import GitHubIssueDraft
from local_voice_harness.integrations import linear
from local_voice_harness.integrations.github import (
    GitHubIssue,
    GitHubIssueCreationResult,
    GitHubOperationAmbiguous,
    GitHubRepository,
    ProvisionedIssue,
)
from local_voice_harness.integrations.herdr import (
    AgentSelection,
    HerdrError,
    PromptOutcome,
)
from local_voice_harness.integrations.linear import (
    LinearError,
    LinearIssue,
    LinearOperationAmbiguous,
    LinearTeam,
    LinearTicketCreationResult,
)
from local_voice_harness.linear_ticket_creation import LinearTicketDraft
from local_voice_harness.local_git import LocalGitRefChanged

WORKER = WorkerOwnership("worker", 42, "boot", "start", "test", 1)
WORKER_2 = WorkerOwnership("worker-2", 43, "boot-2", "start-2", "test", 1)
LEGACY_WORKER = WorkerOwnership("worker", 42, "test-boot", "start", "test", 1)


class _ProvisioningTestAdapter:
    """Legacy-shaped helpers kept only in tests while exercising typed production APIs."""

    JOBS_DIR = Path()
    LEGACY_JOBS_DIR = Path()

    def __init__(self) -> None:
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "HerdrClient", production_jobs.HerdrClient)
        object.__setattr__(self, "GitHubClient", production_jobs.GitHubClient)
        object.__setattr__(self, "HarnessError", production_jobs.HarnessError)
        object.__setattr__(self, "_process_identity", worker_lifecycle.process_identity)
        object.__setattr__(self, "_boot_identity", worker_lifecycle.boot_identity)
        worker_alive = service._worker_is_alive
        object.__setattr__(
            self,
            "_worker_is_alive",
            lambda job: worker_alive(self._coerce(job)),
        )
        object.__setattr__(
            self,
            "_stop_worker",
            lambda job, timeout=2.0: worker_lifecycle.stop_worker(
                self._coerce(job),
                timeout,
                is_alive=lambda _current: self._worker_is_alive(job),
                send_signal=lambda _current, sent_signal, **kwargs: self._signal_worker(
                    job, sent_signal, **kwargs
                ),
            ),
        )
        object.__setattr__(self, "_stop_legacy_worker", service._stop_legacy_worker)
        object.__setattr__(
            self,
            "_settle_fork_operation",
            production_jobs._settle_fork_operation,
        )

    def __getattr__(self, name: str) -> Any:
        if name in {"_worker_is_alive", "_stop_worker", "_stop_legacy_worker"}:
            return getattr(service, name)
        if name == "_process_identity":
            return worker_lifecycle.process_identity
        if name == "_boot_identity":
            return worker_lifecycle.boot_identity
        return getattr(production_jobs, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"JOBS_DIR", "LEGACY_JOBS_DIR"}:
            object.__setattr__(self, name, value)
        elif name in {"_worker_is_alive", "_stop_worker", "_stop_legacy_worker"}:
            object.__setattr__(self, name, value)
            setattr(service, name, value)
        elif name == "_process_identity":
            object.__setattr__(self, name, value)
            worker_lifecycle.process_identity = value
        elif name == "_boot_identity":
            object.__setattr__(self, name, value)
            worker_lifecycle.boot_identity = value
        else:
            object.__setattr__(self, name, value)
            setattr(production_jobs, name, value)

    def _store(self) -> JobStore:
        return JobStore(self.JOBS_DIR, self.LEGACY_JOBS_DIR)

    @staticmethod
    def _typed(raw: dict[str, object]) -> CursorJob:
        values = dict(raw)
        values.setdefault("id", "000000000000")
        values.setdefault("revision", 0)
        values.setdefault("request", "")
        values.setdefault("status", JobStatus.QUEUED.value)
        values.setdefault("delivered", False)
        values.setdefault("created_at", 0.0)
        values.setdefault("queued_at", values["created_at"])
        if any(
            values.get(field) is not None
            for field in ("worker_token", "worker_pid", "worker_process_start")
        ):
            values.setdefault("worker_boot_id", "test-boot")
        status = str(values["status"])
        if status == JobStatus.AWAITING_USER.value:
            values.setdefault("question", "question")
            values.setdefault("result", values["question"])
        if status in {
            JobStatus.BLOCKED.value,
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }:
            values.setdefault("completed_at", values["created_at"])
        return CursorJob.from_dict(values)

    def _coerce(self, job: CursorJob | dict[str, object]) -> CursorJob:
        return job if isinstance(job, CursorJob) else self._typed(job)

    def write_job(self, raw: dict[str, object]) -> None:
        self._store().create(self._typed(raw))

    def read_job(self, job_id: str) -> dict[str, object]:
        return self._store().get(job_id).to_dict()

    def active_jobs(self) -> list[dict[str, object]]:
        return [
            job.to_dict()
            for job in self._store().list()
            if job.status in production_jobs.MODEL_ACTIVE_STATUSES
        ]

    def reserved_targets(self, exclude_job_id: str | None = None) -> set[str]:
        return production_jobs.reserved_targets(self._store(), exclude_job_id)

    def resolve_job_repository(
        self, client: object, raw: dict[str, object], repositories: list[Path]
    ) -> tuple[Path | None, list[Path]]:
        return production_jobs.resolve_job_repository(
            cast(Any, client), self._typed(raw), repositories
        )

    def complete_from_output(
        self, raw: dict[str, object], *, output: str, agent_status: str
    ) -> None:
        values = dict(raw)
        values.setdefault("status", JobStatus.RUNNING.value)
        completed = production_jobs.complete_from_output(
            self._typed(values), output=output, agent_status=agent_status
        )
        raw.clear()
        raw.update(completed.to_dict())

    def read_agent_completion(
        self, client: object, raw: dict[str, object], **kwargs: object
    ) -> tuple[str, str]:
        return production_jobs.read_agent_completion(
            cast(Any, client), self._typed(raw), **cast(Any, kwargs)
        )

    def _worker_complete(self, *args: object, **kwargs: object) -> None:
        production_jobs._worker_complete(
            self._store(), *cast(Any, args), **cast(Any, kwargs)
        )

    def _prepare_pull_request_checkout(
        self,
        job_id: str,
        token: str,
        raw: dict[str, object],
        *args: object,
        **kwargs: object,
    ) -> dict[str, object] | None:
        owner = self._store().get(job_id).worker_ownership
        if owner is None:
            return None
        claim = cast(Any, owner)
        result = production_jobs._prepare_pull_request_checkout(
            self._store(),
            job_id,
            claim,
            self._typed(raw),
            *cast(Any, args),
            **cast(Any, kwargs),
        )
        return result.to_dict() if result is not None else None

    def _reserve_worker_target(self, *args: object, **kwargs: object) -> Any:
        return production_jobs._reserve_worker_target(
            self._store(), *cast(Any, args), **cast(Any, kwargs)
        )

    def _reserve_worker_worktree(self, *args: object, **kwargs: object) -> Any:
        return production_jobs._reserve_worker_worktree(
            self._store(), *cast(Any, args), **cast(Any, kwargs)
        )

    def _settle_worker_worktree(self, *args: object, **kwargs: object) -> Any:
        return production_jobs._settle_worker_worktree(
            self._store(), *cast(Any, args), **cast(Any, kwargs)
        )

    def _worker_command_matches(self, raw: dict[str, object]) -> bool:
        return worker_lifecycle.worker_command_matches(self._typed(raw))

    def _signal_worker(
        self,
        raw: CursorJob | dict[str, object],
        sent_signal: signal.Signals,
        *,
        include_process_group: bool = True,
    ) -> bool:
        return worker_lifecycle.signal_worker(
            self._coerce(raw),
            sent_signal,
            include_process_group=include_process_group,
            is_alive=lambda current: service._worker_is_alive(current),
        )


jobs: Any = _ProvisioningTestAdapter()


def configure_prompt_harness(client: mock.Mock) -> None:
    harness = mock.Mock()
    pending: dict[str, tuple[HarnessTask, bool]] = {}
    tasks = cast(list[HarnessTask], client.__dict__.setdefault("_harness_tasks", []))

    def create(
        request: SessionRequest,
        *,
        before_submit: Callable[[], None],
        **_kwargs: object,
    ) -> HarnessSession:
        before_submit()
        recording_owned = client.__dict__.get("_recording_owned")
        selection = client.start_agent.return_value
        if isinstance(recording_owned, dict):
            existing = recording_owned.get(request.name)
            if isinstance(existing, AgentSelection):
                selection = existing
        if isinstance(selection, AgentSelection):
            if isinstance(client.start_agent.return_value, AgentSelection):
                started = client.start_agent(
                    Path(selection.cwd),
                    Path(selection.cwd).name,
                    request.launch_context["pane_id"],
                    request.launch_context["workspace_id"],
                    name=request.name,
                    checkpoint=mock.ANY,
                )
            else:
                started = selection
        else:
            started = AgentSelection(
                request.name,
                request.launch_context["pane_id"],
                request.launch_context["workspace_id"],
                "",
                request.name,
                provider="cursor/herdr",
                provider_session_id=f"{request.name}-session",
                state_sequence=7,
            )
        if isinstance(recording_owned, dict):
            recording_owned[started.target] = started
        return HarnessSession(
            started.provider or "cursor/herdr",
            started.provider_session_id or "test-session",
            started.target,
            started.state_sequence or 0,
            {
                "pane_id": started.pane_id,
                "workspace_id": started.workspace_id,
                "cwd": started.cwd,
            },
        )

    def submit(
        session: HarnessSession,
        task: HarnessTask,
        *,
        before_submit: Callable[[int], None],
        accepted: Callable[[], None],
        **_kwargs: object,
    ) -> TaskSubmission:
        submit_error = client.__dict__.get("_harness_submit_error")
        if isinstance(submit_error, BaseException):
            raise submit_error
        assert task.baseline_sequence is not None
        before_submit(task.baseline_sequence)
        accepted()
        tasks.append(task)
        pending[task.correlation_id] = (task, False)
        return TaskSubmission(
            session,
            task.correlation_id,
            task.baseline_sequence,
            1,
        )

    def reply(
        session: HarnessSession,
        task: HarnessTask,
        *,
        before_submit: Callable[[int], None],
        accepted: Callable[[], None],
        **_kwargs: object,
    ) -> TaskSubmission:
        submission = submit(
            session,
            task,
            before_submit=before_submit,
            accepted=accepted,
        )
        pending[task.correlation_id] = (task, True)
        return submission

    def stream(submission: TaskSubmission, **_kwargs: object) -> Iterator[HarnessEvent]:
        task, clarification = pending[submission.correlation_id]
        error = client.__dict__.get("_harness_error")
        if isinstance(error, BaseException):
            raise error
        outcomes = client.__dict__.get("_harness_outcomes")
        if isinstance(outcomes, list) and outcomes:
            outcome = outcomes.pop(0)
        else:
            behavior = client.__dict__.get("_harness_behavior")
            if not callable(behavior):
                raise AssertionError("test harness has no configured task outcome")
            outcome = behavior(
                submission.session.target,
                task.text,
                token=task.correlation_id,
                baseline_sequence=task.baseline_sequence,
                expected_agent_session=task.expected_session_id,
                before_submit=lambda _baseline: None,
                accepted=lambda: None,
                checkpoint=lambda: None,
                allow_enter_fallback=task.allow_fallback_submit,
                clarification_reply=clarification,
            )
        assert isinstance(outcome, PromptOutcome)
        sequence = outcome.state_change_sequence or submission.baseline_sequence + 1
        recording_owned = client.__dict__.get("_recording_owned")
        if isinstance(recording_owned, dict):
            current = recording_owned.get(submission.session.target)
            if isinstance(current, AgentSelection):
                recording_owned[submission.session.target] = replace(
                    current,
                    provider_session_id=(
                        outcome.agent_session or submission.session.session_id
                    ),
                    state_sequence=sequence,
                )
        yield HarnessEvent(
            (
                HarnessEventKind.CLARIFICATION
                if outcome.question
                else HarnessEventKind.SUCCEEDED
            ),
            HarnessSession(
                submission.session.provider,
                outcome.agent_session or submission.session.session_id,
                submission.session.target,
                sequence,
                submission.session.metadata,
            ),
            outcome.status,
            outcome.output,
            outcome.summary,
            outcome.question,
            revision=outcome.revision,
        )

    def reconcile(target: str, *, expected_session_id: str) -> SessionReconciliation:
        observation = cast(
            SessionReconciliation,
            client.reconcile_session(
                target,
                expected_session_id=expected_session_id,
            ),
        )
        return observation

    harness.create_session.side_effect = create
    harness.submit_task.side_effect = submit
    harness.reply_to_clarification.side_effect = reply
    harness.stream_events.side_effect = stream
    harness.reconcile.side_effect = reconcile
    client.harness = harness


def configure_tiered_outcomes(
    client: mock.Mock,
    checkout: Path,
    *,
    tier: str = "simple",
) -> None:
    job_id = "123456789abc"
    owned: dict[str, AgentSelection] = {}
    client._recording_owned = owned
    initial = client.ensure_agent.return_value
    if isinstance(initial, AgentSelection):
        initial = replace(
            initial,
            worktree_path=initial.cwd,
        )
        owned[initial.target] = initial

        def ensure_agent(repository: Path, **kwargs: object) -> AgentSelection:
            branch = str(kwargs.get("worktree_branch") or "voice/test-checkout")
            reserve_worktree = cast(Callable[..., None], kwargs["reserve_worktree"])
            settle_worktree = cast(Callable[..., None], kwargs["settle_worktree"])
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            settle = cast(Callable[[AgentSelection], None], kwargs["settle"])
            checkout = Path(initial.worktree_path or initial.cwd)
            reserve_worktree(repository, branch, checkout, "planned")
            settle_worktree(
                checkout,
                initial.workspace_id,
                f"root-{initial.pane_id}",
            )
            provisional = replace(
                initial,
                provider=None,
                provider_session_id=None,
                state_sequence=None,
            )
            reserve(provisional, True)
            if bool(kwargs.get("start_session", True)):
                settle(initial)
                return initial
            return provisional

        client.ensure_agent.side_effect = ensure_agent
        client.get_agent.side_effect = lambda target: {
            "agent_status": "working",
            "state_change_seq": owned[target].state_sequence,
            "agent_session": owned[target].provider_session_id,
        }
    outcomes = [
        PromptOutcome(
            "idle",
            None,
            None,
            f"WORKFLOW_TIER[{job_id}-1]: {tier}\n"
            f"WORKFLOW_REASON[{job_id}-1]: test classification",
        )
    ]
    if tier != "simple":
        outcomes.extend(
            [
                PromptOutcome(
                    "idle",
                    None,
                    None,
                    f"WORKFLOW_PLAN[{job_id}-2]: implement safely",
                    boundary_marker="WORKFLOW_PLAN",
                    revision=2,
                ),
                PromptOutcome(
                    "idle",
                    None,
                    None,
                    f"WORKFLOW_REVIEW_DECISION[{job_id}-3]: approve\n"
                    f"WORKFLOW_REVIEW[{job_id}-3]: plan is safe",
                ),
            ]
        )
    final_turn = 2 if tier == "simple" else 4
    outcomes.append(
        PromptOutcome(
            "idle",
            "done",
            None,
            f"VOICE_SUMMARY[{job_id}-{final_turn}]: done",
        )
    )
    client._harness_outcomes = outcomes
    configure_prompt_harness(client)

    def reconcile(target: str, *, expected_session_id: str) -> SessionReconciliation:
        selection = owned[target]
        return SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession(
                selection.provider or "cursor/herdr",
                expected_session_id,
                target,
                (selection.state_sequence or 0) + 1,
                {
                    "pane_id": selection.pane_id,
                    "workspace_id": selection.workspace_id,
                    "cwd": selection.cwd,
                },
            ),
            "active",
            True,
        )

    client.reconcile_session.side_effect = reconcile

    def start_fresh(
        _checkout: Path,
        _label: str,
        workspace: str,
        *,
        role: str,
        reserve: Callable[[AgentSelection, bool], None],
        settle: Callable[[AgentSelection], None],
        **_kwargs: object,
    ) -> AgentSelection:
        target = str(_kwargs.get("name") or role)
        selection = AgentSelection(
            target,
            f"pane-{role}",
            workspace,
            str(checkout),
            role,
            str(checkout),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        before_pane_submit = cast(Callable[[], None], _kwargs["before_pane_submit"])
        pane_accepted = cast(Callable[[str, str], None], _kwargs["pane_accepted"])
        provisional = replace(
            selection,
            provider=None,
            provider_session_id=None,
            state_sequence=None,
        )
        reserve(provisional, True)
        before_pane_submit()
        pane_accepted(selection.pane_id, selection.workspace_id)
        if bool(_kwargs.get("start_session", True)):
            owned[selection.target] = selection
            settle(selection)
            return selection
        owned[selection.target] = selection
        return provisional

    client.start_fresh_agent.side_effect = start_fresh


class DurablePromptOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = JobStore(root / "jobs", root / "legacy")
        execute_phase_prompt = production_jobs._execute_phase_prompt

        def execute_with_harness(*args: object, **kwargs: object) -> object:
            configure_prompt_harness(cast(mock.Mock, args[3]))
            return execute_phase_prompt(*cast(Any, args), **cast(Any, kwargs))

        self.prompt_patch = mock.patch.object(
            production_jobs,
            "_execute_phase_prompt",
            side_effect=execute_with_harness,
        )
        self.prompt_patch.start()

    def tearDown(self) -> None:
        self.prompt_patch.stop()
        self.temporary.cleanup()

    def create(self, state: str = "none") -> CursorJob:
        values: dict[str, object] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "id": "123456789abc",
            "revision": 0,
            "request": "test",
            "status": "running",
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
            "turn_token": "123456789abc-1",
            "workflow_turn_phase": "classifying",
            "herdr_target": "planner",
            "herdr_pane_id": "planner-pane",
            "herdr_workspace_id": "workspace",
            "agent_operation_checkout": "/checkout",
            "agent_provider": "cursor/herdr",
            "agent_provider_session_id": "planner-session",
            "agent_state_sequence": 7,
            "planner_target": "planner",
            "active_participant": "planner",
            "agent_dispatch_state": "ready",
            "prompt_operation_state": state,
        }
        if state != "none":
            values.update(
                prompt_operation_phase="classifying",
                prompt_operation_turn=1,
                prompt_operation_target="planner",
                prompt_operation_agent_session="planner-session",
                prompt_baseline_sequence=7,
            )
        return self.store.create(CursorJob.from_dict(values))

    def create_approved_plan(self) -> CursorJob:
        created = self.store.create(
            CursorJob.from_dict(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "id": "123456789abc",
                    "revision": 0,
                    "request": "implement reviewed plan",
                    "status": "running",
                    "created_at": 1,
                    "delivered": False,
                    "worker_token": "worker",
                    "worker_pid": 42,
                    "worker_boot_id": "boot",
                    "worker_process_start": "start",
                    "worker_claim_operation": "test",
                    "worker_claimed_at": 1,
                    "workflow_tier": "medium",
                    "workflow_classification_reason": "reviewed",
                    "workflow_phase": "reviewing",
                    "review_round": 0,
                    "turn": 4,
                    "turn_token": "123456789abc-4",
                    "workflow_turn_phase": "reviewing",
                    "herdr_target": "planner",
                    "herdr_pane_id": "planner-pane",
                    "herdr_workspace_id": "workspace",
                    "agent_operation_checkout": "/checkout",
                    "agent_provider": "cursor/herdr",
                    "agent_provider_session_id": "planner-session",
                    "agent_state_sequence": 7,
                    "planner_target": "planner",
                    "active_participant": "planner",
                    "agent_dispatch_state": "ready",
                    "prompt_operation_state": "none",
                    "plan_approval_state": "boundary",
                    "plan_approval_id": "gate-id",
                    "plan_approval_agent_session": "planner-session",
                    "plan_approval_state_change_sequence": 7,
                    "plan_approval_revision": 3,
                    "plan_approval_counted": False,
                }
            )
        )
        plan = "Implement the reviewed plan."
        plan_reference = self.store.write_artifact(created.id, "plan", 0, plan)
        review_reference = self.store.write_artifact(
            created.id,
            "review",
            0,
            "The plan is safe.",
            source_text=plan,
        )
        updated = self.store.update(
            created.id,
            lambda job: job.evolve(
                workflow_phase="implementing",
                workflow_turn_phase="implementing",
                plan_artifact=plan_reference,
                review_artifact=review_reference,
                review_decision="approve",
                review_approved=True,
                review_approval_source="reviewer",
                plan_approval_state="approved",
                plan_approval_source="explicit",
                plan_approval_plan_artifact=plan_reference,
                plan_approval_review_artifact=review_reference,
            ),
        )
        assert updated is not None
        return updated

    def test_explicit_plan_approval_uses_api_without_enter_and_counts_acceptance(
        self,
    ) -> None:
        job = self.create_approved_plan()
        preferences_path = Path(self.temporary.name) / "approval.json"
        client = mock.Mock()
        agent = {
            "agent_status": "blocked",
            "state_change_seq": 7,
            "revision": 3,
            "agent_session": "planner-session",
            "interactive_ready": True,
        }
        client.get_agent.return_value = agent

        def prompt(*_args: object, **kwargs: object) -> PromptOutcome:
            before_submit = cast(Callable[[int], None], kwargs["before_submit"])
            accepted = cast(Callable[[], None], kwargs["accepted"])
            before_submit(7)
            accepted()
            return PromptOutcome(
                "idle",
                "done",
                None,
                "VOICE_SUMMARY[123456789abc-4]: done",
                agent_session="planner-session",
                state_change_sequence=8,
                revision=4,
            )

        client._harness_behavior = prompt
        with mock.patch.dict(
            os.environ,
            {"VOICE_HARNESS_PLAN_APPROVAL_FILE": str(preferences_path)},
        ):
            outcome = production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="lgtm. Implement the approved plan.",
                token="123456789abc-4",
            )
            preferences = user_config.load_plan_approval_preferences(preferences_path)

        self.assertEqual(
            outcome,
            ("VOICE_SUMMARY[123456789abc-4]: done", "idle"),
        )
        current = self.store.get(job.id)
        self.assertEqual(current.plan_approval_state, "observed")
        self.assertTrue(current.plan_approval_counted)
        self.assertEqual(preferences.explicit_approval_count, 1)
        task = cast(list[HarnessTask], client._harness_tasks)[-1]
        self.assertTrue(task.allow_fallback_submit)
        self.assertEqual(task.expected_session_id, "planner-session")
        client.prompt_and_wait.assert_not_called()

    def test_unaccepted_plan_approval_is_not_counted(self) -> None:
        job = self.create_approved_plan()
        preferences_path = Path(self.temporary.name) / "approval.json"
        client = mock.Mock()
        client.get_agent.return_value = {
            "agent_status": "blocked",
            "state_change_seq": 7,
            "revision": 3,
            "agent_session": "planner-session",
            "interactive_ready": True,
        }
        client._harness_submit_error = HerdrError(
            "not accepted",
            code="agent_prompt_stalled",
        )

        with (
            mock.patch.dict(
                os.environ,
                {"VOICE_HARNESS_PLAN_APPROVAL_FILE": str(preferences_path)},
            ),
            self.assertRaises(HerdrError),
        ):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="lgtm. Implement the approved plan.",
                token="123456789abc-4",
            )

        preferences = user_config.load_plan_approval_preferences(preferences_path)
        self.assertEqual(preferences.explicit_approval_count, 0)
        self.assertFalse(self.store.get(job.id).plan_approval_counted)
        client.prompt_and_wait.assert_not_called()

    def test_crash_after_approval_submit_recovers_without_duplicate_prompt(
        self,
    ) -> None:
        created = self.create_approved_plan()
        job = self.store.update(
            created.id,
            lambda current: current.evolve(
                prompt_operation_state="submitting",
                prompt_operation_phase="implementing",
                prompt_operation_turn=4,
                prompt_operation_target="planner",
                prompt_operation_agent_session="planner-session",
                prompt_baseline_sequence=7,
            ),
        )
        assert job is not None
        preferences_path = Path(self.temporary.name) / "approval.json"
        client = mock.Mock()
        client.get_agent.return_value = {
            "state_change_seq": 8,
            "agent_session": "planner-session",
        }
        client.wait_for_stable_completion.return_value = PromptOutcome(
            "idle",
            "done",
            None,
            "VOICE_SUMMARY[123456789abc-4]: done",
            agent_session="planner-session",
            state_change_sequence=8,
            revision=4,
        )

        with mock.patch.dict(
            os.environ,
            {"VOICE_HARNESS_PLAN_APPROVAL_FILE": str(preferences_path)},
        ):
            outcome = production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="lgtm. Implement the approved plan.",
                token="123456789abc-4",
            )

        self.assertEqual(
            outcome,
            ("VOICE_SUMMARY[123456789abc-4]: done", "idle"),
        )
        client.prompt_and_wait.assert_not_called()
        current = self.store.get(job.id)
        self.assertEqual(current.plan_approval_state, "observed")
        self.assertTrue(current.plan_approval_counted)
        self.assertEqual(
            user_config.load_plan_approval_preferences(
                preferences_path
            ).explicit_approval_count,
            1,
        )

    def test_stale_same_owner_prompt_observation_is_rejected(self) -> None:
        created = self.create_approved_plan()
        job = self.store.update(
            created.id,
            lambda current: current.evolve(
                prompt_operation_state="submitting",
                prompt_operation_phase="implementing",
                prompt_operation_turn=4,
                prompt_operation_target="planner",
                prompt_operation_agent_session="planner-session",
                prompt_baseline_sequence=7,
            ),
        )
        assert job is not None
        client = mock.Mock()

        def observe(_target: str) -> dict[str, object]:
            self.store.update(job.id, lambda current: current.evolve(reconcile=True))
            return {
                "state_change_seq": 8,
                "agent_session": "planner-session",
            }

        client.get_agent.side_effect = observe

        outcome = production_jobs._execute_phase_prompt(
            self.store,
            job,
            WORKER,
            client,
            lambda: None,
            target="planner",
            prompt="lgtm. Implement the approved plan.",
            token="123456789abc-4",
        )

        self.assertIsNone(outcome)
        current = self.store.get(job.id)
        self.assertTrue(current.reconcile)
        self.assertEqual(current.prompt_operation_state, "submitting")
        client.wait_for_stable_completion.assert_not_called()

    def test_approval_submit_recovery_rejects_replaced_agent_session(self) -> None:
        created = self.create_approved_plan()
        job = self.store.update(
            created.id,
            lambda current: current.evolve(
                prompt_operation_state="submitting",
                prompt_operation_phase="implementing",
                prompt_operation_turn=4,
                prompt_operation_target="planner",
                prompt_operation_agent_session="planner-session",
                prompt_baseline_sequence=7,
            ),
        )
        assert job is not None
        client = mock.Mock()
        client.get_agent.return_value = {
            "state_change_seq": 8,
            "agent_session": "replacement-session",
        }

        with self.assertRaisesRegex(
            production_jobs.HarnessError, "requires manual reconciliation"
        ):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="lgtm. Implement the approved plan.",
                token="123456789abc-4",
            )

        current = self.store.get(job.id)
        self.assertEqual(current.prompt_operation_state, "ambiguous")
        self.assertFalse(current.plan_approval_counted)
        client.wait_for_stable_completion.assert_not_called()

    def test_preference_read_failure_defers_third_approval_completion(self) -> None:
        created = self.create_approved_plan()
        job = self.store.update(
            created.id,
            lambda current: current.evolve(
                plan_approval_state="observed",
                plan_approval_counted=True,
                prompt_operation_state="submitted",
                prompt_operation_phase="implementing",
                prompt_operation_turn=4,
                prompt_operation_target="planner",
                prompt_operation_agent_session="planner-session",
                prompt_baseline_sequence=7,
            ),
        )
        assert job is not None

        with mock.patch.object(
            production_jobs,
            "load_plan_approval_preferences",
            side_effect=OSError("temporary read failure"),
        ):
            production_jobs._worker_complete(
                self.store,
                job.id,
                WORKER,
                output="VOICE_SUMMARY[123456789abc-4]: done",
                agent_status="idle",
                expected_revision=job.revision,
            )

        deferred = self.store.get(job.id)
        self.assertEqual(deferred.status, JobStatus.QUEUED)
        self.assertTrue(deferred.reconcile)
        self.assertTrue(deferred.plan_approval_completion_pending)
        self.assertEqual(deferred.workflow_phase.value, "finished")
        self.assertEqual(deferred.prompt_operation_state, "none")
        self.assertEqual(deferred.result, "done")
        self.assertIsNone(deferred.worker_token)

        reclaimed = self.store.update(
            job.id,
            lambda current: current.evolve(
                status=JobStatus.RECONCILING,
                worker_token="worker-2",
                worker_pid=43,
                worker_boot_id="boot-2",
                worker_process_start="start-2",
                worker_claim_operation="test",
                worker_claimed_at=1,
            ),
        )
        assert reclaimed is not None
        pending = user_config.PlanApprovalPreferences(
            explicit_approval_ids=("first", "second", "gate-id"),
            offer_pending_id="gate-id",
        )
        with mock.patch.object(
            production_jobs,
            "load_plan_approval_preferences",
            return_value=pending,
        ):
            client = mock.Mock()
            production_jobs._run_tiered_workflow(
                self.store,
                reclaimed,
                WORKER_2,
                client,
                lambda: None,
            )

        offered = self.store.get(job.id)
        self.assertEqual(offered.status, JobStatus.AWAITING_USER)
        self.assertEqual(offered.clarification_kind, "workflow_plan_auto_offer")
        self.assertFalse(offered.plan_approval_completion_pending)
        client.get_agent.assert_not_called()
        client.wait_for_stable_completion.assert_not_called()

    def test_preference_write_failure_retries_accepted_approval_locally(self) -> None:
        created = self.create_approved_plan()
        job = self.store.update(
            created.id,
            lambda current: current.evolve(
                plan_approval_state="observed",
                prompt_operation_state="submitted",
                prompt_operation_phase="implementing",
                prompt_operation_turn=4,
                prompt_operation_target="planner",
                prompt_operation_agent_session="planner-session",
                prompt_baseline_sequence=7,
            ),
        )
        assert job is not None

        with mock.patch.object(
            production_jobs,
            "record_explicit_plan_approval",
            side_effect=OSError("temporary write failure"),
        ):
            production_jobs._worker_complete(
                self.store,
                job.id,
                WORKER,
                output="VOICE_SUMMARY[123456789abc-4]: done",
                agent_status="idle",
                expected_revision=job.revision,
            )

        deferred = self.store.get(job.id)
        self.assertTrue(deferred.plan_approval_completion_pending)
        self.assertFalse(deferred.plan_approval_counted)
        reclaimed = self.store.update(
            job.id,
            lambda current: current.evolve(
                status=JobStatus.RECONCILING,
                worker_token="worker-2",
                worker_pid=43,
                worker_boot_id="boot-2",
                worker_process_start="start-2",
                worker_claim_operation="test",
                worker_claimed_at=1,
            ),
        )
        assert reclaimed is not None
        recorded = user_config.PlanApprovalPreferences(
            explicit_approval_ids=("gate-id",),
        )
        with mock.patch.object(
            production_jobs,
            "record_explicit_plan_approval",
            return_value=recorded,
        ):
            client = mock.Mock()
            production_jobs._run_tiered_workflow(
                self.store,
                reclaimed,
                WORKER_2,
                client,
                lambda: None,
            )

        completed = self.store.get(job.id)
        self.assertEqual(completed.terminal_intent_status, JobStatus.COMPLETED)
        self.assertTrue(completed.plan_approval_counted)
        self.assertFalse(completed.plan_approval_completion_pending)
        client.get_agent.assert_not_called()
        client.wait_for_stable_completion.assert_not_called()

    def test_planned_prompt_persists_submit_and_accept_boundaries(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {
            "state_change_seq": 7,
            "agent_session": "planner-session",
        }

        def prompt(*_args: object, **kwargs: object) -> PromptOutcome:
            before_submit = cast(Callable[[int], None], kwargs["before_submit"])
            accepted = cast(Callable[[], None], kwargs["accepted"])
            before_submit(7)
            self.assertEqual(
                self.store.get(job.id).prompt_operation_state, "submitting"
            )
            accepted()
            return PromptOutcome("idle", None, None, "output")

        client._harness_behavior = prompt

        outcome = production_jobs._execute_phase_prompt(
            self.store,
            job,
            WORKER,
            client,
            lambda: None,
            target="planner",
            prompt="prompt",
            token="123456789abc-1",
        )

        self.assertEqual(outcome, ("output", "idle"))
        self.assertEqual(self.store.get(job.id).prompt_operation_state, "submitted")
        client.prompt_and_wait.assert_not_called()

    def _execute_continuation_with_session(self, session: str) -> tuple[bool, str]:
        created = self.create()
        job = self.store.update(
            created.id,
            lambda current: current.evolve(
                continuation=True,
                continuation_answer="Question: Format?\nUser answered: JSON",
                prompt_context_sessions={"planner": "planner-session"},
            ),
        )
        assert job is not None
        client = mock.Mock()
        client.get_agent.return_value = {
            "state_change_seq": 7,
            "agent_session": session,
        }

        def submit(*args: object, **kwargs: object) -> PromptOutcome:
            cast(Callable[[int], None], kwargs["before_submit"])(7)
            cast(Callable[[], None], kwargs["accepted"])()
            return PromptOutcome("idle", None, None, "output")

        client._harness_behavior = submit
        decisions: list[bool] = []

        def factory(identity: str, full: bool) -> production_jobs.PromptPayload:
            decisions.append(full)
            text = "full request and artifacts" if full else "delta only"
            return production_jobs.bounded_prompt_payload(
                text,
                phase="classifying",
                session_identity=identity,
                full_rehydration=full,
                sections={"request": "test"} if full else {"clarification": "JSON"},
            )

        production_jobs._execute_phase_prompt(
            self.store,
            job,
            WORKER,
            client,
            lambda: None,
            target="planner",
            prompt_factory=factory,
            token="123456789abc-1",
        )
        sent = cast(list[HarnessTask], client._harness_tasks)[-1].text
        return decisions[0], sent

    def test_same_session_continuation_submits_only_delta(self) -> None:
        full_rehydration, sent = self._execute_continuation_with_session(
            "planner-session"
        )

        self.assertFalse(full_rehydration)
        self.assertEqual(sent, "delta only")
        manifest = self.store.get("123456789abc").prompt_manifest
        assert manifest is not None
        self.assertFalse(manifest["full_rehydration"])

    def test_replaced_session_rehydrates_full_context(self) -> None:
        full_rehydration, sent = self._execute_continuation_with_session(
            "replacement-session"
        )

        self.assertTrue(full_rehydration)
        self.assertEqual(sent, "full request and artifacts")
        current = self.store.get("123456789abc")
        self.assertEqual(
            current.prompt_context_sessions["planner"], "replacement-session"
        )

    def test_repeated_clarification_history_does_not_grow_delta_prompt(self) -> None:
        created = self.create()
        record = {
            "question_id": "question-1",
            "question": "Format?",
            "answer": "JSON",
            "turn_token": "123456789abc-1",
        }
        one = created.evolve(
            continuation=True,
            continuation_answer="Question: Format?\nUser answered: JSON",
            clarifications=[record],
        )
        many = one.evolve(clarifications=[record] * 50)

        one_payload = production_jobs._WorkflowPromptFactory(
            one, one.workflow_phase, one.turn_token or "", ()
        )("session", False)
        many_payload = production_jobs._WorkflowPromptFactory(
            many, many.workflow_phase, many.turn_token or "", ()
        )("session", False)

        self.assertEqual(one_payload.text, many_payload.text)
        self.assertEqual(
            one_payload.manifest["total_chars"],
            many_payload.manifest["total_chars"],
        )
        self.assertNotIn(created.request, one_payload.text)

    def test_fresh_session_payload_rehydrates_request_and_current_answer(self) -> None:
        created = self.create()
        continued = created.evolve(
            continuation=True,
            continuation_answer="Question: Format?\nUser answered: JSON",
        )

        payload = production_jobs._WorkflowPromptFactory(
            continued,
            continued.workflow_phase,
            continued.turn_token or "",
            (),
        )("replacement-session", True)

        self.assertIn("User request: test", payload.text)
        self.assertIn("User answered: JSON", payload.text)
        self.assertTrue(payload.manifest["full_rehydration"])

    def test_prompt_failure_before_dispatch_remains_retryable(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {
            "state_change_seq": 7,
            "agent_session": "planner-session",
        }
        client._harness_submit_error = HerdrError("timeout")

        with self.assertRaises(HerdrError):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="prompt",
                token="123456789abc-1",
            )

        self.assertEqual(self.store.get(job.id).prompt_operation_state, "submitting")
        with sqlite3.connect(self.store.db_path) as connection:
            self.assertEqual(
                connection.execute("SELECT status FROM outbox").fetchone(),
                ("failed-retryable",),
            )
        client.prompt_and_wait.assert_not_called()

    def test_questionnaire_after_submit_fence_is_ambiguous(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {
            "state_change_seq": 7,
            "agent_session": "planner-session",
        }
        client._harness_error = HerdrError(
            "questionnaire",
            code="interactive_questionnaire",
        )

        with self.assertRaises(HerdrError):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="prompt",
                token="123456789abc-1",
            )

        current = self.store.get(job.id)
        self.assertEqual(current.prompt_operation_state, "ambiguous")
        self.assertEqual(current.manual_reconcile_operation, "prompt")
        client.prompt_and_wait.assert_not_called()

    def test_terminal_intent_rejects_observation_after_submit(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {
            "state_change_seq": 7,
            "agent_session": "planner-session",
        }
        submitted = False

        def prompt(*_args: object, **kwargs: object) -> PromptOutcome:
            nonlocal submitted
            before_submit = cast(Callable[[int], None], kwargs["before_submit"])
            self.store.update(
                job.id,
                lambda current: stage_terminal_intent(
                    current,
                    JobStatus.CANCELLED,
                    now=2,
                    result="cancelled",
                ),
            )
            before_submit(7)
            submitted = True
            return PromptOutcome("idle", None, None, "output")

        client._harness_behavior = prompt

        with self.assertRaises(production_jobs.WorkerCancelled):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="prompt",
                token="123456789abc-1",
            )

        self.assertTrue(submitted)
        current = self.store.get(job.id)
        self.assertEqual(current.terminal_intent_status, JobStatus.CANCELLED)
        self.assertEqual(current.prompt_operation_state, "submitting")
        client.prompt_and_wait.assert_not_called()

    def test_participant_plan_uses_observed_checkout_before_job_settlement(
        self,
    ) -> None:
        job = self.create()
        checkout = Path("/worktrees/voice-task")

        planned = production_jobs._plan_participant_creation(
            self.store,
            job,
            WORKER,
            WorkflowParticipant.PLANNER,
            target="planner",
            label="task-planner",
            workspace_id="workspace",
            checkout=checkout,
        )

        self.assertEqual(planned.participant_creation_state, "planned")
        self.assertEqual(planned.participant_creation_checkout, str(checkout))
        operation = planned.participant_pane_operation
        assert operation is not None
        self.assertEqual(
            operation.spec.checkout,
            str(checkout),
        )

    def test_terminal_intent_invalidates_pane_before_create_callback(self) -> None:
        job = self.create()
        planned = self.store.update(
            job.id,
            lambda current: current.evolve(
                participant_creation_state="planned",
                participant_creation_participant="reviewer",
                participant_creation_target="reviewer",
                participant_creation_label="task-reviewer",
                participant_creation_workspace_id="workspace",
                participant_creation_checkout="/checkout",
            ),
        )
        assert planned is not None
        before_create, _accepted, _revision = (
            production_jobs._participant_pane_callbacks(
                self.store,
                job.id,
                WORKER,
                "reviewer",
                revision_state=[planned.revision],
            )
        )
        self.store.update(
            job.id,
            lambda current: stage_terminal_intent(
                current,
                JobStatus.FAILED,
                now=2,
                result="failed",
                error="failed",
            ),
        )

        with self.assertRaises(production_jobs.WorkerCancelled):
            before_create()

        current = self.store.get(job.id)
        self.assertEqual(current.terminal_intent_status, JobStatus.FAILED)
        self.assertEqual(current.participant_creation_state, "planned")

    def test_stale_same_owner_participant_callback_is_rejected(self) -> None:
        job = self.create()
        planned = self.store.update(
            job.id,
            lambda current: current.evolve(
                participant_creation_state="planned",
                participant_creation_participant="reviewer",
                participant_creation_target="reviewer",
                participant_creation_label="task-reviewer",
                participant_creation_workspace_id="workspace",
                participant_creation_checkout="/checkout",
            ),
        )
        assert planned is not None
        before_create, _accepted, _revision = (
            production_jobs._participant_pane_callbacks(
                self.store,
                job.id,
                WORKER,
                "reviewer",
                revision_state=[planned.revision],
            )
        )
        mutated = self.store.update(
            job.id, lambda current: current.evolve(reconcile=True)
        )
        assert mutated is not None

        with self.assertRaises(production_jobs.WorkerCancelled):
            before_create()

        current = self.store.get(job.id)
        self.assertEqual(current.revision, mutated.revision)
        self.assertEqual(current.participant_creation_state, "planned")

    def test_submitted_prompt_observes_without_resubmitting(self) -> None:
        job = self.create("submitted")
        client = mock.Mock()
        client.wait_for_stable_completion.return_value = PromptOutcome(
            status="idle",
            summary=None,
            question=None,
            output="output",
        )

        outcome = production_jobs._execute_phase_prompt(
            self.store,
            job,
            WORKER,
            client,
            lambda: None,
            target="planner",
            prompt="prompt",
            token="123456789abc-1",
        )

        self.assertEqual(outcome, ("output", "idle"))
        client.prompt_and_wait.assert_not_called()

    def test_submitted_prompt_read_failure_keeps_reservation_for_retry(self) -> None:
        job = self.create("submitted")
        client = mock.Mock()
        client.wait_for_stable_completion.side_effect = HerdrError(
            "temporarily unavailable"
        )

        with self.assertRaises(production_jobs.WorkerCancelled):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                WORKER,
                client,
                lambda: None,
                target="planner",
                prompt="prompt",
                token="123456789abc-1",
            )

        deferred = self.store.get(job.id)
        self.assertEqual(deferred.status, JobStatus.QUEUED)
        self.assertEqual(deferred.prompt_operation_state, "submitted")
        self.assertEqual(deferred.herdr_target, "planner")
        self.assertTrue(deferred.reconcile)


class CursorJobStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.jobs_patch = mock.patch.object(jobs, "JOBS_DIR", Path(self.temporary.name))
        self.legacy_jobs_patch = mock.patch.object(
            jobs, "LEGACY_JOBS_DIR", Path(self.temporary.name) / "legacy"
        )
        self.job_logs_patch = mock.patch.object(
            service, "JOB_LOGS_DIR", Path(self.temporary.name) / "logs"
        )
        self.service_jobs_patch = mock.patch.object(
            service, "JOBS_DIR", Path(self.temporary.name)
        )
        self.service_legacy_patch = mock.patch.object(
            service, "LEGACY_JOBS_DIR", Path(self.temporary.name) / "legacy"
        )
        self.jobs_patch.start()
        self.legacy_jobs_patch.start()
        self.service_jobs_patch.start()
        self.service_legacy_patch.start()
        self.job_logs_patch.start()

    def tearDown(self) -> None:
        self.job_logs_patch.stop()
        self.service_legacy_patch.stop()
        self.service_jobs_patch.stop()
        self.legacy_jobs_patch.stop()
        self.jobs_patch.stop()
        self.temporary.cleanup()

    def write_exhausted_review_job(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "change recovery ownership",
                "status": "awaiting_user",
                "question": "Approve or abort?",
                "result": "Approve or abort?",
                "clarification_kind": "workflow_review_exhausted",
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "reviewing",
                "review_round": 2,
                "review_decision": "revise",
                "herdr_target": "planner",
                "planner_target": "planner",
                "active_participant": "planner",
                "plan_approval_state": "boundary",
                "plan_approval_id": "gate-id",
                "plan_approval_agent_session": "planner-session",
                "plan_approval_state_change_sequence": 2,
                "plan_approval_revision": 2,
            }
        )
        store = jobs._store()
        plan = "Preserve ownership during recovery."
        plan_reference = store.write_artifact(
            "123456789abc",
            "plan",
            2,
            plan,
        )
        review_reference = store.write_artifact(
            "123456789abc",
            "review",
            2,
            "Ownership remains unresolved.",
            source_text=plan,
        )
        store.update(
            "123456789abc",
            lambda job: job.evolve(
                plan_artifact=plan_reference,
                review_artifact=review_reference,
            ),
        )

    def test_malformed_job_file_is_quarantined_from_collection_read(self) -> None:
        (Path(self.temporary.name) / "123456789abc.json").write_text("{not json")

        with self.assertWarnsRegex(
            JobQuarantineWarning, "123456789abc.json: legacy import is quarantined"
        ):
            self.assertEqual(jobs.active_jobs(), [])

    def test_invalid_job_file_is_not_silently_skipped(self) -> None:
        jobs.write_job(
            {
                "id": "aaaaaaaaaaaa",
                "status": "queued",
                "request": "valid",
                "delivered": False,
            }
        )
        (Path(self.temporary.name) / "bbbbbbbbbbbb.json").write_text(
            json.dumps({"id": "bbbbbbbbbbbb", "status": "unknown"})
        )

        with self.assertWarnsRegex(
            JobQuarantineWarning, "bbbbbbbbbbbb.json: legacy import is quarantined"
        ):
            self.assertEqual(
                [job["id"] for job in jobs.active_jobs()], ["aaaaaaaaaaaa"]
            )

    def test_repository_reply_preserves_original_task(self) -> None:
        job: dict[str, object] = {
            "id": "123456789abc",
            "request": "Fix APP-42",
            "status": "awaiting_user",
            "clarification_kind": "repository",
        }
        jobs.write_job(job)
        with mock.patch.object(service, "launch_worker") as launch:
            service.reply_job("123456789abc", "example-repo")
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["request"], "Fix APP-42")
        self.assertEqual(updated["repository_hint"], "example-repo")
        launch.assert_called_once_with("123456789abc")

    def test_github_repository_reply_preserves_explicit_fork_intent(self) -> None:
        job: dict[str, object] = {
            "id": "123456789abc",
            "request": "Fork this repo and add a feature",
            "fork_requested": True,
            "status": "awaiting_user",
            "clarification_kind": "github_repository",
        }
        jobs.write_job(job)
        with mock.patch.object(service, "launch_worker"):
            service.reply_job("123456789abc", "example/project")
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["github_repository"], "example/project")
        self.assertTrue(updated["fork_requested"])
        self.assertEqual(updated["request"], "Fork this repo and add a feature")

    def test_medium_review_revision_escalates_to_user_without_implementation(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "change several components",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "medium",
                "workflow_classification_reason": "cross-component",
                "workflow_phase": "reviewing",
                "planner_target": "planner",
                "reviewer_target": "reviewer",
                "active_participant": "reviewer",
                "herdr_target": "reviewer",
                "turn_token": "turn",
            }
        )
        store = jobs._store()
        plan = "Preserve compatibility while changing the components."
        plan_reference = store.write_artifact("123456789abc", "plan", 0, plan)
        store.update(
            "123456789abc",
            lambda job: job.evolve(plan_artifact=plan_reference),
        )

        production_jobs._advance_workflow_output(
            store,
            store.get("123456789abc"),
            LEGACY_WORKER,
            "WORKFLOW_REVIEW_DECISION[turn]: revise\n"
            "WORKFLOW_REVIEW[turn]: choose the compatibility policy",
            "idle",
        )

        updated = store.get("123456789abc")
        self.assertEqual(updated.status, JobStatus.AWAITING_USER)
        self.assertEqual(updated.workflow_phase.value, "reviewing")
        self.assertIsNone(updated.participant_target(WorkflowParticipant.IMPLEMENTER))
        with mock.patch.object(service, "launch_worker"):
            service.reply_job("123456789abc", "keep the old format")
        resumed = store.get("123456789abc")
        self.assertEqual(resumed.workflow_tier.value, "high-risk")
        self.assertEqual(resumed.workflow_phase.value, "revising")
        self.assertEqual(resumed.herdr_target, "reviewer")
        self.assertEqual(
            resumed.active_participant,
            WorkflowParticipant.REVIEWER,
        )
        self.assertEqual(resumed.request, "change several components")
        self.assertEqual(len(resumed.clarifications), 1)
        self.assertEqual(resumed.clarifications[0]["answer"], "keep the old format")
        self.assertIn("keep the old format", str(resumed.continuation_answer))

    def test_auto_mode_approves_only_the_reviewed_plan_gate(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "update ordinary application behavior",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "medium",
                "workflow_classification_reason": "cross-component",
                "workflow_phase": "reviewing",
                "planner_target": "planner",
                "reviewer_target": "reviewer",
                "active_participant": "reviewer",
                "herdr_target": "reviewer",
                "turn_token": "turn",
                "plan_approval_state": "boundary",
                "plan_approval_id": "gate-id",
                "plan_approval_agent_session": "planner-session",
                "plan_approval_state_change_sequence": 7,
                "plan_approval_revision": 3,
            }
        )
        store = jobs._store()
        plan = "Update behavior with compatibility tests."
        plan_reference = store.write_artifact("123456789abc", "plan", 0, plan)
        store.update(
            "123456789abc",
            lambda job: job.evolve(plan_artifact=plan_reference),
        )
        before_review = store.get("123456789abc")

        with mock.patch.object(
            production_jobs,
            "load_plan_approval_preferences",
            return_value=user_config.PlanApprovalPreferences(
                mode=user_config.PlanApprovalMode.AUTO
            ),
        ):
            advanced = production_jobs._advance_workflow_output(
                store,
                before_review,
                LEGACY_WORKER,
                "WORKFLOW_REVIEW_DECISION[turn]: approve\n"
                "WORKFLOW_REVIEW[turn]: plan is safe",
                "idle",
            )

        assert advanced is not None
        self.assertEqual(advanced.revision, before_review.revision + 1)
        self.assertEqual(advanced.workflow_phase.value, "implementing")
        self.assertEqual(advanced.plan_approval_state, "approved")
        self.assertEqual(advanced.plan_approval_source, "auto")
        self.assertFalse(advanced.plan_approval_counted)
        self.assertEqual(advanced.herdr_target, "reviewer")

    def test_review_approval_and_voice_question_commit_atomically(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "update ordinary application behavior",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "medium",
                "workflow_classification_reason": "cross-component",
                "workflow_phase": "reviewing",
                "planner_target": "planner",
                "reviewer_target": "reviewer",
                "active_participant": "reviewer",
                "herdr_target": "reviewer",
                "turn_token": "turn",
                "plan_approval_state": "boundary",
                "plan_approval_id": "gate-id",
                "plan_approval_agent_session": "planner-session",
                "plan_approval_state_change_sequence": 7,
                "plan_approval_revision": 3,
            }
        )
        store = jobs._store()
        plan = "Update behavior with compatibility tests."
        plan_reference = store.write_artifact("123456789abc", "plan", 0, plan)
        store.update(
            "123456789abc",
            lambda job: job.evolve(plan_artifact=plan_reference),
        )
        before_review = store.get("123456789abc")

        with mock.patch.object(
            production_jobs,
            "load_plan_approval_preferences",
            return_value=user_config.PlanApprovalPreferences(),
        ):
            advanced = production_jobs._advance_workflow_output(
                store,
                before_review,
                LEGACY_WORKER,
                "WORKFLOW_REVIEW_DECISION[turn]: approve\n"
                "WORKFLOW_REVIEW[turn]: plan is safe",
                "idle",
            )

        self.assertIsNone(advanced)
        awaiting = store.get("123456789abc")
        self.assertEqual(awaiting.revision, before_review.revision + 1)
        self.assertEqual(awaiting.status, JobStatus.AWAITING_USER)
        self.assertEqual(awaiting.clarification_kind, "workflow_plan_approval")
        self.assertEqual(awaiting.plan_approval_state, "awaiting")
        self.assertTrue(awaiting.review_approved)
        self.assertIsNotNone(awaiting.review_artifact)

    def test_auto_mode_does_not_bypass_destructive_confirmation(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "request": "drop every table",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "irreversible data loss",
                "workflow_phase": "reviewing",
                "plan_artifact": ".artifacts/123456789abc/plan-0.json",
                "review_artifact": ".artifacts/123456789abc/review-0.json",
                "review_decision": "approve",
                "review_approved": True,
                "review_approval_source": "reviewer",
                "plan_approval_state": "boundary",
                "plan_approval_id": "gate-id",
                "plan_approval_agent_session": "planner-session",
                "plan_approval_state_change_sequence": 7,
                "plan_approval_revision": 3,
            }
        )

        with mock.patch.object(
            production_jobs,
            "load_plan_approval_preferences",
            return_value=user_config.PlanApprovalPreferences(
                mode=user_config.PlanApprovalMode.AUTO
            ),
        ):
            self.assertFalse(
                production_jobs._auto_plan_approval_allowed(
                    job,
                    plan="Apply the approved DDL.",
                    review="approve",
                )
            )

    def test_deterministic_hard_risk_terms_override_simple_classification(
        self,
    ) -> None:
        tier = production_jobs._classified_tier(
            "simple", "add OAuth authorization", "localized"
        )
        self.assertEqual(tier.value, "high-risk")
        self.assertEqual(
            production_jobs._classified_tier(
                "simple",
                "rename a label",
                "localized",
                "The issue requires a schema migration.",
            ).value,
            "high-risk",
        )
        self.assertEqual(
            production_jobs._classified_tier(
                "simple",
                "rename a label",
                "Touches recovery ownership",
            ).value,
            "high-risk",
        )
        bounded_prefix = "x" * production_jobs._MAX_RISK_EVIDENCE_BYTES
        self.assertEqual(
            production_jobs._classified_tier(
                "simple",
                f"{bounded_prefix} security",
                "localized",
            ).value,
            "simple",
        )
        self.assertEqual(
            production_jobs._hard_risk_evidence(
                "rename a label",
                "localized",
                plan="Rotate the deployment credential.",
            ),
            "approved plan contains 'credential'",
        )
        self.assertIsNone(
            production_jobs._hard_risk_evidence(
                "rename a label",
                "localized",
                review="No security or permission concerns.",
            )
        )
        self.assertEqual(
            production_jobs._hard_risk_evidence(
                "rename a label",
                "localized",
                review="Requires a permission confirmation.",
            ),
            "review contains 'permission'",
        )
        self.assertEqual(
            production_jobs._classified_tier(
                "simple",
                "create one fixture note",
                "Localized and reversible, with no persistence, concurrency, "
                "or API concerns.",
            ).value,
            "simple",
        )
        for value in (
            "No persistence, concurrency, or security concerns.",
            "Without known migration or database risks.",
            "This does not introduce persistence risks.",
            "Read-only audit requiring no edits or external writes.",
            "No schema migration or database changes are needed.",
            "External writes are not required.",
        ):
            with self.subTest(value=value):
                self.assertIsNone(
                    production_jobs._hard_risk_evidence(value, "localized")
                )
        self.assertEqual(
            production_jobs._hard_risk_evidence(
                "rename a label",
                "Requires persistence and concurrency changes.",
            ),
            "classification or promotion reason contains 'persistence'",
        )
        self.assertEqual(
            production_jobs._hard_risk_evidence(
                "rename a label",
                "No schema migration, but external writes are required.",
            ),
            "classification or promotion reason contains 'external write'",
        )
        self.assertEqual(
            production_jobs._hard_risk_evidence(
                "rename a label",
                "This requires not only security review but external writes.",
            ),
            "classification or promotion reason contains 'security'",
        )
        with self.assertRaises(jobs.HarnessError):
            production_jobs._classified_tier("unknown", "rename text", "localized")

    def test_malformed_classification_output_blocks_without_implementation(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "rename text",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_phase": "classifying",
                "planner_target": "planner",
                "active_participant": "planner",
                "herdr_target": "planner",
                "turn_token": "current",
            }
        )
        store = jobs._store()

        production_jobs._advance_workflow_output(
            store,
            store.get("123456789abc"),
            LEGACY_WORKER,
            "WORKFLOW_TIER[stale]: simple\nWORKFLOW_REASON[stale]: old",
            "idle",
        )

        updated = store.get("123456789abc")
        self.assertEqual(updated.status, JobStatus.BLOCKED)
        self.assertIsNone(updated.participant_target(WorkflowParticipant.IMPLEMENTER))

    def test_classification_user_question_pauses_current_phase(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "unclear request",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_phase": "classifying",
                "planner_target": "planner",
                "active_participant": "planner",
                "herdr_target": "planner",
                "turn_token": "current",
                "phase_prompt_active": True,
            }
        )
        store = jobs._store()

        production_jobs._advance_workflow_output(
            store,
            store.get("123456789abc"),
            LEGACY_WORKER,
            "VOICE_QUESTION[current]: Which behavior should win?",
            "idle",
        )

        updated = store.get("123456789abc")
        self.assertEqual(updated.status, JobStatus.AWAITING_USER)
        self.assertEqual(updated.prompt_operation_state, "none")

    def test_same_owner_stale_workflow_callback_cannot_advance_phase(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "rename text",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_phase": "classifying",
                "planner_target": "planner",
                "active_participant": "planner",
                "herdr_target": "planner",
                "turn_token": "current",
            }
        )
        store = jobs._store()
        observed = store.get("123456789abc")
        store.update(observed.id, lambda current: current.evolve(reconcile=True))

        advanced = production_jobs._advance_workflow_output(
            store,
            observed,
            LEGACY_WORKER,
            "WORKFLOW_TIER[current]: simple\n"
            "WORKFLOW_REASON[current]: localized change",
            "idle",
        )

        self.assertIsNone(advanced)
        self.assertEqual(
            store.get(observed.id).workflow_phase, WorkflowPhase.CLASSIFYING
        )

    def test_malformed_review_output_blocks_approval_gate(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "change recovery",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "reviewing",
                "reviewer_target": "reviewer",
                "active_participant": "reviewer",
                "herdr_target": "reviewer",
                "turn_token": "current",
            }
        )
        store = jobs._store()

        production_jobs._advance_workflow_output(
            store,
            store.get("123456789abc"),
            LEGACY_WORKER,
            "WORKFLOW_REVIEW_DECISION[current]: maybe",
            "idle",
        )

        self.assertEqual(store.get("123456789abc").status, JobStatus.BLOCKED)

    def test_implementation_rejects_nonincreasing_promotion(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "rename text",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "simple",
                "workflow_classification_reason": "localized",
                "workflow_phase": "implementing",
                "planner_target": "planner",
                "implementer_target": "implementer",
                "active_participant": "implementer",
                "herdr_target": "implementer",
                "turn_token": "current",
            }
        )
        store = jobs._store()

        production_jobs._advance_workflow_output(
            store,
            store.get("123456789abc"),
            LEGACY_WORKER,
            "WORKFLOW_PROMOTE[current]: simple\n"
            "WORKFLOW_REASON[current]: no additional risk",
            "idle",
        )

        self.assertEqual(store.get("123456789abc").status, JobStatus.BLOCKED)

    def test_promotion_carries_clarification_without_mutating_request(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "rename text",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "simple",
                "workflow_classification_reason": "localized",
                "workflow_phase": "implementing",
                "planner_target": "planner",
                "implementer_target": "implementer",
                "active_participant": "implementer",
                "herdr_target": "implementer",
                "turn_token": "current",
                "continuation_answer": "User answered: preserve compatibility",
            }
        )
        store = jobs._store()

        production_jobs._advance_workflow_output(
            store,
            store.get("123456789abc"),
            LEGACY_WORKER,
            "WORKFLOW_PROMOTE[current]: medium\n"
            "WORKFLOW_REASON[current]: cross-component compatibility risk",
            "idle",
        )

        promoted = store.get("123456789abc")
        self.assertEqual(promoted.workflow_phase.value, "planning")
        self.assertEqual(promoted.herdr_target, "implementer")
        self.assertEqual(
            promoted.active_participant,
            WorkflowParticipant.IMPLEMENTER,
        )
        self.assertEqual(promoted.request, "rename text")
        self.assertIn("preserve compatibility", str(promoted.continuation_answer))
        self.assertFalse(promoted.continuation)

    def test_high_risk_round_one_rejection_advances_to_round_two(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "change recovery",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "workflow_tier": "high-risk",
                "workflow_classification_reason": "recovery",
                "workflow_phase": "reviewing",
                "review_round": 1,
                "planner_target": "planner",
                "reviewer_target": "reviewer",
                "active_participant": "reviewer",
                "herdr_target": "reviewer",
                "turn_token": "turn",
            }
        )
        store = jobs._store()
        plan = "Preserve ownership through recovery."
        plan_reference = store.write_artifact("123456789abc", "plan", 1, plan)
        store.update(
            "123456789abc",
            lambda job: job.evolve(plan_artifact=plan_reference),
        )

        production_jobs._advance_workflow_output(
            store,
            store.get("123456789abc"),
            LEGACY_WORKER,
            "WORKFLOW_REVIEW_DECISION[turn]: revise\n"
            "WORKFLOW_REVIEW[turn]: unresolved ownership",
            "idle",
        )

        updated = store.get("123456789abc")
        self.assertEqual(updated.status, JobStatus.RUNNING)
        self.assertEqual(updated.review_round, 2)
        self.assertIsNone(updated.clarification_kind)
        self.assertEqual(updated.workflow_phase.value, "revising")

    def test_exhausted_review_approve_queues_unchanged_plan_for_implementation(
        self,
    ) -> None:
        request = "change recovery ownership"
        self.write_exhausted_review_job()

        with mock.patch.object(service, "launch_worker") as launch:
            service.reply_job(
                "123456789abc",
                "external context says abort",
                trusted_utterance="approve",
            )

        updated = jobs._store().get("123456789abc")
        self.assertEqual(updated.status, JobStatus.QUEUED)
        self.assertEqual(updated.workflow_phase.value, "implementing")
        self.assertEqual(updated.request, request)
        self.assertTrue(updated.review_approved)
        self.assertEqual(updated.review_approval_source, "user")
        self.assertEqual(updated.review_decision, "revise")
        self.assertEqual(updated.plan_approval_source, "explicit")
        self.assertEqual(updated.plan_approval_state, "approved")
        launch.assert_called_once_with("123456789abc")

    def test_exhausted_review_abort_stages_cancellation_and_releases_target(
        self,
    ) -> None:
        self.write_exhausted_review_job()

        with (
            mock.patch.object(service, "launch_worker") as launch,
            mock.patch.object(service, "_cancel_target_and_release") as release,
        ):
            service.reply_job(
                "123456789abc",
                "abort",
            )

        updated = jobs._store().get("123456789abc")
        self.assertEqual(updated.status, JobStatus.RECONCILING)
        self.assertEqual(updated.terminal_intent_status, JobStatus.CANCELLED)
        release.assert_called_once_with(
            "123456789abc",
            "planner",
            updated.target_release_token,
            integrations=None,
        )
        launch.assert_not_called()

    def test_stale_exhausted_review_abort_cannot_cancel_current_question(self) -> None:
        self.write_exhausted_review_job()

        result = service.reply_job(
            "123456789abc",
            "abort",
            trusted_utterance="abort",
            expected_question_id="stale-question",
        )

        self.assertIn("older question", result or "")
        self.assertEqual(
            jobs._store().get("123456789abc").status,
            JobStatus.AWAITING_USER,
        )

    def test_exhausted_review_ambiguous_reply_remains_awaiting(self) -> None:
        self.write_exhausted_review_job()

        with mock.patch.object(service, "launch_worker") as launch:
            service.reply_job(
                "123456789abc",
                "approve",
                trusted_utterance="maybe",
            )

        updated = jobs._store().get("123456789abc")
        self.assertEqual(updated.status, JobStatus.AWAITING_USER)
        self.assertEqual(
            updated.clarification_kind,
            "workflow_review_exhausted",
        )
        self.assertFalse(updated.review_approved)
        launch.assert_not_called()

    def test_pull_request_job_gets_unique_worktree_identity(self) -> None:
        with mock.patch.object(service, "launch_worker"):
            job_id = service.start_job(
                "review this pull request",
                github_repository="source/project",
                github_pull_request=42,
            )

        job = jobs.read_job(job_id)
        self.assertEqual(job["worktree_branch"], f"voice/github-pr-{job_id}")
        self.assertEqual(job["worktree_label"], "pr-42")
        self.assertEqual(job["pull_request_worktree_state"], "pending")

    def test_worker_checks_out_pull_request_only_in_reserved_worktree(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        worktree = Path(self.temporary.name) / "worktrees" / "pr-42"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: shared/worktrees/pr-42\n")
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "review pull request 42",
                "github_repository": "source/project",
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "worktree_label": "pr-42",
                "pull_request_worktree_state": "pending",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.inspect_repository.return_value = source
        github.pull_request_details.return_value = {"headRefOid": "a" * 40}
        github.ensure_repository_clone.return_value = repository
        github.local_git.checkout_remote_ref.return_value = (
            "voice/github-pr-123456789abc"
        )
        client = mock.Mock()
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(worktree),
            "agent",
            str(worktree),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        configure_tiered_outcomes(client, worktree)

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        self.assertEqual(
            github.inspect_repository.call_args_list,
            [mock.call("source/project"), mock.call("source/project")],
        )
        github.ensure_repository_clone.assert_called_once_with(
            source, checkpoint=mock.ANY
        )
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-pr-123456789abc",
            worktree_label="pr-42",
            mode="plan",
            checkpoint=mock.ANY,
            reserve=mock.ANY,
            settle=mock.ANY,
            fail_agent=mock.ANY,
            reserve_worktree=mock.ANY,
            settle_worktree=mock.ANY,
            fail_worktree=mock.ANY,
            plan_participant=mock.ANY,
            before_pane_submit=mock.ANY,
            pane_accepted=mock.ANY,
            participant_name=None,
            start_session=False,
        )
        github.local_git.checkout_remote_ref.assert_called_once_with(
            worktree,
            remote_url="https://github.com/source/project",
            remote_ref="refs/pull/42/head",
            branch="voice/github-pr-123456789abc",
            expected_oid="a" * 40,
            checkpoint=mock.ANY,
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed", updated)
        self.assertEqual(updated["worktree_path"], str(worktree))
        self.assertEqual(updated["pull_request_worktree_state"], "retained")

    def test_pull_request_shared_clone_is_quarantined_without_checkout(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        repository.mkdir(parents=True)
        (repository / ".git").mkdir()
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "repository": str(repository),
                "worktree_path": str(repository),
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "pull_request_worktree_state": "provisioning",
            }
        )
        github = mock.Mock()

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            self.assertRaisesRegex(jobs.HarnessError, "shared repository clone"),
        ):
            jobs._prepare_pull_request_checkout(
                "123456789abc",
                LEGACY_WORKER,
                jobs.read_job("123456789abc"),
            )

        github.local_git.checkout_remote_ref.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["pull_request_worktree_state"], "quarantined")

    def test_legacy_pull_request_job_resolves_and_persists_checkout_inputs(
        self,
    ) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        worktree = Path(self.temporary.name) / "worktrees" / "legacy-pr"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: shared\n")
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "repository": str(repository),
                "worktree_path": str(worktree),
                "github_repository": "source/project",
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "pull_request_worktree_state": "provisioning",
            }
        )
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github = mock.Mock()
        github.inspect_repository.return_value = source
        github.pull_request_details.return_value = {"headRefOid": "a" * 40}
        github.local_git.checkout_remote_ref.return_value = (
            "voice/github-pr-123456789abc"
        )

        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            jobs._prepare_pull_request_checkout(
                "123456789abc",
                LEGACY_WORKER,
                jobs.read_job("123456789abc"),
            )

        updated = jobs.read_job("123456789abc")
        self.assertEqual(
            updated["pull_request_remote_url"],
            "https://github.com/source/project",
        )
        self.assertEqual(updated["pull_request_head_ref"], "refs/pull/42/head")
        self.assertEqual(updated["pull_request_head_oid"], "a" * 40)
        self.assertEqual(updated["pull_request_worktree_state"], "ready")

    def test_pull_request_head_move_is_re_resolved_once_before_quarantine(
        self,
    ) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        worktree = Path(self.temporary.name) / "worktrees" / "moving-pr"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: shared\n")
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "repository": str(repository),
                "worktree_path": str(worktree),
                "github_repository": "source/project",
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "pull_request_worktree_state": "provisioning",
            }
        )
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github = mock.Mock()
        github.inspect_repository.return_value = source
        github.pull_request_details.side_effect = [
            {"headRefOid": "a" * 40},
            {"headRefOid": "b" * 40},
        ]
        github.local_git.checkout_remote_ref.side_effect = [
            LocalGitRefChanged("head moved"),
            "voice/github-pr-123456789abc",
        ]

        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            jobs._prepare_pull_request_checkout(
                "123456789abc",
                LEGACY_WORKER,
                jobs.read_job("123456789abc"),
            )

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["pull_request_head_oid"], "b" * 40)
        self.assertEqual(updated["pull_request_worktree_state"], "ready")
        self.assertEqual(github.local_git.checkout_remote_ref.call_count, 2)

    def test_pull_request_checkout_rejects_non_voice_branch_before_mutation(
        self,
    ) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        worktree = Path(self.temporary.name) / "worktrees" / "unsafe-pr"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: shared\n")
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "repository": str(repository),
                "worktree_path": str(worktree),
                "github_repository": "source/project",
                "github_pull_request": 42,
                "worktree_branch": "main",
                "pull_request_worktree_state": "provisioning",
                "pull_request_remote_url": "https://github.com/source/project",
                "pull_request_head_ref": "refs/pull/42/head",
                "pull_request_head_oid": "a" * 40,
            }
        )
        github = mock.Mock()

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            self.assertRaisesRegex(jobs.HarnessError, "invalid voice"),
        ):
            jobs._prepare_pull_request_checkout(
                "123456789abc",
                LEGACY_WORKER,
                jobs.read_job("123456789abc"),
            )

        github.local_git.checkout_remote_ref.assert_not_called()
        self.assertEqual(
            jobs.read_job("123456789abc")["pull_request_worktree_state"],
            "quarantined",
        )

    def test_concurrent_pull_requests_prepare_distinct_worktrees(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        barrier = threading.Barrier(2)
        calls: list[tuple[Path, str, str]] = []
        for job_id, number in (("aaaaaaaaaaaa", 41), ("bbbbbbbbbbbb", 42)):
            worktree = Path(self.temporary.name) / "worktrees" / job_id
            worktree.mkdir(parents=True)
            (worktree / ".git").write_text("gitdir: shared\n")
            jobs.write_job(
                {
                    "id": job_id,
                    "status": "routing",
                    "worker_token": job_id,
                    "worker_pid": 42,
                    "worker_process_start": f"start-{job_id}",
                    "worker_claim_operation": "test",
                    "worker_claimed_at": 1,
                    "repository": str(repository),
                    "worktree_path": str(worktree),
                    "github_repository": "source/project",
                    "github_pull_request": number,
                    "worktree_branch": f"voice/github-pr-{job_id}",
                    "pull_request_worktree_state": "provisioning",
                    "pull_request_remote_url": "https://github.com/source/project",
                    "pull_request_head_ref": f"refs/pull/{number}/head",
                    "pull_request_head_oid": "a" * 40,
                }
            )
        github = mock.Mock()
        github.inspect_repository.return_value = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github.pull_request_details.return_value = {"headRefOid": "a" * 40}

        def checkout(
            path: Path,
            *,
            remote_ref: str,
            branch: str,
            **_kwargs: object,
        ) -> str:
            calls.append((path, remote_ref, branch))
            barrier.wait()
            return branch

        github.local_git.checkout_remote_ref.side_effect = checkout

        def prepare(job_id: str) -> None:
            jobs._prepare_pull_request_checkout(
                job_id,
                job_id,
                jobs.read_job(job_id),
            )

        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            threads = [
                threading.Thread(target=prepare, args=(job_id,))
                for job_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len({path for path, _ref, _branch in calls}), 2)
        self.assertEqual(len({branch for _path, _ref, branch in calls}), 2)
        self.assertTrue(
            all(
                jobs.read_job(job_id)["pull_request_worktree_state"] == "ready"
                for job_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
            )
        )

    def test_worker_provisions_fork_and_uses_dedicated_worktree(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        fork = GitHubRepository(
            "me/project",
            "https://github.com/me/project",
            False,
            "main",
            "source/project",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "Fork this repo and add a feature",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "worktree_branch": "voice/github-123456789abc",
                "worktree_label": "github-123456",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.prepare_public_fork.return_value = (source, "me", "me/project")

        def ensure_fork(
            *_args: object, before_submit: Callable[[], None], **_kwargs: object
        ) -> GitHubRepository:
            before_submit()
            return fork

        github.ensure_fork.side_effect = ensure_fork
        github.ensure_clone.return_value = repository
        client = mock.Mock()
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(repository),
            "agent",
            "/worktree",
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        configure_tiered_outcomes(client, repository, tier="high-risk")
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        github.prepare_public_fork.assert_called_once_with("source/project")
        github.ensure_fork.assert_called_once_with(
            source,
            "me",
            checkpoint=mock.ANY,
            before_submit=mock.ANY,
        )
        github.ensure_clone.assert_called_once_with(source, fork, checkpoint=mock.ANY)
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-123456789abc",
            worktree_label="github-123456",
            mode="plan",
            checkpoint=mock.ANY,
            reserve=mock.ANY,
            settle=mock.ANY,
            fail_agent=mock.ANY,
            reserve_worktree=mock.ANY,
            settle_worktree=mock.ANY,
            fail_worktree=mock.ANY,
            plan_participant=mock.ANY,
            before_pane_submit=mock.ANY,
            pane_accepted=mock.ANY,
            participant_name=None,
            start_session=False,
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "awaiting_user")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertEqual(updated["repository"], str(repository))
        self.assertEqual(updated["workflow_tier"], "high-risk")
        self.assertEqual(updated["workflow_phase"], "reviewing")
        self.assertEqual(updated["clarification_kind"], "workflow_plan_approval")
        self.assertEqual(updated["plan_approval_state"], "awaiting")
        self.assertTrue(updated["plan_artifact"])
        self.assertTrue(updated["review_artifact"])
        self.assertEqual(
            [call.kwargs["role"] for call in client.start_fresh_agent.call_args_list],
            ["reviewer"],
        )
        self.assertEqual(
            [call.kwargs["mode"] for call in client.start_fresh_agent.call_args_list],
            ["ask"],
        )
        client.close_owned_pane.assert_called_once_with(
            "agent",
            "pane",
            "workspace",
        )

        with mock.patch.object(service, "launch_worker"):
            service.reply_job("123456789abc", "yes", trusted_utterance="yes")
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed", updated)
        self.assertEqual(
            [call.kwargs["role"] for call in client.start_fresh_agent.call_args_list],
            ["reviewer", "implementer"],
        )
        self.assertEqual(
            [call.kwargs["mode"] for call in client.start_fresh_agent.call_args_list],
            ["ask", None],
        )
        participant_targets = [
            call.kwargs["name"] for call in client.start_fresh_agent.call_args_list
        ]
        self.assertEqual(
            client.close_owned_pane.call_args_list,
            [
                mock.call("agent", "pane", "workspace"),
                mock.call(participant_targets[0], "pane-reviewer", "workspace"),
                mock.call(participant_targets[1], "pane-implementer", "workspace"),
            ],
        )
        implementation_prompt_text = cast(list[HarnessTask], client._harness_tasks)[
            -1
        ].text
        self.assertNotIn("lgtm", implementation_prompt_text)
        self.assertIn(
            "Implement only from this approved plan", implementation_prompt_text
        )

    def test_reconciled_fork_is_never_resubmitted(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        fork = GitHubRepository(
            "me/project",
            "https://github.com/me/project",
            False,
            "main",
            "source/project",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "continue after fork recovery",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "fork_operation_state": "exists",
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_target": "me/project",
                "fork_repository": "me/project",
                "worktree_branch": "voice/github-123456789abc",
                "worktree_label": "github-123456",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.prepare_public_fork.return_value = (source, "me", "me/project")
        github.inspect_public_repository.return_value = source
        github.reconcile_fork.return_value = fork
        github.ensure_clone.return_value = repository
        client = mock.Mock()
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(repository),
            "agent",
            "/worktree",
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        configure_tiered_outcomes(client, repository)

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        github.reconcile_fork.assert_called_once_with(source, "me/project")
        github.prepare_public_fork.assert_not_called()
        github.ensure_fork.assert_not_called()
        github.ensure_clone.assert_called_once_with(source, fork, checkpoint=mock.ANY)

    def test_worker_requires_confirmation_before_provisioning_fork(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork this repo",
                "trusted_utterance": "fork this repo",
                "github_repository": "source/project",
                "fork_requested": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        client = mock.Mock()
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        github.prepare_public_fork.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "awaiting_user")
        self.assertEqual(updated["clarification_kind"], "fork_confirmation")
        self.assertIn("source/project", str(updated["question"]))

    def test_worker_drafts_issue_before_requesting_confirmation(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create an issue about startup",
                "trusted_utterance": "create an issue about startup",
                "issue_provider": "github",
                "github_repository": "source/project",
                "github_issue_create_requested": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github = mock.Mock()
        github.inspect_repository.return_value = source
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(
                production_jobs,
                "draft_github_issue",
                return_value=GitHubIssueDraft(
                    "source/project",
                    "Fix startup",
                    "Startup fails after reboot.",
                ),
            ),
        ):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "awaiting_user")
        self.assertEqual(
            updated["clarification_kind"],
            "github_issue_create_confirmation",
        )
        self.assertEqual(updated["github_issue_create_title"], "Fix startup")
        self.assertEqual(updated["github_issue_create_operation_state"], "planned")
        github.submit_issue.assert_not_called()

    def test_confirmed_issue_creation_completes_without_starting_herdr(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create an issue about startup",
                "trusted_utterance": "create an issue about startup",
                "issue_provider": "github",
                "github_repository": "source/project",
                "github_issue_create_requested": True,
                "github_issue_create_confirmed": True,
                "github_issue_create_title": "Fix startup",
                "github_issue_create_body": "Startup fails after reboot.",
                "github_issue_create_marker": "a" * 32,
                "github_issue_create_operation_state": "planned",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        result = GitHubIssueCreationResult(
            GitHubIssue("source", "project", 42),
            "https://github.com/source/project/issues/42",
            "a" * 32,
        )
        github = mock.Mock()
        github.inspect_repository.return_value = source
        github.submit_issue.return_value = result
        herdr = mock.Mock()
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=herdr),
        ):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["github_issue_created_number"], 42)
        self.assertEqual(updated["github_issue_create_operation_state"], "created")
        herdr.assert_not_called()

    def test_stale_same_owner_issue_submission_result_is_rejected(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create an issue about startup",
                "trusted_utterance": "create an issue about startup",
                "issue_provider": "github",
                "github_repository": "source/project",
                "github_issue_create_requested": True,
                "github_issue_create_confirmed": True,
                "github_issue_create_title": "Fix startup",
                "github_issue_create_body": "Startup fails after reboot.",
                "github_issue_create_marker": "a" * 32,
                "github_issue_create_operation_state": "planned",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        result = GitHubIssueCreationResult(
            GitHubIssue("source", "project", 42),
            "https://github.com/source/project/issues/42",
            "a" * 32,
        )
        github = mock.Mock()
        github.inspect_repository.return_value = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )

        def submit_issue(
            *_args: object, **_kwargs: object
        ) -> GitHubIssueCreationResult:
            jobs._store().update(
                "123456789abc", lambda current: current.evolve(reconcile=True)
            )
            return result

        github.submit_issue.side_effect = submit_issue

        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["github_issue_create_operation_state"], "submitted")
        self.assertIsNone(updated.get("github_issue_created_number"))
        self.assertTrue(updated["reconcile"])

    def test_timed_out_issue_creation_is_reconciled_without_resubmission(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create an issue",
                "trusted_utterance": "create an issue",
                "issue_provider": "github",
                "github_repository": "source/project",
                "github_issue_create_requested": True,
                "github_issue_create_confirmed": True,
                "github_issue_create_title": "Fix startup",
                "github_issue_create_body": "Startup fails.",
                "github_issue_create_marker": "a" * 32,
                "github_issue_create_operation_state": "planned",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.inspect_repository.return_value = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github.submit_issue.side_effect = GitHubOperationAmbiguous("timed out")
        github.observe_issue.return_value = None
        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertEqual(updated["github_issue_create_operation_state"], "ambiguous")
        self.assertEqual(github.submit_issue.call_count, 1)

    def test_worker_drafts_linear_ticket_before_requesting_confirmation(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create a Linear ticket about startup",
                "trusted_utterance": "create a Linear ticket about startup",
                "issue_provider": "linear",
                "linear_ticket_create_requested": True,
                "linear_ticket_create_team": "API",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        provider = linear.LinearIntegration()
        herdr = mock.Mock()
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=herdr),
            mock.patch.object(
                production_jobs,
                "issue_provider",
                return_value=provider,
            ),
            mock.patch.object(
                provider,
                "resolve_team",
                return_value=LinearTeam("team-id-api", "API", "Application Platform"),
            ),
            mock.patch.object(
                production_jobs,
                "draft_linear_ticket",
                return_value=LinearTicketDraft(
                    "API",
                    "Fix startup",
                    "Startup fails after reboot.",
                ),
            ),
        ):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "awaiting_user")
        self.assertEqual(
            updated["clarification_kind"],
            "linear_ticket_create_confirmation",
        )
        self.assertEqual(updated["linear_ticket_create_title"], "Fix startup")
        self.assertEqual(updated["linear_ticket_create_operation_state"], "planned")
        herdr.ensure_router.assert_not_called()

    def test_worker_says_it_could_not_find_the_linear_team(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create a Linear ticket in team API about startup",
                "trusted_utterance": "create a Linear ticket in team API about startup",
                "issue_provider": "linear",
                "linear_ticket_create_requested": True,
                "linear_ticket_create_team": "API",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        provider = linear.LinearIntegration()
        herdr = mock.Mock()
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=herdr),
            mock.patch.object(
                production_jobs,
                "issue_provider",
                return_value=provider,
            ),
            mock.patch.object(
                provider,
                "resolve_team",
                side_effect=LinearError("I couldn't find Linear team API."),
            ),
            mock.patch.object(production_jobs, "draft_linear_ticket") as draft,
        ):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["result"], "I couldn't find Linear team API.")
        draft.assert_not_called()
        herdr.ensure_router.assert_not_called()

    def test_confirmed_linear_ticket_creation_completes(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create a Linear ticket about startup",
                "trusted_utterance": "create a Linear ticket about startup",
                "issue_provider": "linear",
                "linear_ticket_create_requested": True,
                "linear_ticket_create_confirmed": True,
                "linear_ticket_create_team": "API",
                "linear_ticket_create_team_id": "team-id-api",
                "linear_ticket_create_title": "Fix startup",
                "linear_ticket_create_description": "Startup fails after reboot.",
                "linear_ticket_create_marker": "a" * 32,
                "linear_ticket_create_operation_state": "planned",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        provider = linear.LinearIntegration()
        result = LinearTicketCreationResult(
            LinearIssue("API-42"),
            "https://linear.app/acme/issue/API-42/fix-startup",
            "a" * 32,
        )
        herdr = mock.Mock()

        def submit(*_args: object, **kwargs: object) -> LinearTicketCreationResult:
            before_submit = kwargs["before_submit"]
            accepted = kwargs["accepted"]
            assert callable(before_submit)
            assert callable(accepted)
            before_submit("voice-router", "router-session", "prompt-token", 7)
            accepted()
            return result

        with (
            mock.patch.object(jobs, "HerdrClient", return_value=herdr),
            mock.patch.object(
                production_jobs,
                "issue_provider",
                return_value=provider,
            ),
            mock.patch.object(
                provider,
                "submit_ticket_creation",
                side_effect=submit,
            ) as submit,
            mock.patch.object(
                provider,
                "observe_ticket_creation",
                return_value=result,
            ) as observe,
        ):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["linear_ticket_created_identifier"], "API-42")
        self.assertEqual(updated["linear_ticket_create_operation_state"], "created")
        submit.assert_called_once_with(
            herdr,
            mock.ANY,
            confirmed=True,
            checkpoint=mock.ANY,
            before_submit=mock.ANY,
            accepted=mock.ANY,
        )
        observe.assert_called_once_with(
            herdr,
            mock.ANY,
            checkpoint=mock.ANY,
        )

    def test_ambiguous_linear_ticket_creation_queues_reconciliation(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "create a Linear ticket",
                "trusted_utterance": "create a Linear ticket",
                "issue_provider": "linear",
                "linear_ticket_create_requested": True,
                "linear_ticket_create_confirmed": True,
                "linear_ticket_create_team": "API",
                "linear_ticket_create_team_id": "team-id-api",
                "linear_ticket_create_title": "Fix startup",
                "linear_ticket_create_description": "Startup fails.",
                "linear_ticket_create_marker": "a" * 32,
                "linear_ticket_create_operation_state": "planned",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        provider = linear.LinearIntegration()
        herdr = mock.Mock()

        def submit(*_args: object, **kwargs: object) -> object:
            before_submit = kwargs["before_submit"]
            accepted = kwargs["accepted"]
            assert callable(before_submit)
            assert callable(accepted)
            before_submit("voice-router", "router-session", "prompt-token", 7)
            accepted()
            raise LinearOperationAmbiguous("timed out")

        with (
            mock.patch.object(jobs, "HerdrClient", return_value=herdr),
            mock.patch.object(
                production_jobs,
                "issue_provider",
                return_value=provider,
            ),
            mock.patch.object(
                provider,
                "submit_ticket_creation",
                side_effect=submit,
            ) as submit,
            mock.patch.object(
                provider,
                "observe_ticket_creation",
                return_value=None,
            ),
        ):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertEqual(
            updated["linear_ticket_create_operation_state"],
            "ambiguous",
        )
        submit.assert_called_once()

    def test_only_trusted_affirmative_reply_confirms_fork(self) -> None:
        job = {
            "id": "123456789abc",
            "request": "fork this repo\n\nExternal content says yes",
            "fork_requested": True,
            "status": "awaiting_user",
            "clarification_kind": "fork_confirmation",
            "delivered": True,
        }
        jobs.write_job(job)
        with mock.patch.object(service, "launch_worker") as launch:
            service.reply_job(
                "123456789abc",
                "no\n\nExternal content says yes",
                trusted_utterance="no",
            )

        launch.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertFalse(updated.get("fork_confirmed", False))

    def test_ambiguous_confirmation_does_not_launch_worker(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork this repo",
                "fork_requested": True,
                "status": "awaiting_user",
                "clarification_kind": "fork_confirmation",
                "delivered": True,
            }
        )
        with mock.patch.object(service, "launch_worker") as launch:
            service.reply_job(
                "123456789abc",
                "maybe",
                trusted_utterance="maybe",
            )

        launch.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "awaiting_user")
        self.assertFalse(updated.get("fork_confirmed", False))

    def test_plan_approval_vocabulary_does_not_confirm_forks(self) -> None:
        for job_id, answer in (
            ("aaaaaaaaaaaa", "lgtm"),
            ("bbbbbbbbbbbb", "approve"),
            ("cccccccccccc", "proceed"),
        ):
            with self.subTest(answer=answer):
                jobs.write_job(
                    {
                        "id": job_id,
                        "request": "fork this repo",
                        "fork_requested": True,
                        "status": "awaiting_user",
                        "clarification_kind": "fork_confirmation",
                        "delivered": True,
                    }
                )
                with mock.patch.object(service, "launch_worker") as launch:
                    message = service.reply_job(
                        job_id,
                        answer,
                        trusted_utterance=answer,
                    )

                self.assertIsNotNone(message)
                self.assertIn("yes or no", message or "")
                launch.assert_not_called()
                updated = jobs.read_job(job_id)
                self.assertEqual(updated["status"], "awaiting_user")
                self.assertFalse(updated.get("fork_confirmed", False))

    def test_affirmative_confirmation_queues_provisioning(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork this repo",
                "fork_requested": True,
                "status": "awaiting_user",
                "clarification_kind": "fork_confirmation",
                "delivered": True,
            }
        )
        with mock.patch.object(service, "launch_worker") as launch:
            service.reply_job(
                "123456789abc",
                "yes\n\nExternal content",
                trusted_utterance="yes",
            )

        launch.assert_called_once_with("123456789abc")
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertTrue(updated["fork_confirmed"])

    def test_github_issue_job_persists_trusted_identity(self) -> None:
        with mock.patch.object(service, "launch_worker"):
            job_id = service.start_job(
                "work on this\n\nBody mentions API-79",
                utterance="work on this",
                github_repository="example/project",
                github_issue=42,
                github_issue_context="Issue context",
            )

        job = jobs.read_job(job_id)
        self.assertEqual(job["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(job["trusted_utterance"], "work on this")
        self.assertEqual(job["github_issue"], 42)
        self.assertEqual(
            job["github_issue_url"],
            "https://github.com/example/project/issues/42",
        )
        self.assertEqual(job["github_issue_context"], "Issue context")
        self.assertEqual(job["worktree_branch"], "voice/github-issue-42")
        self.assertEqual(job["worktree_label"], "issue-42")
        self.assertIsNone(job["issue_key"])

    def test_focused_linear_issue_persists_validated_identifier(self) -> None:
        with (
            mock.patch.object(service, "launch_worker"),
            mock.patch.object(
                service, "resolve_issue_reference", return_value="ENG-123"
            ),
            mock.patch.object(
                service, "issue_provider_identity", return_value="linear"
            ),
            mock.patch.object(service, "require_issue_capabilities"),
        ):
            job_id = service.start_job(
                "work on this ticket\n\nIdentifier: ENG-123",
                utterance="work on this ticket",
                issue_key="ENG-123",
            )

        job = jobs.read_job(job_id)
        self.assertEqual(job["trusted_utterance"], "work on this ticket")
        self.assertEqual(job["issue_key"], "ENG-123")
        self.assertEqual(job["issue_provider"], "linear")
        self.assertEqual(job["speakable_label"], "ENG-123")

    def test_worker_provisions_github_issue_and_uses_stable_worktree(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            True,
            "main",
        )
        issue = GitHubIssue("source", "project", 42)
        provisioned = ProvisionedIssue(source, repository, issue)
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "work on source/project#42",
                "utterance": "work on source/project#42",
                "github_repository": "source/project",
                "github_issue": 42,
                "github_issue_context": "Title: Fix it",
                "worktree_branch": "voice/github-issue-42",
                "worktree_label": "issue-42",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.provision_issue.return_value = provisioned
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(repository),
            "agent",
            "/worktree/issue-42",
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        configure_tiered_outcomes(client, repository)
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        github.provision_issue.assert_called_once_with(
            issue, candidates=[repository], checkpoint=mock.ANY
        )
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-issue-42",
            worktree_label="issue-42",
            mode="plan",
            checkpoint=mock.ANY,
            reserve=mock.ANY,
            settle=mock.ANY,
            fail_agent=mock.ANY,
            reserve_worktree=mock.ANY,
            settle_worktree=mock.ANY,
            fail_worktree=mock.ANY,
            plan_participant=mock.ANY,
            before_pane_submit=mock.ANY,
            pane_accepted=mock.ANY,
            participant_name=None,
            start_session=False,
        )
        prompt = cast(list[HarnessTask], client._harness_tasks)[-1].text
        self.assertIn("Title: Fix it", prompt)
        self.assertIn(
            'the summary must be exactly "I\'ve finished working on issue 42"',
            prompt,
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["github_issue_url"], issue.url)

    def test_reservation_rejects_an_active_shared_worktree(self) -> None:
        jobs.write_job(
            {
                "id": "aaaaaaaaaaaa",
                "status": "running",
                "worker_token": "first-worker",
                "worker_pid": 41,
                "worker_process_start": "first-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "herdr_target": "first-agent",
                "worktree_path": "/worktree/issue-42",
            }
        )
        jobs.write_job(
            {
                "id": "bbbbbbbbbbbb",
                "status": "routing",
                "worker_token": "worker-token",
                "worker_pid": 42,
                "worker_process_start": "second-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
            }
        )
        selection = AgentSelection(
            "second-agent",
            "pane",
            "workspace",
            "/repo",
            "second-agent",
            "/worktree/issue-42",
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )

        reserved = jobs._reserve_worker_target(
            "bbbbbbbbbbbb",
            "worker-token",
            selection,
            Path("/repo"),
            None,
            expected_revision=0,
        )

        self.assertIsNone(reserved)
        self.assertNotIn("herdr_target", jobs.read_job("bbbbbbbbbbbb"))

    def test_submit_starts_new_job_despite_existing_session(self) -> None:
        with (
            mock.patch.object(
                service, "start_job", return_value="abcdef123456"
            ) as start,
            mock.patch.object(
                service,
                "read_job",
                return_value=jobs._typed(
                    {
                        "id": "abcdef123456",
                        "status": "completed",
                        "result": "done",
                    }
                ),
            ),
            mock.patch.object(service, "claim_delivery", return_value=None),
            mock.patch.object(service, "reply_job") as reply,
        ):
            result, session = service.cursor_turn(
                "Work on APP-43", session_id="oldjob123456"
            )

        start.assert_called_once_with(
            "Work on APP-43",
            repository=None,
            github_repository=None,
            github_issue=None,
            github_issue_context=None,
            github_issue_create_requested=False,
            linear_team=None,
            linear_ticket_create_requested=False,
            fork_requested=False,
            github_pull_request=None,
            agent=None,
            utterance=None,
            context_repository=None,
            issue_key=None,
            foreground_seconds=5.0,
            integrations=mock.ANY,
        )
        reply.assert_not_called()
        self.assertEqual(result, "done")
        self.assertIsNone(session)

    def test_spoken_repository_precedes_focused_context_fallback(self) -> None:
        spoken = Path("/repos/spoken")
        focused = Path("/repos/focused")
        client = mock.Mock()
        client.resolve_repository.return_value = (spoken, [spoken])
        job: dict[str, object] = {
            "request": "work on this\n\nRepository: owner/focused",
            "utterance": "work on the spoken repository",
            "context_repository": "owner/focused",
        }

        repository, candidates = jobs.resolve_job_repository(
            client, job, [spoken, focused]
        )

        self.assertEqual(repository, spoken)
        self.assertEqual(candidates, [spoken])
        client.resolve_repository.assert_called_once_with(
            None, "work on the spoken repository", [spoken, focused]
        )

    def test_focused_repository_is_used_when_utterance_has_no_match(self) -> None:
        focused = Path("/repos/focused")
        client = mock.Mock()
        client.resolve_repository.side_effect = [
            (None, []),
            (focused, [focused]),
        ]
        job: dict[str, object] = {
            "request": "work on this\n\nRepository: owner/focused",
            "utterance": "work on this task",
            "context_repository": "owner/focused",
        }

        repository, candidates = jobs.resolve_job_repository(client, job, [focused])

        self.assertEqual(repository, focused)
        self.assertEqual(candidates, [focused])
        self.assertEqual(
            client.resolve_repository.call_args_list,
            [
                mock.call(None, "work on this task", [focused]),
                mock.call("owner/focused", "", [focused]),
            ],
        )

    def test_latest_voice_marker_controls_terminal_state(self) -> None:
        job: dict[str, object] = {"id": "123456789abc", "turn_token": "token"}
        jobs.complete_from_output(
            job,
            output=(
                "VOICE_QUESTION[token]: old question\n"
                "VOICE_SUMMARY[token]: finished successfully"
            ),
            agent_status="idle",
        )
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"], "finished successfully")

    def test_blocked_without_marker_requires_attention(self) -> None:
        job: dict[str, object] = {
            "id": "123456789abc",
            "turn_token": "token",
            "herdr_target": "agent",
        }
        jobs.complete_from_output(job, output="plain output", agent_status="idle")
        self.assertEqual(job["status"], "blocked")
        self.assertIn("needs attention", str(job["result"]))

    def test_working_agent_cannot_complete_from_stale_summary(self) -> None:
        job: dict[str, object] = {
            "id": "123456789abc",
            "turn_token": "token",
            "herdr_target": "agent",
        }
        jobs.complete_from_output(
            job,
            output="VOICE_SUMMARY[token]: stale result",
            agent_status="working",
        )

        self.assertEqual(job["status"], "blocked")
        self.assertNotEqual(job["result"], "stale result")

    def test_worker_spawns_package_module(self) -> None:
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 1234
        process.wait.return_value = 0
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        with (
            mock.patch("subprocess.Popen", return_value=process) as popen,
            mock.patch("threading.Thread") as thread,
            mock.patch.object(jobs, "_process_identity", return_value="worker-start"),
        ):
            service.launch_worker("123456789abc")
        command = popen.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "local_voice_harness.cursor.worker"])
        thread.assert_called_once()

    def test_worker_uses_rofi_repository_selection_before_prompting(self) -> None:
        repository = Path(self.temporary.name) / "cloned-project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "Use Cursor to fix the bug",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.repository_roots.return_value = []
        client.resolve_repository.return_value = (None, [])
        client.choose_or_clone_repository.return_value = (repository, "")
        client.ensure_agent.return_value = AgentSelection(
            target="cursor-agent",
            pane_id="pane",
            workspace_id="workspace",
            cwd=str(repository),
            name="cursor-agent",
            worktree_path=str(repository),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        configure_tiered_outcomes(client, repository)

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["repository"], str(repository))
        client.choose_or_clone_repository.assert_called_once_with(
            [], checkpoint=mock.ANY
        )

    def test_worker_checks_issue_capability_before_herdr_side_effects(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "work on ENG-123",
                "issue_key": "ENG-123",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(
                production_jobs, "resolve_issue_reference", return_value="ENG-123"
            ),
            mock.patch.object(production_jobs, "require_issue_provider"),
            mock.patch.object(
                production_jobs,
                "require_issue_capabilities",
                side_effect=jobs.HarnessError("Linear MCP requires authentication"),
            ),
        ):
            service.run_worker("123456789abc")

        failed = jobs.read_job("123456789abc")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("requires authentication", str(failed["error"]))
        client.ensure_server.assert_not_called()

    def test_router_mcp_unavailability_never_opens_rofi_fallback(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "work on JOS-17",
                "issue_key": "JOS-17",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.repository_roots.return_value = [Path("/repos/tiered")]
        client.resolve_repository.return_value = (None, [])
        client.ensure_router.return_value = mock.Mock(target="router")
        client.prompt_and_wait.return_value = mock.Mock(
            output=(
                "JOS-17 was unavailable through Linear MCP; generic repository "
                "selected as fallback."
            )
        )

        def route(
            routed_client: Any,
            reference: str,
            repositories: list[Path],
            **kwargs: object,
        ) -> tuple[Path | None, str, str]:
            return linear.LinearIntegration().route_repository(
                routed_client,
                reference,
                repositories,
                token=str(kwargs["token"]),
                reserved=cast(set[str], kwargs["reserved"]),
                checkpoint=kwargs.get("checkpoint"),
            )

        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(
                production_jobs, "resolve_issue_reference", return_value="JOS-17"
            ),
            mock.patch.object(production_jobs, "require_issue_provider"),
            mock.patch.object(production_jobs, "require_issue_capabilities"),
            mock.patch.object(
                production_jobs, "route_issue_repository", side_effect=route
            ),
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(self.temporary.name) / "linear-router.lock",
            ),
        ):
            service.run_worker("123456789abc")

        failed = jobs.read_job("123456789abc")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("refusing unrelated repository fallback", str(failed["error"]))
        client.choose_or_clone_repository.assert_not_called()

    def test_linear_routing_precedes_conflicting_focused_repository(self) -> None:
        focused = Path("/repos/local-voice-harness")
        intended = Path("/repos/local-voice-harness-tiered-batch-fixture")
        repositories = [focused, intended]
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": (
                    "work on JOS-20\n\n"
                    "Ticket description names "
                    "joshua-mo-143/local-voice-harness-tiered-batch-fixture."
                ),
                "context_repository": "joshua-mo-143/local-voice-harness",
                "issue_key": "JOS-20",
                "issue_provider": "linear",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.repository_roots.return_value = repositories
        client.ensure_router.return_value = mock.Mock(target="voice-router")
        client.ensure_agent.return_value = AgentSelection(
            "cursor-agent",
            "pane",
            "workspace",
            str(intended),
            "cursor-agent",
            str(intended),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )

        def resolve(
            hint: str | None, _task: str, _repositories: list[Path]
        ) -> tuple[Path | None, list[Path]]:
            if hint == intended.name:
                return intended, [intended]
            return focused, [focused]

        client.resolve_repository.side_effect = resolve
        configure_tiered_outcomes(client, intended)
        client.prompt_and_wait.return_value = PromptOutcome(
            "idle",
            None,
            None,
            "ROUTE_REPO[123456789abc-route]: "
            "local-voice-harness-tiered-batch-fixture\n"
            "ROUTE_CONFIDENCE[123456789abc-route]: high\n"
            "ROUTE_REASON[123456789abc-route]: named in the Linear ticket",
        )
        client._harness_outcomes = [
            PromptOutcome(
                "idle",
                None,
                None,
                "WORKFLOW_TIER[123456789abc-1]: simple\n"
                "WORKFLOW_REASON[123456789abc-1]: test classification",
            ),
            PromptOutcome(
                "idle",
                "done",
                None,
                "VOICE_SUMMARY[123456789abc-2]: done",
            ),
        ]

        def route(
            routed_client: Any,
            reference: str,
            available: list[Path],
            **kwargs: object,
        ) -> tuple[Path | None, str, str]:
            return linear.LinearIntegration().route_repository(
                routed_client,
                reference,
                available,
                token=str(kwargs["token"]),
                reserved=cast(set[str], kwargs["reserved"]),
                checkpoint=kwargs.get("checkpoint"),
            )

        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(
                production_jobs, "resolve_issue_reference", return_value="JOS-20"
            ),
            mock.patch.object(production_jobs, "require_issue_provider"),
            mock.patch.object(production_jobs, "require_issue_capabilities"),
            mock.patch.object(production_jobs, "prompt_instructions", return_value=()),
            mock.patch.object(
                production_jobs, "route_issue_repository", side_effect=route
            ),
            mock.patch.object(
                linear,
                "LINEAR_ROUTER_LOCK",
                Path(self.temporary.name) / "linear-router.lock",
            ),
        ):
            service.run_worker("123456789abc")

        completed = jobs.read_job("123456789abc")
        self.assertEqual(completed["status"], "completed", completed.get("error"))
        self.assertEqual(completed["repository"], str(intended))
        client.ensure_router.assert_called_once()
        client.prompt_and_wait.assert_called_once()
        client.resolve_repository.assert_called_once_with(
            intended.name, "", repositories
        )
        client.choose_or_clone_repository.assert_not_called()

    def test_explicit_repository_hint_precedes_linear_routing(self) -> None:
        hinted = Path("/repos/explicit-repository")
        repositories = [hinted, Path("/repos/ticket-repository")]
        request = "work on JOS-20"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": request,
                "repository_hint": str(hinted),
                "context_repository": "owner/conflicting-focused-repository",
                "issue_key": "JOS-20",
                "issue_provider": "linear",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.repository_roots.return_value = repositories
        client.resolve_repository.return_value = (hinted, [hinted])
        client.ensure_agent.return_value = AgentSelection(
            "cursor-agent",
            "pane",
            "workspace",
            str(hinted),
            "cursor-agent",
            str(hinted),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        configure_tiered_outcomes(client, hinted)

        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(
                production_jobs, "resolve_issue_reference", return_value="JOS-20"
            ),
            mock.patch.object(production_jobs, "require_issue_provider"),
            mock.patch.object(production_jobs, "require_issue_capabilities"),
            mock.patch.object(production_jobs, "prompt_instructions", return_value=()),
            mock.patch.object(
                production_jobs, "route_issue_repository"
            ) as route_repository,
        ):
            service.run_worker("123456789abc")

        completed = jobs.read_job("123456789abc")
        self.assertEqual(completed["status"], "completed", completed.get("error"))
        self.assertEqual(completed["repository"], str(hinted))
        client.resolve_repository.assert_called_once_with(
            str(hinted), request, repositories
        )
        route_repository.assert_not_called()

    def test_prompt_timeout_retains_target_reservation_for_cancellation(self) -> None:
        repository = Path(self.temporary.name) / "project"
        checkout = Path(self.temporary.name) / "worktree"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "status": "queued",
                "repository": str(repository),
                "worktree_path": str(checkout),
                "herdr_target": "cursor-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_name": "cursor-agent",
                "agent_dispatch_state": "ready",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client._harness_error = HerdrError(
            "prompt timed out",
            code="operation_timeout",
        )
        configure_prompt_harness(client)

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        failed = jobs.read_job("123456789abc")
        self.assertEqual(failed["status"], "reconciling")
        self.assertEqual(failed["terminal_intent_status"], "failed")
        self.assertTrue(failed["target_release_pending"])
        self.assertTrue(failed["cancellation_reconciliation_pending"])
        self.assertEqual(failed["prompt_operation_state"], "ambiguous")
        self.assertIn("cursor-agent", jobs.reserved_targets())

    def test_stalled_agent_requires_prompt_reconciliation(self) -> None:
        repository = Path(self.temporary.name) / "project"
        checkout = Path(self.temporary.name) / "worktree"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "status": "queued",
                "repository": str(repository),
                "worktree_path": str(checkout),
                "herdr_target": "cursor-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_name": "cursor-agent",
                "agent_dispatch_state": "ready",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client._harness_error = HerdrError(
            "Herdr agent cursor-agent exceeded its inactivity timeout",
            code="agent_stalled",
        )
        configure_prompt_harness(client)

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        blocked = jobs.read_job("123456789abc")
        self.assertEqual(blocked["status"], "reconciling")
        self.assertEqual(blocked["manual_reconcile_operation"], "prompt")
        client.cancel_agent.assert_not_called()

    def test_uncertain_stalled_agent_cancellation_is_reconciled(self) -> None:
        repository = Path(self.temporary.name) / "project"
        checkout = Path(self.temporary.name) / "worktree"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "status": "queued",
                "repository": str(repository),
                "worktree_path": str(checkout),
                "herdr_target": "cursor-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_name": "cursor-agent",
                "agent_dispatch_state": "ready",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client._harness_error = HerdrError(
            "Herdr agent cursor-agent exceeded its maximum runtime",
            code="agent_stalled",
        )
        client.cancel_agent.side_effect = HerdrError("agent did not stop")
        configure_prompt_harness(client)

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        failed = jobs.read_job("123456789abc")
        self.assertEqual(failed["status"], "reconciling")
        self.assertEqual(failed["terminal_intent_status"], "failed")
        self.assertTrue(failed["target_release_pending"])
        self.assertTrue(failed["cancellation_reconciliation_pending"])

    def test_dead_worker_reconciles_existing_agent(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "created_at": 1,
                "worker_pid": 999999,
                "worker_process_start": "old-worker",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "worker_token": "old-claim",
                "herdr_target": "cursor-agent",
                "delivered": False,
            }
        )
        with (
            mock.patch("time.time", return_value=100),
            mock.patch.object(jobs, "_process_identity", return_value=None),
            mock.patch.object(service, "launch_worker") as launch,
        ):
            self.assertEqual(service.pending_results(), [])
        updated = jobs.read_job("123456789abc")
        self.assertTrue(updated["reconcile"])
        launch.assert_called_once_with("123456789abc")

    def test_dispatch_crash_retries_same_reserved_agent_identity(self) -> None:
        repository = Path(self.temporary.name) / "project"
        worktree = Path(self.temporary.name) / "worktree"
        selection = AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(worktree),
            "planned-agent",
            str(worktree),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "status": "queued",
                "repository": str(repository),
                "worktree_branch": "voice/test-checkout",
                "worktree_path": str(worktree),
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "root-pane",
                "worktree_provision_state": "ready",
                "herdr_target": "planned-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_name": "planned-agent",
                "agent_dispatch_state": "dispatching",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            {"state_change_seq": 8, "agent_session": "test-session"},
        ]
        client.start_agent.return_value = selection
        client.reconcile_session.return_value = SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession(
                "cursor/herdr",
                "test-session",
                "planned-agent",
                9,
                {
                    "pane_id": "pane",
                    "workspace_id": "workspace",
                    "cwd": str(worktree),
                },
            ),
            "active",
            True,
        )
        client._harness_outcomes = [
            PromptOutcome(
                "idle",
                "done",
                None,
                "VOICE_SUMMARY[123456789abc-1]: done",
            )
        ]
        configure_prompt_harness(client)

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        client.harness.create_session.assert_called_once_with(
            mock.ANY,
            before_submit=mock.ANY,
        )
        client.ensure_agent.assert_not_called()
        recovered = jobs.read_job("123456789abc")
        self.assertEqual(recovered["status"], "completed", recovered)

    def test_dispatch_reconciliation_defers_on_transient_lookup_failure(self) -> None:
        repository = Path(self.temporary.name) / "project"
        worktree = Path(self.temporary.name) / "worktree"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "status": "queued",
                "repository": str(repository),
                "worktree_path": str(worktree),
                "herdr_target": "planned-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_dispatch_state": "dispatching",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("server unavailable")

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        client.start_agent.assert_not_called()
        client.prompt_and_wait.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertEqual(updated["agent_dispatch_state"], "dispatching")

    def test_stale_worker_cannot_overwrite_cancellation(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "turn_token": "turn",
                "worker_token": "worker-claim",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "delivered": False,
            }
        )

        service.cancel_job("123456789abc")
        jobs._worker_complete(
            "123456789abc",
            "worker-claim",
            output="VOICE_SUMMARY[turn]: stale success",
            agent_status="idle",
            expected_revision=0,
        )

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "cancelled")
        self.assertNotEqual(updated["result"], "stale success")

    def test_cancellation_during_repository_resolution_stops_routing(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix the bug",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        client = mock.Mock()

        def repository_roots() -> list[Path]:
            entered.set()
            self.assertTrue(release.wait(2))
            return []

        client.repository_roots.side_effect = repository_roots
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            worker.start()
            self.assertTrue(entered.wait(2))
            service.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        client.resolve_repository.assert_not_called()
        client.choose_or_clone_repository.assert_not_called()
        client.ensure_agent.assert_not_called()

    def test_fork_commit_wins_before_cancellation(self) -> None:
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        fork = GitHubRepository(
            "me/project",
            "https://github.com/me/project",
            False,
            "main",
            "source/project",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork it",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        settled = threading.Event()
        continue_after_settle = threading.Event()
        github = mock.Mock()

        github.prepare_public_fork.return_value = (source, "me", "me/project")

        def ensure_fork(
            *_args: object, before_submit: Callable[[], None], **_kwargs: object
        ) -> GitHubRepository:
            before_submit()
            entered.set()
            self.assertTrue(release.wait(2))
            return fork

        github.ensure_fork.side_effect = ensure_fork
        original_settle = jobs._settle_fork_operation

        def settle_fork(
            store: JobStore,
            job_id: str,
            token: str,
            value: GitHubRepository | None,
            *,
            expected_revision: int,
            ambiguous: bool = False,
            failed_observing: bool = False,
        ) -> dict[str, object] | None:
            result = original_settle(
                store,
                job_id,
                token,
                value,
                expected_revision=expected_revision,
                ambiguous=ambiguous,
                failed_observing=failed_observing,
            )
            if value is fork:
                settled.set()
                self.assertTrue(continue_after_settle.wait(2))
            return result

        client = mock.Mock()
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_settle_fork_operation", side_effect=settle_fork),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(jobs.HarnessError, "committed fork submission"):
                service.cancel_job("123456789abc")
            committed = jobs.read_job("123456789abc")
            self.assertEqual(committed["status"], "routing")
            self.assertEqual(committed["fork_operation_state"], "submitted")
            release.set()
            self.assertTrue(settled.wait(2))
            self.assertEqual(
                jobs.read_job("123456789abc")["fork_operation_state"], "exists"
            )
            service.cancel_job("123456789abc")
            continue_after_settle.set()
            worker.join(2)
            with mock.patch.object(jobs, "_worker_is_alive", return_value=False):
                service.recover_jobs()

        self.assertFalse(worker.is_alive())
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["fork_operation_state"], "exists")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertEqual(updated["status"], "cancelled")
        self.assertFalse(updated["target_release_pending"])
        github.ensure_clone.assert_not_called()
        client.ensure_agent.assert_not_called()

    def test_cancellation_wins_before_fork_commit(self) -> None:
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork it",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        submitted = threading.Event()
        github = mock.Mock()
        client = mock.Mock()
        github.prepare_public_fork.return_value = (source, "me", "me/project")

        def ensure_fork(
            *_args: object, before_submit: Callable[[], None], **_kwargs: object
        ) -> GitHubRepository:
            entered.set()
            self.assertTrue(release.wait(2))
            before_submit()
            submitted.set()
            raise AssertionError("fork submission should not run")

        github.ensure_fork.side_effect = ensure_fork
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
            mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=AssertionError("real service operation attempted"),
            ) as service_operation,
        ):
            worker.start()
            try:
                self.assertTrue(
                    entered.wait(2),
                    f"worker stopped before fork routing: {jobs.read_job('123456789abc')}",
                )
                client.ensure_server.assert_called_once_with()
                service.cancel_job("123456789abc")
                self.assertEqual(jobs.read_job("123456789abc")["status"], "reconciling")
                self.assertEqual(
                    jobs.read_job("123456789abc")["terminal_intent_status"],
                    "cancelled",
                )
            finally:
                release.set()
                worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(submitted.is_set())
        self.assertEqual(jobs.read_job("123456789abc")["status"], "reconciling")
        self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
        github.ensure_clone.assert_not_called()
        client.ensure_agent.assert_not_called()
        service_operation.assert_not_called()

    def test_worker_persists_herdr_failure_before_fork_routing(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork it",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.ensure_server.side_effect = HerdrError(
            "Herdr command failed: executable not found",
            code="operation_spawn_failed",
        )
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "GitHubClient") as github_client,
        ):
            service.run_worker("123456789abc")

        failed = jobs.read_job("123456789abc")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "Herdr command failed: executable not found")
        self.assertNotIn("fork_operation_state", failed)
        github_client.assert_not_called()

    def test_post_create_cancellation_retains_reserved_worktree(self) -> None:
        repository = Path(self.temporary.name) / "project"
        checkout = Path(self.temporary.name) / "worktrees" / "task"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "worktree_branch": "voice/task",
                "worktree_label": "task",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve_worktree = cast(
                Callable[[Path, str, Path, str], None],
                kwargs["reserve_worktree"],
            )
            settle_worktree = cast(
                Callable[[Path, str | None, str | None], None],
                kwargs["settle_worktree"],
            )
            checkpoint = cast(Callable[[], None], kwargs["checkpoint"])
            reserve_worktree(repository, "voice/task", checkout, "planned")
            checkpoint()
            reserve_worktree(repository, "voice/task", checkout, "dispatching")
            entered.set()
            self.assertTrue(release.wait(2))
            settle_worktree(checkout, "workspace", "pane")
            checkpoint()
            raise AssertionError("cancelled worker continued after worktree settle")

        client.ensure_agent.side_effect = ensure_agent
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            dispatching = jobs.read_job("123456789abc")
            self.assertEqual(dispatching["worktree_path"], str(checkout))
            self.assertEqual(dispatching["worktree_provision_state"], "dispatching")
            service.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        retained = jobs.read_job("123456789abc")
        self.assertEqual(retained["worktree_provision_state"], "dispatching")
        self.assertEqual(retained["status"], "reconciling")
        self.assertEqual(retained["terminal_intent_status"], "cancelled")
        self.assertTrue(retained["target_release_pending"])
        client.prompt_and_wait.assert_not_called()
        with mock.patch.object(jobs, "_worker_is_alive", return_value=False):
            service.recover_jobs()
        self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
        self.assertEqual(jobs.read_job("123456789abc")["status"], "reconciling")

    def test_cancellation_before_agent_dispatch_uses_planned_reservation(self) -> None:
        repository = Path(self.temporary.name) / "project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "repository": str(repository),
                "worktree_branch": "voice/test-checkout",
                "worktree_path": str(repository),
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "root-pane",
                "worktree_provision_state": "ready",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        dispatched = threading.Event()
        selection = AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(repository),
            "planned-agent",
            str(repository),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])
        client.get_agent.return_value = {
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": str(repository),
        }
        client.reconcile_session.return_value = SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession(
                "cursor/herdr",
                "test-session",
                "planned-agent",
                8,
                {
                    "pane_id": "pane",
                    "workspace_id": "workspace",
                    "cwd": str(repository),
                },
            ),
            "active",
            True,
        )
        configure_prompt_harness(client)

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            checkpoint = cast(Callable[[], None], kwargs["checkpoint"])
            reserve(selection, True)
            entered.set()
            self.assertTrue(release.wait(2))
            checkpoint()
            dispatched.set()
            return selection

        client.ensure_agent.side_effect = ensure_agent
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            worker.start()
            self.assertTrue(entered.wait(2))
            self.assertEqual(
                jobs.read_job("123456789abc")["agent_dispatch_state"],
                "dispatching",
            )
            service.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(dispatched.is_set())
        client.close_owned_pane.assert_called_once_with(
            "planned-agent", "pane", "workspace"
        )
        client.prompt_and_wait.assert_not_called()

    def test_stale_agent_failure_after_cancelled_startup_is_rejected(self) -> None:
        repository = Path(self.temporary.name) / "project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        selection = AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(repository),
            "planned-agent",
            str(repository),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])
        client.get_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            {
                "name": "planned-agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
            },
        ]

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            reserve_worktree = cast(Callable[..., None], kwargs["reserve_worktree"])
            settle_worktree = cast(Callable[..., None], kwargs["settle_worktree"])
            fail_agent = cast(Callable[[HerdrError], None], kwargs["fail_agent"])
            reserve_worktree(
                repository,
                "voice/test-checkout",
                repository,
                "planned",
            )
            settle_worktree(repository, "workspace", "root-pane")
            reserve(selection, True)
            entered.set()
            self.assertTrue(release.wait(2))
            error = HerdrError("startup timed out", code="operation_timeout")
            fail_agent(error)
            raise error

        client.ensure_agent.side_effect = ensure_agent
        reconciliation_count = 0

        def reconcile_session(
            target: str, *, expected_session_id: str
        ) -> SessionReconciliation:
            nonlocal reconciliation_count
            reconciliation_count += 1
            if reconciliation_count == 1:
                return SessionReconciliation(
                    ReconciliationState.MISSING,
                    None,
                    "not yet visible",
                    False,
                )
            return SessionReconciliation(
                ReconciliationState.ACTIVE,
                HarnessSession(
                    "cursor/herdr",
                    expected_session_id,
                    target,
                    8,
                    {
                        "name": target,
                        "pane_id": "pane",
                        "workspace_id": "workspace",
                        "cwd": str(repository),
                    },
                ),
                "active",
                True,
            )

        client.reconcile_session.side_effect = reconcile_session
        configure_prompt_harness(client)
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
            mock.patch("time.time", return_value=90),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            service.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        cancelled = jobs.read_job("123456789abc")
        self.assertEqual(cancelled["status"], "reconciling")
        self.assertEqual(cancelled["terminal_intent_status"], "cancelled")
        self.assertEqual(cancelled["agent_dispatch_state"], "dispatching")
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_worker_is_alive", return_value=False),
        ):
            with mock.patch("time.time", return_value=100):
                service.recover_jobs()
            with mock.patch("time.time", return_value=105):
                service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["agent_dispatch_state"], "confirmed_absent")
        self.assertEqual(updated["status"], "cancelled")
        self.assertFalse(updated["target_release_pending"])
        self.assertEqual(client.close_owned_pane.call_count, 1)

    def test_cancellation_during_pr_checkout_prevents_prompt_submission(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        worktree = Path(self.temporary.name) / "worktrees" / "pr-42"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: shared\n")
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "review pull request 42",
                "github_repository": "source/project",
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "pull_request_worktree_state": "pending",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        github = mock.Mock()
        github.inspect_repository.return_value = source
        github.pull_request_details.return_value = {"headRefOid": "a" * 40}
        github.ensure_repository_clone.return_value = repository

        def checkout(*_args: object, **_kwargs: object) -> str:
            entered.set()
            self.assertTrue(release.wait(2))
            return "voice/github-pr-123456789abc"

        github.local_git.checkout_remote_ref.side_effect = checkout
        selection = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(worktree),
            "agent",
            str(worktree),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        client = mock.Mock()

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            reserve(selection, False)
            return selection

        client.ensure_agent.side_effect = ensure_agent
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            service.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        client.prompt_and_wait.assert_not_called()
        self.assertEqual(
            jobs.read_job("123456789abc")["pull_request_worktree_state"],
            "retained",
        )

    def test_cancellation_after_prompt_dispatch_rejects_late_observation(self) -> None:
        repository = Path(self.temporary.name) / "project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        submitted = threading.Event()
        selection = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(repository),
            "agent",
            str(repository),
            provider="cursor/herdr",
            provider_session_id="test-session",
            state_sequence=7,
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            reserve(selection, False)
            return selection

        def prompt(*_args: object, **kwargs: object) -> PromptOutcome:
            entered.set()
            self.assertTrue(release.wait(2))
            checkpoint = cast(Callable[[], None], kwargs["checkpoint"])
            checkpoint()
            submitted.set()
            return PromptOutcome("idle", "done", None, "")

        client.ensure_agent.side_effect = ensure_agent
        client._harness_behavior = prompt
        configure_prompt_harness(client)
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            worker.start()
            self.assertTrue(entered.wait(2))
            service.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(submitted.is_set())
        client.close_owned_pane.assert_not_called()
        self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])

    def test_worker_signal_requires_full_process_identity(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_pid": 42,
            "worker_process_start": "start",
            "worker_claim_operation": "test",
            "worker_claimed_at": 1,
            "worker_token": "claim",
        }
        wrong_command = b"\0".join(
            (
                b"/usr/bin/python",
                b"-m",
                b"local_voice_harness.cursor.worker",
                b"123456789abc",
                b"--claim",
                b"different",
                b"",
            )
        )
        with (
            mock.patch.object(jobs, "_process_identity", return_value="start"),
            mock.patch.object(Path, "read_bytes", return_value=wrong_command),
            mock.patch("os.killpg") as kill_group,
            mock.patch("os.kill") as kill_process,
        ):
            self.assertFalse(jobs._signal_worker(job, jobs.signal.SIGTERM))

        kill_group.assert_not_called()
        kill_process.assert_not_called()

    def test_critical_operation_never_escalates_to_process_group_kill(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_pid": 42,
            "worker_process_start": "start",
            "worker_claim_operation": "test",
            "worker_claimed_at": 1,
            "worker_token": "claim",
            "worker_operation": "fork_create",
        }
        with (
            mock.patch.object(jobs, "_worker_is_alive", side_effect=[True, True, True]),
            mock.patch.object(
                jobs, "_signal_worker", return_value=True
            ) as signal_worker,
        ):
            self.assertFalse(jobs._stop_worker(job, timeout=0))

        signal_worker.assert_called_once_with(
            job,
            jobs.signal.SIGTERM,
            include_process_group=False,
        )

    def test_only_one_concurrent_reply_transitions_job(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "original",
                "status": "awaiting_user",
                "clarification_kind": "agent",
                "delivered": True,
            }
        )
        barrier = threading.Barrier(2)
        successes: list[str] = []
        errors: list[Exception] = []

        def reply(text: str) -> None:
            barrier.wait()
            try:
                service.reply_job("123456789abc", text)
                successes.append(text)
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(service, "launch_worker"):
            threads = [
                threading.Thread(target=reply, args=("first",)),
                threading.Thread(target=reply, args=("second",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len(successes), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(jobs.read_job("123456789abc")["status"], "queued")

    def test_delivery_claim_has_single_concurrent_winner(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "completed",
                "result": "done",
                "completed_at": 1,
                "delivered": False,
            }
        )
        barrier = threading.Barrier(2)
        claims: list[DeliveryClaim | None] = []

        def claim() -> None:
            barrier.wait()
            claims.append(service.claim_delivery())

        with mock.patch("time.time", return_value=100):
            threads = [threading.Thread(target=claim) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(sum(claim is not None for claim in claims), 1)

    def test_launch_failure_is_persisted_for_delivery(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        diagnostic = "spawn failed: Authorization: Bearer launch-secret"
        with (
            mock.patch("subprocess.Popen", side_effect=OSError(diagnostic)),
            self.assertRaisesRegex(OSError, "spawn failed"),
        ):
            service.launch_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "failed")
        self.assertFalse(updated["delivered"])
        self.assertNotIn("launch-secret", str(updated["error"]))
        self.assertIn("[REDACTED]", str(updated["error"]))
        self.assertEqual(
            updated["result"],
            "Cursor job failed to start. Check the job log for details.",
        )

    def test_abandoned_queued_job_is_launched_only_once(self) -> None:
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 4321
        process.wait.return_value = 0
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        with (
            mock.patch("subprocess.Popen", return_value=process) as popen,
            mock.patch("threading.Thread"),
            mock.patch.object(jobs, "_process_identity", return_value="worker-start"),
        ):
            service.recover_jobs()
            service.recover_jobs()

        popen.assert_called_once()
        updated = jobs.read_job("123456789abc")
        self.assertTrue(updated["worker_token"])

    def test_interrupted_routing_worker_restarts_before_dispatch(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "old",
                "worker_pid": 999999,
                "worker_process_start": "old-worker",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "created_at": 1,
                "delivered": False,
            }
        )
        with (
            mock.patch.object(jobs, "_process_identity", return_value=None),
            mock.patch.object(service, "launch_worker") as launch,
        ):
            service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertNotIn("reconcile", updated)
        launch.assert_called_once_with("123456789abc")

    def test_legacy_worker_pid_without_boot_identity_is_never_alive(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_pid": 42,
        }
        command = b"\0".join(
            (
                b"/usr/bin/python",
                b"-m",
                b"local_voice_harness.cursor.worker",
                b"123456789abc",
                b"",
            )
        )
        with mock.patch.object(Path, "read_bytes", return_value=command):
            self.assertFalse(jobs._worker_is_alive(job))

    def test_worker_liveness_requires_matching_linux_boot_identity(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_token": "claim",
            "worker_pid": 42,
            "worker_boot_id": "old-boot",
            "worker_process_start": "start",
            "worker_claim_operation": "test",
            "worker_claimed_at": 1,
        }
        with (
            mock.patch.object(jobs, "_boot_identity", return_value="new-boot"),
            mock.patch.object(jobs, "_process_identity", return_value="start"),
        ):
            self.assertFalse(jobs._worker_is_alive(job))
        with (
            mock.patch.object(jobs, "_boot_identity", return_value="old-boot"),
            mock.patch.object(jobs, "_process_identity", return_value="start"),
        ):
            self.assertTrue(jobs._worker_is_alive(job))

    def test_cancel_keeps_target_reserved_until_interrupt_finishes(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "herdr_target": "cursor-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "anchor",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "github_pull_request": 42,
                "worktree_path": "/worktree/pr-42",
                "pull_request_worktree_state": "provisioning",
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.get_agent.return_value = {
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/worktree/pr-42",
        }
        reserved_during_cancel: set[str] = set()

        def inspect_reservation(_target: str) -> None:
            reserved_during_cancel.update(jobs.reserved_targets())

        client.close_owned_pane.side_effect = lambda _target, _pane, _workspace: (
            inspect_reservation(_target)
        )
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.cancel_job("123456789abc")
            service.cancel_job("123456789abc")

        self.assertIn("cursor-agent", reserved_during_cancel)
        self.assertNotIn("cursor-agent", jobs.reserved_targets())
        client.close_owned_pane.assert_called_once()
        self.assertEqual(
            jobs.read_job("123456789abc")["pull_request_worktree_state"],
            "retained",
        )

    def test_failed_agent_interrupt_keeps_reservation_for_recovery(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "herdr_target": "cursor-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree/task",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "anchor",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "delivered": False,
            }
        )
        failing = mock.Mock()
        failing.get_agent.return_value = {
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/worktree/task",
        }
        failing.close_owned_pane.side_effect = HerdrError("still working")
        recovered = mock.Mock()
        recovered.get_agent.return_value = {
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/worktree/task",
        }
        with (
            mock.patch.object(jobs, "_stop_worker", return_value=True),
            mock.patch.object(jobs, "HerdrClient", side_effect=[failing, recovered]),
        ):
            service.cancel_job("123456789abc")
            cancelled = jobs.read_job("123456789abc")
            self.assertTrue(cancelled["target_release_pending"])
            self.assertIn("cursor-agent", jobs.reserved_targets())
            service.recover_jobs()

        recovered.close_owned_pane.assert_called_once_with(
            "cursor-agent", "pane", "workspace"
        )
        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])

    def test_provisional_agent_fence_survives_first_not_found(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "herdr_target": "planned-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "worktree_path": "/worktree/task",
                "worktree_workspace_id": "workspace",
                "worktree_root_pane_id": "anchor",
                "agent_dispatch_state": "dispatching",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        close_attempts = 0

        def close_owned(*_args: object) -> None:
            nonlocal close_attempts
            close_attempts += 1
            if close_attempts == 1:
                raise HerdrError("close not confirmed")

        client.close_owned_pane.side_effect = close_owned
        client.get_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            {
                "name": "planned-agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
                "cwd": "/worktree/task",
            },
            {
                "name": "planned-agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
                "cwd": "/worktree/task",
            },
            {
                "name": "planned-agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
                "cwd": "/worktree/task",
            },
        ]
        with (
            mock.patch.object(jobs, "_stop_worker", return_value=True),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.cancel_job("123456789abc")
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            with mock.patch("time.time", return_value=100):
                service.recover_jobs()
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            with mock.patch("time.time", return_value=104):
                service.recover_jobs()
            client.close_owned_pane.assert_not_called()

        unresolved = jobs.read_job("123456789abc")
        self.assertTrue(unresolved["target_release_pending"])
        self.assertEqual(unresolved["agent_dispatch_state"], "ambiguous")

    def test_truly_absent_agent_releases_after_bounded_backoff(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "herdr_target": "missing-agent",
                "agent_dispatch_state": "failed_observing",
                "agent_dispatch_exited": True,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("not found", code="agent_not_found")
        client.close_owned_pane.side_effect = HerdrError(
            "not found", code="agent_not_found"
        )
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            for now in (100, 104, 105, 114):
                with mock.patch("time.time", return_value=now):
                    service.recover_jobs()
            pending = jobs.read_job("123456789abc")
            self.assertEqual(pending["agent_reconcile_attempts"], 2)
            self.assertEqual(pending["agent_next_reconcile_at"], 115)
            with mock.patch("time.time", return_value=115):
                service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["agent_dispatch_state"], "confirmed_absent")
        self.assertEqual(updated["agent_reconcile_attempts"], 3)
        self.assertFalse(updated["target_release_pending"])

    def test_truly_absent_fork_releases_after_bounded_backoff(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "fork_operation_state": "failed_observing",
                "fork_dispatch_exited": True,
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_source_private": False,
                "fork_operation_target": "me/project",
                "target_release_pending": True,
                "delivered": True,
            }
        )
        github = mock.Mock()
        github.inspect_public_repository.return_value = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github.reconcile_fork.return_value = None
        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            for now in (100, 105, 115):
                with mock.patch("time.time", return_value=now):
                    service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["fork_operation_state"], "confirmed_absent")
        self.assertEqual(updated["fork_reconcile_attempts"], 3)
        self.assertFalse(updated["target_release_pending"])

    def test_failed_fork_becomes_visible_after_first_observation(self) -> None:
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        fork = GitHubRepository(
            "me/project",
            "https://github.com/me/project",
            False,
            "main",
            "source/project",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "fork_operation_state": "failed_observing",
                "fork_dispatch_exited": True,
                "fork_operation_source": source.name_with_owner,
                "fork_operation_source_url": source.url,
                "fork_operation_source_default_branch": source.default_branch,
                "fork_operation_source_private": False,
                "fork_operation_target": fork.name_with_owner,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        github = mock.Mock()
        github.inspect_public_repository.return_value = source
        github.reconcile_fork.side_effect = [None, fork]
        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            with mock.patch("time.time", return_value=100):
                service.recover_jobs()
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            with mock.patch("time.time", return_value=105):
                service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["fork_operation_state"], "exists")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertFalse(updated["target_release_pending"])

    def test_truly_absent_worktree_releases_after_bounded_backoff(self) -> None:
        checkout = Path(self.temporary.name) / "missing-worktree"
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "repository": str(Path(self.temporary.name) / "repository"),
                "worktree_branch": "voice/task",
                "worktree_path": str(checkout),
                "worktree_provision_state": "failed_observing",
                "worktree_dispatch_exited": True,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.run_json.return_value = {"worktrees": []}
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            for now in (100, 105, 115):
                with mock.patch("time.time", return_value=now):
                    service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["worktree_provision_state"], "confirmed_absent")
        self.assertEqual(updated["worktree_reconcile_attempts"], 3)
        self.assertFalse(updated["target_release_pending"])

    def test_dispatching_agent_backoff_is_capped_then_requires_manual_review(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "error": "agent startup failed; external operation reconciliation is pending",
                "result": "agent startup failed; external operation reconciliation is pending",
                "herdr_target": "late-agent",
                "agent_dispatch_state": "dispatching",
                "agent_reconcile_attempts": 4,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("not found", code="agent_not_found")
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_worker_is_alive", return_value=False),
        ):
            with mock.patch("time.time", return_value=100):
                service.recover_jobs()
            capped = jobs.read_job("123456789abc")
            self.assertEqual(capped["agent_reconcile_attempts"], 5)
            self.assertEqual(capped["agent_next_reconcile_at"], 160)

            with mock.patch("time.time", return_value=160):
                service.recover_jobs()
            manual = jobs.read_job("123456789abc")
            self.assertEqual(manual["agent_dispatch_state"], "manual_required")
            self.assertEqual(manual["agent_reconcile_attempts"], 6)
            self.assertTrue(manual["target_release_pending"])
            self.assertEqual(
                manual["error"],
                "agent startup failed; external operation reconciliation is pending",
            )
            self.assertEqual(
                manual["result"],
                "agent startup failed; external operation reconciliation is pending",
            )
            self.assertEqual(
                manual["reconciliation_base_error"], "agent startup failed"
            )

            with mock.patch("time.time", return_value=10_000):
                service.recover_jobs()

        self.assertEqual(client.get_agent.call_count, 2)
        client.close_owned_pane.assert_not_called()

    def test_ambiguous_fork_requires_manual_review_and_can_confirm_absence(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "error": "fork outcome unknown; external operation reconciliation is pending",
                "result": "fork outcome unknown; external operation reconciliation is pending",
                "fork_operation_state": "ambiguous",
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_source_private": False,
                "fork_operation_target": "me/project",
                "fork_reconcile_attempts": 5,
                "herdr_target": "unrelated-agent",
                "target_release_pending": True,
                "delivered": True,
            }
        )
        github = mock.Mock()
        github.inspect_public_repository.return_value = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        github.reconcile_fork.return_value = None
        client = mock.Mock()
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_worker_is_alive", return_value=False),
            mock.patch("time.time", return_value=100),
        ):
            service.recover_jobs()

        manual = jobs.read_job("123456789abc")
        self.assertEqual(manual["fork_operation_state"], "manual_required")
        self.assertTrue(manual["target_release_pending"])
        self.assertTrue(manual["target_release_manual_required"])
        client.close_owned_pane.assert_not_called()
        token = str(manual["manual_reconcile_token"])
        with self.assertRaisesRegex(jobs.HarnessError, "fence is stale"):
            service.resolve_manual_reconciliation(
                "123456789abc", "fork", "stale-token", "confirmed_absent"
            )
        with self.assertRaisesRegex(jobs.HarnessError, "fence is stale"):
            service.resolve_manual_reconciliation(
                "123456789abc", "worktree", token, "confirmed_absent"
            )
        resolved = service.resolve_manual_reconciliation(
            "123456789abc", "fork", token, "confirmed_absent"
        )
        self.assertEqual(resolved.fork_operation_state, "confirmed_absent")
        self.assertEqual(
            resolved.error,
            "fork outcome unknown; external operation reconciliation is pending",
        )
        self.assertEqual(github.reconcile_fork.call_count, 1)

    def test_manual_materialized_agent_is_retained_without_external_action(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "error": (
                    "agent startup failed; manual reconciliation required for "
                    "Herdr agent retained-agent"
                ),
                "result": (
                    "agent startup failed; manual reconciliation required for "
                    "Herdr agent retained-agent"
                ),
                "reconciliation_base_error": "agent startup failed",
                "herdr_target": "retained-agent",
                "agent_dispatch_state": "manual_required",
                "agent_provider": "cursor/herdr",
                "agent_provider_session_id": "retained-session",
                "agent_state_sequence": 1,
                "manual_reconcile_operation": "agent",
                "manual_reconcile_token": "manual-token",
                "target_release_pending": True,
                "cancellation_reconciliation_pending": True,
                "delivered": True,
            }
        )

        resolved = service.resolve_manual_reconciliation(
            "123456789abc", "agent", "manual-token", "materialized"
        )

        self.assertEqual(resolved.agent_dispatch_state, "retained")
        self.assertEqual(resolved.herdr_target, "retained-agent")
        self.assertFalse(resolved.target_release_pending)
        self.assertFalse(resolved.cancellation_reconciliation_pending)
        self.assertEqual(
            resolved.error,
            (
                "agent startup failed; manual reconciliation required for "
                "Herdr agent retained-agent"
            ),
        )
        self.assertEqual(
            resolved.result,
            (
                "agent startup failed; manual reconciliation required for "
                "Herdr agent retained-agent"
            ),
        )

    def test_quarantined_worktree_releases_target_but_blocks_path(self) -> None:
        repository = Path(self.temporary.name) / "repository"
        checkout = Path(self.temporary.name) / "unexpected-worktree"
        checkout.mkdir()
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "repository": str(repository),
                "herdr_target": "agent",
                "worktree_branch": "voice/task",
                "worktree_path": str(checkout),
                "worktree_provision_state": "ambiguous",
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.run_json.return_value = {"worktrees": []}
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch("time.time", return_value=100),
        ):
            service.recover_jobs()

        quarantined = jobs.read_job("123456789abc")
        self.assertEqual(quarantined["worktree_provision_state"], "quarantined")
        self.assertTrue(quarantined["worktree_manual_inspection_required"])
        self.assertTrue(quarantined["target_release_pending"])
        self.assertTrue(quarantined["target_release_manual_required"])
        client.close_owned_pane.assert_not_called()

        jobs.write_job(
            {
                "id": "bbbbbbbbbbbb",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
            }
        )
        self.assertIsNone(
            jobs._reserve_worker_worktree(
                "bbbbbbbbbbbb",
                WorkerOwnership("worker", 42, "test-boot", "worker-start", "test", 1),
                repository,
                "voice/task",
                checkout,
                expected_revision=0,
                state="planned",
            )
        )

        service.acknowledge_worktree_quarantine("123456789abc")
        self.assertEqual(
            jobs.read_job("123456789abc")["worktree_provision_state"],
            "ambiguous",
        )
        self.assertIsNone(
            jobs._reserve_worker_worktree(
                "bbbbbbbbbbbb",
                WorkerOwnership("worker", 42, "test-boot", "worker-start", "test", 1),
                repository,
                "voice/task",
                checkout,
                expected_revision=0,
                state="planned",
            )
        )

    def test_recovery_retries_worker_stop_without_target_reservation(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "worker_claim_operation": "test",
                "worker_claimed_at": 1,
                "delivered": False,
            }
        )
        alive = True
        stops = 0

        def stop_worker(_job: dict[str, object]) -> bool:
            nonlocal alive, stops
            stops += 1
            if stops == 2:
                alive = False
                return True
            return False

        with (
            mock.patch.object(jobs, "_worker_is_alive", side_effect=lambda _job: alive),
            mock.patch.object(jobs, "_stop_worker", side_effect=stop_worker),
        ):
            service.cancel_job("123456789abc")
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            service.recover_jobs()

        self.assertEqual(stops, 2)
        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])

    def test_cancel_retires_unfenced_legacy_worker_first(self) -> None:
        (Path(self.temporary.name) / "123456789abc.json").write_text(
            json.dumps(
                {
                    "id": "123456789abc",
                    "status": "running",
                    "worker_pid": 42,
                    "created_at": 1,
                    "delivered": False,
                }
            )
        )
        with (
            mock.patch.object(jobs, "_worker_is_alive", return_value=True),
            mock.patch.object(
                jobs, "_stop_legacy_worker", return_value=True
            ) as stop_legacy,
        ):
            service.cancel_job("123456789abc")

        stop_legacy.assert_called_once_with("123456789abc")
        self.assertEqual(jobs.read_job("123456789abc")["status"], "cancelled")

    def test_recovery_releases_abandoned_cancellation_fence(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "herdr_target": "cursor-agent",
                "target_release_pending": True,
                "target_release_owner_pid": 999999,
                "target_release_owner_start": "old-process",
                "completed_at": 1,
                "delivered": True,
            }
        )
        client = mock.Mock()
        with (
            mock.patch.object(
                service.worker_lifecycle,
                "process_identity",
                side_effect=lambda pid: (
                    "current-process" if pid == os.getpid() else None
                ),
            ),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertTrue(updated["target_release_pending"])
        self.assertTrue(updated["target_release_manual_required"])
        client.close_owned_pane.assert_not_called()

    def test_foreground_delivery_is_acknowledged_explicitly(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "completed",
                "result": "done",
                "completed_at": 1,
                "delivered": False,
            }
        )
        claims: DeliveryClaims = []
        with mock.patch.object(service, "start_job", return_value="123456789abc"):
            result, session = service.cursor_turn("do it", delivery_claims=claims)

        self.assertEqual(result, "done")
        self.assertIsNone(session)
        self.assertFalse(jobs.read_job("123456789abc")["delivered"])
        self.assertEqual(len(claims), 1)
        service.acknowledge_deliveries(claims)
        self.assertTrue(jobs.read_job("123456789abc")["delivered"])


if __name__ == "__main__":
    unittest.main()
