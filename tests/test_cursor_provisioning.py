from __future__ import annotations

import json
import signal
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest import mock

from local_voice_harness.cursor import provisioning as production_jobs
from local_voice_harness.cursor import service, worker_lifecycle
from local_voice_harness.cursor.delivery import DeliveryClaim, DeliveryClaims
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    WorkflowParticipant,
)
from local_voice_harness.cursor.recovery import stage_terminal_intent
from local_voice_harness.cursor.store import JobQuarantineWarning, JobStore
from local_voice_harness.integrations.github import (
    GitHubIssue,
    GitHubRepository,
    ProvisionedIssue,
    ProvisionedPullRequest,
)
from local_voice_harness.integrations.herdr import (
    AgentSelection,
    HerdrError,
    PromptOutcome,
)


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
        result = production_jobs._prepare_pull_request_checkout(
            self._store(),
            job_id,
            token,
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


def configure_tiered_outcomes(
    client: mock.Mock,
    checkout: Path,
    *,
    tier: str = "simple",
) -> None:
    job_id = "123456789abc"
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
    client.prompt_and_wait.side_effect = outcomes

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
        selection = AgentSelection(
            role,
            f"pane-{role}",
            workspace,
            str(checkout),
            role,
            str(checkout),
        )
        reserve(selection, True)
        settle(selection)
        return selection

    client.start_fresh_agent.side_effect = start_fresh


class DurablePromptOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = JobStore(root / "jobs", root / "legacy")

    def tearDown(self) -> None:
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
            "workflow_phase": "classifying",
            "turn": 1,
            "turn_token": "123456789abc-1",
            "workflow_turn_phase": "classifying",
            "herdr_target": "planner",
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
                prompt_baseline_sequence=7,
            )
        return self.store.create(CursorJob.from_dict(values))

    def test_planned_prompt_persists_submit_and_accept_boundaries(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {"state_change_seq": 7}

        def prompt(*_args: object, **kwargs: object) -> PromptOutcome:
            before_submit = cast(Callable[[int], None], kwargs["before_submit"])
            accepted = cast(Callable[[], None], kwargs["accepted"])
            before_submit(7)
            self.assertEqual(
                self.store.get(job.id).prompt_operation_state, "submitting"
            )
            accepted()
            return PromptOutcome("idle", None, None, "output")

        client.prompt_and_wait.side_effect = prompt

        outcome = production_jobs._execute_phase_prompt(
            self.store,
            job,
            "worker",
            client,
            lambda: None,
            target="planner",
            prompt="prompt",
            token="123456789abc-1",
        )

        self.assertEqual(outcome, ("output", "idle"))
        self.assertEqual(self.store.get(job.id).prompt_operation_state, "submitted")

    def test_prompt_call_failure_is_ambiguous_even_without_callback(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {"state_change_seq": 7}
        client.prompt_and_wait.side_effect = HerdrError("timeout")

        with self.assertRaises(HerdrError):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                "worker",
                client,
                lambda: None,
                target="planner",
                prompt="prompt",
                token="123456789abc-1",
            )

        self.assertEqual(self.store.get(job.id).prompt_operation_state, "ambiguous")

    def test_pre_submit_questionnaire_keeps_planned_prompt_retryable(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {"state_change_seq": 7}
        client.prompt_and_wait.side_effect = HerdrError(
            "questionnaire",
            code="interactive_questionnaire",
        )

        with self.assertRaises(HerdrError):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                "worker",
                client,
                lambda: None,
                target="planner",
                prompt="prompt",
                token="123456789abc-1",
            )

        current = self.store.get(job.id)
        self.assertEqual(current.prompt_operation_state, "planned")
        self.assertIsNone(current.manual_reconcile_operation)

    def test_terminal_intent_invalidates_prompt_before_submit_callback(self) -> None:
        job = self.create()
        client = mock.Mock()
        client.get_agent.return_value = {"state_change_seq": 7}
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

        client.prompt_and_wait.side_effect = prompt

        with self.assertRaises(production_jobs.WorkerCancelled):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                "worker",
                client,
                lambda: None,
                target="planner",
                prompt="prompt",
                token="123456789abc-1",
            )

        self.assertFalse(submitted)
        current = self.store.get(job.id)
        self.assertEqual(current.terminal_intent_status, JobStatus.CANCELLED)
        self.assertEqual(current.prompt_operation_state, "planned")

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
            ),
        )
        assert planned is not None
        before_create, _accepted = production_jobs._participant_pane_callbacks(
            self.store,
            job.id,
            "worker",
            "reviewer",
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

    def test_submitted_prompt_observes_without_resubmitting(self) -> None:
        job = self.create("submitted")
        client = mock.Mock()
        client.get_agent.return_value = {"agent_status": "idle"}
        client.run_text.return_value = "output"

        outcome = production_jobs._execute_phase_prompt(
            self.store,
            job,
            "worker",
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
        client.get_agent.side_effect = HerdrError("temporarily unavailable")

        with self.assertRaises(production_jobs.WorkerCancelled):
            production_jobs._execute_phase_prompt(
                self.store,
                job,
                "worker",
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
                "review_round": 1,
                "review_decision": "revise",
            }
        )
        store = jobs._store()
        plan = "Preserve ownership during recovery."
        plan_reference = store.write_artifact(
            "123456789abc",
            "plan",
            1,
            plan,
        )
        review_reference = store.write_artifact(
            "123456789abc",
            "review",
            1,
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
            JobQuarantineWarning, "123456789abc.json: job is quarantined"
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
            JobQuarantineWarning, "bbbbbbbbbbbb.json: job is quarantined"
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
            "worker",
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
        self.assertEqual(resumed.herdr_target, "planner")
        self.assertIn("change several components", resumed.request)
        self.assertIn("User clarification: keep the old format", resumed.request)

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
            "worker",
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
            "worker",
            "VOICE_QUESTION[current]: Which behavior should win?",
            "idle",
        )

        updated = store.get("123456789abc")
        self.assertEqual(updated.status, JobStatus.AWAITING_USER)
        self.assertEqual(updated.prompt_operation_state, "none")

    def test_malformed_review_output_blocks_approval_gate(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "change recovery",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
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
            "worker",
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
            "worker",
            "WORKFLOW_PROMOTE[current]: simple\n"
            "WORKFLOW_REASON[current]: no additional risk",
            "idle",
        )

        self.assertEqual(store.get("123456789abc").status, JobStatus.BLOCKED)

    def test_promotion_carries_clarification_into_planning_request(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "rename text",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
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
            "worker",
            "WORKFLOW_PROMOTE[current]: medium\n"
            "WORKFLOW_REASON[current]: cross-component compatibility risk",
            "idle",
        )

        promoted = store.get("123456789abc")
        self.assertEqual(promoted.workflow_phase.value, "planning")
        self.assertIn("rename text", promoted.request)
        self.assertIn("preserve compatibility", promoted.request)
        self.assertIsNone(promoted.continuation_answer)
        self.assertFalse(promoted.continuation)

    def test_high_risk_final_rejected_review_awaits_explicit_decision(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "change recovery",
                "status": "running",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "start",
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
            "worker",
            "WORKFLOW_REVIEW_DECISION[turn]: revise\n"
            "WORKFLOW_REVIEW[turn]: unresolved ownership",
            "idle",
        )

        updated = store.get("123456789abc")
        self.assertEqual(updated.status, JobStatus.AWAITING_USER)
        self.assertEqual(updated.review_round, 1)
        self.assertEqual(
            updated.clarification_kind,
            "workflow_review_exhausted",
        )
        self.assertEqual(updated.workflow_phase.value, "reviewing")

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
        launch.assert_called_once_with("123456789abc")

    def test_exhausted_review_abort_uses_job_cancellation(self) -> None:
        self.write_exhausted_review_job()

        with mock.patch.object(service, "launch_worker") as launch:
            service.reply_job(
                "123456789abc",
                "abort",
            )

        self.assertEqual(
            jobs._store().get("123456789abc").status,
            JobStatus.CANCELLED,
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
        github.provision_pull_request.return_value = ProvisionedPullRequest(
            source, repository, 42
        )
        github.checkout_pull_request.return_value = "voice/github-pr-123456789abc"
        client = mock.Mock()
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(worktree),
            "agent",
            str(worktree),
        )
        configure_tiered_outcomes(client, worktree)

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        github.provision_pull_request.assert_called_once_with(
            "source/project", 42, checkpoint=mock.ANY
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
        )
        github.checkout_pull_request.assert_called_once_with(
            worktree,
            42,
            branch="voice/github-pr-123456789abc",
            checkpoint=mock.ANY,
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
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
                "worker",
                jobs.read_job("123456789abc"),
            )

        github.checkout_pull_request.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["pull_request_worktree_state"], "quarantined")

    def test_concurrent_pull_requests_prepare_distinct_worktrees(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        barrier = threading.Barrier(2)
        calls: list[tuple[Path, int, str]] = []
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
                    "repository": str(repository),
                    "worktree_path": str(worktree),
                    "github_pull_request": number,
                    "worktree_branch": f"voice/github-pr-{job_id}",
                    "pull_request_worktree_state": "provisioning",
                }
            )
        github = mock.Mock()

        def checkout(path: Path, number: int, *, branch: str) -> str:
            calls.append((path, number, branch))
            barrier.wait()
            return branch

        github.checkout_pull_request.side_effect = checkout

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

        self.assertEqual(len({path for path, _number, _branch in calls}), 2)
        self.assertEqual(len({branch for _path, _number, branch in calls}), 2)
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
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertEqual(updated["repository"], str(repository))
        self.assertEqual(updated["workflow_tier"], "high-risk")
        self.assertEqual(updated["workflow_phase"], "finished")
        self.assertTrue(updated["plan_artifact"])
        self.assertTrue(updated["review_artifact"])
        self.assertEqual(
            [call.kwargs["role"] for call in client.start_fresh_agent.call_args_list],
            ["reviewer", "implementer"],
        )
        self.assertEqual(
            [call.kwargs["mode"] for call in client.start_fresh_agent.call_args_list],
            ["ask", None],
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
        )
        configure_tiered_outcomes(client, repository)

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.run_worker("123456789abc")

        github.reconcile_fork.assert_called_once_with(source, "me/project")
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
        )
        prompt = client.prompt_and_wait.call_args.args[1]
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
            }
        )
        selection = AgentSelection(
            "second-agent",
            "pane",
            "workspace",
            "/repo",
            "second-agent",
            "/worktree/issue-42",
        )

        reserved = jobs._reserve_worker_target(
            "bbbbbbbbbbbb",
            "worker-token",
            selection,
            Path("/repo"),
            None,
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
            fork_requested=False,
            github_pull_request=None,
            agent=None,
            utterance=None,
            context_repository=None,
            issue_key=None,
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
        client.prompt_and_wait.side_effect = HerdrError(
            "prompt timed out",
            code="operation_timeout",
        )

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        failed = jobs.read_job("123456789abc")
        self.assertEqual(failed["status"], "reconciling")
        self.assertEqual(failed["terminal_intent_status"], "failed")
        self.assertTrue(failed["target_release_pending"])
        self.assertTrue(failed["cancellation_reconciliation_pending"])
        self.assertEqual(failed["prompt_operation_state"], "ambiguous")
        self.assertIn("cursor-agent", jobs.reserved_targets())

    def test_dead_worker_reconciles_existing_agent(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "created_at": 1,
                "worker_pid": 999999,
                "worker_process_start": "old-worker",
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
        updated = json.loads(
            (Path(self.temporary.name) / "123456789abc.json").read_text()
        )
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
        )
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
                "agent_name": "planned-agent",
                "agent_dispatch_state": "dispatching",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            {"state_change_seq": 1},
        ]
        client.start_agent.return_value = selection
        client.prompt_and_wait.return_value = PromptOutcome(
            "idle",
            "done",
            None,
            "VOICE_SUMMARY[123456789abc-1]: done",
        )

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.run_worker("123456789abc")

        client.start_agent.assert_called_once_with(
            worktree,
            "project",
            "pane",
            "workspace",
            name="planned-agent",
            checkpoint=mock.ANY,
        )
        client.ensure_agent.assert_not_called()
        self.assertEqual(jobs.read_job("123456789abc")["status"], "completed")

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
                "delivered": False,
            }
        )

        service.cancel_job("123456789abc")
        jobs._worker_complete(
            "123456789abc",
            "worker-claim",
            output="VOICE_SUMMARY[turn]: stale success",
            agent_status="idle",
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
            ambiguous: bool = False,
            failed_observing: bool = False,
        ) -> dict[str, object] | None:
            result = original_settle(
                store,
                job_id,
                token,
                value,
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
                "local_voice_harness.integrations.herdr.subprocess.run",
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
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])

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
        client.cancel_agent.assert_called_once_with("planned-agent")
        client.prompt_and_wait.assert_not_called()

    def test_late_agent_visibility_after_cancelled_startup(self) -> None:
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
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])
        client.cancel_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            None,
            None,
        ]
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
            fail_agent = cast(Callable[[HerdrError], None], kwargs["fail_agent"])
            reserve(selection, True)
            entered.set()
            self.assertTrue(release.wait(2))
            error = HerdrError("startup timed out", code="operation_timeout")
            fail_agent(error)
            raise error

        client.ensure_agent.side_effect = ensure_agent
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
        self.assertEqual(
            jobs.read_job("123456789abc")["agent_dispatch_state"], "ambiguous"
        )
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
        self.assertEqual(client.cancel_agent.call_count, 2)

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
        github.provision_pull_request.return_value = ProvisionedPullRequest(
            source, repository, 42
        )

        def checkout(*_args: object, **_kwargs: object) -> str:
            entered.set()
            self.assertTrue(release.wait(2))
            return "voice/github-pr-123456789abc"

        github.checkout_pull_request.side_effect = checkout
        selection = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(worktree),
            "agent",
            str(worktree),
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

    def test_cancellation_before_prompt_submission_stops_submission(self) -> None:
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
        client.prompt_and_wait.side_effect = prompt
        worker = threading.Thread(target=service.run_worker, args=("123456789abc",))
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            worker.start()
            self.assertTrue(entered.wait(2))
            service.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(submitted.is_set())
        client.cancel_agent.assert_called_once_with("agent")

    def test_worker_signal_requires_full_process_identity(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_pid": 42,
            "worker_process_start": "start",
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
        with (
            mock.patch("subprocess.Popen", side_effect=OSError("spawn failed")),
            self.assertRaisesRegex(OSError, "spawn failed"),
        ):
            service.launch_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "failed")
        self.assertFalse(updated["delivered"])
        self.assertEqual(updated["error"], "spawn failed")

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
            "worker_pid": 42,
            "worker_boot_id": "old-boot",
            "worker_process_start": "start",
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
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "github_pull_request": 42,
                "worktree_path": "/worktree/pr-42",
                "pull_request_worktree_state": "provisioning",
                "delivered": False,
            }
        )
        client = mock.Mock()
        reserved_during_cancel: set[str] = set()

        def inspect_reservation(_target: str) -> None:
            reserved_during_cancel.update(jobs.reserved_targets())

        client.cancel_agent.side_effect = inspect_reservation
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            service.cancel_job("123456789abc")
            service.cancel_job("123456789abc")

        self.assertIn("cursor-agent", reserved_during_cancel)
        self.assertNotIn("cursor-agent", jobs.reserved_targets())
        client.cancel_agent.assert_called_once_with("cursor-agent")
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
                "worktree_path": "/worktree/task",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "delivered": False,
            }
        )
        failing = mock.Mock()
        failing.cancel_agent.side_effect = HerdrError("still working")
        recovered = mock.Mock()
        with (
            mock.patch.object(jobs, "_stop_worker", return_value=True),
            mock.patch.object(jobs, "HerdrClient", side_effect=[failing, recovered]),
        ):
            service.cancel_job("123456789abc")
            cancelled = jobs.read_job("123456789abc")
            self.assertTrue(cancelled["target_release_pending"])
            self.assertIn("cursor-agent", jobs.reserved_targets())
            service.recover_jobs()

        recovered.cancel_agent.assert_called_once_with("cursor-agent")
        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])

    def test_provisional_agent_fence_survives_first_not_found(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "herdr_target": "planned-agent",
                "worktree_path": "/worktree/task",
                "agent_dispatch_state": "dispatching",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.cancel_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            None,
        ]
        client.get_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            {
                "name": "planned-agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
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
            self.assertEqual(client.get_agent.call_count, 1)
            with mock.patch("time.time", return_value=105):
                service.recover_jobs()

        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])
        self.assertEqual(client.get_agent.call_count, 2)
        self.assertEqual(client.cancel_agent.call_count, 2)

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
        client.cancel_agent.side_effect = HerdrError(
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
            self.assertIn(
                "manual reconciliation required for Herdr agent late-agent",
                str(manual["error"]),
            )
            self.assertNotIn("reconciliation is pending", str(manual["result"]))

            with mock.patch("time.time", return_value=10_000):
                service.recover_jobs()

        self.assertEqual(client.get_agent.call_count, 2)
        client.cancel_agent.assert_not_called()

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
        self.assertFalse(manual["target_release_pending"])
        client.cancel_agent.assert_called_once_with("unrelated-agent")
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
        self.assertEqual(resolved.error, "fork outcome unknown")
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
        self.assertEqual(resolved.error, "agent startup failed")
        self.assertEqual(resolved.result, "agent startup failed")

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
        self.assertFalse(quarantined["target_release_pending"])
        client.cancel_agent.assert_called_once_with("agent")

        jobs.write_job(
            {
                "id": "bbbbbbbbbbbb",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
            }
        )
        self.assertIsNone(
            jobs._reserve_worker_worktree(
                "bbbbbbbbbbbb",
                "worker",
                repository,
                "voice/task",
                checkout,
                state="planned",
            )
        )

        service.acknowledge_worktree_quarantine("123456789abc")
        self.assertEqual(
            jobs.read_job("123456789abc")["worktree_provision_state"],
            "retained",
        )
        self.assertIsNotNone(
            jobs._reserve_worker_worktree(
                "bbbbbbbbbbbb",
                "worker",
                repository,
                "voice/task",
                checkout,
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
            mock.patch.object(jobs, "_process_identity", return_value=None),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            service.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertFalse(updated["target_release_pending"])
        client.cancel_agent.assert_called_once_with("cursor-agent")

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
