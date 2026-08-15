from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.agents import (
    HarnessSession,
    ReconciliationState,
    SessionReconciliation,
)
from local_voice_harness.cursor.model import CursorJob, JobStatus
from local_voice_harness.cursor.recovery import (
    recover_jobs,
    relinquish_session_control,
    resume_session_control,
)
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations.herdr import HerdrError
from local_voice_harness.job_lifecycle import SessionControlMode
from local_voice_harness.prompt_operations import (
    AmbiguousPrompt,
    PlannedPrompt,
    PromptIdentity,
    SubmittingPrompt,
    observe_prompt_submission,
)


def _ready_job_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": "123456789abc",
        "status": "queued",
        "request": "implement the change",
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
        "agent_dispatch_state": "ready",
        "agent_provider": "cursor/herdr",
        "agent_provider_session_id": "session",
        "agent_state_sequence": 7,
        "agent_operation_target": "agent",
        "agent_operation_checkout": "/repo-worktree",
        "agent_operation_workspace_id": "workspace",
        "agent_operation_pane_id": "pane",
        "participant_admission_state": "held",
    }
    values.update(changes)
    return values


class SessionTakeoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = JobStore(self.root / "jobs", self.root / "legacy")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(
        self,
        *,
        prompt_operation: object | None = None,
        **changes: object,
    ) -> CursorJob:
        job = self.store.create(CursorJob.from_dict(_ready_job_values(**changes)))
        if prompt_operation is None:
            return job
        return (
            self.store.update(
                job.id,
                lambda current: current.evolve(prompt_operation=prompt_operation),
            )
            or job
        )

    def test_idle_takeover_retains_session_checkout_and_reservations(self) -> None:
        job = self.create()
        updated = relinquish_session_control(self.store, job.id, now=10)

        self.assertEqual(updated.session_control, SessionControlMode.USER_OWNED.value)
        self.assertEqual(updated.session_control_generation, 1)
        self.assertEqual(updated.status, JobStatus.QUEUED)
        self.assertEqual(updated.herdr_target, "agent")
        self.assertEqual(updated.agent_provider_session_id, "session")
        self.assertEqual(updated.herdr_workspace_id, "workspace")
        self.assertEqual(updated.worktree_path, "/repo-worktree")
        self.assertEqual(updated.participant_admission_state, "held")
        self.assertFalse(updated.target_release_pending)
        self.assertIsNone(updated.terminal_intent_status)
        self.assertEqual(updated.prompt_operation_state, "none")

    def test_queued_undispatched_prompt_is_not_launched_while_user_owned(
        self,
    ) -> None:
        identity = PromptIdentity(
            job_id="123456789abc",
            phase="implementing",
            turn=1,
            turn_token="123456789abc-1",
            target="agent",
            agent_session="session",
            baseline_sequence=7,
        )
        job = self.create(
            prompt_operation=PlannedPrompt(identity),
            turn=1,
            turn_token="123456789abc-1",
        )
        relinquish_session_control(self.store, job.id, now=10)
        launch = mock.Mock()

        recover_jobs(
            self.store,
            launch_worker=launch,
            herdr_factory=lambda: mock.Mock(),
            is_worker_alive=lambda _job: False,
            now=20,
        )

        recovered = self.store.get(job.id)
        self.assertEqual(recovered.session_control, SessionControlMode.USER_OWNED.value)
        self.assertEqual(recovered.prompt_operation_state, "planned")
        self.assertEqual(recovered.status, JobStatus.QUEUED)
        launch.assert_not_called()

    def test_in_flight_prompt_is_not_accepted_from_sequence_after_takeover(
        self,
    ) -> None:
        identity = PromptIdentity(
            job_id="123456789abc",
            phase="implementing",
            turn=1,
            turn_token="123456789abc-1",
            target="agent",
            agent_session="session",
            baseline_sequence=7,
        )
        job = self.create(
            prompt_operation=SubmittingPrompt(identity),
            turn=1,
            turn_token="123456789abc-1",
        )
        updated = relinquish_session_control(self.store, job.id, now=10)

        self.assertIsInstance(updated.prompt_operation, AmbiguousPrompt)
        self.assertEqual(updated.manual_reconcile_operation, "prompt")
        self.assertIsNotNone(updated.manual_reconcile_token)
        self.assertEqual(
            observe_prompt_submission(
                SubmittingPrompt(identity),
                identity,
                target="agent",
                agent_session="session",
                state_sequence=9,
                sequence_evidence_trusted=False,
            ),
            AmbiguousPrompt(identity),
        )

    def test_session_replacement_blocks_hand_back(self) -> None:
        job = self.create()
        relinquish_session_control(self.store, job.id, now=10)
        client = mock.Mock()
        client.harness.reconcile.return_value = SessionReconciliation(
            ReconciliationState.CHANGED,
            HarnessSession("cursor/herdr", "replacement", "agent", 12),
            "active",
            False,
            "provider session identity changed",
        )

        with self.assertRaises(HarnessError) as raised:
            resume_session_control(
                self.store, job.id, now=20, herdr_factory=lambda: client
            )

        self.assertIn("explicit reconciliation", str(raised.exception))
        current = self.store.get(job.id)
        self.assertEqual(current.session_control, SessionControlMode.USER_OWNED.value)
        self.assertEqual(current.agent_provider_session_id, "session")
        self.assertEqual(current.herdr_target, "agent")
        self.assertEqual(current.worktree_path, "/repo-worktree")

    def test_hand_back_sets_observed_baseline_and_resumes_idle_automation(
        self,
    ) -> None:
        identity = PromptIdentity(
            job_id="123456789abc",
            phase="implementing",
            turn=1,
            turn_token="123456789abc-1",
            target="agent",
            agent_session="session",
            baseline_sequence=7,
        )
        job = self.create(
            prompt_operation=PlannedPrompt(identity),
            turn=1,
            turn_token="123456789abc-1",
        )
        relinquish_session_control(self.store, job.id, now=10)
        client = mock.Mock()
        client.harness.reconcile.return_value = SessionReconciliation(
            ReconciliationState.ACTIVE,
            HarnessSession("cursor/herdr", "session", "agent", 12),
            "active",
            True,
        )

        resumed = resume_session_control(
            self.store, job.id, now=20, herdr_factory=lambda: client
        )

        self.assertEqual(resumed.session_control, SessionControlMode.AUTOMATED.value)
        self.assertEqual(resumed.session_control_generation, 2)
        self.assertEqual(resumed.agent_state_sequence, 12)
        self.assertEqual(resumed.prompt_operation_state, "planned")
        self.assertEqual(resumed.prompt_baseline_sequence, 12)
        self.assertEqual(resumed.herdr_target, "agent")
        self.assertEqual(resumed.agent_provider_session_id, "session")
        client.harness.reconcile.assert_called_once_with(
            "agent", expected_session_id="session"
        )

    def test_hand_back_fails_closed_when_prompt_is_confusable(self) -> None:
        identity = PromptIdentity(
            job_id="123456789abc",
            phase="implementing",
            turn=1,
            turn_token="123456789abc-1",
            target="agent",
            agent_session="session",
            baseline_sequence=7,
        )
        job = self.create(
            prompt_operation=SubmittingPrompt(identity),
            turn=1,
            turn_token="123456789abc-1",
        )
        relinquish_session_control(self.store, job.id, now=10)

        with self.assertRaises(HarnessError) as raised:
            resume_session_control(self.store, job.id, now=20)

        self.assertIn("confused with manual activity", str(raised.exception))
        current = self.store.get(job.id)
        self.assertEqual(current.session_control, SessionControlMode.USER_OWNED.value)
        self.assertEqual(current.prompt_operation_state, "ambiguous")

    def test_restart_while_manually_controlled_does_not_dispatch(self) -> None:
        job = self.create()
        relinquish_session_control(self.store, job.id, now=10)
        launch = mock.Mock()

        recover_jobs(
            self.store,
            launch_worker=launch,
            herdr_factory=lambda: mock.Mock(),
            is_worker_alive=lambda _job: False,
            now=30,
        )

        recovered = self.store.get(job.id)
        self.assertEqual(recovered.session_control, SessionControlMode.USER_OWNED.value)
        self.assertEqual(recovered.herdr_target, "agent")
        self.assertEqual(recovered.worktree_path, "/repo-worktree")
        self.assertFalse(recovered.target_release_pending)
        launch.assert_not_called()

    def test_transient_reconcile_failure_does_not_resume(self) -> None:
        job = self.create()
        relinquish_session_control(self.store, job.id, now=10)
        client = mock.Mock()
        client.ensure_server.side_effect = HerdrError("herdr unavailable")

        with self.assertRaises(HarnessError) as raised:
            resume_session_control(
                self.store, job.id, now=20, herdr_factory=lambda: client
            )

        self.assertIn("could not reconcile", str(raised.exception))
        current = self.store.get(job.id)
        self.assertEqual(current.session_control, SessionControlMode.USER_OWNED.value)

    def test_terminal_job_cannot_relinquish_control(self) -> None:
        job = self.create(
            status="completed",
            result="done",
            completed_at=2,
            workflow_phase="finished",
            participant_admission_state="released",
        )

        with self.assertRaises(HarnessError) as raised:
            relinquish_session_control(self.store, job.id, now=10)

        self.assertIn("already completed", str(raised.exception))
        current = self.store.get(job.id)
        self.assertEqual(current.session_control, SessionControlMode.AUTOMATED.value)
        self.assertEqual(current.session_control_generation, 0)
