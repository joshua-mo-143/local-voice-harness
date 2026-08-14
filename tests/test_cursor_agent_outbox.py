from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_voice_harness.agents.harness import (
    Accepted,
    BeforeDispatch,
    BeforeSubmit,
    Checkpoint,
    HarnessCapability,
    HarnessEvent,
    HarnessEventKind,
    HarnessSession,
    HarnessTask,
    ReconciliationState,
    SessionReconciliation,
    SessionRequest,
    TaskSubmission,
)
from local_voice_harness.cursor.agent_outbox import (
    CLARIFICATION_REPLY,
    SESSION_CREATE,
    TASK_SUBMIT,
    agent_effect_handlers,
    consume_agent_results,
    session_payload,
)
from local_voice_harness.cursor.coordinator import (
    OUTBOX_RETRY_SECONDS,
    CoordinatorCommand,
    CoordinatorDecision,
    DurableEffect,
    OutboxResult,
)
from local_voice_harness.cursor.model import CURRENT_SCHEMA_VERSION, CursorJob
from local_voice_harness.cursor.outbox import drain_outbox
from local_voice_harness.cursor.store import JobStore


def _job(job_id: str = "aaaaaaaaaaaa") -> CursorJob:
    return CursorJob.from_dict(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "id": job_id,
            "revision": 0,
            "request": "exercise agent effects",
            "status": "queued",
            "created_at": 1,
            "queued_at": 1,
            "delivered": False,
        }
    )


def _admit(
    store: JobStore,
    kind: str,
    payload: dict[str, object],
    *,
    key: str | None = None,
) -> str:
    created = store.create(_job())
    idempotency_key = key or f"{kind}:{created.id}:1"
    updated = store.apply(
        CoordinatorCommand(
            job_id=created.id,
            expected_revision=created.revision,
            command_id=f"admit:{idempotency_key}",
            kind=f"{kind}.admit",
        ),
        lambda job: CoordinatorDecision(
            job=job.evolve(speakable_label="agent effect"),
            effects=(
                DurableEffect(
                    kind=kind,
                    idempotency_key=idempotency_key,
                    concurrency_key="cursor/herdr:voice",
                    payload=payload,
                ),
            ),
        ),
    )
    assert updated is not None
    return updated.id


def _row(store: JobStore) -> sqlite3.Row:
    with sqlite3.connect(store.db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM outbox").fetchone()
    assert row is not None
    return row


class RecordingHarness:
    provider = "cursor/herdr"
    capabilities = frozenset(HarnessCapability)

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.session = HarnessSession(
            provider=self.provider,
            session_id="session-1",
            target="voice",
            state_sequence=4,
            metadata={
                "pane_id": "pane-1",
                "workspace_id": "workspace-1",
                "cwd": "/checkout",
            },
        )
        self.event_sequence = 5
        self.crash_before_submit = False
        self.crash_after_submit = False
        self.last_task: HarnessTask | None = None
        self.submission_override: TaskSubmission | None = None

    def create_session(
        self,
        request: SessionRequest,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: BeforeDispatch | None = None,
    ) -> HarnessSession:
        self.calls.append(f"create:{request.name}")
        if self.crash_before_submit:
            raise RuntimeError("before create")
        assert before_submit is not None
        before_submit()
        if self.crash_after_submit:
            raise RuntimeError("after create")
        return self.session

    def submit_task(
        self,
        session: HarnessSession,
        task: HarnessTask,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: BeforeSubmit | None = None,
        accepted: Accepted | None = None,
    ) -> TaskSubmission:
        return self._submit("submit", session, task, before_submit, accepted)

    def reply_to_clarification(
        self,
        session: HarnessSession,
        task: HarnessTask,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: BeforeSubmit | None = None,
        accepted: Accepted | None = None,
    ) -> TaskSubmission:
        return self._submit("reply", session, task, before_submit, accepted)

    def _submit(
        self,
        name: str,
        session: HarnessSession,
        task: HarnessTask,
        before_submit: BeforeSubmit | None,
        accepted: Accepted | None,
    ) -> TaskSubmission:
        self.calls.append(f"{name}:{task.correlation_id}")
        self.last_task = task
        if self.crash_before_submit:
            raise RuntimeError("before submit")
        assert before_submit is not None
        assert task.baseline_sequence is not None
        before_submit(task.baseline_sequence)
        if self.crash_after_submit:
            raise RuntimeError("after submit")
        assert accepted is not None
        accepted()
        if self.submission_override is not None:
            return self.submission_override
        return TaskSubmission(
            session=session,
            correlation_id=task.correlation_id,
            baseline_sequence=task.baseline_sequence,
            started_at=1,
        )

    def stream_events(
        self,
        submission: TaskSubmission,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Iterator[HarnessEvent]:
        self.calls.append(f"stream:{submission.correlation_id}")
        yield HarnessEvent(
            HarnessEventKind.SUCCEEDED,
            HarnessSession(
                provider=self.session.provider,
                session_id=self.session.session_id,
                target=self.session.target,
                state_sequence=self.event_sequence,
                metadata=self.session.metadata,
            ),
            "done",
            output="finished",
            revision=9,
        )

    def cancel(
        self,
        session: HarnessSession,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        raise AssertionError("retained outbox effects must not cancel sessions")

    def reconcile(
        self,
        target: str,
        *,
        expected_session_id: str | None = None,
    ) -> SessionReconciliation:
        return SessionReconciliation(
            ReconciliationState.UNKNOWN,
            None,
            "unknown",
            False,
        )


def _task_payload(harness: RecordingHarness) -> dict[str, object]:
    return {
        "session": session_payload(harness.session),
        "text": "do the work",
        "correlation_id": "turn-1",
        "baseline_sequence": harness.session.state_sequence,
        "expected_revision": 1,
    }


def test_only_retained_agent_effect_handlers_are_registered() -> None:
    handlers = agent_effect_handlers(RecordingHarness())
    assert set(handlers) == {SESSION_CREATE, TASK_SUBMIT, CLARIFICATION_REPLY}


def test_session_create_runs_once_through_harness_after_restart(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    legacy = tmp_path / "legacy"
    store = JobStore(jobs, legacy)
    _admit(
        store,
        SESSION_CREATE,
        {
            "name": "voice",
            "provider": "cursor/herdr",
            "mode": "plan",
            "launch_context": {
                "pane_id": "pane-1",
                "workspace_id": "workspace-1",
            },
            "required_capabilities": [],
            "expected_revision": 1,
        },
    )
    provider = RecordingHarness()

    reopened = JobStore(jobs, legacy)
    assert drain_outbox(reopened, agent_effect_handlers(provider), now=10) == 1
    restarted = RecordingHarness()
    assert drain_outbox(reopened, agent_effect_handlers(restarted), now=11) == 0
    assert provider.calls == ["create:voice"]
    assert restarted.calls == []
    outcome = json.loads(_row(reopened)["outcome_json"])
    assert outcome["outcome"] == "Confirmed"
    assert outcome["session"]["session_id"] == "session-1"


def test_clarification_reply_runs_through_harness_after_restart(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    legacy = tmp_path / "legacy"
    provider = RecordingHarness()
    store = JobStore(jobs, legacy)
    _admit(store, CLARIFICATION_REPLY, _task_payload(provider))

    reopened = JobStore(jobs, legacy)
    assert drain_outbox(reopened, agent_effect_handlers(provider), now=10) == 1
    assert provider.calls == ["reply:turn-1", "stream:turn-1"]
    assert _row(reopened)["status"] == "succeeded"


def test_task_submit_runs_through_harness_after_restart(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    legacy = tmp_path / "legacy"
    provider = RecordingHarness()
    store = JobStore(jobs, legacy)
    _admit(store, TASK_SUBMIT, _task_payload(provider))

    reopened = JobStore(jobs, legacy)
    assert drain_outbox(reopened, agent_effect_handlers(provider), now=10) == 1
    assert provider.calls == ["submit:turn-1", "stream:turn-1"]
    assert _row(reopened)["status"] == "succeeded"


@pytest.mark.parametrize(
    ("kind", "expected_call"),
    [
        (TASK_SUBMIT, "submit:turn-1"),
        (CLARIFICATION_REPLY, "reply:turn-1"),
    ],
)
def test_task_effect_uses_exact_harness_operation(
    tmp_path: Path, kind: str, expected_call: str
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    _admit(store, kind, _task_payload(provider))

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    assert provider.calls == [expected_call, "stream:turn-1"]
    outcome = json.loads(_row(store)["outcome_json"])
    assert outcome["outcome"] == "Confirmed"
    assert outcome["event"]["session"]["state_sequence"] == 5
    assert outcome["event"]["revision"] == 9


def test_nonadvancing_task_event_is_outcome_unknown(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    provider.event_sequence = provider.session.state_sequence
    _admit(store, TASK_SUBMIT, _task_payload(provider))

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    row = _row(store)
    assert row["status"] == "unknown"
    assert json.loads(row["outcome_json"])["outcome"] == "OutcomeUnknown"


def test_mismatched_task_submission_is_outcome_unknown(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    provider.submission_override = TaskSubmission(
        HarnessSession("cursor/herdr", "other-session", "voice", 4),
        "other-turn",
        3,
        1,
    )
    _admit(store, TASK_SUBMIT, _task_payload(provider))

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    assert provider.calls == ["submit:turn-1"]
    row = _row(store)
    assert row["status"] == "unknown"
    assert json.loads(row["outcome_json"])["outcome"] == "OutcomeUnknown"


def test_task_effect_preserves_interactive_boundary_options(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    payload = _task_payload(provider)
    payload.update(
        completion_marker="WORKFLOW_PLAN",
        allow_interactive_boundary=True,
        allow_fallback_submit=False,
    )
    _admit(store, TASK_SUBMIT, payload)

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    assert provider.last_task is not None
    assert provider.last_task.completion_marker == "WORKFLOW_PLAN"
    assert provider.last_task.allow_interactive_boundary
    assert not provider.last_task.allow_fallback_submit


def test_submission_crash_after_dispatch_is_never_replayed(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    provider.crash_after_submit = True
    _admit(store, TASK_SUBMIT, _task_payload(provider))

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    provider.crash_after_submit = False
    assert (
        drain_outbox(
            store,
            agent_effect_handlers(provider),
            now=10 + OUTBOX_RETRY_SECONDS,
        )
        == 0
    )
    assert provider.calls == ["submit:turn-1"]
    assert _row(store)["status"] == "unknown"


def test_session_create_crash_after_dispatch_is_never_replayed(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    provider.crash_after_submit = True
    _admit(
        store,
        SESSION_CREATE,
        {
            "name": "voice",
            "provider": "cursor/herdr",
            "launch_context": {
                "pane_id": "pane-1",
                "workspace_id": "workspace-1",
            },
            "required_capabilities": [],
            "expected_revision": 1,
        },
    )

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    restarted = RecordingHarness()
    assert (
        drain_outbox(
            store,
            agent_effect_handlers(restarted),
            now=10 + OUTBOX_RETRY_SECONDS,
        )
        == 0
    )
    assert provider.calls == ["create:voice"]
    assert restarted.calls == []
    assert _row(store)["status"] == "unknown"


def test_submission_crash_before_dispatch_retries_same_effect(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    provider.crash_before_submit = True
    _admit(store, TASK_SUBMIT, _task_payload(provider))

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    assert _row(store)["status"] == "failed-retryable"
    provider.crash_before_submit = False
    assert (
        drain_outbox(
            store,
            agent_effect_handlers(provider),
            now=10 + OUTBOX_RETRY_SECONDS,
        )
        == 1
    )
    assert provider.calls == ["submit:turn-1", "submit:turn-1", "stream:turn-1"]


def test_superseded_effect_is_rejected_before_dispatch(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    job_id = _admit(store, TASK_SUBMIT, _task_payload(provider))
    store.update(job_id, lambda job: job.evolve(speakable_label="superseded"))

    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 0
    assert provider.calls == []
    row = _row(store)
    assert row["status"] == "failed"
    assert json.loads(row["outcome_json"])["outcome"] == "Failed"


def test_terminal_observation_is_reduced_by_effect_and_revision(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    job_id = _admit(store, TASK_SUBMIT, _task_payload(provider))
    key = f"{TASK_SUBMIT}:{job_id}:1"
    assert store.outbox_result(key) is None
    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    assert store.outbox_result(key) is not None

    def reduce(job: CursorJob, result: OutboxResult) -> CursorJob | None:
        assert result.job_id == job.id
        assert result.kind == TASK_SUBMIT
        assert result.idempotency_key == f"{TASK_SUBMIT}:{job.id}:1"
        event = result.outcome.get("event")
        assert isinstance(event, dict)
        session = event.get("session")
        assert isinstance(session, dict)
        if (
            session.get("session_id") != provider.session.session_id
            or session.get("target") != provider.session.target
            or session.get("state_sequence") != 5
        ):
            return None
        return job.evolve(speakable_label="observed")

    assert consume_agent_results(store, reduce) == 1
    assert store.get(job_id).speakable_label == "observed"
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT disposition FROM outbox_consumptions"
        ).fetchone() == ("applied",)
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM events WHERE kind = ?",
                (f"{TASK_SUBMIT}.observed",),
            ).fetchone()[0]
        )
    assert payload["effect_id"] == _row(store)["effect_id"]
    assert payload["idempotency_key"] == f"{TASK_SUBMIT}:{job_id}:1"


def test_late_observation_cannot_overwrite_newer_revision(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    job_id = _admit(store, TASK_SUBMIT, _task_payload(provider))
    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    store.update(job_id, lambda job: job.evolve(speakable_label="cancelled later"))
    called = False

    def reduce(job: CursorJob, result: OutboxResult) -> CursorJob | None:
        nonlocal called
        called = True
        if job.speakable_label == "cancelled later":
            return None
        return job.evolve(speakable_label="stale success")

    assert consume_agent_results(store, reduce) == 1
    assert called
    assert store.get(job_id).speakable_label == "cancelled later"
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT disposition FROM outbox_consumptions"
        ).fetchone() == ("rejected",)


def test_benign_revision_change_revalidates_terminal_observation(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    job_id = _admit(store, TASK_SUBMIT, _task_payload(provider))
    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    store.update(job_id, lambda job: job.evolve(speakable_label="concurrent update"))

    def reduce(job: CursorJob, result: OutboxResult) -> CursorJob | None:
        if result.kind != TASK_SUBMIT or job.speakable_label != "concurrent update":
            return None
        return job.evolve(speakable_label="revalidated")

    assert consume_agent_results(store, reduce) == 1
    assert store.get(job_id).speakable_label == "revalidated"


def test_consumption_rejects_forged_effect_identity(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    provider = RecordingHarness()
    _admit(store, TASK_SUBMIT, _task_payload(provider))
    assert drain_outbox(store, agent_effect_handlers(provider), now=10) == 1
    result = store.unconsumed_outbox_results((TASK_SUBMIT,))[0]
    forged = OutboxResult(
        effect_id=result.effect_id,
        job_id=result.job_id,
        kind=result.kind,
        idempotency_key="forged",
        payload=result.payload,
        status=result.status,
        outcome=result.outcome,
    )
    assert not store.mark_outbox_consumed(forged, "applied")
