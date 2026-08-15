from __future__ import annotations

import pytest

from local_voice_harness.cursor.lifecycle import (
    CleanupPending,
    CleanupReconciling,
    CleanupSettled,
    MaterializedTerminalOutcome,
    TerminalIntent,
)
from local_voice_harness.cursor.operations import WorkerOwnership
from local_voice_harness.cursor.workflow import WorkflowPhase, WorkflowState
from local_voice_harness.job_lifecycle import (
    AwaitingUserJob,
    BlockedJob,
    CancellationEvent,
    ExecutionComponents,
    FollowUpEvent,
    JobIdentity,
    JobLifecycle,
    JobLifecycleError,
    JobState,
    LifecycleEvent,
    QueuedJob,
    ReconcilingJob,
    RecoveryEvent,
    RoutingProvisioningJob,
    RunningJob,
    SessionControlMode,
    SessionControlState,
    TerminalJob,
    WorkerCallbackEvent,
    WorkerClaim,
    apply_event,
    apply_follow_up,
    legal_edges,
)
from local_voice_harness.prompt_operations import IdlePrompt

WORKER = WorkerClaim("token", 42, "boot", "start", "run", 1)
OWNERSHIP = WorkerOwnership("token", 42, "boot", "start", "run", 1)


def execution() -> ExecutionComponents:
    return ExecutionComponents(
        IdlePrompt(),
        WorkflowState(WorkflowPhase.CLASSIFYING),
    )


def identity(revision: int, *, parent: str | None = None) -> JobIdentity:
    return JobIdentity(
        "123456789abc",
        1,
        parent,
        "do it",
        "test-harness",
        "github",
        revision,
    )


def state(kind: JobState, revision: int) -> JobLifecycle:
    job_identity = identity(revision)
    if kind == JobState.QUEUED:
        return QueuedJob(job_identity, 1, execution())
    if kind == JobState.ROUTING:
        return RoutingProvisioningJob(job_identity, WORKER, execution())
    if kind == JobState.RUNNING:
        return RunningJob(job_identity, WORKER, execution())
    if kind == JobState.AWAITING_USER:
        return AwaitingUserJob(job_identity, "question", "question", execution())
    if kind == JobState.BLOCKED:
        return BlockedJob(job_identity, "blocked", 2, execution())
    if kind == JobState.RECONCILING:
        return ReconcilingJob(
            job_identity,
            execution(),
            CleanupSettled(),
        )
    outcome = MaterializedTerminalOutcome(
        kind.value,
        "failed" if kind == JobState.FAILED else "done",
        "failed" if kind == JobState.FAILED else None,
        2,
    )
    return TerminalJob(job_identity, execution(), outcome)


def test_variants_expose_only_state_specific_top_level_data() -> None:
    queued = state(JobState.QUEUED, 0)
    blocked = state(JobState.BLOCKED, 0)
    terminal = state(JobState.COMPLETED, 0)

    assert isinstance(queued, QueuedJob) and not hasattr(queued, "result")
    assert isinstance(blocked, BlockedJob) and not hasattr(blocked, "outcome")
    assert isinstance(terminal, TerminalJob) and not hasattr(terminal, "cleanup")
    assert not hasattr(queued.execution, "terminal")
    assert not hasattr(queued.execution, "cleanup")
    assert not hasattr(queued.execution, "worker")


def test_reconciling_terminal_intent_requires_pending_cleanup() -> None:
    with pytest.raises(JobLifecycleError, match="release fence"):
        ReconcilingJob(
            identity(0),
            execution(),
            CleanupSettled(),
            TerminalIntent("cancelled", "cancelled", None, 2),
        )


def test_reconciling_accepts_cleanup_reconciliation_without_terminal_intent() -> None:
    job = ReconcilingJob(identity(0), execution(), CleanupReconciling())

    assert isinstance(job.cleanup, CleanupReconciling)


def test_cleanup_reconciliation_cannot_carry_terminal_intent() -> None:
    with pytest.raises(JobLifecycleError, match="release fence"):
        ReconcilingJob(
            identity(0),
            execution(),
            CleanupReconciling(),
            TerminalIntent("cancelled", "cancelled", None, 2),
        )


def test_terminal_and_reconciling_reject_each_others_outcomes() -> None:
    intent = TerminalIntent("cancelled", "cancelled", None, 2)
    outcome = MaterializedTerminalOutcome("cancelled", "cancelled", None, 2)

    with pytest.raises(JobLifecycleError, match="materialized outcome"):
        TerminalJob(identity(0), execution(), intent)  # type: ignore[arg-type]
    with pytest.raises(JobLifecycleError, match="terminal intent is invalid"):
        ReconcilingJob(
            identity(0),
            execution(),
            CleanupPending("release", True),
            outcome,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("source", list(JobState))
@pytest.mark.parametrize("target", list(JobState))
def test_every_standard_top_level_edge_is_explicit(
    source: JobState, target: JobState
) -> None:
    before = state(source, 0)
    after = state(target, 1)
    event = LifecycleEvent(0, after)

    if target == source or target in legal_edges(source):
        assert apply_event(before, event) == after
    else:
        with pytest.raises(JobLifecycleError, match="illegal job lifecycle transition"):
            apply_event(before, event)


def test_recovery_resolution_is_explicit_and_stale_observation_is_rejected() -> None:
    queued = state(JobState.QUEUED, 4)
    completed = state(JobState.COMPLETED, 5)

    assert apply_event(queued, RecoveryEvent(4, completed)) == completed
    with pytest.raises(JobLifecycleError, match="stale"):
        apply_event(queued, RecoveryEvent(3, completed))
    with pytest.raises(JobLifecycleError, match="illegal"):
        apply_event(queued, LifecycleEvent(4, completed))


def test_worker_callback_requires_exact_claim() -> None:
    running = state(JobState.RUNNING, 2)
    queued = state(JobState.QUEUED, 3)

    assert apply_event(running, WorkerCallbackEvent(2, queued, WORKER)) == queued
    stale = WorkerClaim("other", 42, "boot", "start", "run", 1)
    with pytest.raises(JobLifecycleError, match="stale worker callback"):
        apply_event(running, WorkerCallbackEvent(2, queued, stale))


def test_cancellation_keeps_cleanup_intent_distinct_from_terminal_outcome() -> None:
    running = state(JobState.RUNNING, 0)
    intent = TerminalIntent("cancelled", "cancelled", None, 2)
    reconciling = ReconcilingJob(
        identity(1),
        execution(),
        CleanupPending("release", True),
        intent,
    )

    updated = apply_event(running, CancellationEvent(0, reconciling))

    assert isinstance(updated, ReconcilingJob)
    assert updated.terminal_intent == intent
    assert updated.state == JobState.RECONCILING


def test_follow_up_event_fences_parent_completion_and_inherited_identity() -> None:
    parent = state(JobState.COMPLETED, 7)
    child_identity = JobIdentity(
        "aaaaaaaaaaaa",
        3,
        "123456789abc",
        "follow up",
        "test-harness",
        "github",
        0,
    )
    child = QueuedJob(child_identity, 3, execution())

    assert apply_follow_up(parent, FollowUpEvent(7, 2, child)) == child
    with pytest.raises(JobLifecycleError, match="stale follow-up"):
        apply_follow_up(parent, FollowUpEvent(6, 2, child))


def test_session_control_relinquish_and_resume_advance_generation() -> None:
    control = SessionControlState()
    owned = control.relinquish()
    resumed = owned.resume()

    assert control.mode is SessionControlMode.AUTOMATED
    assert owned.mode is SessionControlMode.USER_OWNED
    assert owned.generation == 1
    assert resumed.mode is SessionControlMode.AUTOMATED
    assert resumed.generation == 2
    with pytest.raises(JobLifecycleError, match="already user-owned"):
        owned.relinquish()
    with pytest.raises(JobLifecycleError, match="not user-owned"):
        control.resume()
    with pytest.raises(JobLifecycleError, match="must not be negative"):
        SessionControlState(SessionControlMode.AUTOMATED, -1)
