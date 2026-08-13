"""Agent-neutral top-level lifecycle and event model for durable jobs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TypeAlias

from .cursor.lifecycle import (
    CleanupOwned,
    CleanupPending,
    CleanupReconciling,
    CleanupSettled,
    CleanupState,
    MaterializedTerminalOutcome,
    TerminalIntent,
)
from .cursor.operations import WorkerOwnership
from .cursor.workflow import WorkflowState
from .prompt_operations import PromptOperation


class JobLifecycleError(ValueError):
    """A job lifecycle value or event is incomplete, illegal, or stale."""


class JobState(StrEnum):
    QUEUED = "queued"
    ROUTING = "routing"
    RUNNING = "running"
    AWAITING_USER = "awaiting_user"
    BLOCKED = "blocked"
    RECONCILING = "reconciling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class JobIdentity:
    id: str
    created_at: float
    parent_id: str | None
    request: str
    harness: str
    provider: str | None
    revision: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{12}", self.id) is None:
            raise JobLifecycleError(
                "job id must be 12 lowercase hexadecimal characters"
            )
        if self.parent_id is not None and (
            re.fullmatch(r"[0-9a-f]{12}", self.parent_id) is None
            or self.parent_id == self.id
        ):
            raise JobLifecycleError("job parent identity is invalid")
        if not self.harness.strip():
            raise JobLifecycleError("job harness identity must not be empty")
        if (
            self.provider is not None
            and re.fullmatch(r"[a-z][a-z0-9-]*", self.provider) is None
        ):
            raise JobLifecycleError("job provider identity is invalid")
        if not math.isfinite(self.created_at):
            raise JobLifecycleError("job creation time must be finite")
        if self.revision < 0:
            raise JobLifecycleError("job revision must not be negative")

    def next_revision(self) -> JobIdentity:
        return replace(self, revision=self.revision + 1)


WorkerClaim: TypeAlias = WorkerOwnership


@dataclass(frozen=True, slots=True)
class ExecutionComponents:
    """Prompt and workflow state shared by non-identity lifecycle variants."""

    prompt: PromptOperation | None
    workflow: WorkflowState


@dataclass(frozen=True, slots=True)
class QueuedJob:
    identity: JobIdentity
    queued_at: float
    execution: ExecutionComponents
    worker: WorkerClaim | None = None
    state: JobState = field(default=JobState.QUEUED, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.queued_at):
            raise JobLifecycleError("queued job requires a finite queued time")
        if self.worker is not None and not isinstance(self.worker, WorkerOwnership):
            raise JobLifecycleError("queued worker reservation identity is invalid")


@dataclass(frozen=True, slots=True)
class RoutingProvisioningJob:
    identity: JobIdentity
    worker: WorkerClaim
    execution: ExecutionComponents
    state: JobState = field(default=JobState.ROUTING, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.worker, WorkerOwnership):
            raise JobLifecycleError("routing job requires worker identity")


@dataclass(frozen=True, slots=True)
class RunningJob:
    identity: JobIdentity
    worker: WorkerClaim
    execution: ExecutionComponents
    state: JobState = field(default=JobState.RUNNING, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.worker, WorkerOwnership):
            raise JobLifecycleError("running job requires worker identity")


@dataclass(frozen=True, slots=True)
class AwaitingUserJob:
    identity: JobIdentity
    question: str
    result: str
    execution: ExecutionComponents
    state: JobState = field(default=JobState.AWAITING_USER, init=False)

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.result.strip():
            raise JobLifecycleError("awaiting-user job requires question and result")


@dataclass(frozen=True, slots=True)
class BlockedJob:
    """A retryable stopped state, deliberately distinct from terminal outcomes."""

    identity: JobIdentity
    result: str
    blocked_at: float
    execution: ExecutionComponents
    state: JobState = field(default=JobState.BLOCKED, init=False)

    def __post_init__(self) -> None:
        if not self.result.strip() or not math.isfinite(self.blocked_at):
            raise JobLifecycleError("blocked job requires result and blocked time")


@dataclass(frozen=True, slots=True)
class ReconcilingJob:
    """Cleanup intent before, and separate from, a terminal outcome."""

    identity: JobIdentity
    execution: ExecutionComponents
    cleanup: CleanupState
    terminal_intent: TerminalIntent | None = None
    worker: WorkerClaim | None = None
    state: JobState = field(default=JobState.RECONCILING, init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.cleanup,
            (CleanupSettled, CleanupReconciling, CleanupPending, CleanupOwned),
        ):
            raise JobLifecycleError("reconciling cleanup state is invalid")
        if self.terminal_intent is not None and not isinstance(
            self.terminal_intent, TerminalIntent
        ):
            raise JobLifecycleError("reconciling terminal intent is invalid")
        if self.worker is not None and not isinstance(self.worker, WorkerOwnership):
            raise JobLifecycleError("reconciling worker identity is invalid")
        if self.terminal_intent is not None and not isinstance(
            self.cleanup, (CleanupPending, CleanupOwned)
        ):
            raise JobLifecycleError(
                "terminal intent requires a pending cleanup release fence"
            )


@dataclass(frozen=True, slots=True)
class TerminalJob:
    identity: JobIdentity
    execution: ExecutionComponents
    outcome: MaterializedTerminalOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, MaterializedTerminalOutcome):
            raise JobLifecycleError("terminal job requires materialized outcome")

    @property
    def state(self) -> JobState:
        return JobState(self.outcome.status)


JobLifecycle: TypeAlias = (
    QueuedJob
    | RoutingProvisioningJob
    | RunningJob
    | AwaitingUserJob
    | BlockedJob
    | ReconcilingJob
    | TerminalJob
)


_LEGAL_EDGES: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset(
        {JobState.ROUTING, JobState.RECONCILING, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.ROUTING: frozenset(
        {
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.RECONCILING,
            JobState.AWAITING_USER,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.RUNNING: frozenset(
        {
            JobState.QUEUED,
            JobState.AWAITING_USER,
            JobState.BLOCKED,
            JobState.RECONCILING,
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.RECONCILING: frozenset(
        {
            JobState.QUEUED,
            JobState.AWAITING_USER,
            JobState.BLOCKED,
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.AWAITING_USER: frozenset(
        {
            JobState.QUEUED,
            JobState.RECONCILING,
            JobState.COMPLETED,
            JobState.CANCELLED,
        }
    ),
    JobState.BLOCKED: frozenset(
        {JobState.QUEUED, JobState.RECONCILING, JobState.CANCELLED}
    ),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    expected_revision: int
    next_state: JobLifecycle


@dataclass(frozen=True, slots=True)
class RecoveryEvent(LifecycleEvent):
    """A recovery observation fenced by the revision it inspected."""


@dataclass(frozen=True, slots=True)
class CancellationEvent(LifecycleEvent):
    """A cancellation decision, possibly staging cleanup before finality."""


@dataclass(frozen=True, slots=True)
class WorkerCallbackEvent(LifecycleEvent):
    """A worker callback fenced by both revision and complete worker identity."""

    expected_worker: WorkerClaim


@dataclass(frozen=True, slots=True)
class FollowUpEvent:
    expected_parent_revision: int
    expected_parent_completed_at: float
    child: QueuedJob


JobEvent: TypeAlias = (
    LifecycleEvent | RecoveryEvent | CancellationEvent | WorkerCallbackEvent
)


def legal_edges(state: JobState) -> frozenset[JobState]:
    return _LEGAL_EDGES[state]


def apply_event(current: JobLifecycle, event: JobEvent) -> JobLifecycle:
    if event.expected_revision != current.identity.revision:
        raise JobLifecycleError("stale job lifecycle event revision")
    after = event.next_state
    expected_identity = current.identity.next_revision()
    if after.identity != expected_identity:
        raise JobLifecycleError(
            "job identity, creation, parent, request, harness, provider, "
            "or revision changed outside a transition"
        )
    if isinstance(event, WorkerCallbackEvent):
        current_worker = getattr(current, "worker", None)
        if current_worker != event.expected_worker:
            raise JobLifecycleError("stale worker callback identity")
    if current.state == after.state:
        if isinstance(current, TerminalJob) and isinstance(after, TerminalJob):
            if current.outcome != after.outcome:
                raise JobLifecycleError("materialized terminal outcome is immutable")
        return after
    if isinstance(event, RecoveryEvent) and current.state == JobState.QUEUED:
        # A previously submitted provider operation can be resolved directly by
        # recovery without pretending that a worker ran the job again.
        if after.state in {JobState.COMPLETED, JobState.BLOCKED}:
            return after
    if after.state not in _LEGAL_EDGES[current.state]:
        raise JobLifecycleError(
            f"illegal job lifecycle transition {current.state.value} -> "
            f"{after.state.value}"
        )
    return after


def apply_follow_up(parent: JobLifecycle, event: FollowUpEvent) -> QueuedJob:
    if event.expected_parent_revision != parent.identity.revision:
        raise JobLifecycleError("stale follow-up parent revision")
    if not isinstance(parent, TerminalJob) or parent.state != JobState.COMPLETED:
        raise JobLifecycleError("follow-up requires a completed parent")
    if parent.outcome.completed_at != event.expected_parent_completed_at:
        raise JobLifecycleError("stale follow-up completion identity")
    child = event.child
    if (
        child.identity.parent_id != parent.identity.id
        or child.identity.harness != parent.identity.harness
        or child.identity.provider != parent.identity.provider
        or child.identity.revision != 0
    ):
        raise JobLifecycleError(
            "follow-up must inherit parent, harness, and provider identity"
        )
    return child
