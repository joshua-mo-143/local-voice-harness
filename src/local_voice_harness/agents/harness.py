"""Provider-neutral coding-agent transport and lifecycle contract.

Workspace selection, worktree allocation, pane ownership, and durable job storage
are deliberately outside this module.  A harness receives an already prepared,
opaque launch context and owns only the remote agent session lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class HarnessCapability(StrEnum):
    """Optional behavior an implementation can explicitly advertise."""

    CLARIFICATION_REPLIES = "clarification_replies"
    CANCELLATION = "cancellation"
    MCP_CONNECTORS = "mcp_connectors"
    RECOVERY = "recovery"


class HarnessEventKind(StrEnum):
    """Portable lifecycle events emitted by a harness task."""

    SUCCEEDED = "succeeded"
    CLARIFICATION = "clarification"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReconciliationState(StrEnum):
    """Result of observing a durable session after an interruption."""

    ACTIVE = "active"
    SETTLED = "settled"
    MISSING = "missing"
    CHANGED = "changed"
    UNKNOWN = "unknown"


class HarnessContractError(RuntimeError):
    """Base error raised at the provider-neutral harness boundary."""

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


class UnsupportedCapabilityError(HarnessContractError):
    """Raised before side effects when requested behavior is unavailable."""

    def __init__(self, provider: str, capability: HarnessCapability) -> None:
        super().__init__(
            f"{provider} does not support the required harness capability "
            f"{capability.value!r}; choose a compatible harness or disable that feature",
            code="unsupported_capability",
        )
        self.provider = provider
        self.capability = capability


@dataclass(frozen=True, slots=True)
class SessionRequest:
    """Request a provider session in an already allocated transport context."""

    name: str
    provider: str
    mode: str | None = None
    launch_context: Mapping[str, str] = field(default_factory=dict)
    required_capabilities: frozenset[HarnessCapability] = frozenset()


@dataclass(frozen=True, slots=True)
class HarnessSession:
    """Durable provider identity used to fence every later operation."""

    provider: str
    session_id: str
    target: str
    state_sequence: int
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HarnessTask:
    """One task or clarification turn submitted to an existing session."""

    text: str
    correlation_id: str
    expected_session_id: str
    baseline_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class TaskSubmission:
    """Accepted task identity and its pre-submit observation fence."""

    session: HarnessSession
    correlation_id: str
    baseline_sequence: int
    started_at: float


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    """A provider-neutral task outcome."""

    kind: HarnessEventKind
    session: HarnessSession
    status: str
    output: str = ""
    summary: str | None = None
    question: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SessionReconciliation:
    """Read-only recovery observation; it never authorizes task replay."""

    state: ReconciliationState
    session: HarnessSession | None
    status: str
    recoverable: bool
    detail: str | None = None


Checkpoint = Callable[[], None]
BeforeSubmit = Callable[[int], None]
Accepted = Callable[[], None]


def require_capabilities(
    provider: str,
    available: frozenset[HarnessCapability],
    required: frozenset[HarnessCapability],
) -> None:
    """Fail deterministically before an unsupported operation can have effects."""

    missing = sorted(required - available, key=lambda item: item.value)
    if missing:
        raise UnsupportedCapabilityError(provider, missing[0])


@runtime_checkable
class AgentHarness(Protocol):
    """Stable session transport contract implemented by coding-agent harnesses."""

    @property
    def provider(self) -> str: ...

    @property
    def capabilities(self) -> frozenset[HarnessCapability]: ...

    def create_session(
        self,
        request: SessionRequest,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> HarnessSession: ...

    def submit_task(
        self,
        session: HarnessSession,
        task: HarnessTask,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: BeforeSubmit | None = None,
        accepted: Accepted | None = None,
    ) -> TaskSubmission: ...

    def stream_events(
        self,
        submission: TaskSubmission,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Iterator[HarnessEvent]: ...

    def reply_to_clarification(
        self,
        session: HarnessSession,
        task: HarnessTask,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: BeforeSubmit | None = None,
        accepted: Accepted | None = None,
    ) -> TaskSubmission: ...

    def cancel(
        self,
        session: HarnessSession,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> None: ...

    def reconcile(
        self,
        target: str,
        *,
        expected_session_id: str | None = None,
    ) -> SessionReconciliation: ...
