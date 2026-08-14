"""Typed lifecycle adapters for durable checkout and provider operations.

The job record intentionally remains flat.  These values are the production
boundary which validates complete identities and fences state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class OperationTransitionError(ValueError):
    """A durable operation is incomplete or attempted an illegal transition."""


class OperationState(StrEnum):
    PLANNED = "planned"
    ACTIVE = "active"
    SETTLED = "settled"
    UNKNOWN = "unknown"
    FAILED = "failed"
    MANUAL = "manual"


class CheckoutState(StrEnum):
    PLANNED = "planned"
    DISPATCHING = "dispatching"
    READY = "ready"
    RETAINED = "retained"
    QUARANTINED = "quarantined"
    AMBIGUOUS = "ambiguous"
    FAILED_OBSERVING = "failed_observing"
    CONFIRMED_ABSENT = "confirmed_absent"
    MANUAL_REQUIRED = "manual_required"


class AgentSessionState(StrEnum):
    DISPATCHING = "dispatching"
    READY = "ready"
    RETAINED = "retained"
    AMBIGUOUS = "ambiguous"
    FAILED_OBSERVING = "failed_observing"
    CONFIRMED_ABSENT = "confirmed_absent"
    MANUAL_REQUIRED = "manual_required"


_SETTLED_CHECKOUT_STATES = frozenset({CheckoutState.READY, CheckoutState.RETAINED})
_UNKNOWN_CHECKOUT_STATES = frozenset(
    {
        CheckoutState.QUARANTINED,
        CheckoutState.AMBIGUOUS,
        CheckoutState.FAILED_OBSERVING,
    }
)
_SETTLED_SESSION_STATES = frozenset(
    {AgentSessionState.READY, AgentSessionState.RETAINED}
)
_UNKNOWN_SESSION_STATES = frozenset(
    {AgentSessionState.AMBIGUOUS, AgentSessionState.FAILED_OBSERVING}
)

# Preserve the previous collapsed OperationState edges, plus same-collapse
# distinctions that callers already persisted (ready/retained, quarantined/
# ambiguous/failed_observing).
_CHECKOUT_TRANSITIONS = {
    CheckoutState.PLANNED: frozenset(
        {
            CheckoutState.DISPATCHING,
            CheckoutState.READY,
            CheckoutState.RETAINED,
            CheckoutState.CONFIRMED_ABSENT,
        }
    ),
    CheckoutState.DISPATCHING: frozenset(
        {
            CheckoutState.READY,
            CheckoutState.RETAINED,
            CheckoutState.QUARANTINED,
            CheckoutState.AMBIGUOUS,
            CheckoutState.FAILED_OBSERVING,
            CheckoutState.CONFIRMED_ABSENT,
            CheckoutState.MANUAL_REQUIRED,
        }
    ),
    CheckoutState.QUARANTINED: frozenset(
        {
            CheckoutState.READY,
            CheckoutState.RETAINED,
            CheckoutState.AMBIGUOUS,
            CheckoutState.FAILED_OBSERVING,
            CheckoutState.CONFIRMED_ABSENT,
            CheckoutState.MANUAL_REQUIRED,
        }
    ),
    CheckoutState.AMBIGUOUS: frozenset(
        {
            CheckoutState.READY,
            CheckoutState.RETAINED,
            CheckoutState.QUARANTINED,
            CheckoutState.FAILED_OBSERVING,
            CheckoutState.CONFIRMED_ABSENT,
            CheckoutState.MANUAL_REQUIRED,
        }
    ),
    CheckoutState.FAILED_OBSERVING: frozenset(
        {
            CheckoutState.READY,
            CheckoutState.RETAINED,
            CheckoutState.QUARANTINED,
            CheckoutState.AMBIGUOUS,
            CheckoutState.CONFIRMED_ABSENT,
            CheckoutState.MANUAL_REQUIRED,
        }
    ),
    CheckoutState.MANUAL_REQUIRED: frozenset(
        {
            CheckoutState.READY,
            CheckoutState.RETAINED,
            CheckoutState.CONFIRMED_ABSENT,
        }
    ),
    CheckoutState.READY: frozenset(
        {CheckoutState.RETAINED, CheckoutState.CONFIRMED_ABSENT}
    ),
    CheckoutState.RETAINED: frozenset(
        {CheckoutState.READY, CheckoutState.CONFIRMED_ABSENT}
    ),
    CheckoutState.CONFIRMED_ABSENT: frozenset(),
}

_SESSION_TRANSITIONS = {
    AgentSessionState.DISPATCHING: frozenset(
        {
            AgentSessionState.READY,
            AgentSessionState.RETAINED,
            AgentSessionState.AMBIGUOUS,
            AgentSessionState.FAILED_OBSERVING,
            AgentSessionState.CONFIRMED_ABSENT,
            AgentSessionState.MANUAL_REQUIRED,
        }
    ),
    AgentSessionState.AMBIGUOUS: frozenset(
        {
            AgentSessionState.READY,
            AgentSessionState.RETAINED,
            AgentSessionState.FAILED_OBSERVING,
            AgentSessionState.CONFIRMED_ABSENT,
            AgentSessionState.MANUAL_REQUIRED,
        }
    ),
    AgentSessionState.FAILED_OBSERVING: frozenset(
        {
            AgentSessionState.READY,
            AgentSessionState.RETAINED,
            AgentSessionState.AMBIGUOUS,
            AgentSessionState.CONFIRMED_ABSENT,
            AgentSessionState.MANUAL_REQUIRED,
        }
    ),
    AgentSessionState.MANUAL_REQUIRED: frozenset(
        {
            AgentSessionState.READY,
            AgentSessionState.RETAINED,
            AgentSessionState.AMBIGUOUS,
            AgentSessionState.CONFIRMED_ABSENT,
        }
    ),
    AgentSessionState.READY: frozenset(
        {AgentSessionState.RETAINED, AgentSessionState.CONFIRMED_ABSENT}
    ),
    AgentSessionState.RETAINED: frozenset(
        {AgentSessionState.READY, AgentSessionState.CONFIRMED_ABSENT}
    ),
    AgentSessionState.CONFIRMED_ABSENT: frozenset(),
}


_TRANSITIONS = {
    OperationState.PLANNED: frozenset(
        {OperationState.ACTIVE, OperationState.SETTLED, OperationState.FAILED}
    ),
    OperationState.ACTIVE: frozenset(
        {
            OperationState.SETTLED,
            OperationState.UNKNOWN,
            OperationState.FAILED,
            OperationState.MANUAL,
        }
    ),
    OperationState.UNKNOWN: frozenset(
        {
            OperationState.SETTLED,
            OperationState.FAILED,
            OperationState.MANUAL,
        }
    ),
    OperationState.MANUAL: frozenset({OperationState.SETTLED, OperationState.FAILED}),
    OperationState.SETTLED: frozenset({OperationState.FAILED}),
    OperationState.FAILED: frozenset(),
}


def _required(subject: str, **values: object) -> None:
    missing = [name for name, value in values.items() if value is None or value == ""]
    if missing:
        raise OperationTransitionError(
            f"{subject} requires complete {', '.join(missing)} identity"
        )


@dataclass(frozen=True, slots=True)
class WorkerOwnership:
    token: str
    pid: int
    boot_id: str
    process_start: str
    operation: str
    claimed_at: float

    def __post_init__(self) -> None:
        _required(
            "worker ownership",
            token=self.token,
            pid=self.pid,
            boot_id=self.boot_id,
            process_start=self.process_start,
            operation=self.operation,
            claimed_at=self.claimed_at,
        )
        if self.pid <= 0 or self.claimed_at < 0:
            raise OperationTransitionError(
                "worker ownership has invalid process identity"
            )

    def matches(self, other: WorkerOwnership | None) -> bool:
        return other == self


def worker_ownership_fields(
    ownership: WorkerOwnership | None,
) -> dict[str, object]:
    if ownership is None:
        return {
            "worker_token": None,
            "worker_pid": None,
            "worker_boot_id": None,
            "worker_process_start": None,
            "worker_claim_operation": None,
            "worker_claimed_at": None,
        }
    return {
        "worker_token": ownership.token,
        "worker_pid": ownership.pid,
        "worker_boot_id": ownership.boot_id,
        "worker_process_start": ownership.process_start,
        "worker_claim_operation": ownership.operation,
        "worker_claimed_at": ownership.claimed_at,
    }


def load_worker_ownership(
    *,
    token: str | None,
    pid: int | None,
    boot_id: str | None,
    process_start: str | None,
    operation: str | None,
    claimed_at: float | None,
) -> WorkerOwnership | None:
    values = (token, pid, boot_id, process_start, operation, claimed_at)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise OperationTransitionError("worker ownership is incomplete")
    assert token is not None
    assert pid is not None
    assert boot_id is not None
    assert process_start is not None
    assert operation is not None
    assert claimed_at is not None
    return WorkerOwnership(token, pid, boot_id, process_start, operation, claimed_at)


@dataclass(frozen=True, slots=True)
class CheckoutSpec:
    repository: str
    branch: str
    path: str

    def __post_init__(self) -> None:
        _required(
            "checkout spec",
            repository=self.repository,
            branch=self.branch,
            path=self.path,
        )


@dataclass(frozen=True, slots=True)
class CheckoutOperation:
    state: CheckoutState
    spec: CheckoutSpec
    workspace_id: str | None = None
    root_pane_id: str | None = None

    def __post_init__(self) -> None:
        if bool(self.workspace_id) != bool(self.root_pane_id):
            raise OperationTransitionError(
                "checkout workspace and root-pane identity must be paired"
            )
        if self.state in _SETTLED_CHECKOUT_STATES and (
            not self.workspace_id or not self.root_pane_id
        ):
            raise OperationTransitionError(
                "settled checkout requires workspace and root-pane identity"
            )

    def transition(
        self,
        state: CheckoutState,
        *,
        workspace_id: str | None = None,
        root_pane_id: str | None = None,
    ) -> CheckoutOperation:
        if state not in _CHECKOUT_TRANSITIONS[self.state]:
            raise OperationTransitionError(
                f"checkout cannot transition from {self.state} to {state}"
            )
        return replace(
            self,
            state=state,
            workspace_id=workspace_id or self.workspace_id,
            root_pane_id=root_pane_id or self.root_pane_id,
        )


def checkout_fields(operation: CheckoutOperation) -> dict[str, object]:
    """Flatten a typed checkout without changing the legacy storage schema."""
    return {
        "worktree_provision_state": operation.state.value,
        "repository": operation.spec.repository,
        "worktree_branch": operation.spec.branch,
        "worktree_path": operation.spec.path,
        "worktree_workspace_id": operation.workspace_id,
        "worktree_root_pane_id": operation.root_pane_id,
    }


def load_checkout_operation(
    *,
    state: str | None,
    repository: str | None,
    branch: str | None,
    path: str | None,
    workspace_id: str | None,
    root_pane_id: str | None,
) -> CheckoutOperation | None:
    """Adapt persisted checkout labels into the typed operation."""
    if state is None:
        return None
    try:
        parsed = CheckoutState(state)
    except ValueError as exc:
        raise OperationTransitionError(
            f"invalid checkout operation state {state!r}"
        ) from exc
    return CheckoutOperation(
        parsed,
        CheckoutSpec(repository or "", branch or "", path or ""),
        workspace_id,
        root_pane_id,
    )


def checkout_blocks_reservation(state: CheckoutState | None) -> bool:
    return state in {CheckoutState.QUARANTINED, CheckoutState.MANUAL_REQUIRED}


def checkout_is_usable(state: CheckoutState | None) -> bool:
    return state in _SETTLED_CHECKOUT_STATES


def uncertain_checkout_state(*, ambiguous: bool) -> CheckoutState:
    return CheckoutState.AMBIGUOUS if ambiguous else CheckoutState.FAILED_OBSERVING


@dataclass(frozen=True, slots=True)
class ForkSpec:
    source: str
    source_url: str
    source_default_branch: str
    source_private: bool
    login: str
    target: str
    source_parent: str | None = None

    def __post_init__(self) -> None:
        _required(
            "fork spec",
            source=self.source,
            source_url=self.source_url,
            source_default_branch=self.source_default_branch,
            login=self.login,
            target=self.target,
        )


@dataclass(frozen=True, slots=True)
class ForkOperation:
    state: OperationState
    spec: ForkSpec

    def transition(self, state: OperationState) -> ForkOperation:
        if state not in _TRANSITIONS[self.state]:
            raise OperationTransitionError(
                f"fork cannot transition from {self.state} to {state}"
            )
        return replace(self, state=state)


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    provider: str
    session_id: str
    target: str
    state_sequence: int

    def __post_init__(self) -> None:
        _required(
            "session identity",
            provider=self.provider,
            session_id=self.session_id,
            target=self.target,
        )
        if self.state_sequence < 0:
            raise OperationTransitionError(
                "session state sequence must not be negative"
            )

    def accepts(
        self,
        *,
        provider: str,
        session_id: str,
        target: str,
        state_sequence: int,
    ) -> bool:
        return (
            provider == self.provider
            and session_id == self.session_id
            and target == self.target
            and state_sequence >= self.state_sequence
        )


@dataclass(frozen=True, slots=True)
class AgentSessionSpec:
    target: str
    checkout: str
    workspace_id: str
    pane_id: str

    def __post_init__(self) -> None:
        _required(
            "agent session spec",
            target=self.target,
            checkout=self.checkout,
            workspace_id=self.workspace_id,
            pane_id=self.pane_id,
        )


@dataclass(frozen=True, slots=True)
class AgentSessionOperation:
    state: AgentSessionState
    spec: AgentSessionSpec
    session: SessionIdentity | None = None

    def __post_init__(self) -> None:
        if self.session is not None and self.session.target != self.spec.target:
            raise OperationTransitionError(
                "agent session target does not match its spec"
            )
        if self.state in _SETTLED_SESSION_STATES and self.session is None:
            raise OperationTransitionError(
                "settled agent operation requires provider session identity"
            )

    def transition(
        self,
        state: AgentSessionState,
        *,
        session: SessionIdentity | None = None,
    ) -> AgentSessionOperation:
        if state not in _SESSION_TRANSITIONS[self.state]:
            raise OperationTransitionError(
                f"agent session cannot transition from {self.state} to {state}"
            )
        return replace(self, state=state, session=session or self.session)

    def accepts_observation(self, identity: SessionIdentity) -> bool:
        return self.session is not None and self.session.accepts(
            provider=identity.provider,
            session_id=identity.session_id,
            target=identity.target,
            state_sequence=identity.state_sequence,
        )


def agent_session_fields(operation: AgentSessionOperation) -> dict[str, object]:
    """Flatten a typed session without changing the legacy storage schema."""
    fields: dict[str, object] = {
        "agent_dispatch_state": operation.state.value,
        "agent_operation_target": operation.spec.target,
        "agent_operation_checkout": operation.spec.checkout,
        "agent_operation_workspace_id": operation.spec.workspace_id,
        "agent_operation_pane_id": operation.spec.pane_id,
    }
    if operation.session is not None:
        fields.update(
            agent_provider=operation.session.provider,
            agent_provider_session_id=operation.session.session_id,
            agent_state_sequence=operation.session.state_sequence,
        )
    return fields


def load_agent_session_operation(
    *,
    state: str | None,
    target: str | None,
    checkout: str | None,
    workspace_id: str | None,
    pane_id: str | None,
    provider: str | None,
    session_id: str | None,
    state_sequence: int | None,
) -> AgentSessionOperation | None:
    """Adapt persisted session labels into the typed operation."""
    if state is None:
        return None
    try:
        parsed = AgentSessionState(state)
    except ValueError as exc:
        raise OperationTransitionError(
            f"invalid agent session operation state {state!r}"
        ) from exc
    session = None
    values = (provider, session_id, state_sequence)
    if any(value is not None for value in values):
        if not all(value is not None for value in values):
            raise OperationTransitionError(
                "agent provider session identity is incomplete"
            )
        assert provider is not None
        assert session_id is not None
        assert state_sequence is not None
        session = SessionIdentity(provider, session_id, target or "", state_sequence)
    return AgentSessionOperation(
        parsed,
        AgentSessionSpec(
            target or "", checkout or "", workspace_id or "", pane_id or ""
        ),
        session,
    )


def uncertain_session_state(*, ambiguous: bool) -> AgentSessionState:
    return (
        AgentSessionState.AMBIGUOUS if ambiguous else AgentSessionState.FAILED_OBSERVING
    )


@dataclass(frozen=True, slots=True)
class ParticipantPaneSpec:
    participant: str
    target: str
    label: str
    checkout: str
    workspace_id: str

    def __post_init__(self) -> None:
        _required(
            "participant pane spec",
            participant=self.participant,
            target=self.target,
            label=self.label,
            checkout=self.checkout,
            workspace_id=self.workspace_id,
        )


@dataclass(frozen=True, slots=True)
class ParticipantPaneOperation:
    state: OperationState
    spec: ParticipantPaneSpec
    pane_id: str | None = None

    def __post_init__(self) -> None:
        if self.state == OperationState.SETTLED and not self.pane_id:
            raise OperationTransitionError(
                "settled participant pane requires pane identity"
            )

    def transition(
        self, state: OperationState, *, pane_id: str | None = None
    ) -> ParticipantPaneOperation:
        if state not in _TRANSITIONS[self.state]:
            raise OperationTransitionError(
                f"participant pane cannot transition from {self.state} to {state}"
            )
        return replace(self, state=state, pane_id=pane_id or self.pane_id)
