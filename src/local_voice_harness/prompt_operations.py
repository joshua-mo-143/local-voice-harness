"""Provider-neutral durable prompt-operation state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, TypedDict


class PromptOperationError(ValueError):
    """A prompt operation or transition violates its durable identity fence."""


class PromptOperationState(StrEnum):
    IDLE = "none"
    PLANNED = "planned"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class PromptIdentity:
    job_id: str
    phase: str
    turn: int
    turn_token: str
    target: str
    agent_session: str
    baseline_sequence: int

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.job_id,
                self.phase,
                self.turn_token,
                self.target,
                self.agent_session,
            )
        ):
            raise PromptOperationError("prompt identity fields must not be empty")
        if self.turn <= 0:
            raise PromptOperationError("prompt turn must be positive")
        if self.baseline_sequence < 0:
            raise PromptOperationError("prompt baseline sequence must not be negative")


@dataclass(frozen=True, slots=True)
class IdlePrompt:
    state: PromptOperationState = PromptOperationState.IDLE


@dataclass(frozen=True, slots=True)
class PlannedPrompt:
    identity: PromptIdentity
    state: PromptOperationState = PromptOperationState.PLANNED


@dataclass(frozen=True, slots=True)
class SubmittingPrompt:
    identity: PromptIdentity
    state: PromptOperationState = PromptOperationState.SUBMITTING


@dataclass(frozen=True, slots=True)
class SubmittedPrompt:
    identity: PromptIdentity
    state: PromptOperationState = PromptOperationState.SUBMITTED


@dataclass(frozen=True, slots=True)
class AmbiguousPrompt:
    identity: PromptIdentity
    state: PromptOperationState = PromptOperationState.AMBIGUOUS


PromptOperation: TypeAlias = (
    IdlePrompt | PlannedPrompt | SubmittingPrompt | SubmittedPrompt | AmbiguousPrompt
)


class LegacyPromptFields(TypedDict):
    prompt_operation_state: str
    prompt_operation_phase: str | None
    prompt_operation_turn: int | None
    prompt_operation_target: str | None
    prompt_operation_agent_session: str | None
    prompt_baseline_sequence: int | None


def load_prompt_operation(
    *,
    state: str,
    job_id: str,
    phase: str | None,
    turn: int,
    turn_token: str | None,
    target: str | None,
    agent_session: str | None,
    baseline_sequence: int | None,
) -> PromptOperation:
    """Adapt legacy flat fields into the typed operation at the persistence edge."""
    try:
        parsed_state = PromptOperationState(state)
    except ValueError as exc:
        raise PromptOperationError(f"invalid prompt operation state {state!r}") from exc
    if parsed_state == PromptOperationState.IDLE:
        return IdlePrompt()
    identity = PromptIdentity(
        job_id=job_id,
        phase=phase or "",
        turn=turn,
        turn_token=turn_token or "",
        target=target or "",
        agent_session=agent_session or "",
        baseline_sequence=(baseline_sequence if baseline_sequence is not None else -1),
    )
    constructors = {
        PromptOperationState.PLANNED: PlannedPrompt,
        PromptOperationState.SUBMITTING: SubmittingPrompt,
        PromptOperationState.SUBMITTED: SubmittedPrompt,
        PromptOperationState.AMBIGUOUS: AmbiguousPrompt,
    }
    return constructors[parsed_state](identity)


def legacy_prompt_fields(operation: PromptOperation) -> LegacyPromptFields:
    """Flatten a typed operation without changing the legacy storage schema."""
    if isinstance(operation, IdlePrompt):
        return {
            "prompt_operation_state": "none",
            "prompt_operation_phase": None,
            "prompt_operation_turn": None,
            "prompt_operation_target": None,
            "prompt_operation_agent_session": None,
            "prompt_baseline_sequence": None,
        }
    identity = operation.identity
    return {
        "prompt_operation_state": operation.state.value,
        "prompt_operation_phase": identity.phase,
        "prompt_operation_turn": identity.turn,
        "prompt_operation_target": identity.target,
        "prompt_operation_agent_session": identity.agent_session,
        "prompt_baseline_sequence": identity.baseline_sequence,
    }


def plan_prompt(operation: PromptOperation, identity: PromptIdentity) -> PlannedPrompt:
    if not isinstance(operation, IdlePrompt):
        raise PromptOperationError("only an idle prompt operation can be planned")
    return PlannedPrompt(identity)


def begin_prompt_submission(
    operation: PromptOperation, identity: PromptIdentity
) -> SubmittingPrompt:
    if not isinstance(operation, PlannedPrompt):
        raise PromptOperationError("only a planned prompt can begin submission")
    _require_identity(operation, identity)
    return SubmittingPrompt(identity)


def accept_prompt_submission(
    operation: PromptOperation, identity: PromptIdentity
) -> SubmittedPrompt:
    if not isinstance(operation, SubmittingPrompt):
        raise PromptOperationError("only a submitting prompt can be accepted")
    _require_identity(operation, identity)
    return SubmittedPrompt(identity)


def observe_prompt_submission(
    operation: PromptOperation,
    identity: PromptIdentity,
    *,
    target: str,
    agent_session: str | None,
    state_sequence: int,
) -> SubmittedPrompt | AmbiguousPrompt:
    """Resolve a submit fence from positive acceptance evidence or fail closed."""
    if not isinstance(operation, SubmittingPrompt):
        raise PromptOperationError("only a submitting prompt can be observed")
    _require_identity(operation, identity)
    if (
        target == identity.target
        and agent_session == identity.agent_session
        and state_sequence != identity.baseline_sequence
    ):
        return SubmittedPrompt(identity)
    return AmbiguousPrompt(identity)


def mark_prompt_ambiguous(
    operation: PromptOperation, identity: PromptIdentity
) -> AmbiguousPrompt:
    if not isinstance(operation, PlannedPrompt | SubmittingPrompt | SubmittedPrompt):
        raise PromptOperationError("prompt operation cannot become ambiguous")
    _require_identity(operation, identity)
    return AmbiguousPrompt(identity)


def _require_identity(
    operation: PlannedPrompt | SubmittingPrompt | SubmittedPrompt,
    identity: PromptIdentity,
) -> None:
    if operation.identity != identity:
        raise PromptOperationError("stale prompt operation identity")
