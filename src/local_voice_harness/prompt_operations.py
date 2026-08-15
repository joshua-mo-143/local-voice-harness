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
    OBSERVED = "observed"
    AMBIGUOUS = "ambiguous"
    RESOLVED = "resolved"


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
class ObservedPrompt:
    identity: PromptIdentity
    state: PromptOperationState = PromptOperationState.OBSERVED


@dataclass(frozen=True, slots=True)
class AmbiguousPrompt:
    identity: PromptIdentity
    state: PromptOperationState = PromptOperationState.AMBIGUOUS


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    identity: PromptIdentity
    state: PromptOperationState = PromptOperationState.RESOLVED


PromptOperation: TypeAlias = (
    IdlePrompt
    | PlannedPrompt
    | SubmittingPrompt
    | SubmittedPrompt
    | ObservedPrompt
    | AmbiguousPrompt
    | ResolvedPrompt
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
        PromptOperationState.OBSERVED: ObservedPrompt,
        PromptOperationState.AMBIGUOUS: AmbiguousPrompt,
        PromptOperationState.RESOLVED: ResolvedPrompt,
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


def record_prompt_submitted(
    operation: PromptOperation, identity: PromptIdentity
) -> SubmittedPrompt:
    """Cross the submit fence from planned or submitting. Never retries after it."""
    if not isinstance(operation, PlannedPrompt | SubmittingPrompt):
        raise PromptOperationError(
            "only a planned or submitting prompt can record submission"
        )
    _require_identity(operation, identity)
    return SubmittedPrompt(identity)


def observe_prompt_submission(
    operation: PromptOperation,
    identity: PromptIdentity,
    *,
    target: str,
    agent_session: str | None,
    state_sequence: int,
    sequence_evidence_trusted: bool = True,
    input_provenance: str | None = None,
) -> SubmittedPrompt | AmbiguousPrompt:
    """Resolve a submit fence from positive acceptance evidence or fail closed.

    A session sequence advance is shared by every input on that session. It is
    acceptance evidence only when the caller can trust that no overlapping
    manual activity could have caused it. Provider input provenance is an
    optional stronger signal and is never assumed to exist.
    """
    if not isinstance(operation, SubmittingPrompt):
        raise PromptOperationError("only a submitting prompt can be observed")
    _require_identity(operation, identity)
    provenance_accepted = input_provenance == "harness"
    if (
        target == identity.target
        and agent_session == identity.agent_session
        and state_sequence > identity.baseline_sequence
        and (sequence_evidence_trusted or provenance_accepted)
    ):
        return SubmittedPrompt(identity)
    return AmbiguousPrompt(identity)


def observe_accepted_prompt(
    operation: PromptOperation, identity: PromptIdentity
) -> ObservedPrompt:
    if not isinstance(operation, SubmittedPrompt):
        raise PromptOperationError("only a submitted prompt can be observed")
    _require_identity(operation, identity)
    return ObservedPrompt(identity)


def resolve_prompt(
    operation: PromptOperation, identity: PromptIdentity
) -> ResolvedPrompt:
    if not isinstance(operation, PlannedPrompt | SubmittedPrompt | ObservedPrompt):
        raise PromptOperationError(
            "only a planned, submitted, or observed prompt can resolve"
        )
    _require_identity(operation, identity)
    return ResolvedPrompt(identity)


def replan_unobserved_prompt(
    operation: PromptOperation, identity: PromptIdentity
) -> PlannedPrompt:
    """Retry only after confirmed absence; never replay an observed submit."""
    if not isinstance(operation, SubmittedPrompt):
        raise PromptOperationError(
            "only an unobserved submitted prompt can be replanned"
        )
    _require_identity(operation, identity)
    return PlannedPrompt(identity)


def mark_prompt_ambiguous(
    operation: PromptOperation, identity: PromptIdentity
) -> AmbiguousPrompt:
    if not isinstance(
        operation, PlannedPrompt | SubmittingPrompt | SubmittedPrompt | ObservedPrompt
    ):
        raise PromptOperationError("prompt operation cannot become ambiguous")
    _require_identity(operation, identity)
    return AmbiguousPrompt(identity)


_IDENTITY_STATES = (
    PlannedPrompt
    | SubmittingPrompt
    | SubmittedPrompt
    | ObservedPrompt
    | AmbiguousPrompt
    | ResolvedPrompt
)


def _require_identity(operation: _IDENTITY_STATES, identity: PromptIdentity) -> None:
    if operation.identity != identity:
        raise PromptOperationError("stale prompt operation identity")
