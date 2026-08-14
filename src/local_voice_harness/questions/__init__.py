"""Provider-neutral contracts and deterministic voice-question resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ..prompt_operations import (
    IdlePrompt,
    PromptIdentity,
    PromptOperation,
    PromptOperationState,
    load_prompt_operation,
    observe_accepted_prompt,
    plan_prompt,
    record_prompt_submitted,
    replan_unobserved_prompt,
    resolve_prompt,
)

QUESTION_VERSION = 1


class QuestionError(ValueError):
    """A question payload does not satisfy the broker contract."""


class QuestionKind(StrEnum):
    FREE_TEXT = "free_text"
    MULTIPLE_CHOICE = "multiple_choice"


class QuestionSensitivity(StrEnum):
    UNSPECIFIED = "unspecified"
    ROUTINE = "routine"
    SECURITY = "security"
    DESTRUCTIVE = "destructive"
    ARCHITECTURE = "architecture"
    PRODUCT = "product"


class QuestionState(StrEnum):
    PENDING = "pending"
    DEFERRED = "deferred"
    ANSWERED = "answered"
    DISPATCHING = "dispatching"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class QuestionIdentity:
    job_id: str
    question_id: str
    turn_token: str


class AnswerOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    REPEAT = "repeat"
    DEFERRED = "deferred"


class AnswerProvenance(StrEnum):
    USER_VOICE = "user_voice"
    USER_TEXT = "user_text"
    AUTOMATION = "automation"


@dataclass(frozen=True, slots=True)
class Choice:
    id: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True, slots=True)
class QuestionOrigin:
    provider: str
    job_id: str
    turn_token: str

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "job_id": self.job_id,
            "turn_token": self.turn_token,
        }


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    text: str
    kind: QuestionKind
    sensitivity: QuestionSensitivity
    origin: QuestionOrigin
    choices: tuple[Choice, ...] = ()
    owner: str = "agent"
    state: QuestionState = QuestionState.PENDING
    asked_at: float = 0
    answer: str | None = None
    trusted_answer: str | None = None
    answered_at: float | None = None
    dispatch_token: str | None = None
    prompt_state: PromptOperationState | None = None
    prompt_turn: int | None = None
    prompt_target: str | None = None
    prompt_agent_session: str | None = None
    prompt_baseline_seq: int | None = None
    prompt_submitted_at: float | None = None
    prompt_absent_observations: int = 0

    def __post_init__(self) -> None:
        _validate_question(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": QUESTION_VERSION,
            "id": self.id,
            "text": self.text,
            "kind": self.kind.value,
            "sensitivity": self.sensitivity.value,
            "origin": self.origin.to_dict(),
            "choices": [choice.to_dict() for choice in self.choices],
            "owner": self.owner,
            "state": self.state.value,
            "asked_at": self.asked_at,
            "answer": self.answer,
            "trusted_answer": self.trusted_answer,
            "answered_at": self.answered_at,
            "dispatch_token": self.dispatch_token,
            "prompt_state": self.prompt_state.value if self.prompt_state else None,
            "prompt_turn": self.prompt_turn,
            "prompt_target": self.prompt_target,
            "prompt_agent_session": self.prompt_agent_session,
            "prompt_baseline_seq": self.prompt_baseline_seq,
            "prompt_submitted_at": self.prompt_submitted_at,
            "prompt_absent_observations": self.prompt_absent_observations,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Question:
        if not isinstance(raw, dict) or raw.get("version") != QUESTION_VERSION:
            raise QuestionError("unsupported question envelope")
        origin_raw = raw.get("origin")
        if not isinstance(origin_raw, dict):
            raise QuestionError("question requires an origin")
        try:
            origin = QuestionOrigin(
                provider=_required_string(origin_raw, "provider"),
                job_id=_required_string(origin_raw, "job_id"),
                turn_token=_required_string(origin_raw, "turn_token"),
            )
            choices_raw = raw.get("choices", [])
            if not isinstance(choices_raw, list):
                raise QuestionError("question choices must be a list")
            choices = tuple(
                Choice(
                    id=_required_string(choice, "id"),
                    label=_required_string(choice, "label"),
                )
                for choice in choices_raw
                if isinstance(choice, dict)
            )
            if len(choices) != len(choices_raw):
                raise QuestionError("question choice must be an object")
            question = cls(
                id=_required_string(raw, "id"),
                text=_required_string(raw, "text"),
                kind=QuestionKind(str(raw.get("kind") or "")),
                sensitivity=QuestionSensitivity(
                    str(raw.get("sensitivity") or QuestionSensitivity.UNSPECIFIED)
                ),
                origin=origin,
                choices=choices,
                owner=_required_string(raw, "owner"),
                state=QuestionState(str(raw.get("state") or "")),
                asked_at=_number(raw.get("asked_at"), "asked_at"),
                answer=_optional_string(raw.get("answer"), "answer"),
                trusted_answer=_optional_string(
                    raw.get("trusted_answer"), "trusted_answer"
                ),
                answered_at=_optional_number(raw.get("answered_at"), "answered_at"),
                dispatch_token=_optional_string(
                    raw.get("dispatch_token"), "dispatch_token"
                ),
                prompt_state=(
                    PromptOperationState(str(raw["prompt_state"]))
                    if raw.get("prompt_state") is not None
                    else None
                ),
                prompt_turn=_optional_int(raw.get("prompt_turn"), "prompt_turn"),
                prompt_target=_optional_string(
                    raw.get("prompt_target"), "prompt_target"
                ),
                prompt_agent_session=_optional_string(
                    raw.get("prompt_agent_session"), "prompt_agent_session"
                ),
                prompt_baseline_seq=_optional_int(
                    raw.get("prompt_baseline_seq"), "prompt_baseline_seq"
                ),
                prompt_submitted_at=_optional_number(
                    raw.get("prompt_submitted_at"), "prompt_submitted_at"
                ),
                prompt_absent_observations=_integer(
                    raw.get("prompt_absent_observations", 0),
                    "prompt_absent_observations",
                ),
            )
        except (TypeError, ValueError) as exc:
            raise QuestionError(str(exc)) from exc
        return question


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    text: str
    kind: QuestionKind = QuestionKind.FREE_TEXT
    choices: tuple[Choice, ...] = ()
    sensitivity: QuestionSensitivity = QuestionSensitivity.UNSPECIFIED

    def __post_init__(self) -> None:
        _validate_spec(self)


@dataclass(frozen=True, slots=True)
class AnswerResolution:
    outcome: AnswerOutcome
    answer: str | None = None
    trusted_answer: str | None = None
    choice_label: str | None = None


_LEGAL_QUESTION_TRANSITIONS = {
    QuestionState.PENDING: frozenset(
        {
            QuestionState.DEFERRED,
            QuestionState.ANSWERED,
            QuestionState.RESOLVED,
            QuestionState.CANCELLED,
        }
    ),
    QuestionState.DEFERRED: frozenset(
        {
            QuestionState.DEFERRED,
            QuestionState.ANSWERED,
            QuestionState.RESOLVED,
            QuestionState.CANCELLED,
        }
    ),
    QuestionState.ANSWERED: frozenset(
        {
            QuestionState.DISPATCHING,
            QuestionState.RESOLVED,
            QuestionState.CANCELLED,
        }
    ),
    QuestionState.DISPATCHING: frozenset(
        {
            QuestionState.DISPATCHING,
            QuestionState.RESOLVED,
            QuestionState.CANCELLED,
        }
    ),
    QuestionState.RESOLVED: frozenset(),
    QuestionState.CANCELLED: frozenset(),
}
_IMMUTABLE_QUESTION_FIELDS = frozenset(
    {"id", "text", "kind", "sensitivity", "origin", "choices", "owner", "asked_at"}
)


def question_identity(question: Question) -> QuestionIdentity:
    return QuestionIdentity(
        job_id=question.origin.job_id,
        question_id=question.id,
        turn_token=question.origin.turn_token,
    )


def validate_question_identity(question: Question, identity: QuestionIdentity) -> None:
    if identity != question_identity(question):
        raise QuestionError("stale question identity")


def transition_question(
    question: Question,
    state: QuestionState,
    identity: QuestionIdentity,
    **changes: object,
) -> Question:
    """Apply a legal question transition after checking its originating fence."""
    validate_question_identity(question, identity)
    immutable_changes = _IMMUTABLE_QUESTION_FIELDS.intersection(changes)
    if immutable_changes:
        fields = ", ".join(sorted(immutable_changes))
        raise QuestionError(
            f"question transition cannot change identity fields: {fields}"
        )
    if state not in _LEGAL_QUESTION_TRANSITIONS[question.state]:
        raise QuestionError(
            f"illegal question transition {question.state.value} -> {state.value}"
        )
    if state == QuestionState.DISPATCHING and (
        not question.answer or question.answered_at is None
    ):
        raise QuestionError("dispatching question requires an answer and answered_at")
    return replace(question, state=state, **changes)


_REPEAT = frozenset({"repeat", "repeat that", "say that again", "what was that"})
_DEFER = frozenset(
    {
        "answer later",
        "i'll answer later",
        "ill answer later",
        "ask me later",
        "later",
    }
)
_ORDINALS = {
    "first": 0,
    "one": 0,
    "1": 0,
    "second": 1,
    "two": 1,
    "2": 1,
    "third": 2,
    "three": 2,
    "3": 2,
    "fourth": 3,
    "four": 3,
    "4": 3,
    "fifth": 4,
    "five": 4,
    "5": 4,
}
_ORDINAL_PATTERN = "|".join(
    sorted((re.escape(value) for value in _ORDINALS), key=len, reverse=True)
)
_CHOICE_REFERENCE_PATTERNS = (
    re.compile(rf"\b(?:option|choice)(?: number)? (?P<ordinal>{_ORDINAL_PATTERN})\b"),
    re.compile(rf"\b(?P<ordinal>{_ORDINAL_PATTERN}) (?:option|choice|one)\b"),
)


def parse_question_spec(value: str) -> QuestionSpec:
    """Parse a structured marker payload, accepting legacy plain text."""
    text = value.strip()
    if not text:
        raise QuestionError("question text must not be empty")
    if not text.startswith("{"):
        return QuestionSpec(text=text, sensitivity=QuestionSensitivity.UNSPECIFIED)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QuestionError("structured question is invalid JSON") from exc
    if not isinstance(raw, dict) or raw.get("version") != QUESTION_VERSION:
        raise QuestionError("structured question requires version 1")
    question_text = _required_string(raw, "text")
    try:
        kind = QuestionKind(str(raw.get("kind") or QuestionKind.FREE_TEXT))
        if "sensitivity" not in raw:
            raise QuestionError("structured question requires sensitivity")
        sensitivity = QuestionSensitivity(str(raw["sensitivity"]))
    except ValueError as exc:
        raise QuestionError(str(exc)) from exc
    choices_raw = raw.get("choices", [])
    if not isinstance(choices_raw, list):
        raise QuestionError("question choices must be a list")
    choices: list[Choice] = []
    for index, choice in enumerate(choices_raw):
        if isinstance(choice, str):
            choices.append(Choice(str(index + 1), choice.strip()))
        elif isinstance(choice, dict):
            choices.append(
                Choice(
                    _required_string(choice, "id"),
                    _required_string(choice, "label"),
                )
            )
        else:
            raise QuestionError("question choice must be text or an object")
    return QuestionSpec(question_text, kind, tuple(choices), sensitivity)


def resolve_answer(
    question: Question,
    answer: str,
    *,
    trusted_answer: str | None = None,
    provenance: AnswerProvenance = AnswerProvenance.AUTOMATION,
) -> AnswerResolution:
    """Resolve controls and choices without semantic guessing."""
    control = question_control(trusted_answer or answer)
    if control is not None:
        return AnswerResolution(control)
    if question.sensitivity != QuestionSensitivity.ROUTINE and provenance not in {
        AnswerProvenance.USER_VOICE,
        AnswerProvenance.USER_TEXT,
    }:
        return AnswerResolution(AnswerOutcome.REJECTED)
    normalized = _normalize(trusted_answer or answer)
    if question.kind == QuestionKind.FREE_TEXT:
        value = (trusted_answer or answer).strip()
        if not value:
            return AnswerResolution(AnswerOutcome.AMBIGUOUS)
        return AnswerResolution(
            AnswerOutcome.ACCEPTED,
            value,
            value,
        )
    matches = [
        choice
        for choice in question.choices
        if normalized in {_normalize(choice.id), _normalize(choice.label)}
    ]
    ordinals = _choice_ordinals(normalized)
    if len(ordinals) == 1:
        ordinal = next(iter(ordinals))
        if ordinal >= len(question.choices):
            return AnswerResolution(AnswerOutcome.AMBIGUOUS)
        choice = question.choices[ordinal]
        if choice not in matches:
            matches.append(choice)
    elif ordinals:
        return AnswerResolution(AnswerOutcome.AMBIGUOUS)
    if len(matches) != 1:
        return AnswerResolution(AnswerOutcome.AMBIGUOUS)
    return AnswerResolution(
        AnswerOutcome.ACCEPTED,
        matches[0].id,
        (trusted_answer or answer).strip(),
        matches[0].label,
    )


def question_control(value: str) -> AnswerOutcome | None:
    normalized = _normalize(value)
    if normalized in _REPEAT:
        return AnswerOutcome.REPEAT
    if normalized in _DEFER:
        return AnswerOutcome.DEFERRED
    return None


def choices_prompt(question: Question) -> str:
    options = " ".join(
        f"Option {index} is {choice.label}."
        for index, choice in enumerate(question.choices, start=1)
    )
    return f"{options} Please choose one."


def question_prompt(question: Question) -> str:
    if not question.choices:
        return question.text
    return f"{question.text} {choices_prompt(question)}"


def _choice_ordinals(value: str) -> set[int]:
    direct = _ORDINALS.get(value)
    if direct is not None:
        return {direct}
    return {
        _ORDINALS[match.group("ordinal")]
        for pattern in _CHOICE_REFERENCE_PATTERNS
        for match in pattern.finditer(value)
    }


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^\w\s'-]", "", value.casefold())
    normalized = re.sub(r"\backnowledgement\b", "acknowledgment", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _required_string(values: dict[str, Any], field: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise QuestionError(f"question {field} must be non-empty text")
    return value.strip()


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise QuestionError(f"question {field} must be text")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, int | float):
        raise QuestionError(f"question {field} must be a number")
    return float(value)


def _optional_number(value: object, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise QuestionError(f"question {field} must be an integer")
    return value


def _optional_int(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field)


def _validate_spec(spec: QuestionSpec) -> None:
    if not spec.text.strip() or len(spec.text) > 500:
        raise QuestionError("question text must contain 1 to 500 characters")
    if spec.kind == QuestionKind.MULTIPLE_CHOICE and len(spec.choices) < 2:
        raise QuestionError("multiple-choice question requires at least two choices")
    if len(spec.choices) > 10:
        raise QuestionError("question must not define more than ten choices")
    if spec.kind == QuestionKind.FREE_TEXT and spec.choices:
        raise QuestionError("free-text question must not define choices")
    ids: list[str] = []
    for choice in spec.choices:
        choice_id = choice.id.strip()
        label = choice.label.strip()
        if not choice_id or len(choice_id) > 80:
            raise QuestionError("question choice id must contain 1 to 80 characters")
        if not label or len(label) > 200:
            raise QuestionError(
                "question choice label must contain 1 to 200 characters"
            )
        ids.append(choice_id.casefold())
    if len(ids) != len(set(ids)):
        raise QuestionError("question choice ids must be unique")


def _validate_question(question: Question) -> None:
    _validate_spec(
        QuestionSpec(
            question.text,
            question.kind,
            question.choices,
            question.sensitivity,
        )
    )
    if not all(
        value.strip()
        for value in (
            question.id,
            question.owner,
            question.origin.provider,
            question.origin.job_id,
            question.origin.turn_token,
        )
    ):
        raise QuestionError("question identity fields must not be empty")
    if question.prompt_absent_observations < 0:
        raise QuestionError("prompt absent observations must not be negative")
    prompt_identity = (
        question.prompt_turn,
        question.prompt_target,
        question.prompt_agent_session,
    )
    if any(value is not None for value in prompt_identity) and not all(
        value is not None for value in prompt_identity
    ):
        raise QuestionError("question prompt identity is incomplete")
    if question.state == QuestionState.ANSWERED and (
        not question.answer or question.answered_at is None
    ):
        raise QuestionError(
            f"{question.state.value} question requires an answer and answered_at"
        )
    if question.state == QuestionState.DISPATCHING and (
        not question.dispatch_token or question.prompt_state is None
    ):
        raise QuestionError(
            "dispatching question requires a token and prompt operation state"
        )
    if question.state == QuestionState.DISPATCHING and question.prompt_state not in {
        PromptOperationState.PLANNED,
        PromptOperationState.SUBMITTED,
        PromptOperationState.OBSERVED,
        PromptOperationState.RESOLVED,
    }:
        raise QuestionError(
            "dispatching question has an invalid prompt operation state"
        )


def question_prompt_identity(
    question: Question,
    *,
    job_id: str,
    turn: int,
    target: str,
    agent_session: str,
) -> PromptIdentity:
    """Build the shared submission identity from question and job fences."""
    return PromptIdentity(
        job_id=job_id,
        phase="question",
        turn=turn,
        turn_token=question.dispatch_token or question.origin.turn_token,
        target=target,
        agent_session=agent_session,
        baseline_sequence=(
            question.prompt_baseline_seq
            if question.prompt_baseline_seq is not None
            else 0
        ),
    )


def load_question_prompt(
    question: Question, identity: PromptIdentity
) -> PromptOperation:
    """Adapt question JSON prompt fields at the persistence edge."""
    if question.prompt_state is None:
        return IdlePrompt()
    return load_prompt_operation(
        state=question.prompt_state.value,
        job_id=identity.job_id,
        phase=identity.phase,
        turn=identity.turn,
        turn_token=identity.turn_token,
        target=identity.target,
        agent_session=identity.agent_session,
        baseline_sequence=(
            question.prompt_baseline_seq
            if question.prompt_baseline_seq is not None
            else identity.baseline_sequence
        ),
    )


def bind_question_prompt(question: Question, operation: PromptOperation) -> Question:
    """Write a shared prompt operation back onto question compatibility fields."""
    if isinstance(operation, IdlePrompt):
        return replace(
            question,
            prompt_state=None,
            prompt_turn=None,
            prompt_target=None,
            prompt_agent_session=None,
        )
    return replace(
        question,
        prompt_state=operation.state,
        prompt_turn=operation.identity.turn,
        prompt_target=operation.identity.target,
        prompt_agent_session=operation.identity.agent_session,
    )


def plan_question_prompt(question: Question, identity: PromptIdentity) -> Question:
    return bind_question_prompt(
        question, plan_prompt(load_question_prompt(question, identity), identity)
    )


def submit_question_prompt(question: Question, identity: PromptIdentity) -> Question:
    return bind_question_prompt(
        question,
        record_prompt_submitted(load_question_prompt(question, identity), identity),
    )


def observe_question_prompt(question: Question, identity: PromptIdentity) -> Question:
    return bind_question_prompt(
        question,
        observe_accepted_prompt(load_question_prompt(question, identity), identity),
    )


def resolve_question_prompt(question: Question, identity: PromptIdentity) -> Question:
    return bind_question_prompt(
        question, resolve_prompt(load_question_prompt(question, identity), identity)
    )


def replan_question_prompt(question: Question, identity: PromptIdentity) -> Question:
    return bind_question_prompt(
        question,
        replan_unobserved_prompt(load_question_prompt(question, identity), identity),
    )
