"""Provider-neutral contracts and deterministic voice-question resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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


class PromptOperationState(StrEnum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    OBSERVED = "observed"
    RESOLVED = "resolved"


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
    ordinal = _ORDINALS.get(normalized)
    if ordinal is not None and ordinal < len(question.choices):
        choice = question.choices[ordinal]
        if choice not in matches:
            matches.append(choice)
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
    labels = ", ".join(choice.label for choice in question.choices)
    return f"Please choose exactly one of: {labels}."


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", "", value.casefold())).strip()


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
    if question.prompt_absent_observations < 0:
        raise QuestionError("prompt absent observations must not be negative")
    if question.state == QuestionState.DISPATCHING and (
        not question.dispatch_token or question.prompt_state is None
    ):
        raise QuestionError(
            "dispatching question requires a token and prompt operation state"
        )
