"""Cursor persistence adapter for the provider-neutral question broker."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Protocol

from ..questions import (
    AnswerResolution,
    Question,
    QuestionOrigin,
    QuestionSpec,
    QuestionState,
)
from .model import CursorJob, JobStatus

FIELD = "voice_question"


def current(job: CursorJob) -> Question | None:
    raw = job.voice_question
    if raw is not None:
        return Question.from_dict(raw)
    if job.status != JobStatus.AWAITING_USER or not job.question:
        return None
    return Question(
        id=f"legacy-{job.id}-{job.turn}",
        text=job.question,
        kind=QuestionSpec(job.question).kind,
        sensitivity=QuestionSpec(job.question).sensitivity,
        origin=QuestionOrigin(
            provider="cursor",
            job_id=job.id,
            turn_token=job.turn_token or f"{job.id}-routing-{job.turn}",
        ),
        owner=job.clarification_kind or "agent",
        asked_at=job.updated_at or job.created_at,
    )


def ask(
    job: CursorJob,
    spec: QuestionSpec,
    *,
    owner: str,
    turn_token: str,
    now: float,
    clear_worker: bool = False,
    remove_reconcile: bool = False,
) -> CursorJob:
    question = Question(
        id=uuid.uuid4().hex,
        text=spec.text,
        kind=spec.kind,
        sensitivity=spec.sensitivity,
        choices=spec.choices,
        origin=QuestionOrigin(
            provider="cursor",
            job_id=job.id,
            turn_token=turn_token,
        ),
        owner=owner,
        asked_at=now,
    )
    if clear_worker:
        return job.evolve_for_delivery(
            now=now,
            status=JobStatus.AWAITING_USER,
            remove=frozenset({"reconcile"}) if remove_reconcile else frozenset(),
            question=question.text,
            result=question.text,
            clarification_kind=owner,
            voice_question=question.to_dict(),
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
        )
    return job.evolve_for_delivery(
        now=now,
        status=JobStatus.AWAITING_USER,
        remove=frozenset({"reconcile"}) if remove_reconcile else frozenset(),
        question=question.text,
        result=question.text,
        clarification_kind=owner,
        voice_question=question.to_dict(),
    )


def with_state(
    job: CursorJob,
    question: Question,
    state: QuestionState,
    **changes: object,
) -> CursorJob:
    updated = replace(question, state=state, **changes)
    return job.evolve(voice_question=updated.to_dict())


def envelope(
    question: Question, state: QuestionState, **changes: object
) -> dict[str, object]:
    return replace(question, state=state, **changes).to_dict()


@dataclass(frozen=True, slots=True)
class AnswerContext:
    now: float
    foreground_until: float
    text: str
    trusted_text: str | None


@dataclass(frozen=True, slots=True)
class AnswerTransition:
    job: CursorJob | None
    launch: bool = False
    message: str | None = None


class AnswerHandler(Protocol):
    def __call__(
        self,
        job: CursorJob,
        question: Question,
        resolution: AnswerResolution,
        context: AnswerContext,
    ) -> AnswerTransition: ...


def _answered_envelope(
    question: Question,
    resolution: AnswerResolution,
    now: float,
) -> dict[str, object]:
    return envelope(
        question,
        QuestionState.ANSWERED,
        answer=resolution.answer,
        trusted_answer=resolution.trusted_answer,
        answered_at=now,
    )


def _queue_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
    *,
    continuation: bool,
    continuation_answer: str | None = None,
    repository_hint: str | None = None,
    github_repository: str | None = None,
    fork_confirmed: bool | None = None,
    clear_target: bool = False,
) -> CursorJob:
    return job.evolve(
        status=JobStatus.QUEUED,
        question=None,
        clarification_kind=None,
        delivered=True,
        delivery_claim_token=None,
        delivery_claimed_at=None,
        queued_at=context.now,
        updated_at=context.now,
        foreground_until=context.foreground_until,
        worker_pid=None,
        worker_boot_id=None,
        worker_process_start=None,
        worker_token=None,
        repository_hint=(
            job.repository_hint if repository_hint is None else repository_hint
        ),
        github_repository=(
            job.github_repository if github_repository is None else github_repository
        ),
        fork_confirmed=(
            job.fork_confirmed if fork_confirmed is None else fork_confirmed
        ),
        herdr_target=None if clear_target else job.herdr_target,
        continuation=continuation,
        continuation_answer=continuation_answer,
        voice_question=_answered_envelope(question, resolution, context.now),
    )


def _agent_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    if resolution.choice_label is not None:
        answer = (
            f"Question: {question.text}\nUser selected: "
            f"{resolution.choice_label} (choice id: {resolution.answer})."
        )
    else:
        answer = f"Question: {question.text}\nUser answered: {resolution.answer}"
    return AnswerTransition(
        _queue_answer(
            job,
            question,
            resolution,
            context,
            continuation=True,
            continuation_answer=answer,
        ),
        launch=True,
    )


def _repository_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    return AnswerTransition(
        _queue_answer(
            job,
            question,
            resolution,
            context,
            continuation=False,
            repository_hint=context.text,
            clear_target=True,
        ),
        launch=True,
    )


def _github_repository_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    return AnswerTransition(
        _queue_answer(
            job,
            question,
            resolution,
            context,
            continuation=False,
            github_repository=context.text.strip(),
            clear_target=True,
        ),
        launch=True,
    )


_CONFIRMATIONS = frozenset(
    {
        "yes",
        "yes please",
        "confirm",
        "confirmed",
        "do it",
        "go ahead",
        "create it",
        "create the fork",
        "fork it",
    }
)
_REJECTIONS = frozenset({"no", "no thanks", "do not", "don't", "cancel", "stop"})


def _confirmation(value: str) -> bool | None:
    normalized = " ".join(value.casefold().replace("’", "'").split())
    if normalized in _CONFIRMATIONS:
        return True
    if normalized in _REJECTIONS:
        return False
    return None


def _fork_confirmation_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    confirmation = _confirmation(context.trusted_text or context.text)
    if confirmation is None:
        return AnswerTransition(
            None,
            message="Please answer yes or no. Should I create the GitHub fork?",
        )
    if confirmation:
        return AnswerTransition(
            _queue_answer(
                job,
                question,
                resolution,
                context,
                continuation=False,
                fork_confirmed=True,
                clear_target=True,
            ),
            launch=True,
        )
    completed = job.evolve_for_delivery(
        now=context.now,
        status=JobStatus.COMPLETED,
        question=None,
        clarification_kind=None,
        result="Okay, I did not create a GitHub fork.",
        completed_at=context.now,
        worker_pid=None,
        worker_boot_id=None,
        worker_process_start=None,
        worker_token=None,
        voice_question=envelope(
            question,
            QuestionState.RESOLVED,
            answer="no",
            trusted_answer=context.trusted_text or context.text,
            answered_at=context.now,
        ),
    )
    return AnswerTransition(completed)


_ANSWER_HANDLERS: dict[str, AnswerHandler] = {
    "agent": _agent_answer,
    "repository": _repository_answer,
    "github_repository": _github_repository_answer,
    "fork_confirmation": _fork_confirmation_answer,
}


def register_answer_handler(owner: str, handler: AnswerHandler) -> None:
    if not owner.strip() or owner in _ANSWER_HANDLERS:
        raise ValueError(f"question owner {owner!r} is already registered or invalid")
    _ANSWER_HANDLERS[owner] = handler


def answer_handler(owner: str) -> AnswerHandler | None:
    return _ANSWER_HANDLERS.get(owner)
