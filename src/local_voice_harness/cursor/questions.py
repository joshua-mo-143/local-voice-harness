"""Cursor persistence adapter for the provider-neutral question broker."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ..questions import (
    AnswerResolution,
    Question,
    QuestionOrigin,
    QuestionSpec,
    QuestionState,
)
from ..user_config import (
    PlanApprovalMode,
    UserConfigurationError,
    load_plan_approval_preferences,
    resolve_plan_approval_offer,
)
from .model import (
    CursorJob,
    JobStatus,
    WorkflowParticipant,
    WorkflowPhase,
    WorkflowTier,
)

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
    prompt_operation_state: str | None = None,
    job_changes: Mapping[str, object] | None = None,
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
    changes: dict[str, object] = {
        "question": question.text,
        "result": question.text,
        "clarification_kind": owner,
        "voice_question": question.to_dict(),
        "prompt_operation_state": (
            job.prompt_operation_state
            if prompt_operation_state is None
            else prompt_operation_state
        ),
    }
    changes.update(job_changes or {})
    if clear_worker:
        return job.evolve_for_delivery(
            now=now,
            status=JobStatus.AWAITING_USER,
            remove=frozenset({"reconcile"}) if remove_reconcile else frozenset(),
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            **changes,
        )
    return job.evolve_for_delivery(
        now=now,
        status=JobStatus.AWAITING_USER,
        remove=frozenset({"reconcile"}) if remove_reconcile else frozenset(),
        **changes,
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
    cancel: bool = False
    complete: bool = False


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
        "lgtm",
        "ok",
        "okay",
        "ok then",
        "okay then",
        "sure",
        "sounds good",
        "please do",
        "proceed",
        "approve",
        "create it",
        "create the fork",
        "fork it",
    }
)
_REJECTIONS = frozenset(
    {
        "no",
        "no thanks",
        "nah",
        "nope",
        "do not",
        "don't",
        "cancel",
        "stop",
        "reject",
        "not now",
    }
)
_FORK_CONFIRMATIONS = frozenset(
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
_FORK_REJECTIONS = frozenset(
    {
        "no",
        "no thanks",
        "do not",
        "don't",
        "cancel",
        "stop",
    }
)


def _confirmation(
    value: str,
    *,
    confirmations: frozenset[str] = _CONFIRMATIONS,
    rejections: frozenset[str] = _REJECTIONS,
) -> bool | None:
    normalized = re.sub(r"[^\w\s'’]", "", value.casefold())
    normalized = " ".join(normalized.replace("’", "'").split())
    if normalized in confirmations:
        return True
    if normalized in rejections:
        return False
    return None


def _fork_confirmation_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    confirmation = _confirmation(
        context.trusted_text or context.text,
        confirmations=_FORK_CONFIRMATIONS,
        rejections=_FORK_REJECTIONS,
    )
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


_REVIEW_APPROVALS = {"approve", "approved", "proceed", "go ahead"}
_REVIEW_ABORTS = {"abort", "cancel", "stop"}


def _exhausted_review_decision(utterance: str) -> str | None:
    normalized = re.sub(r"[^\w\s'’]", "", utterance.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip().replace("’", "'")
    if normalized in _REVIEW_APPROVALS:
        return "approve"
    if normalized in _REVIEW_ABORTS:
        return "abort"
    return None


def _workflow_queue_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
    *,
    request_text: str | None = None,
    repository_hint: str | None = None,
    github_repository: str | None = None,
    fork_confirmed: bool | None = None,
    herdr_target: str | None = None,
    continuation: bool,
    workflow_phase: str | None = None,
    workflow_tier: str | None = None,
    workflow_reason: str | None = None,
    review_round: int | None = None,
    active_participant: str | None = None,
    review_approved: bool | None = None,
    review_approval_source: str | None = None,
    clear_target: bool = False,
    extra_changes: Mapping[str, object] | None = None,
) -> CursorJob:
    changes: dict[str, Any] = {
        "question": None,
        "clarification_kind": None,
        "delivered": True,
        "delivery_claim_token": None,
        "delivery_claimed_at": None,
        "queued_at": context.now,
        "updated_at": context.now,
        "foreground_until": context.foreground_until,
        "worker_pid": None,
        "worker_boot_id": None,
        "worker_process_start": None,
        "worker_token": None,
        "request": request_text or job.request,
        "repository_hint": (
            job.repository_hint if repository_hint is None else repository_hint
        ),
        "github_repository": (
            job.github_repository if github_repository is None else github_repository
        ),
        "fork_confirmed": (
            job.fork_confirmed if fork_confirmed is None else fork_confirmed
        ),
        "herdr_target": None if clear_target else (herdr_target or job.herdr_target),
        "continuation": continuation,
        "continuation_answer": None,
        "prompt_operation_state": "none",
        "prompt_operation_phase": None,
        "prompt_operation_turn": None,
        "prompt_operation_target": None,
        "prompt_baseline_sequence": None,
        "voice_question": _answered_envelope(question, resolution, context.now),
    }
    if workflow_phase is not None:
        changes["workflow_phase"] = workflow_phase
    if workflow_tier is not None:
        changes["workflow_tier"] = workflow_tier
    if workflow_reason is not None:
        changes["workflow_classification_reason"] = workflow_reason
    if review_round is not None:
        changes["review_round"] = review_round
    if active_participant is not None:
        changes["active_participant"] = active_participant
    if review_approved is not None:
        changes["review_approved"] = review_approved
    if review_approval_source is not None:
        changes["review_approval_source"] = review_approval_source
    changes.update(extra_changes or {})
    return job.evolve(status=JobStatus.QUEUED, **changes)


def _workflow_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    request_text = f"{job.request}\n\nUser clarification: {context.text}"
    return AnswerTransition(
        _workflow_queue_answer(
            job,
            question,
            resolution,
            context,
            continuation=True,
            request_text=request_text,
        ),
        launch=True,
    )


def _workflow_review_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    planner_target = job.participant_target(WorkflowParticipant.PLANNER)
    if not planner_target:
        return AnswerTransition(
            None,
            message=("The planner is unavailable. Please cancel or restart this job."),
        )
    promoted = (
        WorkflowTier.HIGH_RISK
        if job.workflow_tier == WorkflowTier.MEDIUM
        else job.workflow_tier
    )
    reason = job.workflow_classification_reason or ""
    if job.workflow_tier == WorkflowTier.MEDIUM:
        reason = f"{reason} Promoted after review required a user decision.".strip()
    return AnswerTransition(
        _workflow_queue_answer(
            job,
            question,
            resolution,
            context,
            request_text=f"{job.request}\n\nUser clarification: {context.text}",
            herdr_target=planner_target,
            continuation=True,
            workflow_phase=WorkflowPhase.REVISING.value,
            workflow_tier=promoted.value if promoted is not None else None,
            workflow_reason=reason,
            review_round=max(1, job.review_round),
            active_participant=WorkflowParticipant.PLANNER.value,
            extra_changes={
                "plan_approval_state": "none",
                "plan_approval_id": None,
                "plan_approval_source": None,
                "plan_approval_agent_session": None,
                "plan_approval_state_change_sequence": None,
                "plan_approval_revision": None,
                "plan_approval_counted": False,
            },
        ),
        launch=True,
    )


def _workflow_review_exhausted_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    decision = _exhausted_review_decision(context.trusted_text or context.text)
    if decision == "abort":
        return AnswerTransition(None, cancel=True)
    if decision is None:
        return AnswerTransition(
            None,
            message=(
                "Please say approve to implement the reviewed plan, or abort "
                "to cancel the job."
            ),
        )
    planner_target = job.participant_target(WorkflowParticipant.PLANNER)
    if not planner_target or job.plan_approval_state != "boundary":
        return AnswerTransition(
            None,
            message="The reviewed Plan Mode boundary is no longer available.",
        )
    return AnswerTransition(
        _workflow_queue_answer(
            job,
            question,
            resolution,
            context,
            continuation=False,
            herdr_target=planner_target,
            workflow_phase=WorkflowPhase.IMPLEMENTING.value,
            active_participant=WorkflowParticipant.PLANNER.value,
            review_approved=True,
            review_approval_source="user",
            extra_changes={
                "plan_approval_state": "approved",
                "plan_approval_source": "explicit",
                "plan_approval_counted": False,
            },
        ),
        launch=True,
    )


def _workflow_plan_approval_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    confirmation = _confirmation(context.trusted_text or context.text)
    if confirmation is None:
        return AnswerTransition(
            None,
            message=(
                "Please answer yes to approve this reviewed plan, or no to "
                "cancel the job."
            ),
        )
    if not confirmation:
        return AnswerTransition(None, cancel=True)
    planner_target = job.participant_target(WorkflowParticipant.PLANNER)
    if (
        job.plan_approval_state != "awaiting"
        or not job.review_approved
        or not job.plan_artifact
        or not planner_target
    ):
        return AnswerTransition(
            None,
            message="That reviewed Plan Mode boundary is no longer available.",
        )
    return AnswerTransition(
        _workflow_queue_answer(
            job,
            question,
            resolution,
            context,
            continuation=False,
            herdr_target=planner_target,
            workflow_phase=WorkflowPhase.IMPLEMENTING.value,
            active_participant=WorkflowParticipant.PLANNER.value,
            extra_changes={
                "plan_approval_state": "approved",
                "plan_approval_source": "explicit",
                "plan_approval_counted": False,
            },
        ),
        launch=True,
    )


def _workflow_plan_auto_offer_answer(
    job: CursorJob,
    question: Question,
    resolution: AnswerResolution,
    context: AnswerContext,
) -> AnswerTransition:
    confirmation = _confirmation(context.trusted_text or context.text)
    if confirmation is None:
        return AnswerTransition(
            None,
            message=(
                "Please answer yes to enable automatic approval for ordinary "
                "reviewed plans, or no to keep asking."
            ),
        )
    approval_id = job.plan_approval_id
    if not approval_id or job.workflow_phase != WorkflowPhase.FINISHED:
        return AnswerTransition(
            None,
            message="That automatic plan-approval offer is no longer available.",
        )
    try:
        preferences = resolve_plan_approval_offer(
            approval_id,
            approved=confirmation,
        )
    except (OSError, UserConfigurationError):
        try:
            preferences = load_plan_approval_preferences()
        except (OSError, UserConfigurationError):
            return AnswerTransition(
                None,
                message="I could not safely save that plan-approval preference.",
            )
        if not preferences.offer_completed:
            return AnswerTransition(
                None,
                message="I could not safely save that plan-approval preference.",
            )
    enabled = preferences.mode == PlanApprovalMode.AUTO
    return AnswerTransition(
        None,
        message=(
            "Automatic approval is enabled for ordinary reviewed Cursor plans."
            if enabled
            else "Okay, I will keep asking before Cursor implements reviewed plans."
        ),
        complete=True,
    )


_ANSWER_HANDLERS: dict[str, AnswerHandler] = {
    "agent": _agent_answer,
    "repository": _repository_answer,
    "github_repository": _github_repository_answer,
    "fork_confirmation": _fork_confirmation_answer,
    "workflow": _workflow_answer,
    "workflow_review": _workflow_review_answer,
    "workflow_review_exhausted": _workflow_review_exhausted_answer,
    "workflow_plan_approval": _workflow_plan_approval_answer,
    "workflow_plan_auto_offer": _workflow_plan_auto_offer_answer,
}


def register_answer_handler(owner: str, handler: AnswerHandler) -> None:
    if not owner.strip() or owner in _ANSWER_HANDLERS:
        raise ValueError(f"question owner {owner!r} is already registered or invalid")
    _ANSWER_HANDLERS[owner] = handler


def answer_handler(owner: str) -> AnswerHandler | None:
    return _ANSWER_HANDLERS.get(owner)
