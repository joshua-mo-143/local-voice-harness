"""Deterministic readback policy for identity-sensitive ticket actions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import StrEnum

from .browser_context import RequestContext
from .questions import (
    AnswerOutcome,
    AnswerProvenance,
    Choice,
    Question,
    QuestionKind,
    QuestionOrigin,
    QuestionSensitivity,
    resolve_answer,
)
from .responses import AssistantResponse, ResponseLike, as_assistant_response
from .ticket_targets import TicketExtraction, extract_ticket_targets

READBACK_TIMEOUT_SECONDS = 30.0

_GITHUB_TARGET = re.compile(
    r"(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#(?P<number>[1-9]\d*)"
)
_LINEAR_TARGET = re.compile(r"(?P<team>[A-Z][A-Z0-9]+)-(?P<number>[1-9]\d*)")
_AFFIRMATIVE = frozenset(
    {"yes", "yes please", "confirm", "confirmed", "go ahead", "do it"}
)
_NEGATIVE = frozenset({"no", "no thanks", "cancel", "stop", "do not", "don't"})


class ReadbackReply(StrEnum):
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    EXPIRED = "expired"
    COMPETING = "competing"


@dataclass(frozen=True, slots=True)
class CriticalTarget:
    provider: str
    repository: str
    ticket: str
    action: str

    @property
    def canonical(self) -> str:
        if self.provider == "github":
            return f"{self.repository}#{self.ticket}"
        return f"{self.repository}-{self.ticket}"


@dataclass(frozen=True, slots=True)
class TargetSelection:
    target: CriticalTarget
    context_binding: tuple[str | None, ...]
    readback_required: bool = True


@dataclass(frozen=True, slots=True)
class ReadbackCandidate:
    target: CriticalTarget
    question: Question
    context_binding: tuple[str | None, ...]
    created_at: float

    @property
    def origin_turn(self) -> str:
        return self.question.origin.turn_token


@dataclass(frozen=True, slots=True)
class ReadbackResolution:
    reply: ReadbackReply
    replacement: CriticalTarget | None = None


def parse_target(canonical: str, *, action: str = "start_work") -> CriticalTarget:
    github = _GITHUB_TARGET.fullmatch(canonical)
    if github is not None:
        return CriticalTarget(
            "github",
            github.group("repository"),
            github.group("number"),
            action,
        )
    linear = _LINEAR_TARGET.fullmatch(canonical)
    if linear is not None:
        return CriticalTarget(
            "linear",
            linear.group("team"),
            linear.group("number"),
            action,
        )
    raise ValueError(f"unsupported canonical ticket target: {canonical}")


def context_binding(context: RequestContext) -> tuple[str | None, ...]:
    """Return the trusted identity fields that must remain stable."""

    return (
        context.focused_repository,
        context.focused_issue,
        context.focused_issue_page,
        context.issue_scope_source,
        context.issue_scope,
    )


def select_submit_target(
    extraction: TicketExtraction,
    context: RequestContext,
) -> TargetSelection | None:
    """Select one identity-sensitive ticket and its readback policy."""

    if extraction.batch_requested:
        return None
    resolved = [
        reference.canonical
        for reference in extraction.references
        if reference.canonical is not None
    ]
    if len(resolved) == 1:
        target = parse_target(resolved[0])
        # Explicit text is the authority. Bind ambient context only when it
        # contributed scope or presents a conflicting identity.
        reference = next(
            reference
            for reference in extraction.references
            if reference.canonical is not None
        )
        conflict = bool(
            context.focused_issue
            and context.focused_issue.casefold() != target.canonical.casefold()
        )
        binding = (
            context_binding(context) if reference.scoped or conflict else (None,) * 5
        )
        return TargetSelection(target, binding)
    if extraction.requested_count:
        return None
    if context.focused_issue:
        try:
            target = parse_target(context.focused_issue)
        except ValueError:
            return None
        exact_focused_page = (
            context.focused_issue_page is not None
            and context.focused_issue_page.casefold() == target.canonical.casefold()
        )
        return TargetSelection(
            target,
            context_binding(context),
            readback_required=not exact_focused_page,
        )
    return None


def new_candidate(
    selection: TargetSelection,
    *,
    origin_turn: str,
    now: float | None = None,
) -> ReadbackCandidate:
    target = selection.target
    question = Question(
        id=f"critical-target:{origin_turn}",
        text=f"Start work on {target.canonical}?",
        kind=QuestionKind.MULTIPLE_CHOICE,
        sensitivity=QuestionSensitivity.PRODUCT,
        origin=QuestionOrigin(
            provider=target.provider,
            job_id=target.canonical,
            turn_token=origin_turn,
        ),
        choices=(Choice("yes", "Start work"), Choice("no", "Cancel")),
        owner="critical_target_readback",
        asked_at=time.monotonic() if now is None else now,
    )
    return ReadbackCandidate(
        target,
        question,
        selection.context_binding,
        question.asked_at,
    )


def readback_response(candidate: ReadbackCandidate) -> AssistantResponse:
    target = candidate.target
    if target.provider == "github":
        short_repository = target.repository.rsplit("/", 1)[-1]
        spoken = f"Work on issue {target.ticket} in {short_repository}?"
    else:
        spoken = f"Work on {target.repository} issue {target.ticket}?"
    return AssistantResponse(
        spoken_text=spoken,
        display_text=(
            f"Confirm start work on {target.canonical}. "
            "Reply yes, no, or correct the target."
        ),
    )


def identified_target_response(
    target: CriticalTarget,
    response: ResponseLike,
) -> AssistantResponse:
    """Name the canonical fast-path target without changing the job outcome."""

    rendered = as_assistant_response(response)
    spoken_target = f"{target.repository} issue {target.ticket}"
    return AssistantResponse(
        spoken_text=f"For {spoken_target}: {rendered.spoken_text}",
        display_text=f"Target {target.canonical}: {rendered.display_text}",
    )


def resolve_readback(
    candidate: ReadbackCandidate,
    reply: str,
    context: RequestContext,
    *,
    now: float | None = None,
) -> ReadbackResolution:
    current_time = time.monotonic() if now is None else now
    if current_time - candidate.created_at > READBACK_TIMEOUT_SECONDS or (
        candidate.context_binding != (None,) * 5
        and context_binding(context) != candidate.context_binding
    ):
        return ReadbackResolution(ReadbackReply.EXPIRED)

    correction = extract_ticket_targets(
        reply,
        scope_source=candidate.target.provider,
        scope=candidate.target.repository,
    )
    corrected = [
        reference.canonical
        for reference in correction.references
        if reference.canonical is not None
    ]
    if not correction.batch_requested and len(corrected) == 1:
        return ReadbackResolution(
            ReadbackReply.CORRECTION,
            parse_target(corrected[0], action=candidate.target.action),
        )

    normalized = re.sub(r"[^\w\s']", "", reply.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    broker_answer = (
        "yes"
        if normalized in _AFFIRMATIVE
        else ("no" if normalized in _NEGATIVE else reply)
    )
    resolution = resolve_answer(
        candidate.question,
        broker_answer,
        trusted_answer=broker_answer,
        provenance=AnswerProvenance.USER_VOICE,
    )
    if resolution.outcome == AnswerOutcome.ACCEPTED:
        if resolution.answer == "yes":
            return ReadbackResolution(ReadbackReply.AFFIRMATIVE)
        if resolution.answer == "no":
            return ReadbackResolution(ReadbackReply.NEGATIVE)
    return ReadbackResolution(ReadbackReply.COMPETING)
