"""Background announcement policy, classification, and bounded digest drain."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dt_time
from enum import StrEnum
from zoneinfo import ZoneInfo

from ..notifications import notify as desktop_notify
from ..responses import AssistantResponse
from ..user_config import AnnouncementMode, AnnouncementSettings
from . import inbox
from .delivery import (
    ANNOUNCEMENT_BATCH_LIMIT,
    DeliveryClaim,
    acknowledge_deferred_delivery,
    acknowledge_desktop_delivery,
    announcement_drain_lock,
    pending_deliveries,
    suppressed_jobs,
)
from .model import (
    AnnouncementAck,
    CursorJob,
    JobStatus,
)
from .store import JobStore

DIGEST_SPOKEN_ITEMS = 4


class AnnouncementKind(StrEnum):
    COMPLETION = "completion"
    QUESTION = "question"
    FAILURE = "failure"
    CANCELLATION = "cancellation"
    INFORMATIONAL = "informational"


class AnnouncementDisposition(StrEnum):
    SPEAK = "speak"
    DESKTOP = "desktop"
    DEFER = "defer"


ACTION_REQUIRED_KINDS = frozenset(
    {
        AnnouncementKind.QUESTION,
        AnnouncementKind.FAILURE,
        AnnouncementKind.INFORMATIONAL,
    }
)


@dataclass(frozen=True, slots=True)
class DrainResult:
    speak: tuple[DeliveryClaim, ...] = ()
    desktop: tuple[DeliveryClaim, ...] = ()
    deferred: tuple[DeliveryClaim, ...] = ()


def classify(job: CursorJob) -> AnnouncementKind:
    if job.status == JobStatus.COMPLETED:
        return AnnouncementKind.COMPLETION
    if job.status == JobStatus.AWAITING_USER:
        return AnnouncementKind.QUESTION
    if job.status == JobStatus.FAILED:
        return AnnouncementKind.FAILURE
    if job.status == JobStatus.CANCELLED:
        return AnnouncementKind.CANCELLATION
    if job.status == JobStatus.BLOCKED:
        return AnnouncementKind.INFORMATIONAL
    raise ValueError(f"{job.status.value} job cannot be classified for announcement")


def local_datetime(when: float, timezone_name: str) -> datetime:
    instant = datetime.fromtimestamp(when, tz=UTC)
    if timezone_name:
        return instant.astimezone(ZoneInfo(timezone_name))
    return instant.astimezone()


def parse_clock(value: str) -> dt_time:
    hour, minute = value.split(":", 1)
    return dt_time(int(hour), int(minute))


def in_quiet_hours(
    settings: AnnouncementSettings,
    *,
    now: float | None = None,
) -> bool:
    if not settings.quiet_hours_start or not settings.quiet_hours_end:
        return False
    current = local_datetime(
        time.time() if now is None else now,
        settings.timezone,
    ).time()
    start = parse_clock(settings.quiet_hours_start)
    end = parse_clock(settings.quiet_hours_end)
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def disposition(
    settings: AnnouncementSettings,
    kind: AnnouncementKind,
    *,
    now: float | None = None,
) -> AnnouncementDisposition:
    if settings.mode == AnnouncementMode.QUIET or in_quiet_hours(settings, now=now):
        return AnnouncementDisposition.DEFER
    if settings.mode == AnnouncementMode.DESKTOP_ONLY:
        return AnnouncementDisposition.DESKTOP
    if settings.mode == AnnouncementMode.ACTION_REQUIRED:
        if kind in ACTION_REQUIRED_KINDS:
            return AnnouncementDisposition.SPEAK
        return AnnouncementDisposition.DEFER
    return AnnouncementDisposition.SPEAK


def auto_eligible(
    settings: AnnouncementSettings,
    *,
    now: float | None = None,
) -> Callable[[CursorJob], bool]:
    def eligible(job: CursorJob) -> bool:
        ack = job.announcement_ack
        chosen = disposition(settings, classify(job), now=now)
        if chosen == AnnouncementDisposition.SPEAK:
            return ack in {
                AnnouncementAck.PENDING.value,
                AnnouncementAck.DEFERRED.value,
                AnnouncementAck.DESKTOP.value,
            }
        if chosen == AnnouncementDisposition.DESKTOP:
            return ack in {
                AnnouncementAck.PENDING.value,
                AnnouncementAck.DEFERRED.value,
            }
        return ack == AnnouncementAck.PENDING.value

    return eligible


def digest_eligible(job: CursorJob) -> bool:
    return job.announcement_ack in {
        AnnouncementAck.DEFERRED.value,
        AnnouncementAck.DESKTOP.value,
    }


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _bounded_labels(jobs: list[CursorJob]) -> str:
    labels = [inbox.speakable_label_for(job) for job in jobs]
    shown = labels[:DIGEST_SPOKEN_ITEMS]
    remainder = len(labels) - len(shown)
    if remainder > 0:
        shown.append(f"{remainder} more")
    return _join(shown)


def render_digest(
    jobs: list[CursorJob],
    *,
    missed: bool = False,
    render_job: Callable[[CursorJob], AssistantResponse],
) -> AssistantResponse:
    if not jobs:
        spoken = (
            "You didn't miss any background announcements."
            if missed
            else "There are no background announcements."
        )
        return AssistantResponse(spoken_text=spoken, display_text=spoken)
    grouped: dict[AnnouncementKind, list[CursorJob]] = {}
    for job in jobs:
        grouped.setdefault(classify(job), []).append(job)
    phrases: list[str] = []
    for kind, label in (
        (AnnouncementKind.QUESTION, "waiting for you"),
        (AnnouncementKind.FAILURE, "failed"),
        (AnnouncementKind.INFORMATIONAL, "need attention"),
        (AnnouncementKind.COMPLETION, "finished"),
        (AnnouncementKind.CANCELLATION, "cancelled"),
    ):
        bucket = grouped.get(kind, [])
        if not bucket:
            continue
        phrases.append(f"{label}: {_bounded_labels(bucket)}")
    count = len(jobs)
    noun = "update" if count == 1 else "updates"
    if missed:
        spoken = f"You missed {count} background {noun}. " + ". ".join(phrases) + "."
    else:
        spoken = f"{count} background {noun}. " + ". ".join(phrases) + "."
    display_parts = [render_job(job).display_text for job in jobs]
    return AssistantResponse(
        spoken_text=spoken,
        display_text="\n".join(display_parts),
    )


def _apply_desktop(
    store: JobStore,
    claim: DeliveryClaim,
    *,
    render_job: Callable[[CursorJob], AssistantResponse],
    notify: Callable[..., None],
    now: float,
) -> bool:
    rendered = render_job(claim.job)
    notify(
        rendered.display_text,
        error=classify(claim.job) == AnnouncementKind.FAILURE,
    )
    return acknowledge_desktop_delivery(store, claim.job.id, claim.token, now=now)


def _apply_deferred(store: JobStore, claim: DeliveryClaim, *, now: float) -> bool:
    return acknowledge_deferred_delivery(store, claim.job.id, claim.token, now=now)


def drain_background_announcements(
    store: JobStore,
    settings: AnnouncementSettings,
    *,
    now: float | None = None,
    notify: Callable[..., None] = desktop_notify,
    render_job: Callable[[CursorJob], AssistantResponse] | None = None,
) -> DrainResult:
    """Claim due announcements under one drain lock and apply non-spoken policy."""

    from .service import render_job_announcement

    renderer = render_job or render_job_announcement
    applied_at = time.time() if now is None else now
    with announcement_drain_lock(store) as acquired:
        if not acquired:
            return DrainResult()
        claims = pending_deliveries(
            store,
            limit=ANNOUNCEMENT_BATCH_LIMIT,
            now=now,
            eligible=auto_eligible(settings, now=now),
        )
        speak: list[DeliveryClaim] = []
        desktop: list[DeliveryClaim] = []
        deferred: list[DeliveryClaim] = []
        for claim in claims:
            chosen = disposition(settings, classify(claim.job), now=now)
            if chosen == AnnouncementDisposition.SPEAK:
                speak.append(claim)
            elif chosen == AnnouncementDisposition.DESKTOP:
                if _apply_desktop(
                    store,
                    claim,
                    render_job=renderer,
                    notify=notify,
                    now=applied_at,
                ):
                    desktop.append(claim)
            elif _apply_deferred(store, claim, now=applied_at):
                deferred.append(claim)
        return DrainResult(
            speak=tuple(speak),
            desktop=tuple(desktop),
            deferred=tuple(deferred),
        )


def claim_missed_announcements(
    store: JobStore,
    *,
    now: float | None = None,
) -> list[DeliveryClaim]:
    """Claim suppressed results for an explicit 'what did I miss?' delivery."""

    with announcement_drain_lock(store) as acquired:
        if not acquired:
            return []
        return pending_deliveries(
            store,
            limit=ANNOUNCEMENT_BATCH_LIMIT,
            now=now,
            eligible=digest_eligible,
        )


def inspect_missed_announcements(store: JobStore) -> list[CursorJob]:
    return suppressed_jobs(store)
