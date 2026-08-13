from __future__ import annotations

import fcntl
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .model import (
    SUPPRESSED_ANNOUNCEMENT_ACKS,
    CursorJob,
    JobStatus,
)
from .store import JobStore

DELIVERY_CLAIM_SECONDS = 300.0
DELIVERY_RENEW_SECONDS = DELIVERY_CLAIM_SECONDS / 2
DELIVERY_RETRY_SECONDS = 5.0
DELIVERY_WINDOW = 1
ANNOUNCEMENT_BATCH_LIMIT = 8
ANNOUNCEMENT_DRAIN_LOCK = ".announcement-drain.lock"
DELIVERABLE_STATUSES = frozenset(
    {
        JobStatus.AWAITING_USER,
        JobStatus.BLOCKED,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    job: CursorJob
    token: str


DeliveryClaims = list[DeliveryClaim]


def _claim_is_live(job: CursorJob, now: float) -> bool:
    return (
        bool(job.delivery_claim_token)
        and job.delivery_claimed_at is not None
        and now - job.delivery_claimed_at < DELIVERY_CLAIM_SECONDS
    )


def claim_delivery(
    store: JobStore,
    job_id: str | None = None,
    *,
    foreground: bool = False,
    now: float | None = None,
    eligible: Callable[[CursorJob], bool] | None = None,
) -> DeliveryClaim | None:
    claimed_at = time.time() if now is None else now

    def claim(job: CursorJob) -> CursorJob | None:
        if job.status not in DELIVERABLE_STATUSES or job.delivered:
            return None
        if eligible is not None and not eligible(job):
            return None
        if not foreground and claimed_at < job.foreground_until:
            return None
        if claimed_at < job.delivery_retry_at:
            return None
        completed_age = claimed_at - (job.completed_at or claimed_at)
        if (
            not foreground
            and job.status not in {JobStatus.AWAITING_USER, JobStatus.BLOCKED}
            and completed_age < 1
        ):
            return None
        if _claim_is_live(job, claimed_at):
            return None
        return job.claim_delivery(uuid.uuid4().hex, claimed_at=claimed_at)

    if job_id is not None:
        updated = store.update(job_id, claim)
        if updated is None:
            return None
        assert updated.delivery_claim_token is not None
        return DeliveryClaim(updated, updated.delivery_claim_token)

    jobs = sorted(
        store.list(),
        key=lambda job: job.completed_at or job.created_at,
    )
    for job in jobs:
        updated = store.update(job.id, claim)
        if updated is not None:
            assert updated.delivery_claim_token is not None
            return DeliveryClaim(updated, updated.delivery_claim_token)
    return None


def renew_delivery(
    store: JobStore,
    job_id: str,
    token: str,
    *,
    now: float | None = None,
) -> bool:
    renewed_at = time.time() if now is None else now

    def renew(job: CursorJob) -> CursorJob | None:
        if (
            job.delivery_claim_token != token
            or job.delivered
            or not _claim_is_live(job, renewed_at)
        ):
            return None
        return job.renew_delivery(claimed_at=renewed_at)

    return store.update(job_id, renew) is not None


def acknowledge_delivery(
    store: JobStore,
    job_id: str,
    token: str,
    *,
    now: float | None = None,
) -> bool:
    delivered_at = time.time() if now is None else now

    def acknowledge(job: CursorJob) -> CursorJob | None:
        if (
            job.delivery_claim_token != token
            or job.delivered
            or not _claim_is_live(job, delivered_at)
        ):
            return None
        return job.acknowledge_delivery(delivered_at=delivered_at)

    return store.update(job_id, acknowledge) is not None


def acknowledge_desktop_delivery(
    store: JobStore,
    job_id: str,
    token: str,
) -> bool:
    def acknowledge(job: CursorJob) -> CursorJob | None:
        if job.delivery_claim_token != token or job.delivered:
            return None
        return job.acknowledge_desktop_delivery()

    return store.update(job_id, acknowledge) is not None


def acknowledge_deferred_delivery(
    store: JobStore,
    job_id: str,
    token: str,
) -> bool:
    def acknowledge(job: CursorJob) -> CursorJob | None:
        if job.delivery_claim_token != token or job.delivered:
            return None
        return job.acknowledge_deferred_delivery()

    return store.update(job_id, acknowledge) is not None


def release_delivery(
    store: JobStore,
    job_id: str,
    token: str,
    *,
    retry: bool = True,
    now: float | None = None,
) -> bool:
    released_at = time.time() if now is None else now

    def release(job: CursorJob) -> CursorJob | None:
        if job.delivery_claim_token != token or job.delivered:
            return None
        retry_at = released_at + DELIVERY_RETRY_SECONDS if retry else 0
        return job.release_delivery(retry_at=retry_at)

    return store.update(job_id, release) is not None


def acknowledge_deliveries(
    store: JobStore, claims: DeliveryClaims
) -> list[DeliveryClaim]:
    acknowledged: list[DeliveryClaim] = []
    for claim in claims:
        if acknowledge_delivery(store, claim.job.id, claim.token):
            acknowledged.append(claim)
    claims.clear()
    return acknowledged


def release_deliveries(store: JobStore, claims: DeliveryClaims) -> None:
    for claim in claims:
        release_delivery(store, claim.job.id, claim.token)
    claims.clear()


def pending_deliveries(
    store: JobStore,
    *,
    limit: int = DELIVERY_WINDOW,
    now: float | None = None,
    eligible: Callable[[CursorJob], bool] | None = None,
) -> list[DeliveryClaim]:
    if limit <= 0:
        raise ValueError("delivery limit must be positive")
    claims: list[DeliveryClaim] = []
    while len(claims) < limit and (
        claim := claim_delivery(store, now=now, eligible=eligible)
    ):
        claims.append(claim)
    return claims


def suppressed_jobs(store: JobStore) -> list[CursorJob]:
    """Return deferred and desktop-notified results in durable completion order."""

    jobs = [
        job
        for job in store.list()
        if job.status in DELIVERABLE_STATUSES
        and not job.delivered
        and job.announcement_ack in SUPPRESSED_ANNOUNCEMENT_ACKS
    ]
    return sorted(jobs, key=lambda job: job.completed_at or job.created_at)


@contextmanager
def announcement_drain_lock(store: JobStore) -> Iterator[bool]:
    """Non-blocking exclusive drain lease so concurrent daemons have one winner."""

    store.durable_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = store.durable_dir / ANNOUNCEMENT_DRAIN_LOCK
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        pass
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except BlockingIOError:
        yield False
    finally:
        os.close(fd)
