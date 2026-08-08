"""Deprecated compatibility facade for the decomposed Cursor job subsystem."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypeAlias, cast

from ..config import JOBS_DIR, LEGACY_JOBS_DIR
from ..errors import HarnessError
from ..integrations.github import GitHubClient
from ..integrations.herdr import HerdrClient
from . import delivery, provisioning
from .delivery import DeliveryClaim
from .model import (
    ACTIVE_STATUSES as MODEL_ACTIVE_STATUSES,
)
from .model import CURRENT_SCHEMA_VERSION, CursorJob, JobStatus, JobValidationError
from .model import (
    TERMINAL_STATUSES as MODEL_TERMINAL_STATUSES,
)
from .model import (
    WORKER_STATUSES as MODEL_WORKER_STATUSES,
)
from .service import (
    CursorTurnRequest,
    CursorTurnResult,
    StartJobRequest,
    acknowledge_worktree_quarantine,
    cancel_job,
    cursor_turn,
    job_status,
    launch_worker,
    recover_jobs,
    reply_job,
    run_worker,
    start_job,
)
from .store import JobStore

DeliveryClaims: TypeAlias = list[tuple[str, str] | DeliveryClaim]
ACTIVE_STATUSES = {status.value for status in MODEL_ACTIVE_STATUSES}
WORKER_STATUSES = {status.value for status in MODEL_WORKER_STATUSES}
TERMINAL_STATUSES = {status.value for status in MODEL_TERMINAL_STATUSES}
DELIVERABLE_STATUSES = TERMINAL_STATUSES | {
    JobStatus.AWAITING_USER.value,
    JobStatus.BLOCKED.value,
}
DELIVERY_CLAIM_SECONDS = delivery.DELIVERY_CLAIM_SECONDS
DELIVERY_RETRY_SECONDS = delivery.DELIVERY_RETRY_SECONDS
FORK_CONFIRMATIONS = {
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
FORK_REJECTIONS = {"no", "no thanks", "do not", "don't", "cancel", "stop"}
repository_question = provisioning.repository_question
run_claimed_worker = provisioning.run_claimed_worker
WorkerCancelled = provisioning.WorkerCancelled
ReservationConflict = provisioning.ReservationConflict


def _store() -> JobStore:
    return JobStore(JOBS_DIR, LEGACY_JOBS_DIR)


def _compat_job(job: dict[str, object], *, default_status: str = "queued") -> CursorJob:
    values = dict(job)
    values.setdefault("id", "000000000000")
    values.setdefault("status", default_status)
    values.setdefault("request", "")
    values.setdefault("created_at", 0)
    values.setdefault("queued_at", values["created_at"])
    values.setdefault("delivered", False)
    return CursorJob.from_dict(values)


def job_path(job_id: str) -> Path:
    return _store().path(job_id)


def read_job(job_id: str) -> dict[str, object]:
    try:
        return _store().get(job_id).to_dict()
    except (OSError, JobValidationError) as exc:
        raise HarnessError(f"could not read Cursor job {job_id}: {exc}") from exc


def write_job(job: dict[str, object]) -> None:
    job_id = str(job.get("id") or "")
    store = _store()
    try:
        store.path(job_id)
        try:
            current = store.get(job_id)
        except FileNotFoundError:
            store.create(CursorJob.from_dict(job))
            return
        base_revision = job.get("revision")

        def apply_edit(persisted: CursorJob) -> CursorJob:
            if (
                not isinstance(base_revision, int)
                or isinstance(base_revision, bool)
                or base_revision != persisted.revision
            ):
                raise JobValidationError(
                    f"stale Cursor job revision {base_revision!r}; "
                    f"expected {persisted.revision}"
                )
            values = dict(job)
            values["schema_version"] = CURRENT_SCHEMA_VERSION
            values["revision"] = persisted.revision + 1
            return CursorJob.from_dict(values)

        store.update(current.id, apply_edit)
    except (OSError, JobValidationError) as exc:
        raise HarnessError(f"could not write Cursor job {job_id}: {exc}") from exc


def active_jobs() -> list[dict[str, object]]:
    return [
        job.to_dict() for job in _store().list() if job.status in MODEL_ACTIVE_STATUSES
    ]


def reserved_targets(exclude_job_id: str | None = None) -> set[str]:
    return provisioning.reserved_targets(_store(), exclude_job_id)


def decide_fork_confirmation(utterance: str) -> bool | None:
    normalized = re.sub(r"[^\w\s'’]", "", utterance.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip().replace("’", "'")
    if normalized in FORK_CONFIRMATIONS:
        return True
    if normalized in FORK_REJECTIONS:
        return False
    return None


def resolve_job_repository(
    client: HerdrClient,
    job: dict[str, object],
    repositories: list[Path],
) -> tuple[Path | None, list[Path]]:
    return provisioning.resolve_job_repository(client, _compat_job(job), repositories)


def complete_from_output(
    job: dict[str, object], *, output: str, agent_status: str
) -> None:
    completed = provisioning.complete_from_output(
        _compat_job(job, default_status="running"),
        output=output,
        agent_status=agent_status,
    )
    job.clear()
    job.update(completed.to_dict())


def read_agent_completion(
    client: HerdrClient,
    job: dict[str, object],
    **kwargs: object,
) -> tuple[str, str]:
    return provisioning.read_agent_completion(
        client, _compat_job(job), **cast(Any, kwargs)
    )


def mark_delivered(job_id: str) -> dict[str, object]:
    from .service import mark_delivered as typed_mark_delivered

    return typed_mark_delivered(job_id).to_dict()


def claim_delivery(
    job_id: str | None = None, *, foreground: bool = False
) -> dict[str, object] | None:
    claim = delivery.claim_delivery(_store(), job_id, foreground=foreground)
    if claim is None:
        return None
    result = claim.job.to_dict()
    result["_delivery_token"] = claim.token
    return result


def acknowledge_delivery(job_id: str, token: str) -> bool:
    return delivery.acknowledge_delivery(_store(), job_id, token)


def release_delivery(job_id: str, token: str, *, retry: bool = True) -> bool:
    return delivery.release_delivery(_store(), job_id, token, retry=retry)


def acknowledge_deliveries(claims: DeliveryClaims) -> None:
    for claim in claims:
        job_id, token = (
            (claim.job.id, claim.token) if isinstance(claim, DeliveryClaim) else claim
        )
        acknowledge_delivery(job_id, token)
    claims.clear()


def release_deliveries(claims: DeliveryClaims) -> None:
    for claim in claims:
        job_id, token = (
            (claim.job.id, claim.token) if isinstance(claim, DeliveryClaim) else claim
        )
        release_delivery(job_id, token)
    claims.clear()


def pending_results() -> list[dict[str, object]]:
    recover_jobs()
    results: list[dict[str, object]] = []
    for claim in delivery.pending_deliveries(_store()):
        value = claim.job.to_dict()
        value["_delivery_token"] = claim.token
        results.append(value)
    return results


pending_deliveries = pending_results


def resolve_manual_reconciliation(
    job_id: str, operation: str, token: str, outcome: str
) -> dict[str, object]:
    from .service import resolve_manual_reconciliation as typed_resolve

    return typed_resolve(job_id, operation, token, outcome).to_dict()


__all__ = [
    "ACTIVE_STATUSES",
    "CursorTurnRequest",
    "CursorTurnResult",
    "DELIVERABLE_STATUSES",
    "DELIVERY_CLAIM_SECONDS",
    "DELIVERY_RETRY_SECONDS",
    "DeliveryClaim",
    "DeliveryClaims",
    "GitHubClient",
    "HarnessError",
    "HerdrClient",
    "ReservationConflict",
    "StartJobRequest",
    "TERMINAL_STATUSES",
    "WORKER_STATUSES",
    "WorkerCancelled",
    "acknowledge_deliveries",
    "acknowledge_delivery",
    "acknowledge_worktree_quarantine",
    "active_jobs",
    "cancel_job",
    "claim_delivery",
    "complete_from_output",
    "cursor_turn",
    "decide_fork_confirmation",
    "job_path",
    "job_status",
    "launch_worker",
    "mark_delivered",
    "pending_deliveries",
    "pending_results",
    "read_agent_completion",
    "read_job",
    "recover_jobs",
    "release_deliveries",
    "release_delivery",
    "reply_job",
    "repository_question",
    "reserved_targets",
    "resolve_job_repository",
    "resolve_manual_reconciliation",
    "run_claimed_worker",
    "run_worker",
    "start_job",
    "write_job",
]
