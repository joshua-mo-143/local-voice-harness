from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import NamedTuple

from ..config import CURSOR_FOREGROUND_SECONDS, JOB_LOGS_DIR, JOBS_DIR, LEGACY_JOBS_DIR
from ..errors import HarnessError
from ..integrations.herdr import extract_linear_issue
from . import delivery, inbox, provisioning, recovery, worker_lifecycle
from .delivery import DeliveryClaim, DeliveryClaims
from .model import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    WORKER_STATUSES,
    CursorJob,
    JobStatus,
    NewCursorJob,
)
from .provisioning import run_claimed_worker
from .store import JobStore

DELIVERY_RETRY_SECONDS = 5.0
FOREGROUND_GRACE_SECONDS = 2.0
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


@dataclass(frozen=True, slots=True)
class StartJobRequest:
    text: str
    repository: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_context: str | None = None
    fork_requested: bool = False
    github_pull_request: int | None = None
    agent: str | None = None
    utterance: str | None = None
    context_repository: str | None = None


@dataclass(frozen=True, slots=True)
class CursorTurnRequest:
    text: str
    session_id: str | None = None
    repository: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_context: str | None = None
    fork_requested: bool = False
    github_pull_request: int | None = None
    agent: str | None = None
    utterance: str | None = None
    context_repository: str | None = None
    action: str = "submit"
    job_id: str | None = None
    reference: str | None = None


class CursorTurnResult(NamedTuple):
    text: str
    session_id: str | None


def _job_store() -> JobStore:
    return JobStore(JOBS_DIR, LEGACY_JOBS_DIR)


def read_job(job_id: str) -> CursorJob:
    return _job_store().get(job_id)


def decide_fork_confirmation(utterance: str) -> bool | None:
    normalized = re.sub(r"[^\w\s'’]", "", utterance.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip().replace("’", "'")
    if normalized in FORK_CONFIRMATIONS:
        return True
    if normalized in FORK_REJECTIONS:
        return False
    return None


def _worker_is_alive(job: CursorJob) -> bool:
    return worker_lifecycle.worker_is_alive(
        job,
        get_boot_identity=worker_lifecycle.boot_identity,
        get_process_identity=worker_lifecycle.process_identity,
    )


def _stop_worker(job: CursorJob, timeout: float = 2.0) -> bool:
    return worker_lifecycle.stop_worker(job, timeout)


def _stop_legacy_worker(job_id: str, timeout: float = 2.0) -> bool:
    job = _job_store().get(job_id)
    if not worker_lifecycle.has_legacy_worker_claim(job):
        return True
    return (
        worker_lifecycle.inspect_and_stop_legacy_worker(
            job,
            timeout,
            get_process_identity=worker_lifecycle.process_identity,
            command_matches=worker_lifecycle.legacy_worker_command_matches,
        )
        != "unsafe"
    )


def run_worker(job_id: str, claim_token: str | None = None) -> None:
    worker_lifecycle.run_worker(
        _job_store(),
        job_id,
        claim_token,
        run_claimed_worker,
    )


def launch_worker(job_id: str) -> None:
    def prepare_failure(job: CursorJob, message: str, failed_at: float) -> CursorJob:
        return job.evolve(
            status=JobStatus.FAILED,
            error=message,
            result=message,
            completed_at=failed_at,
            delivered=False,
            delivery_generation=job.delivery_generation + 1,
            delivery_claim_token=None,
            delivery_claimed_at=None,
            delivery_retry_at=0,
            delivery_attempts=0,
            updated_at=failed_at,
            worker_token=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
        )

    worker_lifecycle.launch_worker(
        _job_store(),
        JOB_LOGS_DIR,
        job_id,
        prepare_failure=prepare_failure,
        get_boot_identity=worker_lifecycle.boot_identity,
        get_process_identity=worker_lifecycle.process_identity,
    )


def start_job(
    request: StartJobRequest | str,
    *,
    repository: str | None = None,
    github_repository: str | None = None,
    github_issue: int | None = None,
    github_issue_context: str | None = None,
    fork_requested: bool = False,
    github_pull_request: int | None = None,
    agent: str | None = None,
    utterance: str | None = None,
    context_repository: str | None = None,
) -> str:
    if isinstance(request, StartJobRequest):
        text = request.text
        repository = request.repository
        github_repository = request.github_repository
        github_issue = request.github_issue
        github_issue_context = request.github_issue_context
        fork_requested = request.fork_requested
        github_pull_request = request.github_pull_request
        agent = request.agent
        utterance = request.utterance
        context_repository = request.context_repository
    else:
        text = request
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    spoken_text = utterance if utterance is not None else text
    issue_repository = (github_repository or "").strip()
    github_issue_url = (
        f"https://github.com/{issue_repository}/issues/{github_issue}"
        if issue_repository and github_issue
        else None
    )
    job = CursorJob.new(
        NewCursorJob(
            id=job_id,
            request=text,
            created_at=now,
            foreground_until=(
                now + CURSOR_FOREGROUND_SECONDS + FOREGROUND_GRACE_SECONDS
            ),
            utterance=utterance,
            trusted_utterance=spoken_text,
            repository_hint=repository,
            context_repository=context_repository,
            github_repository=github_repository,
            github_issue=github_issue,
            github_issue_url=github_issue_url,
            github_issue_context=github_issue_context,
            fork_requested=fork_requested,
            github_pull_request=github_pull_request,
            worktree_branch=(
                f"voice/github-{job_id}"
                if fork_requested
                else (
                    f"voice/github-issue-{github_issue}"
                    if github_issue
                    else (f"voice/github-pr-{job_id}" if github_pull_request else None)
                )
            ),
            worktree_label=(
                f"github-{job_id[:6]}"
                if fork_requested
                else (
                    f"issue-{github_issue}"
                    if github_issue
                    else (f"pr-{github_pull_request}" if github_pull_request else None)
                )
            ),
            pull_request_worktree_state=("pending" if github_pull_request else None),
            agent_hint=agent,
            issue_key=extract_linear_issue(spoken_text),
            speakable_label=inbox.build_speakable_label(
                text,
                issue_key=extract_linear_issue(spoken_text),
                github_repository=github_repository,
                github_issue=github_issue,
                github_pull_request=github_pull_request,
            ),
        )
    )
    _job_store().create(job)
    launch_worker(job_id)
    return job_id


def reply_job(job_id: str, text: str, *, trusted_utterance: str | None = None) -> None:
    now = time.time()
    should_launch = True

    def reply(job: CursorJob) -> CursorJob | None:
        nonlocal should_launch
        if job.status != JobStatus.AWAITING_USER:
            return None

        def queue(
            *,
            request_text: str = job.request,
            repository_hint: str | None = job.repository_hint,
            github_repository: str | None = job.github_repository,
            fork_confirmed: bool = job.fork_confirmed,
            herdr_target: str | None = job.herdr_target,
            continuation: bool,
        ) -> CursorJob:
            return job.evolve(
                status=JobStatus.QUEUED,
                question=None,
                clarification_kind=None,
                delivered=True,
                delivery_claim_token=None,
                delivery_claimed_at=None,
                queued_at=now,
                updated_at=now,
                foreground_until=(
                    now + CURSOR_FOREGROUND_SECONDS + FOREGROUND_GRACE_SECONDS
                ),
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                request=request_text,
                repository_hint=repository_hint,
                github_repository=github_repository,
                fork_confirmed=fork_confirmed,
                herdr_target=herdr_target,
                continuation=continuation,
            )

        if job.clarification_kind == "repository":
            return queue(
                repository_hint=text,
                herdr_target=None,
                continuation=False,
            )
        if job.clarification_kind == "github_repository":
            return queue(
                github_repository=text.strip(),
                herdr_target=None,
                continuation=False,
            )
        if job.clarification_kind == "fork_confirmation":
            confirmation = decide_fork_confirmation(trusted_utterance or "")
            if confirmation is False:
                should_launch = False
                return job.evolve_for_delivery(
                    now=now,
                    status=JobStatus.COMPLETED,
                    question=None,
                    clarification_kind=None,
                    result="Okay, I did not create a GitHub fork.",
                    completed_at=now,
                    worker_pid=None,
                    worker_boot_id=None,
                    worker_process_start=None,
                    worker_token=None,
                )
            if confirmation is None:
                should_launch = False
                question = "Please answer yes or no. Should I create the GitHub fork?"
                return job.evolve_for_delivery(
                    now=now, question=question, result=question
                )
            return queue(
                fork_confirmed=True,
                herdr_target=None,
                continuation=False,
            )
        return queue(continuation=True, request_text=text)

    if _job_store().update(job_id, reply) is None:
        raise HarnessError(f"Cursor job {job_id} is not waiting for a reply")
    if should_launch:
        launch_worker(job_id)


def _cancel_target_and_release(
    job_id: str,
    target: str,
    release_token: str,
    *,
    worker_stopped: bool = True,
) -> None:
    recovery.cancel_target_and_release(
        _job_store(),
        job_id,
        target,
        release_token,
        worker_stopped=worker_stopped,
        herdr_factory=provisioning.HerdrClient,
    )


def cancel_job(job_id: str) -> str:
    initial = read_job(job_id)
    legacy_worker_stopped = False
    if initial.status in WORKER_STATUSES and worker_lifecycle.has_legacy_worker_claim(
        initial
    ):
        legacy_worker_stopped = _stop_legacy_worker(job_id)
    if not legacy_worker_stopped and worker_lifecycle.has_legacy_worker_claim(initial):
        raise HarnessError(f"could not safely stop legacy Cursor worker for {job_id}")
    target = ""
    worker: CursorJob | None = None
    release_token = uuid.uuid4().hex
    cancelled_at = time.time()

    def cancel(job: CursorJob) -> CursorJob | None:
        nonlocal target, worker
        if job.status == JobStatus.CANCELLED:
            return None
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            raise HarnessError(f"Cursor job {job_id} is already {job.status.value}")
        if job.status not in ACTIVE_STATUSES:
            raise HarnessError(f"Cursor job {job_id} cannot be cancelled")
        if job.fork_committed and job.fork_operation_state in {
            "submitted",
            "ambiguous",
            "failed_observing",
        }:
            raise HarnessError(
                f"Cursor job {job_id} cannot be cancelled while its committed "
                "fork submission is being reconciled"
            )
        target = job.herdr_target or ""
        worker = job
        release_pending = bool(
            target
            or job.worktree_path
            or job.worker_token
            or job.has_uncertain_operation()
        )
        reconciliation_pending = job.has_uncertain_operation()
        release_owner_start = (
            worker_lifecycle.process_identity(os.getpid()) if release_pending else None
        )
        clear_unfenced = legacy_worker_stopped or (
            job.status in WORKER_STATUSES and not job.worker_token
        )
        return job.evolve_for_delivery(
            now=cancelled_at,
            status=JobStatus.CANCELLED,
            remove=frozenset({"reconcile"}),
            result=(
                f"Cursor job {job_id} was cancelled; external operation "
                "reconciliation is pending."
                if reconciliation_pending
                else f"Cursor job {job_id} was cancelled."
            ),
            completed_at=cancelled_at,
            foreground_until=(
                cancelled_at + CURSOR_FOREGROUND_SECONDS + FOREGROUND_GRACE_SECONDS
            ),
            target_release_pending=release_pending,
            target_release_token=release_token if release_pending else None,
            target_release_owner_pid=os.getpid() if release_owner_start else None,
            target_release_owner_boot_id=(
                worker_lifecycle.boot_identity() if release_owner_start else None
            ),
            target_release_owner_start=release_owner_start,
            cancellation_reconciliation_pending=reconciliation_pending,
            pull_request_worktree_state=(
                "retained"
                if job.github_pull_request
                and job.worktree_path
                and job.pull_request_worktree_state != "quarantined"
                else job.pull_request_worktree_state
            ),
            worker_pid=None if clear_unfenced else job.worker_pid,
            worker_boot_id=None if clear_unfenced else job.worker_boot_id,
            worker_process_start=None if clear_unfenced else job.worker_process_start,
            worker_token=None if clear_unfenced else job.worker_token,
        )

    updated = _job_store().update(job_id, cancel)
    if updated is None:
        current = read_job(job_id)
        return current.result or f"Cursor job {job_id} was cancelled."
    worker_stopped = True
    if worker is not None and worker.worker_token:
        worker_stopped = _stop_worker(worker)

    def clear_stopped_worker(job: CursorJob) -> CursorJob | None:
        if job.status != JobStatus.CANCELLED or not worker_stopped:
            return None
        return job.clear_worker()

    _job_store().update(job_id, clear_stopped_worker)
    if updated.target_release_pending:
        _cancel_target_and_release(
            job_id,
            target,
            release_token,
            worker_stopped=worker_stopped,
        )
    return updated.result or f"Cursor job {job_id} was cancelled."


def job_status(job_id: str | None = None) -> str:
    if job_id:
        job = read_job(job_id)
        return f"Cursor job {job_id} is {job.status.value.replace('_', ' ')}."
    jobs = [job for job in _job_store().list() if job.status in ACTIVE_STATUSES]
    if not jobs:
        return "There are no active Cursor jobs."
    return (
        "Active Cursor jobs: "
        + "; ".join(f"{job.id} is {job.status.value.replace('_', ' ')}" for job in jobs)
        + "."
    )


def acknowledge_worktree_quarantine(job_id: str) -> None:
    recovery.acknowledge_worktree_quarantine(_job_store(), job_id)


def resolve_manual_reconciliation(
    job_id: str,
    operation: str,
    token: str,
    outcome: str,
) -> CursorJob:
    return recovery.resolve_manual_reconciliation(
        _job_store(), job_id, operation, token, outcome
    )


def mark_delivered(job_id: str) -> CursorJob:
    def deliver(job: CursorJob) -> CursorJob:
        return job.evolve(
            delivered=True,
            delivery_claim_token=None,
            delivery_claimed_at=None,
            delivery_retry_at=0,
        )

    delivered = _job_store().update(job_id, deliver)
    assert delivered is not None
    return delivered


def claim_delivery(
    job_id: str | None = None, *, foreground: bool = False
) -> DeliveryClaim | None:
    return delivery.claim_delivery(_job_store(), job_id, foreground=foreground)


def acknowledge_delivery(job_id: str, token: str) -> bool:
    return delivery.acknowledge_delivery(_job_store(), job_id, token)


def release_delivery(job_id: str, token: str, *, retry: bool = True) -> bool:
    return delivery.release_delivery(_job_store(), job_id, token, retry=retry)


def acknowledge_deliveries(claims: DeliveryClaims) -> None:
    delivery.acknowledge_deliveries(_job_store(), claims)


def release_deliveries(claims: DeliveryClaims) -> None:
    delivery.release_deliveries(_job_store(), claims)


def _defer_or_acknowledge(
    job_id: str, claims: DeliveryClaims | None
) -> CursorJob | None:
    claimed = claim_delivery(job_id, foreground=True)
    if claimed is None:
        return None
    if claims is None:
        acknowledge_delivery(job_id, claimed.token)
    else:
        claims.append(claimed)
    return claimed.job


def recover_jobs() -> None:
    recovery.recover_jobs(
        _job_store(),
        launch_worker=launch_worker,
        herdr_factory=provisioning.HerdrClient,
        github_factory=provisioning.GitHubClient,
        is_worker_alive=_worker_is_alive,
        stop_owned_worker=_stop_worker,
        stop_unfenced_worker=lambda _store, job_id: _stop_legacy_worker(job_id),
        get_boot_identity=worker_lifecycle.boot_identity,
        get_process_identity=worker_lifecycle.process_identity,
        inspect_legacy_worker=lambda job: (
            worker_lifecycle.inspect_and_stop_legacy_worker(
                job,
                get_process_identity=worker_lifecycle.process_identity,
                command_matches=worker_lifecycle.legacy_worker_command_matches,
            )
        ),
    )


def pending_results() -> list[DeliveryClaim]:
    recover_jobs()
    return delivery.pending_deliveries(_job_store())


ANNOUNCEABLE_STATUSES = delivery.DELIVERABLE_STATUSES


@dataclass(frozen=True, slots=True)
class ResolvedReference:
    job_id: str | None
    clarification: str | None


def _resolve_reference(
    reference: str,
    *,
    statuses: frozenset[JobStatus],
    action: str,
    empty_message: str,
    job_id: str | None = None,
    session_id: str | None = None,
) -> ResolvedReference:
    """Map a spoken reference (with context fallbacks) onto a single job.

    Explicit textual matches win. When the reference is ambiguous we ask for
    clarification rather than guessing. With no textual match we fall back to an
    explicit id, the active session, or the sole candidate before giving up and
    listing the options.
    """

    jobs = [job for job in _job_store().list() if job.status in statuses]
    resolution = inbox.resolve_reference(jobs, reference or "")
    if resolution.unique is not None:
        return ResolvedReference(resolution.unique.id, None)
    if resolution.ambiguous:
        return ResolvedReference(None, inbox.clarify(list(resolution.matches), action))
    for candidate in (job_id, session_id):
        if candidate and any(job.id == candidate for job in jobs):
            return ResolvedReference(candidate, None)
    if len(jobs) == 1:
        return ResolvedReference(jobs[0].id, None)
    if not jobs:
        return ResolvedReference(None, empty_message)
    return ResolvedReference(None, inbox.clarify(inbox.summarize_all(jobs), action))


def list_jobs() -> str:
    return inbox.describe_inbox(_job_store().list())


def _status_message(job_id: str) -> str:
    summary = inbox.summarize(read_job(job_id))
    state = summary.status.value.replace("_", " ")
    message = f"{summary.label} is {state}"
    if summary.detail and summary.status not in {
        JobStatus.QUEUED,
        JobStatus.ROUTING,
        JobStatus.RUNNING,
        JobStatus.RECONCILING,
    }:
        message += f". {summary.detail}"
    return message + "."


def dismiss_announcement(job_id: str) -> str:
    now = time.time()

    def dismiss(job: CursorJob) -> CursorJob | None:
        if job.announcement_dismissed and job.delivered:
            return None
        return job.evolve(
            announcement_dismissed=True,
            delivered=True,
            delivery_claim_token=None,
            delivery_claimed_at=None,
            delivery_retry_at=0,
            delivered_at=now,
        )

    updated = _job_store().update(job_id, dismiss)
    label = inbox.speakable_label_for(updated or read_job(job_id))
    return f"Dismissed the update for {label}."


def repeat_announcement(job_id: str) -> str:
    now = time.time()

    def repeat(job: CursorJob) -> CursorJob | None:
        if job.status not in ANNOUNCEABLE_STATUSES:
            return None
        return job.evolve_for_delivery(
            now=now,
            announcement_repeated=True,
            announcement_dismissed=False,
        )

    updated = _job_store().update(job_id, repeat)
    if updated is None:
        raise HarnessError(f"Cursor job {job_id} has no update to repeat")
    return f"I'll repeat the update for {inbox.speakable_label_for(updated)}."


def cursor_turn(
    request: CursorTurnRequest | str,
    session_id: str | None = None,
    *,
    repository: str | None = None,
    github_repository: str | None = None,
    github_issue: int | None = None,
    github_issue_context: str | None = None,
    fork_requested: bool = False,
    github_pull_request: int | None = None,
    agent: str | None = None,
    utterance: str | None = None,
    context_repository: str | None = None,
    action: str = "submit",
    job_id: str | None = None,
    reference: str | None = None,
    delivery_claims: DeliveryClaims | None = None,
) -> CursorTurnResult:
    if isinstance(request, CursorTurnRequest):
        text = request.text
        session_id = request.session_id
        repository = request.repository
        github_repository = request.github_repository
        github_issue = request.github_issue
        github_issue_context = request.github_issue_context
        fork_requested = request.fork_requested
        github_pull_request = request.github_pull_request
        agent = request.agent
        utterance = request.utterance
        context_repository = request.context_repository
        action = request.action
        job_id = request.job_id
        reference = request.reference
    else:
        text = request
    if action == "list":
        return CursorTurnResult(list_jobs(), session_id)
    if action == "status":
        resolved = _resolve_reference(
            reference or text,
            statuses=ACTIVE_STATUSES | TERMINAL_STATUSES,
            action="check",
            empty_message="There are no Cursor jobs.",
            job_id=job_id,
            session_id=session_id,
        )
        if resolved.clarification is not None:
            return CursorTurnResult(resolved.clarification, session_id)
        assert resolved.job_id is not None
        return CursorTurnResult(_status_message(resolved.job_id), session_id)
    if action == "cancel":
        resolved = _resolve_reference(
            reference or text,
            statuses=ACTIVE_STATUSES,
            action="cancel",
            empty_message="There are no active Cursor jobs to cancel.",
            job_id=job_id,
            session_id=session_id,
        )
        if resolved.clarification is not None:
            return CursorTurnResult(resolved.clarification, session_id)
        assert resolved.job_id is not None
        result = cancel_job(resolved.job_id)
        _defer_or_acknowledge(resolved.job_id, delivery_claims)
        return CursorTurnResult(result, None)
    if action in {"dismiss", "repeat"}:
        resolved = _resolve_reference(
            reference or text,
            statuses=ANNOUNCEABLE_STATUSES,
            action=action,
            empty_message=f"There are no updates to {action}.",
            job_id=job_id,
            session_id=session_id,
        )
        if resolved.clarification is not None:
            return CursorTurnResult(resolved.clarification, session_id)
        assert resolved.job_id is not None
        message = (
            dismiss_announcement(resolved.job_id)
            if action == "dismiss"
            else repeat_announcement(resolved.job_id)
        )
        return CursorTurnResult(message, session_id)
    if action == "reply":
        reply_id = job_id or session_id
        if not reply_id:
            resolved = _resolve_reference(
                reference or text,
                statuses=frozenset({JobStatus.AWAITING_USER}),
                action="answer",
                empty_message="No Cursor job is waiting for a reply.",
            )
            if resolved.clarification is not None:
                return CursorTurnResult(resolved.clarification, session_id)
            reply_id = resolved.job_id
        assert reply_id is not None
        reply_job(reply_id, text, trusted_utterance=utterance)
        job_id = reply_id
    else:
        job_id = start_job(
            text,
            repository=repository,
            github_repository=github_repository,
            github_issue=github_issue,
            github_issue_context=github_issue_context,
            fork_requested=fork_requested,
            github_pull_request=github_pull_request,
            agent=agent,
            utterance=utterance,
            context_repository=context_repository,
        )
    started = time.perf_counter()
    deadline = time.monotonic() + CURSOR_FOREGROUND_SECONDS
    while time.monotonic() < deadline:
        job = read_job(job_id)
        if job.status == JobStatus.COMPLETED:
            claimed = _defer_or_acknowledge(job_id, delivery_claims)
            print(
                json.dumps(
                    {
                        "stage": "cursor",
                        "job_id": job_id,
                        "background": False,
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
            )
            result = claimed.result if claimed is not None else job.result
            return CursorTurnResult(str(result or "").strip(), None)
        if job.status == JobStatus.AWAITING_USER:
            claimed = _defer_or_acknowledge(job_id, delivery_claims)
            question = (
                claimed.question or claimed.result
                if claimed is not None
                else job.question or job.result
            )
            return CursorTurnResult(str(question or "").strip(), job_id)
        if job.status == JobStatus.BLOCKED:
            claimed = _defer_or_acknowledge(job_id, delivery_claims)
            result = claimed.result if claimed is not None else job.result
            return CursorTurnResult(
                str(result or "Cursor needs attention in Herdr"), None
            )
        if job.status == JobStatus.FAILED:
            claimed = _defer_or_acknowledge(job_id, delivery_claims)
            error = claimed.error if claimed is not None else job.error
            raise HarnessError(str(error or "Cursor failed"))
        if job.status == JobStatus.CANCELLED:
            claimed = _defer_or_acknowledge(job_id, delivery_claims)
            result = claimed.result if claimed is not None else job.result
            return CursorTurnResult(str(result or "Cursor job was cancelled"), None)
        time.sleep(0.1)
    _job_store().update(
        job_id,
        lambda job: job.evolve(foreground_until=0, updated_at=time.time()),
    )
    print(
        json.dumps(
            {
                "stage": "cursor",
                "job_id": job_id,
                "background": True,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    )
    return CursorTurnResult(
        f"Cursor is still working on job {job_id}. I will report back when it finishes.",
        None,
    )
