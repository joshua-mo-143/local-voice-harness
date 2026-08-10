from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, NamedTuple

from ..config import (
    CURSOR_FOREGROUND_SECONDS,
    JOB_LOGS_DIR,
    JOBS_DIR,
    LEGACY_JOBS_DIR,
)
from ..errors import HarnessError
from ..integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubIssue,
    format_issue_context,
    github_issue_from_url,
)
from ..integrations.registry import (
    extract_issue_reference,
    integration_enabled,
    require_issue_capabilities,
    resolve_issue_reference,
)
from ..questions import (
    AnswerOutcome,
    AnswerProvenance,
    QuestionState,
    choices_prompt,
    question_prompt,
    resolve_answer,
)
from ..ticket_targets import TicketExtraction, TicketReference, extract_ticket_targets
from ..user_config import load_user_config
from . import delivery, inbox, provisioning, questions, recovery, worker_lifecycle
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
from .store import (
    FollowUpCheckoutBusy,
    FollowUpUnavailable,
    JobMaintenanceError,
    JobStore,
    MaintenanceLease,
)

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
    issue_key: str | None = None
    foreground: bool = True


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
    issue_key: str | None = None
    issue_scope: str | None = None
    issue_scope_source: str | None = None
    action: str = "submit"
    job_id: str | None = None
    reference: str | None = None
    expected_question_id: str | None = None
    expected_question_turn: str | None = None
    answer_provenance: AnswerProvenance = AnswerProvenance.USER_TEXT
    expected_completed_at: float | None = None
    on_follow_up_started: Callable[[], None] | None = None
    on_job_started: Callable[[], None] | None = None


class CursorTurnResult(NamedTuple):
    text: str
    session_id: str | None


TicketStartStatus = Literal["accepted", "rejected", "start-failed"]


@dataclass(frozen=True, slots=True)
class TicketJobRequest:
    target: str
    request: StartJobRequest


@dataclass(frozen=True, slots=True)
class TicketStartOutcome:
    target: str
    status: TicketStartStatus
    job_id: str | None = None
    detail: str | None = None


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
    issue_key: str | None = None,
    foreground: bool = True,
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
        issue_key = request.issue_key
        foreground = request.foreground
    else:
        text = request
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    spoken_text = utterance if utterance is not None else text
    resolved_issue_key = (
        resolve_issue_reference(issue_key)
        if issue_key is not None
        else extract_issue_reference(spoken_text)
    )
    if resolved_issue_key:
        require_issue_capabilities(resolved_issue_key)
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
                if foreground
                else 0
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
            issue_key=resolved_issue_key,
            speakable_label=inbox.build_speakable_label(
                text,
                issue_key=resolved_issue_key,
                github_repository=github_repository,
                github_issue=github_issue,
                github_pull_request=github_pull_request,
            ),
        )
    )
    _job_store().create(job)
    launch_worker(job_id)
    return job_id


def _start_error_detail(error: BaseException) -> str:
    detail = re.sub(r"\s+", " ", str(error) or type(error).__name__).strip()
    return detail[:240]


def start_jobs(
    requests: tuple[TicketJobRequest, ...],
    *,
    concurrency: int | None = None,
) -> tuple[TicketStartOutcome, ...]:
    """Start ordinary durable jobs concurrently without foreground waiting."""
    if not requests:
        return ()
    outcomes: list[TicketStartOutcome | None] = [None] * len(requests)

    def start(request: TicketJobRequest) -> TicketStartOutcome:
        try:
            job_id = start_job(request.request)
        except Exception as exc:  # noqa: BLE001 - every child needs an outcome
            return TicketStartOutcome(
                request.target,
                "start-failed",
                detail=_start_error_detail(exc),
            )
        try:
            current = read_job(job_id)
        except Exception:  # noqa: BLE001 - handoff succeeded; observation is optional
            current = None
        if current is not None and current.status == JobStatus.FAILED:
            return TicketStartOutcome(
                request.target,
                "start-failed",
                job_id=job_id,
                detail=_start_error_detail(
                    HarnessError(
                        str(current.error or current.result or "worker failed")
                    )
                ),
            )
        return TicketStartOutcome(request.target, "accepted", job_id=job_id)

    configured_concurrency = (
        load_user_config().platform.agent_job_start_concurrency
        if concurrency is None
        else concurrency
    )
    with ThreadPoolExecutor(max_workers=max(1, configured_concurrency)) as executor:
        futures = {
            executor.submit(start, request): index
            for index, request in enumerate(requests)
        }
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()
    return tuple(outcome for outcome in outcomes if outcome is not None)


def _scoped_request_text(base: StartJobRequest, target: str, source: str) -> str:
    provider = "GitHub issue" if source == "github" else "Linear issue"
    original = (base.utterance or base.text).strip()
    return (
        f"Work only on {provider} {target}. Do not work on any other ticket "
        "mentioned in the original request.\n\n"
        f"Original user request: {original}"
    )


def _rejected(reference: TicketReference, detail: str) -> TicketStartOutcome:
    return TicketStartOutcome(
        reference.label,
        "rejected",
        detail=_start_error_detail(HarnessError(detail)),
    )


def _github_target(
    reference: TicketReference,
    base: StartJobRequest,
    client: GitHubClient,
    *,
    foreground: bool,
) -> TicketJobRequest | TicketStartOutcome:
    assert reference.canonical is not None
    repository, separator, number_text = reference.canonical.rpartition("#")
    if not separator:
        return _rejected(reference, "GitHub issue reference is invalid")
    owner, repository_separator, name = repository.partition("/")
    if not repository_separator:
        return _rejected(reference, "GitHub issue reference is invalid")
    issue = GitHubIssue(owner, name, int(number_text))
    try:
        details = client.issue_details(issue)
    except GitHubError as exc:
        return _rejected(reference, str(exc))

    detail_number = details.get("number")
    if isinstance(detail_number, int) and detail_number != issue.number:
        return _rejected(reference, "GitHub returned a different issue number")
    detail_url = str(details.get("url") or "").strip()
    if detail_url:
        canonical_issue = github_issue_from_url(detail_url)
        if (
            canonical_issue is None
            or canonical_issue.number != issue.number
            or canonical_issue.name_with_owner.casefold()
            != issue.name_with_owner.casefold()
        ):
            return _rejected(reference, "GitHub returned a different issue identity")
        issue = canonical_issue

    target = issue.reference
    return TicketJobRequest(
        target,
        StartJobRequest(
            text=_scoped_request_text(base, target, "github"),
            repository=base.repository,
            github_repository=issue.name_with_owner,
            github_issue=issue.number,
            github_issue_context=format_issue_context(issue, details),
            agent=base.agent,
            utterance=f"Work only on GitHub issue {target}.",
            context_repository=issue.name_with_owner,
            foreground=foreground,
        ),
    )


def _linear_target(
    reference: TicketReference,
    base: StartJobRequest,
    *,
    foreground: bool,
) -> TicketJobRequest | TicketStartOutcome:
    canonical = resolve_issue_reference(reference.canonical)
    if canonical is None:
        return _rejected(
            reference,
            "Linear integration is disabled or the issue key is invalid",
        )
    return TicketJobRequest(
        canonical,
        StartJobRequest(
            text=_scoped_request_text(base, canonical, "linear"),
            repository=base.repository,
            agent=base.agent,
            utterance=f"Work only on Linear issue {canonical}.",
            context_repository=base.context_repository,
            issue_key=canonical,
            foreground=foreground,
        ),
    )


def _preflight_ticket_targets(
    extraction: TicketExtraction,
    base: StartJobRequest,
    *,
    foreground: bool,
) -> tuple[
    list[TicketStartOutcome | None],
    list[tuple[int, TicketJobRequest]],
]:
    """Validate every unique reference before returning any startable child."""
    slots: list[TicketStartOutcome | None] = [None] * len(extraction.references)
    prepared: list[tuple[int, TicketJobRequest]] = []
    github_client = GitHubClient()
    github_available = integration_enabled("github")
    linear_capability_error: str | None = None
    linear_capability_checked = False

    for index, reference in enumerate(extraction.references):
        if reference.error is not None:
            slots[index] = _rejected(reference, reference.error)
            continue
        if reference.canonical is None or reference.source is None:
            slots[index] = _rejected(reference, "ticket reference is invalid")
            continue
        if reference.source == "github":
            candidate: TicketJobRequest | TicketStartOutcome
            candidate = (
                _github_target(
                    reference,
                    base,
                    github_client,
                    foreground=foreground,
                )
                if github_available
                else _rejected(reference, "GitHub integration is disabled")
            )
        else:
            if not linear_capability_checked:
                try:
                    require_issue_capabilities(reference.canonical)
                except HarnessError as exc:
                    linear_capability_error = str(exc)
                linear_capability_checked = True
            candidate = (
                _rejected(reference, linear_capability_error)
                if linear_capability_error is not None
                else _linear_target(reference, base, foreground=foreground)
            )
        if isinstance(candidate, TicketStartOutcome):
            slots[index] = candidate
        else:
            prepared.append((index, candidate))
    return slots, prepared


def _ticket_start_summary(outcomes: tuple[TicketStartOutcome, ...]) -> str:
    parts: list[str] = []
    for outcome in outcomes:
        if outcome.status == "accepted":
            parts.append(f"{outcome.target}: accepted as job {outcome.job_id}")
        else:
            detail = f" ({outcome.detail})" if outcome.detail else ""
            parts.append(f"{outcome.target}: {outcome.status}{detail}")
    return "Ticket starts: " + "; ".join(parts) + "."


def _submit_extracted_targets(
    extraction: TicketExtraction,
    base: StartJobRequest,
    *,
    foreground: bool,
) -> tuple[TicketStartOutcome, ...]:
    slots, prepared = _preflight_ticket_targets(
        extraction,
        base,
        foreground=foreground,
    )
    started = start_jobs(tuple(request for _index, request in prepared))
    for (index, _request), outcome in zip(prepared, started, strict=True):
        slots[index] = outcome
    return tuple(outcome for outcome in slots if outcome is not None)


def reply_job(
    job_id: str,
    text: str,
    *,
    trusted_utterance: str | None = None,
    expected_question_id: str | None = None,
    expected_question_turn: str | None = None,
    answer_provenance: AnswerProvenance = AnswerProvenance.USER_TEXT,
    on_started: Callable[[], None] | None = None,
) -> str | None:
    now = time.time()
    should_launch = False
    should_cancel = False
    immediate: str | None = None

    def reply(job: CursorJob) -> CursorJob | None:
        nonlocal immediate, should_cancel, should_launch
        if job.status != JobStatus.AWAITING_USER:
            return None
        question = questions.current(job)
        if question is None:
            return None
        if (
            expected_question_id is not None and question.id != expected_question_id
        ) or (
            expected_question_turn is not None
            and question.origin.turn_token != expected_question_turn
        ):
            immediate = "That answer belongs to an older question, so I did not use it."
            should_launch = False
            return None
        resolution = resolve_answer(
            question,
            text,
            trusted_answer=trusted_utterance,
            provenance=answer_provenance,
        )
        if resolution.outcome == AnswerOutcome.REPEAT:
            immediate = question_prompt(question)
            should_launch = False
            return None
        if resolution.outcome == AnswerOutcome.DEFERRED:
            immediate = "Okay, I'll keep that question for later."
            should_launch = False
            return job.evolve(
                delivered=True,
                delivery_claim_token=None,
                delivery_claimed_at=None,
                updated_at=now,
                voice_question=questions.envelope(question, QuestionState.DEFERRED),
            )
        if resolution.outcome == AnswerOutcome.AMBIGUOUS:
            immediate = (
                choices_prompt(question)
                if question.choices
                else "I could not tell what your answer was. Please answer again."
            )
            should_launch = False
            return None
        if resolution.outcome == AnswerOutcome.REJECTED:
            immediate = (
                "That decision requires a direct user answer, so I did not use "
                "an automated response."
            )
            return None
        handler = questions.answer_handler(question.owner)
        if handler is None:
            immediate = (
                f"I cannot safely route an answer for question owner {question.owner}."
            )
            return None
        transition = handler(
            job,
            question,
            resolution,
            questions.AnswerContext(
                now=now,
                foreground_until=(
                    now + CURSOR_FOREGROUND_SECONDS + FOREGROUND_GRACE_SECONDS
                ),
                text=text,
                trusted_text=trusted_utterance,
            ),
        )
        if transition.cancel:
            should_cancel = True
            should_launch = False
            return recovery.stage_terminal_intent(
                job,
                JobStatus.CANCELLED,
                now=now,
                result=f"Cursor job {job_id} was cancelled.",
                voice_question=questions.envelope(question, QuestionState.CANCELLED),
            )
        should_launch = transition.launch
        immediate = transition.message
        return transition.job

    updated = _job_store().update(job_id, reply)
    if updated is None and immediate is None:
        raise HarnessError(f"Cursor job {job_id} is not waiting for a reply")
    if should_cancel:
        assert updated is not None
        if updated.target_release_pending:
            _cancel_target_and_release(
                job_id,
                updated.herdr_target or "",
                updated.target_release_token or "",
            )
        return None
    if should_launch:
        launch_worker(job_id)
        if on_started is not None:
            on_started()
    return immediate


def start_follow_up(
    parent_job_id: str,
    text: str,
    *,
    expected_completed_at: float | None = None,
    utterance: str | None = None,
    on_created: Callable[[], None] | None = None,
) -> str:
    """Create and launch a child job that reuses a completed parent's checkout."""
    now = time.time()
    child_id = uuid.uuid4().hex[:12]
    spoken = utterance if utterance is not None else text
    store = _job_store()

    def build(parent: CursorJob) -> CursorJob:
        active_issue_key = resolve_issue_reference(parent.issue_key)
        if active_issue_key:
            require_issue_capabilities(active_issue_key)
        return CursorJob.new(
            NewCursorJob(
                id=child_id,
                parent_job_id=parent.id,
                request=text,
                created_at=now,
                foreground_until=(
                    now + CURSOR_FOREGROUND_SECONDS + FOREGROUND_GRACE_SECONDS
                ),
                utterance=utterance,
                trusted_utterance=spoken,
                repository=parent.repository,
                context_repository=parent.repository,
                worktree_branch=parent.worktree_branch,
                worktree_path=parent.worktree_path,
                worktree_label=parent.worktree_label,
                worktree_workspace_id=parent.worktree_workspace_id,
                worktree_root_pane_id=parent.worktree_root_pane_id,
                worktree_provision_state="ready",
                issue_key=parent.issue_key,
                speakable_label=parent.speakable_label,
            )
        )

    created = store.create_follow_up(
        parent_job_id, build, expected_completed_at=expected_completed_at
    )
    if on_created is not None:
        on_created()
    launch_worker(created.id)
    return created.id


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
    cancelled_at = time.time()

    def cancel(job: CursorJob) -> CursorJob | None:
        nonlocal target, worker
        if job.status == JobStatus.CANCELLED:
            return None
        if job.terminal_intent_status == JobStatus.CANCELLED:
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
        question = questions.current(job)
        return recovery.stage_terminal_intent(
            job,
            JobStatus.CANCELLED,
            now=cancelled_at,
            result=f"Cursor job {job_id} was cancelled.",
            clear_worker=legacy_worker_stopped,
            voice_question=(
                questions.envelope(question, QuestionState.CANCELLED)
                if question is not None
                else None
            ),
        )

    updated = _job_store().update(job_id, cancel)
    if updated is None:
        current = read_job(job_id)
        return current.result or f"Cursor job {job_id} was cancelled."
    worker_stopped = True
    if worker is not None and worker.worker_token:
        worker_stopped = _stop_worker(worker)

    if updated.target_release_pending:
        _cancel_target_and_release(
            job_id,
            target,
            updated.target_release_token or "",
            worker_stopped=worker_stopped,
        )
    current = read_job(job_id)
    return (
        current.result
        or current.terminal_intent_result
        or f"Cursor job {job_id} was cancelled."
    )


def job_status(job_id: str | None = None) -> str:
    if job_id:
        job = read_job(job_id)
        workflow = (
            f", {job.workflow_tier.value} tier" if job.workflow_tier is not None else ""
        )
        return (
            f"Cursor job {job_id} is {job.status.value.replace('_', ' ')}, "
            f"in {job.workflow_phase.value.replace('_', ' ')}{workflow}."
        )
    jobs = [job for job in _job_store().list() if job.status in ACTIVE_STATUSES]
    if not jobs:
        return "There are no active Cursor jobs."
    return (
        "Active Cursor jobs: "
        + "; ".join(
            f"{job.id} is {job.status.value.replace('_', ' ')} "
            f"({job.workflow_phase.value.replace('_', ' ')})"
            for job in jobs
        )
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


def acknowledge_deliveries(claims: DeliveryClaims) -> list[DeliveryClaim]:
    return delivery.acknowledge_deliveries(_job_store(), claims)


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


def count_jobs() -> int:
    """Return how many durable Cursor jobs currently exist."""
    return len(_job_store().list())


def nuke_jobs() -> str:
    """Fence claims, drain workers, and delete only fully reconciled jobs."""
    store = _job_store()
    owner_pid = os.getpid()
    owner_boot = worker_lifecycle.boot_identity()
    owner_start = worker_lifecycle.process_identity(owner_pid)
    if not owner_boot or not owner_start:
        raise HarnessError("could not establish job deletion process identity")
    lease = MaintenanceLease(
        token=uuid.uuid4().hex,
        started_at=time.time(),
        owner_pid=owner_pid,
        owner_boot_id=owner_boot,
        owner_process_start=owner_start,
    )
    original: dict[str, CursorJob] = {}

    def stage(job: CursorJob) -> CursorJob | None:
        original[job.id] = job
        if (
            worker_lifecycle.has_legacy_worker_claim(job)
            or job.status not in ACTIVE_STATUSES
            or job.terminal_intent_status is not None
        ):
            return None
        return recovery.stage_terminal_intent(
            job,
            JobStatus.CANCELLED,
            now=time.time(),
            result=f"Cursor job {job.id} was cancelled before deletion.",
            preserve_worker_operation=True,
        )

    def maintenance_owner_alive(existing: MaintenanceLease) -> bool | None:
        return worker_lifecycle.process_owner_alive(
            existing.owner_pid,
            existing.owner_boot_id,
            existing.owner_process_start,
        )

    def abort_owned_lease() -> None:
        try:
            store.abort_maintenance(lease.token)
        except JobMaintenanceError:
            # A malformed or replaced fence is not ours to remove.
            pass

    try:
        store.begin_maintenance(
            lease,
            stage,
            owner_alive=maintenance_owner_alive,
        )
        stop_failures: list[str] = []
        stopped: dict[str, bool] = {}
        for snapshot in original.values():
            current = store.get(snapshot.id)
            if worker_lifecycle.has_legacy_worker_claim(current):
                disposition = worker_lifecycle.inspect_and_stop_legacy_worker(
                    current,
                    get_process_identity=worker_lifecycle.process_identity,
                    command_matches=worker_lifecycle.legacy_worker_command_matches,
                )
                stopped[current.id] = disposition != "unsafe"
                if disposition == "unsafe":
                    stop_failures.append(
                        f"{current.id}: legacy worker identity or exit could not "
                        "be verified"
                    )
                elif current.status in ACTIVE_STATUSES:

                    def stage_legacy(job: CursorJob) -> CursorJob | None:
                        if not worker_lifecycle.has_legacy_worker_claim(job):
                            return None
                        return recovery.stage_terminal_intent(
                            job,
                            JobStatus.CANCELLED,
                            now=time.time(),
                            result=(
                                f"Cursor job {job.id} was cancelled before deletion."
                            ),
                            clear_worker=True,
                            preserve_worker_operation=True,
                        )

                    store.update(current.id, stage_legacy)
                else:

                    def clear_legacy(job: CursorJob) -> CursorJob | None:
                        if not worker_lifecycle.has_legacy_worker_claim(job):
                            return None
                        return job.evolve(
                            worker_token=None,
                            worker_pid=None,
                            worker_boot_id=None,
                            worker_process_start=None,
                        )

                    store.update(current.id, clear_legacy)
                continue
            if not current.worker_token:
                stopped[current.id] = True
                continue
            ownership = (
                current.worker_token,
                current.worker_pid,
                current.worker_boot_id,
                current.worker_process_start,
            )
            worker_stopped = _stop_worker(current)
            latest = store.get(current.id)
            latest_ownership = (
                latest.worker_token,
                latest.worker_pid,
                latest.worker_boot_id,
                latest.worker_process_start,
            )
            if latest_ownership != ownership and any(
                value is not None for value in latest_ownership
            ):
                worker_stopped = False
                stop_failures.append(
                    f"{current.id}: worker ownership changed while stopping"
                )
            elif (
                latest.worker_pid is not None
                and latest.worker_boot_id is not None
                and latest.worker_process_start is not None
            ):
                owner_alive = worker_lifecycle.process_owner_alive(
                    latest.worker_pid,
                    latest.worker_boot_id,
                    latest.worker_process_start,
                )
                worker_stopped = owner_alive is False
                if not worker_stopped:
                    worker_stopped = False
                    reason = (
                        "worker is still running"
                        if owner_alive
                        else "worker exit could not be verified"
                    )
                    stop_failures.append(f"{current.id}: {reason}")
            elif not worker_stopped:
                stop_failures.append(
                    f"{current.id}: worker did not exit within the safe timeout"
                )
            if worker_stopped and latest_ownership == ownership:

                def clear_worker(
                    job: CursorJob,
                    expected_ownership: tuple[str | int | None, ...] = ownership,
                ) -> CursorJob | None:
                    current_ownership = (
                        job.worker_token,
                        job.worker_pid,
                        job.worker_boot_id,
                        job.worker_process_start,
                    )
                    if current_ownership != expected_ownership:
                        return None
                    return job.evolve(
                        worker_token=None,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                    )

                store.update(current.id, clear_worker)
            stopped[current.id] = worker_stopped

        for current in store.list():
            if not current.target_release_pending or not stopped.get(current.id, True):
                continue
            _cancel_target_and_release(
                current.id,
                current.herdr_target or "",
                current.target_release_token or "",
                worker_stopped=True,
            )

        if stop_failures:
            raise HarnessError(
                "Cursor jobs were preserved because workers could not be stopped "
                "safely: "
                + "; ".join(stop_failures)
                + ". Resolve the worker or external operation, then retry."
            )
        removed = store.finalize_maintenance(lease.token)
    except JobMaintenanceError as exc:
        abort_owned_lease()
        raise HarnessError(str(exc)) from exc
    except Exception:
        abort_owned_lease()
        raise

    count = len(removed)
    if not count:
        return "There were no Cursor jobs to delete."
    noun = "job" if count == 1 else "jobs"
    return f"Deleted all {count} Cursor {noun}."


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
    issue_key: str | None = None,
    issue_scope: str | None = None,
    issue_scope_source: str | None = None,
    action: str = "submit",
    job_id: str | None = None,
    reference: str | None = None,
    delivery_claims: DeliveryClaims | None = None,
) -> CursorTurnResult:
    on_follow_up_started: Callable[[], None] | None = None
    on_job_started: Callable[[], None] | None = None
    expected_question_id: str | None = None
    expected_question_turn: str | None = None
    answer_provenance = AnswerProvenance.AUTOMATION
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
        issue_key = request.issue_key
        issue_scope = request.issue_scope
        issue_scope_source = request.issue_scope_source
        action = request.action
        job_id = request.job_id
        reference = request.reference
        expected_question_id = request.expected_question_id
        expected_question_turn = request.expected_question_turn
        answer_provenance = request.answer_provenance
        expected_completed_at = request.expected_completed_at
        on_follow_up_started = request.on_follow_up_started
        on_job_started = request.on_job_started
    else:
        text = request
        expected_completed_at = None
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
        immediate = reply_job(
            reply_id,
            text,
            trusted_utterance=utterance,
            expected_question_id=expected_question_id,
            expected_question_turn=expected_question_turn,
            answer_provenance=answer_provenance,
            on_started=on_job_started,
        )
        if immediate is not None:
            pending = questions.current(read_job(reply_id))
            next_session = (
                None
                if pending is not None and pending.state == QuestionState.DEFERRED
                else reply_id
            )
            return CursorTurnResult(immediate, next_session)
        job_id = reply_id
    elif action == "follow_up":
        if not job_id:
            return CursorTurnResult(
                "I don't have a recent completed Cursor job to follow up on.", None
            )
        try:
            job_id = start_follow_up(
                job_id,
                text,
                expected_completed_at=expected_completed_at,
                utterance=utterance,
                on_created=on_follow_up_started,
            )
            if on_job_started is not None:
                on_job_started()
        except FollowUpCheckoutBusy:
            return CursorTurnResult(
                "That checkout is busy with another Cursor job right now.", None
            )
        except FollowUpUnavailable:
            return CursorTurnResult(
                "I can no longer follow up on that Cursor job.", None
            )
    else:
        extraction = extract_ticket_targets(
            utterance or text,
            scope_source=issue_scope_source,
            scope=issue_scope,
        )
        use_extracted_targets = extraction.batch_requested or bool(
            issue_scope
            and extraction.requested_count == 1
            and extraction.references
            and extraction.references[0].scoped
        )
        if use_extracted_targets:
            base = StartJobRequest(
                text=text,
                repository=repository,
                github_repository=github_repository,
                github_issue=github_issue,
                github_issue_context=github_issue_context,
                fork_requested=fork_requested,
                github_pull_request=github_pull_request,
                agent=agent,
                utterance=utterance,
                context_repository=context_repository,
                issue_key=issue_key,
                foreground=not extraction.batch_requested,
            )
            outcomes = _submit_extracted_targets(
                extraction,
                base,
                foreground=not extraction.batch_requested,
            )
            accepted = tuple(
                outcome for outcome in outcomes if outcome.status == "accepted"
            )
            if accepted and on_job_started is not None:
                on_job_started()
            if extraction.batch_requested or not accepted:
                return CursorTurnResult(_ticket_start_summary(outcomes), None)
            assert len(accepted) == 1 and accepted[0].job_id is not None
            return _await_foreground(accepted[0].job_id, delivery_claims)
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
            issue_key=issue_key,
        )
        if on_job_started is not None:
            on_job_started()
    return _await_foreground(job_id, delivery_claims)


def _await_foreground(
    job_id: str, delivery_claims: DeliveryClaims | None
) -> CursorTurnResult:
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
            awaiting = claimed if claimed is not None else job
            pending = questions.current(awaiting)
            rendered_question = (
                question_prompt(pending)
                if pending is not None
                else str(awaiting.question or awaiting.result or "").strip()
            )
            return CursorTurnResult(rendered_question, job_id)
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


# Compatibility aliases for callers migrating to the agent-neutral service API.
StartAgentJobRequest = StartJobRequest
AgentTurnRequest = CursorTurnRequest
AgentTurnResult = CursorTurnResult
agent_turn = cursor_turn
