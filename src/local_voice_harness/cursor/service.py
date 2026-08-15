from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, overload

from ..config import JOB_LOGS_DIR, JOBS_DIR, LEGACY_JOBS_DIR
from ..diagnostic_safety import redact_diagnostic
from ..errors import HarnessError
from ..integrations import registry as integration_registry_module
from ..integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubIssue,
    GitHubIssueLookupError,
    GitHubIssueLookupReason,
    GitHubProvider,
    format_issue_context,
    github_issue_from_url,
)
from ..integrations.herdr import HerdrClient, HerdrError
from ..integrations.linear import (
    LinearIssue,
    LinearIssueLookupError,
    LinearIssueLookupReason,
    parse_linear_issue_reference,
)
from ..integrations.registry import (
    IntegrationRegistry,
    extract_issue_reference,
    integration_enabled,
    issue_provider,
    issue_provider_identity,
    require_harness_capabilities,
    require_issue_capabilities,
    require_issue_provider,
    resolve_issue_reference,
    route_issue_repository,
)
from ..questions import (
    AnswerOutcome,
    AnswerProvenance,
    Question,
    QuestionError,
    QuestionIdentity,
    QuestionKind,
    QuestionOrigin,
    QuestionSensitivity,
    QuestionSpec,
    QuestionState,
    choices_prompt,
    question_prompt,
    resolve_answer,
    validate_question_identity,
)
from ..responses import (
    AssistantResponse,
    ResponseLike,
    spoken_utterance_slice,
    with_spoken_utterance_ack,
)
from ..ticket_targets import TicketExtraction, TicketReference, extract_ticket_targets
from ..user_config import PlatformSettings, default_user_config
from . import (
    announcements,
    delivery,
    inbox,
    outbox,
    provisioning,
    questions,
    recovery,
    worker_lifecycle,
)
from .delivery import DeliveryClaim, DeliveryClaims
from .model import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    WORKER_STATUSES,
    CursorJob,
    HarnessKind,
    JobStatus,
    JobValidationError,
    NewCursorJob,
)
from .provisioning import run_claimed_worker
from .store import (
    ActiveTicketConflict,
    FollowUpCheckoutBusy,
    FollowUpUnavailable,
    JobMaintenanceError,
    JobStore,
    MaintenanceLease,
    QuarantineEvidence,
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
    github_issue_create_requested: bool = False
    linear_team: str | None = None
    linear_ticket_create_requested: bool = False
    fork_requested: bool = False
    github_pull_request: int | None = None
    agent: str | None = None
    utterance: str | None = None
    context_repository: str | None = None
    issue_key: str | None = None
    foreground: bool = True
    harness_kind: HarnessKind | None = None


@dataclass(frozen=True, slots=True)
class CursorTurnRequest:
    text: str
    session_id: str | None = None
    repository: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_context: str | None = None
    github_issue_create_requested: bool = False
    linear_team: str | None = None
    linear_ticket_create_requested: bool = False
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
    harness_kind: HarnessKind | None = None
    answer_provenance: AnswerProvenance = AnswerProvenance.USER_TEXT
    expected_parent_revision: int | None = None
    expected_completed_at: float | None = None
    on_follow_up_started: Callable[[], None] | None = None
    on_job_started: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class CursorTurnResult:
    text: ResponseLike
    session_id: str | None
    mutated: bool = False

    def __iter__(self):
        # Keep the long-standing ``response, session = cursor_turn(...)`` contract.
        yield self.text
        yield self.session_id

    def __len__(self) -> int:
        return 2

    @overload
    def __getitem__(self, index: Literal[0]) -> ResponseLike: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> str | None: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ResponseLike | str | None, ...]: ...

    @overload
    def __getitem__(self, index: int) -> ResponseLike | str | None: ...

    def __getitem__(self, index: int | slice):
        return (self.text, self.session_id)[index]


TicketStartStatus = Literal[
    "accepted", "queued", "awaiting-clarification", "rejected", "start-failed"
]


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
    github_lookup_reason: GitHubIssueLookupReason | None = None
    linear_lookup_reason: LinearIssueLookupReason | None = None


GROUPED_REPOSITORY_OWNER = "grouped_repository"
GROUPED_REPOSITORY_TARGETS_FIELD = "grouped_repository_targets"
GROUPED_REPOSITORY_CANDIDATES_FIELD = "grouped_repository_candidates"
GROUPED_REPOSITORY_QUESTION_LIMIT = 500


def _job_store() -> JobStore:
    return JobStore(JOBS_DIR, LEGACY_JOBS_DIR)


def _integration_registry(
    registry: IntegrationRegistry | None,
) -> IntegrationRegistry:
    if registry is not None:
        return registry
    defaults = integration_registry_module._registry(None)
    return replace(defaults, github_client=GitHubClient)


def read_job(job_id: str) -> CursorJob:
    return _job_store().get(job_id)


def _speakable_label(job_id: str) -> str:
    try:
        return inbox.speakable_label_for(read_job(job_id))
    except (FileNotFoundError, OSError, JobValidationError):
        return "that job"


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
    try:
        worker_lifecycle.run_worker(
            _job_store(),
            job_id,
            claim_token,
            run_claimed_worker,
        )
    finally:
        _dispatch_waiting_jobs()


def launch_worker(job_id: str) -> None:
    def prepare_failure(job: CursorJob, message: str, failed_at: float) -> CursorJob:
        diagnostic = redact_diagnostic(message, limit=500)
        return job.evolve_for_delivery(
            now=failed_at,
            status=JobStatus.FAILED,
            error=diagnostic,
            result="Cursor job failed to start. Check the job log for details.",
            completed_at=failed_at,
            worker_token=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            participant_admission_state="released",
        )

    worker_lifecycle.launch_worker(
        _job_store(),
        JOB_LOGS_DIR,
        job_id,
        prepare_failure=prepare_failure,
        get_boot_identity=worker_lifecycle.boot_identity,
        get_process_identity=worker_lifecycle.process_identity,
    )


def _participant_capacity(integrations: IntegrationRegistry | None = None) -> int:
    registry = _integration_registry(integrations)
    runtime = registry.platform or default_user_config().platform
    return runtime.agent_job_start_concurrency


def _dispatch_waiting_jobs(
    *,
    integrations: IntegrationRegistry | None = None,
    requested_ids: tuple[str, ...] = (),
    capacity: int | None = None,
) -> frozenset[str]:
    """Claim global participant capacity and launch only admitted jobs."""

    store = _job_store()
    claimed = store.claim_participant_capacity(
        capacity if capacity is not None else _participant_capacity(integrations)
    )
    candidates = [*requested_ids, *(job.id for job in claimed)]
    launched: set[str] = set()
    for job_id in dict.fromkeys(candidates):
        try:
            current = store.get(job_id)
        except FileNotFoundError:
            continue
        if current.participant_admission_state != "held":
            continue
        if current.session_control == "user_owned":
            continue
        try:
            launch_worker(job_id)
        except Exception:
            continue
        launched.add(job_id)
    return frozenset(launched)


def _build_start_job(
    request: StartJobRequest,
    *,
    job_id: str,
    foreground_seconds: float,
    integrations: IntegrationRegistry | None,
) -> CursorJob:
    registry = _integration_registry(integrations)
    now = time.time()
    spoken_text = request.utterance or request.text
    candidate_issue_key = (
        None
        if (
            request.github_issue_create_requested
            or request.linear_ticket_create_requested
        )
        else (
            request.issue_key
            if request.issue_key is not None
            else extract_issue_reference(spoken_text, registry)
        )
    )
    resolved_issue_key = resolve_issue_reference(candidate_issue_key, registry)
    issue_provider = (
        issue_provider_identity(resolved_issue_key, registry)
        if resolved_issue_key
        else (
            "linear"
            if request.linear_ticket_create_requested
            else (
                "github"
                if (
                    request.github_issue is not None
                    or request.github_issue_create_requested
                )
                else None
            )
        )
    )
    if request.harness_kind is not None:
        harness_kind = request.harness_kind
    elif isinstance(registry.platform, PlatformSettings):
        harness_kind = HarnessKind(registry.platform.default_harness)
    else:
        harness_kind = HarnessKind.CURSOR
    require_harness_capabilities(
        harness_kind,
        provider=issue_provider,
        linear_ticket_create_requested=request.linear_ticket_create_requested,
        integrations=registry,
    )
    if resolved_issue_key:
        assert issue_provider is not None
        require_issue_capabilities(
            resolved_issue_key,
            registry,
            provider=issue_provider,
        )
    elif request.issue_key is not None:
        raise HarnessError("selected issue provider is unavailable")
    if issue_provider == "github" or request.linear_ticket_create_requested:
        require_issue_provider(issue_provider, registry)
    if harness_kind == HarnessKind.OPENCODE:
        client = registry.herdr_client()
        client.bind_harness_kind(harness_kind.value)
        client.require_harness_ready()
    issue_repository = (request.github_repository or "").strip()
    github_issue_url = (
        f"https://github.com/{issue_repository}/issues/{request.github_issue}"
        if issue_repository and request.github_issue
        else None
    )
    return CursorJob.new(
        NewCursorJob(
            id=job_id,
            request=request.text,
            created_at=now,
            foreground_until=(
                now + foreground_seconds + FOREGROUND_GRACE_SECONDS
                if request.foreground
                else 0
            ),
            utterance=request.utterance,
            trusted_utterance=spoken_text,
            repository_hint=request.repository,
            context_repository=request.context_repository,
            github_repository=request.github_repository,
            github_issue=request.github_issue,
            github_issue_url=github_issue_url,
            github_issue_context=request.github_issue_context,
            github_issue_create_requested=request.github_issue_create_requested,
            linear_ticket_create_requested=request.linear_ticket_create_requested,
            linear_ticket_create_team=request.linear_team,
            fork_requested=request.fork_requested,
            github_pull_request=request.github_pull_request,
            worktree_branch=(
                f"voice/github-{job_id}"
                if request.fork_requested
                else (
                    f"voice/github-issue-{request.github_issue}"
                    if request.github_issue
                    else (
                        f"voice/github-pr-{job_id}"
                        if request.github_pull_request
                        else None
                    )
                )
            ),
            worktree_label=(
                f"github-{job_id[:6]}"
                if request.fork_requested
                else (
                    f"issue-{request.github_issue}"
                    if request.github_issue
                    else (
                        f"pr-{request.github_pull_request}"
                        if request.github_pull_request
                        else None
                    )
                )
            ),
            pull_request_worktree_state=(
                "pending" if request.github_pull_request else None
            ),
            agent_hint=request.agent,
            harness_kind=harness_kind,
            issue_key=resolved_issue_key,
            issue_provider=issue_provider,
            speakable_label=inbox.build_speakable_label(
                request.text,
                issue_key=resolved_issue_key,
                github_repository=request.github_repository,
                github_issue=request.github_issue,
                github_pull_request=request.github_pull_request,
            ),
        )
    )


def start_job(
    request: StartJobRequest | str,
    *,
    repository: str | None = None,
    github_repository: str | None = None,
    github_issue: int | None = None,
    github_issue_context: str | None = None,
    github_issue_create_requested: bool = False,
    linear_team: str | None = None,
    linear_ticket_create_requested: bool = False,
    fork_requested: bool = False,
    github_pull_request: int | None = None,
    agent: str | None = None,
    utterance: str | None = None,
    context_repository: str | None = None,
    issue_key: str | None = None,
    foreground: bool = True,
    harness_kind: HarnessKind | None = None,
    foreground_seconds: float = 5.0,
    integrations: IntegrationRegistry | None = None,
) -> str:
    normalized = (
        request
        if isinstance(request, StartJobRequest)
        else StartJobRequest(
            request,
            repository=repository,
            github_repository=github_repository,
            github_issue=github_issue,
            github_issue_context=github_issue_context,
            github_issue_create_requested=github_issue_create_requested,
            linear_team=linear_team,
            linear_ticket_create_requested=linear_ticket_create_requested,
            fork_requested=fork_requested,
            github_pull_request=github_pull_request,
            agent=agent,
            utterance=utterance,
            context_repository=context_repository,
            issue_key=issue_key,
            foreground=foreground,
            harness_kind=harness_kind,
        )
    )
    job_id = uuid.uuid4().hex[:12]
    job = _build_start_job(
        normalized,
        job_id=job_id,
        foreground_seconds=foreground_seconds,
        integrations=integrations,
    )
    _job_store().create(job, enforce_unique_ticket=True)
    _dispatch_waiting_jobs(integrations=integrations)
    return job_id


def _start_error_detail(error: BaseException) -> str:
    return redact_diagnostic(str(error) or type(error).__name__, limit=240)


def start_jobs(
    requests: tuple[TicketJobRequest, ...],
    *,
    concurrency: int = 3,
    foreground_seconds: float = 5.0,
    integrations: IntegrationRegistry | None = None,
) -> tuple[TicketStartOutcome, ...]:
    """Create durable jobs and dispatch them through global participant capacity."""
    if not requests:
        return ()
    outcomes: list[TicketStartOutcome | None] = [None] * len(requests)

    def start(request: TicketJobRequest) -> TicketStartOutcome:
        try:
            job_id = start_job(
                request.request,
                foreground_seconds=foreground_seconds,
                integrations=integrations,
            )
        except ActiveTicketConflict as exc:
            return TicketStartOutcome(
                request.target,
                "rejected",
                job_id=exc.active_job_id,
                detail=_start_error_detail(exc),
            )
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
        status = (
            "queued"
            if current is not None and current.participant_admission_state == "waiting"
            else "accepted"
        )
        return TicketStartOutcome(request.target, status, job_id=job_id)

    for index, request in enumerate(requests):
        outcomes[index] = start(request)
    return tuple(outcome for outcome in outcomes if outcome is not None)


def _scoped_request_text(base: StartJobRequest, target: str, source: str) -> str:
    provider = "GitHub issue" if source == "github" else "Linear issue"
    original = (base.utterance or base.text).strip()
    return (
        f"Work only on {provider} {target}. Do not work on any other ticket "
        "mentioned in the original request.\n\n"
        f"Original user request: {original}"
    )


def _rejected(
    reference: TicketReference,
    detail: str,
    *,
    github_lookup_reason: GitHubIssueLookupReason | None = None,
    linear_lookup_reason: LinearIssueLookupReason | None = None,
) -> TicketStartOutcome:
    return TicketStartOutcome(
        reference.label,
        "rejected",
        detail=_start_error_detail(HarnessError(detail)),
        github_lookup_reason=github_lookup_reason,
        linear_lookup_reason=linear_lookup_reason,
    )


def resolve_linear_issue(
    reference: str,
    integrations: IntegrationRegistry,
) -> LinearIssue:
    """Resolve one Linear issue's existence and accessibility before admission."""

    parsed = parse_linear_issue_reference(reference)
    integration = issue_provider("linear", integrations)
    resolve = getattr(integration, "resolve_issue", None)
    factory = getattr(integrations, "herdr_client", None)
    if not callable(resolve) or not callable(factory):
        raise LinearIssueLookupError(
            LinearIssueLookupReason.TRANSIENT,
            "Linear lookup client is unavailable",
        )
    resolved = resolve(factory(), parsed.identifier)
    if not isinstance(resolved, LinearIssue):
        raise LinearIssueLookupError(
            LinearIssueLookupReason.UNKNOWN,
            "Linear lookup returned an invalid issue identity",
        )
    return resolved


def _github_target(
    reference: TicketReference,
    base: StartJobRequest,
    provider: GitHubProvider,
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
        details = provider.resolve_issue(issue)
    except GitHubError as exc:
        if isinstance(exc, GitHubIssueLookupError):
            return _rejected(
                reference,
                exc.voice_message,
                github_lookup_reason=exc.reason,
            )
        return _rejected(
            reference,
            str(exc),
            github_lookup_reason=GitHubIssueLookupReason.UNKNOWN,
        )

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
            harness_kind=base.harness_kind,
        ),
    )


def _linear_target(
    reference: TicketReference,
    base: StartJobRequest,
    *,
    foreground: bool,
    integrations: IntegrationRegistry,
) -> TicketJobRequest | TicketStartOutcome:
    try:
        parsed = parse_linear_issue_reference(reference.canonical or reference.raw)
    except LinearIssueLookupError as exc:
        return _rejected(
            reference,
            exc.voice_message,
            linear_lookup_reason=exc.reason,
        )
    canonical = resolve_issue_reference(parsed.identifier, integrations)
    if canonical is None:
        return _rejected(
            reference,
            "Linear integration is disabled or the issue key is invalid",
        )
    try:
        issue = resolve_linear_issue(canonical, integrations)
    except LinearIssueLookupError as exc:
        return _rejected(
            reference,
            exc.voice_message,
            linear_lookup_reason=exc.reason,
        )
    except HarnessError as exc:
        return _rejected(
            reference,
            str(exc),
            linear_lookup_reason=LinearIssueLookupReason.UNKNOWN,
        )
    identity = issue.identifier
    return TicketJobRequest(
        identity,
        StartJobRequest(
            text=_scoped_request_text(base, identity, "linear"),
            repository=base.repository,
            agent=base.agent,
            utterance=f"Work only on Linear issue {identity}.",
            context_repository=base.context_repository,
            issue_key=identity,
            foreground=foreground,
            harness_kind=base.harness_kind,
        ),
    )


def _preflight_ticket_targets(
    extraction: TicketExtraction,
    base: StartJobRequest,
    *,
    foreground: bool,
    integrations: IntegrationRegistry,
) -> tuple[
    list[TicketStartOutcome | None],
    list[tuple[int, TicketJobRequest]],
]:
    """Validate every unique reference before returning any startable child."""
    slots: list[TicketStartOutcome | None] = [None] * len(extraction.references)
    prepared: list[tuple[int, TicketJobRequest]] = []
    github_available = integration_enabled("github", integrations)
    github_provider = (
        GitHubProvider(integrations.github_client()) if github_available else None
    )
    linear_capability_error: str | None = None
    linear_capability_checked = False

    for index, reference in enumerate(extraction.references):
        if reference.error is not None:
            linear_reason = None
            if reference.source == "linear":
                try:
                    parse_linear_issue_reference(reference.canonical or reference.raw)
                except LinearIssueLookupError as exc:
                    linear_reason = exc.reason
            slots[index] = _rejected(
                reference,
                reference.error,
                linear_lookup_reason=linear_reason,
            )
            continue
        if reference.canonical is None or reference.source is None:
            slots[index] = _rejected(reference, "ticket reference is invalid")
            continue
        if reference.source == "github":
            candidate: TicketJobRequest | TicketStartOutcome
            if github_available:
                assert github_provider is not None
                candidate = _github_target(
                    reference,
                    base,
                    github_provider,
                    foreground=foreground,
                )
            else:
                candidate = _rejected(reference, "GitHub integration is disabled")
        else:
            if not linear_capability_checked:
                try:
                    require_issue_capabilities(reference.canonical, integrations)
                except HarnessError as exc:
                    linear_capability_error = str(exc)
                linear_capability_checked = True
            candidate = (
                _rejected(reference, linear_capability_error)
                if linear_capability_error is not None
                else _linear_target(
                    reference,
                    base,
                    foreground=foreground,
                    integrations=integrations,
                )
            )
        if isinstance(candidate, TicketStartOutcome):
            slots[index] = candidate
        else:
            prepared.append((index, candidate))
    return slots, prepared


def _serialize_start_request(request: StartJobRequest) -> dict[str, object]:
    return {
        "text": request.text,
        "repository": request.repository,
        "github_repository": request.github_repository,
        "github_issue": request.github_issue,
        "github_issue_context": request.github_issue_context,
        "fork_requested": request.fork_requested,
        "github_pull_request": request.github_pull_request,
        "agent": request.agent,
        "utterance": request.utterance,
        "context_repository": request.context_repository,
        "issue_key": request.issue_key,
        "foreground": request.foreground,
        "harness_kind": (
            request.harness_kind.value if request.harness_kind is not None else None
        ),
    }


def _deserialize_start_request(raw: object) -> StartJobRequest | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        return None
    string_fields = (
        "repository",
        "github_repository",
        "github_issue_context",
        "agent",
        "utterance",
        "context_repository",
        "issue_key",
    )
    if any(
        raw.get(field) is not None and not isinstance(raw.get(field), str)
        for field in string_fields
    ):
        return None
    integer_fields = ("github_issue", "github_pull_request")
    if any(
        raw.get(field) is not None
        and (not isinstance(raw.get(field), int) or isinstance(raw.get(field), bool))
        for field in integer_fields
    ):
        return None
    if not isinstance(raw.get("fork_requested", False), bool) or not isinstance(
        raw.get("foreground", False), bool
    ):
        return None
    raw_kind = raw.get("harness_kind")
    harness_kind: HarnessKind | None
    if raw_kind is None:
        harness_kind = None
    elif isinstance(raw_kind, str):
        try:
            harness_kind = HarnessKind(raw_kind)
        except ValueError:
            return None
    else:
        return None
    return StartJobRequest(
        text=str(raw["text"]),
        repository=raw.get("repository"),
        github_repository=raw.get("github_repository"),
        github_issue=raw.get("github_issue"),
        github_issue_context=raw.get("github_issue_context"),
        fork_requested=bool(raw.get("fork_requested", False)),
        github_pull_request=raw.get("github_pull_request"),
        agent=raw.get("agent"),
        utterance=raw.get("utterance"),
        context_repository=raw.get("context_repository"),
        issue_key=raw.get("issue_key"),
        foreground=bool(raw.get("foreground", False)),
        harness_kind=harness_kind,
    )


def _grouped_repository_question(
    targets: list[dict[str, object]], repositories: list[Path]
) -> str:
    identities = ", ".join(str(target["target"]) for target in targets)
    question = (
        "Reply TARGET: REPOSITORY for each ticket. "
        f"Targets in request order: {identities}. "
        "Use an available local repository name or path."
    )
    if len(question) > GROUPED_REPOSITORY_QUESTION_LIMIT:
        raise HarnessError("grouped repository target identities exceed question limit")

    names = [repository.name for repository in repositories]
    if not names:
        return question
    prefix = " Available repositories include: "
    budget = GROUPED_REPOSITORY_QUESTION_LIMIT - len(question) - len(prefix)
    if budget <= 0:
        return question

    included: list[str] = []
    used = 0
    for name in names:
        separator = 2 if included else 0
        if used + separator + len(name) > budget:
            break
        included.append(name)
        used += separator + len(name)
    if not included:
        return question

    summary = ", ".join(included)
    omitted = len(names) - len(included)
    if omitted:
        suffix = f" (+{omitted} more)"
        while included and len(summary) + len(suffix) > budget:
            included.pop()
            summary = ", ".join(included)
            omitted = len(names) - len(included)
            suffix = f" (+{omitted} more)"
        if not included:
            return question
        summary += suffix
    return question + prefix + summary


def _create_grouped_repository_clarification(
    targets: list[dict[str, object]],
    repositories: list[Path],
    *,
    original_request: StartJobRequest,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    question = _grouped_repository_question(targets, repositories)
    coordinator = CursorJob.new(
        NewCursorJob(
            id=job_id,
            request=original_request.text,
            utterance=original_request.utterance,
            trusted_utterance=original_request.utterance or original_request.text,
            created_at=now,
            foreground_until=0,
            speakable_label="grouped ticket repository clarification",
        )
    )
    coordinator = CursorJob.from_dict(
        {
            **coordinator.to_dict(),
            "participant_admission_state": "released",
        }
    )
    envelope = Question(
        id=uuid.uuid4().hex,
        text=question,
        kind=QuestionKind.FREE_TEXT,
        sensitivity=QuestionSensitivity.ROUTINE,
        origin=QuestionOrigin(
            provider="cursor",
            job_id=job_id,
            turn_token=f"{job_id}-repository-group",
        ),
        owner=GROUPED_REPOSITORY_OWNER,
        asked_at=now,
    )
    coordinator_values = coordinator.to_dict()
    coordinator_values.update(
        {
            "status": JobStatus.AWAITING_USER.value,
            "question": question,
            "result": question,
            "clarification_kind": GROUPED_REPOSITORY_OWNER,
            "voice_question": envelope.to_dict(),
            GROUPED_REPOSITORY_TARGETS_FIELD: targets,
            GROUPED_REPOSITORY_CANDIDATES_FIELD: [
                str(repository) for repository in repositories
            ],
        }
    )
    coordinator = CursorJob.from_dict(coordinator_values)
    _job_store().create(coordinator)
    return job_id


def _preflight_batch_repositories(
    prepared: list[tuple[int, TicketJobRequest]],
    slots: list[TicketStartOutcome | None],
    *,
    integrations: IntegrationRegistry,
) -> tuple[list[tuple[int, TicketJobRequest]], list[dict[str, object]], list[Path]]:
    """Resolve provider-backed repositories without allowing child Rofi prompts."""
    routable = [
        item
        for item in prepared
        if item[1].request.issue_key and not item[1].request.repository
    ]
    if not routable:
        return prepared, [], []

    startable: list[tuple[int, TicketJobRequest]] = []
    ambiguous: list[dict[str, object]] = []
    routable_indexes = {index for index, _request in routable}
    client = HerdrClient()
    try:
        repositories = client.repository_roots()
        reserved = provisioning.reserved_targets(_job_store())
    except (HarnessError, HerdrError) as exc:
        for index, ticket in prepared:
            if index in routable_indexes:
                slots[index] = TicketStartOutcome(
                    ticket.target,
                    "rejected",
                    detail=_start_error_detail(exc),
                )
            else:
                startable.append((index, ticket))
        return startable, [], []

    for index, ticket in prepared:
        if index not in routable_indexes:
            startable.append((index, ticket))
            continue
        issue_key = ticket.request.issue_key
        assert issue_key is not None
        try:
            routed = route_issue_repository(
                client,
                issue_key,
                repositories,
                token=f"batch-{uuid.uuid4().hex[:12]}",
                reserved=reserved,
                integrations=integrations,
                provider="linear",
            )
        except (HarnessError, HerdrError) as exc:
            slots[index] = TicketStartOutcome(
                ticket.target,
                "rejected",
                detail=_start_error_detail(exc),
            )
            continue
        if routed is not None and routed[0] is not None:
            repository = routed[0]
            startable.append(
                (
                    index,
                    TicketJobRequest(
                        ticket.target,
                        replace(ticket.request, repository=str(repository)),
                    ),
                )
            )
            continue
        ambiguous.append(
            {
                "index": index,
                "target": ticket.target,
                "request": _serialize_start_request(ticket.request),
            }
        )
    return startable, ambiguous, repositories


def _counted(count: int, singular: str, plural: str | None = None) -> str:
    quantity = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
    }.get(count, str(count))
    return f"{quantity} {singular if count == 1 else plural or singular + 's'}"


def _ticket_start_spoken(
    outcomes: tuple[TicketStartOutcome, ...],
    utterance: str | None = None,
) -> str:
    lookup_reason = outcomes[0].github_lookup_reason if len(outcomes) == 1 else None
    if lookup_reason is not None:
        outcome = outcomes[0]
        _repository, separator, number = outcome.target.rpartition("#")
        identity = f"GitHub issue {number}" if separator else "that GitHub issue"
        messages = {
            GitHubIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE: (
                f"I couldn't find or access {identity}."
            ),
            GitHubIssueLookupReason.UNAUTHORIZED: (
                f"I couldn't access {identity} because GitHub authorization "
                "is required."
            ),
            GitHubIssueLookupReason.TRANSIENT: (
                f"GitHub is temporarily unavailable while checking {identity}."
            ),
            GitHubIssueLookupReason.UNKNOWN: f"I couldn't verify {identity}.",
        }
        return messages.get(
            lookup_reason,
            f"I couldn't verify {identity}.",
        )
    if len(outcomes) == 1 and outcomes[0].linear_lookup_reason is not None:
        return LinearIssueLookupError(
            outcomes[0].linear_lookup_reason,
            outcomes[0].detail or "",
        ).voice_message
    accepted = sum(outcome.status == "accepted" for outcome in outcomes)
    queued = sum(outcome.status == "queued" for outcome in outcomes)
    github_lookup_failures = sum(
        outcome.github_lookup_reason is not None for outcome in outcomes
    )
    linear_lookup_failures = sum(
        outcome.linear_lookup_reason is not None for outcome in outcomes
    )
    rejected = sum(
        outcome.status == "rejected"
        and outcome.github_lookup_reason is None
        and outcome.linear_lookup_reason is None
        for outcome in outcomes
    )
    start_failed = sum(outcome.status == "start-failed" for outcome in outcomes)
    awaiting = sum(outcome.status == "awaiting-clarification" for outcome in outcomes)
    parts: list[str] = []
    if accepted:
        parts.append(f"{_counted(accepted, 'job')} started")
    if queued:
        parts.append(f"{_counted(queued, 'job')} accepted and queued")
    if github_lookup_failures:
        parts.append(
            f"{_counted(github_lookup_failures, 'GitHub issue')} could not be accessed"
        )
    if linear_lookup_failures:
        parts.append(
            f"{_counted(linear_lookup_failures, 'Linear issue')} could not be accessed"
        )
    if rejected:
        parts.append(f"{_counted(rejected, 'ticket')} rejected")
    if start_failed:
        parts.append(f"{_counted(start_failed, 'job')} failed to start")
    if awaiting:
        parts.append(
            f"{_counted(awaiting, 'ticket')} waiting for one grouped clarification"
        )
    if not parts:
        return "No ticket jobs were started."
    sentence = "; ".join(parts) + "."
    sentence = sentence[0].upper() + sentence[1:]
    if accepted or queued:
        return with_spoken_utterance_ack(sentence, utterance or "")
    return sentence


def _ticket_display_detail(outcome: TicketStartOutcome) -> str | None:
    if outcome.github_lookup_reason is not None:
        classified = {
            GitHubIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE: (
                "issue not found or inaccessible"
            ),
            GitHubIssueLookupReason.UNAUTHORIZED: "GitHub authorization required",
            GitHubIssueLookupReason.TRANSIENT: "GitHub temporarily unavailable",
            GitHubIssueLookupReason.UNKNOWN: "GitHub issue could not be verified",
        }
        return classified[outcome.github_lookup_reason]
    if outcome.linear_lookup_reason is not None:
        classified = {
            LinearIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE: (
                "issue not found or inaccessible"
            ),
            LinearIssueLookupReason.UNAUTHORIZED: "Linear authorization required",
            LinearIssueLookupReason.TRANSIENT: "Linear temporarily unavailable",
            LinearIssueLookupReason.MALFORMED: "Linear issue key is malformed",
            LinearIssueLookupReason.NONPOSITIVE: "Linear issue number must be positive",
            LinearIssueLookupReason.UNKNOWN: "Linear issue could not be verified",
        }
        return classified[outcome.linear_lookup_reason]
    if outcome.status == "start-failed":
        if outcome.job_id:
            return f"job {outcome.job_id} failed to start; check its log"
        return "job could not be started; check the harness logs"
    if outcome.status == "rejected":
        if outcome.job_id:
            return f"active job {outcome.job_id} already exists"
        return "request could not be accepted; verify the ticket reference and access"
    if outcome.status == "awaiting-clarification":
        return "waiting for a repository in the grouped clarification"
    if outcome.status == "queued":
        return "accepted and queued for participant capacity"
    return None


def _ticket_start_summary(
    outcomes: tuple[TicketStartOutcome, ...],
    utterance: str | None = None,
) -> AssistantResponse:
    parts: list[str] = []
    for outcome in outcomes:
        if outcome.status in {"accepted", "queued"}:
            state = (
                "accepted" if outcome.status == "accepted" else "accepted and queued"
            )
            parts.append(f"{outcome.target}: {state} as job {outcome.job_id}")
        else:
            display_detail = _ticket_display_detail(outcome)
            detail = f" ({display_detail})" if display_detail else ""
            parts.append(f"{outcome.target}: {outcome.status}{detail}")
    display = "Ticket starts: " + "; ".join(parts) + "."
    full = " ".join((utterance or "").split())
    spoken_slice = spoken_utterance_slice(utterance or "")
    if full and full != spoken_slice:
        display = f"{display} Request: {full}"
    return AssistantResponse(
        spoken_text=_ticket_start_spoken(outcomes, utterance),
        display_text=display,
    )


def _queued_accept_response(
    label: str,
    job_id: str,
    utterance: str | None,
) -> AssistantResponse:
    display = f"Cursor job {job_id} was accepted and queued."
    full = " ".join((utterance or "").split())
    if full:
        display = f"{display} Request: {full}"
    return AssistantResponse(
        spoken_text=with_spoken_utterance_ack(
            f"Cursor accepted {label} and queued it.",
            utterance or "",
        ),
        display_text=display,
    )


def _submit_extracted_targets(
    extraction: TicketExtraction,
    base: StartJobRequest,
    *,
    foreground: bool,
    foreground_seconds: float,
    concurrency: int,
    integrations: IntegrationRegistry,
) -> tuple[TicketStartOutcome, ...]:
    slots, prepared = _preflight_ticket_targets(
        extraction,
        base,
        foreground=foreground,
        integrations=integrations,
    )
    ambiguous: list[dict[str, object]] = []
    repositories: list[Path] = []
    if extraction.batch_requested:
        prepared, ambiguous, repositories = _preflight_batch_repositories(
            prepared,
            slots,
            integrations=integrations,
        )
        available: list[dict[str, object]] = []
        store = _job_store()
        for target in ambiguous:
            request = _deserialize_start_request(target.get("request"))
            identity = (
                ("linear", request.issue_key.casefold())
                if request is not None and request.issue_key
                else None
            )
            owner = (
                store.ticket_reservation_owner(identity)
                if identity is not None
                else None
            )
            if owner is None:
                available.append(target)
                continue
            raw_index = target.get("index")
            if not isinstance(raw_index, int):
                raise HarnessError("grouped clarification target index is invalid")
            slots[raw_index] = TicketStartOutcome(
                str(target.get("target") or ""),
                "rejected",
                job_id=owner,
                detail=f"ticket is already active as Cursor job {owner}",
            )
        ambiguous = available
    clarification_job_id: str | None = None
    if ambiguous:
        clarification_job_id = _create_grouped_repository_clarification(
            ambiguous,
            repositories,
            original_request=base,
        )
        for target in ambiguous:
            raw_index = target["index"]
            if not isinstance(raw_index, int):
                raise HarnessError("grouped clarification target index is invalid")
            slots[raw_index] = TicketStartOutcome(
                str(target["target"]),
                "awaiting-clarification",
                job_id=clarification_job_id,
            )
    started = start_jobs(
        tuple(request for _index, request in prepared),
        concurrency=concurrency,
        foreground_seconds=foreground_seconds,
        integrations=integrations,
    )
    for (index, _request), outcome in zip(prepared, started, strict=True):
        slots[index] = outcome
    return tuple(outcome for outcome in slots if outcome is not None)


def _grouped_repository_assignments(
    text: str, targets: list[dict[str, object]]
) -> dict[str, str]:
    identities = [str(target.get("target") or "") for target in targets]
    identities = [identity for identity in identities if identity]
    if not identities:
        return {}
    identity_pattern = "|".join(
        re.escape(identity) for identity in sorted(identities, key=len, reverse=True)
    )
    marker = re.compile(
        rf"(?P<target>{identity_pattern})\s*(?::|=|\bis\b|\buses?\b)\s*",
        re.IGNORECASE,
    )
    matches = list(marker.finditer(text))
    assignments: dict[str, str] = {}
    duplicates: set[str] = set()
    canonical = {identity.casefold(): identity for identity in identities}
    for position, match in enumerate(matches):
        target = canonical[match.group("target").casefold()]
        end = (
            matches[position + 1].start() if position + 1 < len(matches) else len(text)
        )
        value = text[match.end() : end].strip(" \t\r\n,;.")
        value = re.sub(r"^(?:repository|repo)\s+", "", value, flags=re.IGNORECASE)
        if target in assignments:
            duplicates.add(target)
        elif value:
            assignments[target] = value
    for target in duplicates:
        assignments.pop(target, None)
    return assignments


def _reply_repository_list(
    job: CursorJob,
    *,
    expected_question_id: str | None,
    expected_question_turn: str | None,
) -> str:
    question = questions.current(job)
    if question is None:
        raise HarnessError(f"Cursor job {job.id} has no repository clarification")
    try:
        validate_question_identity(
            question,
            QuestionIdentity(
                job.id,
                expected_question_id or question.id,
                expected_question_turn or question.origin.turn_token,
            ),
        )
    except QuestionError:
        return "That answer belongs to an older question, so I did not use it."
    remaining = list(job.grouped_repository_candidates or ())
    page, rest = provisioning.repository_name_page(remaining)
    if not page:
        return (
            "I don't have more repository names to list. "
            "Say a local repository name or path."
        )
    now = time.time()
    next_question = provisioning.repository_question(
        [],
        names=page,
        remaining=len(rest),
    )

    def update(current: CursorJob) -> CursorJob | None:
        pending = questions.current(current)
        if (
            current.status != JobStatus.AWAITING_USER
            or pending is None
            or pending.id != question.id
        ):
            return None
        return questions.ask(
            current,
            QuestionSpec(
                next_question,
                sensitivity=QuestionSensitivity.ROUTINE,
            ),
            owner="repository",
            turn_token=f"{current.id}-repository-list-{current.revision + 1}",
            now=now,
            job_changes={
                "participant_admission_state": "waiting",
                "grouped_repository_candidates": rest,
            },
        )

    updated = _job_store().update(job.id, update)
    if updated is None:
        return "That answer belongs to an older question, so I did not use it."
    return next_question


def _reply_grouped_repository(
    job: CursorJob,
    text: str,
    *,
    trusted_utterance: str | None,
    expected_question_id: str | None,
    expected_question_turn: str | None,
    on_started: Callable[[], None] | None,
    foreground_seconds: float,
    concurrency: int,
    integrations: IntegrationRegistry | None,
) -> str:
    question = questions.current(job)
    if question is None:
        raise HarnessError(f"Cursor job {job.id} has no grouped clarification")
    try:
        validate_question_identity(
            question,
            QuestionIdentity(
                job.id,
                expected_question_id or question.id,
                expected_question_turn or question.origin.turn_token,
            ),
        )
    except QuestionError:
        return "That answer belongs to an older question, so I did not use it."
    raw_targets = job.grouped_repository_targets
    raw_candidates = job.grouped_repository_candidates
    if raw_targets is None or raw_candidates is None:
        raise HarnessError("grouped repository clarification state is invalid")
    targets = raw_targets
    candidates = [Path(candidate) for candidate in raw_candidates]
    assignments = _grouped_repository_assignments(
        trusted_utterance or text,
        targets,
    )
    client = HerdrClient()
    selected: dict[str, Path] = {}
    for target in targets:
        identity = str(target.get("target") or "")
        answer = assignments.get(identity)
        if answer is None:
            continue
        repository, _matches = client.resolve_repository(answer, "", candidates)
        if repository is not None:
            selected[identity] = repository
    if not selected:
        return (
            "I could not map any canonical ticket target to one available "
            "repository. No ticket jobs were started."
        )

    requests: list[TicketJobRequest] = []
    children: list[CursorJob] = []
    launch_records: list[dict[str, object]] = []
    resolved_targets: set[str] = set()
    for target in targets:
        identity = str(target.get("target") or "")
        repository = selected.get(identity)
        request = _deserialize_start_request(target.get("request"))
        if repository is None or request is None:
            continue
        selected_request = replace(
            request,
            repository=str(repository),
            foreground=False,
        )
        child_id = uuid.uuid4().hex[:12]
        requests.append(TicketJobRequest(identity, selected_request))
        child = _build_start_job(
            selected_request,
            job_id=child_id,
            foreground_seconds=foreground_seconds,
            integrations=integrations,
        )
        child_values = child.to_dict()
        child_values["grouped_repository_coordinator_id"] = job.id
        children.append(CursorJob.from_dict(child_values))
        launch_records.append(
            {
                "target": identity,
                "job_id": child_id,
                "state": "pending",
            }
        )
        resolved_targets.add(identity)
    if not requests:
        return "The grouped clarification is invalid. No ticket jobs were started."

    remaining = [
        target
        for target in targets
        if str(target.get("target") or "") not in resolved_targets
    ]
    now = time.time()

    def record(current: CursorJob) -> CursorJob | None:
        pending = questions.current(current)
        if (
            current.status != JobStatus.AWAITING_USER
            or pending is None
            or pending.id != question.id
        ):
            return None
        if remaining:
            return questions.ask(
                current,
                QuestionSpec(
                    _grouped_repository_question(remaining, candidates),
                    sensitivity=QuestionSensitivity.ROUTINE,
                ),
                owner=GROUPED_REPOSITORY_OWNER,
                turn_token=f"{current.id}-repository-group-{current.revision + 1}",
                now=now,
                job_changes={
                    GROUPED_REPOSITORY_TARGETS_FIELD: remaining,
                    "grouped_repository_launches": [
                        *current.grouped_repository_launches,
                        *launch_records,
                    ],
                },
            )
        return current.evolve_for_delivery(
            now=now,
            status=JobStatus.COMPLETED,
            question=None,
            clarification_kind=None,
            result="Grouped ticket repository clarification was applied.",
            completed_at=now,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            voice_question=questions.envelope(
                pending,
                QuestionState.RESOLVED,
                job=current,
                answer=text,
                trusted_answer=trusted_utterance or text,
                answered_at=now,
            ),
            grouped_repository_targets=[],
            grouped_repository_launches=[
                *current.grouped_repository_launches,
                *launch_records,
            ],
        )

    store = _job_store()
    try:
        updated = store.stage_grouped_children(job.id, record, tuple(children))
    except JobValidationError as exc:
        if "resource is reserved" not in str(exc):
            raise
        return (
            "One or more tickets became active before this answer was applied. "
            "No duplicate ticket jobs were started."
        )
    if updated is None:
        return "That grouped clarification was already changed, so I did not use it."
    outcomes = _launch_grouped_children(
        job.id,
        concurrency=concurrency,
    )
    if (
        any(outcome.status in {"accepted", "queued"} for outcome in outcomes)
        and on_started
    ):
        on_started()
    message = _ticket_start_spoken(outcomes, trusted_utterance)
    if remaining:
        message += (
            f" {_counted(len(remaining), 'ticket')} still needs a repository in "
            "the grouped clarification."
        )
    return message


def _launch_grouped_children(
    coordinator_id: str,
    *,
    concurrency: int,
) -> tuple[TicketStartOutcome, ...]:
    store = _job_store()
    coordinator = store.get(coordinator_id)
    pending = [
        launch
        for launch in coordinator.grouped_repository_launches
        if launch.get("state") == "pending"
        and isinstance(launch.get("target"), str)
        and isinstance(launch.get("job_id"), str)
    ]
    if not pending:
        return ()
    child_ids = tuple(str(entry["job_id"]) for entry in pending)
    launched = _dispatch_waiting_jobs(requested_ids=child_ids, capacity=concurrency)
    outcomes: list[TicketStartOutcome] = []
    for entry in pending:
        target = str(entry["target"])
        child_id = str(entry["job_id"])
        child = store.get(child_id)
        if child.status == JobStatus.FAILED:
            outcomes.append(
                TicketStartOutcome(
                    target,
                    "start-failed",
                    job_id=child_id,
                    detail=_start_error_detail(
                        HarnessError(
                            str(child.error or child.result or "worker failed")
                        )
                    ),
                )
            )
            continue
        outcomes.append(
            TicketStartOutcome(
                target,
                "accepted" if child_id in launched else "queued",
                job_id=child_id,
            )
        )
    completed_ids = {
        str(entry["job_id"])
        for entry in pending
        if isinstance(entry.get("job_id"), str)
    }

    def complete(current: CursorJob) -> CursorJob:
        launches = [
            {
                **entry,
                "state": (
                    "completed"
                    if str(entry.get("job_id") or "") in completed_ids
                    else entry.get("state")
                ),
            }
            for entry in current.grouped_repository_launches
        ]
        return current.evolve(grouped_repository_launches=launches)

    store.update(coordinator_id, complete)
    return tuple(outcomes)


def reply_job(
    job_id: str,
    text: str,
    *,
    trusted_utterance: str | None = None,
    expected_question_id: str | None = None,
    expected_question_turn: str | None = None,
    answer_provenance: AnswerProvenance = AnswerProvenance.USER_TEXT,
    on_started: Callable[[], None] | None = None,
    foreground_seconds: float = 5.0,
    concurrency: int = 3,
    integrations: IntegrationRegistry | None = None,
) -> str | None:
    current = read_job(job_id)
    if current.clarification_kind == GROUPED_REPOSITORY_OWNER:
        return _reply_grouped_repository(
            current,
            text,
            trusted_utterance=trusted_utterance,
            expected_question_id=expected_question_id,
            expected_question_turn=expected_question_turn,
            on_started=on_started,
            foreground_seconds=foreground_seconds,
            concurrency=concurrency,
            integrations=integrations,
        )
    if (
        current.clarification_kind == "repository"
        and provisioning.is_repository_list_request(trusted_utterance or text)
    ):
        return _reply_repository_list(
            current,
            expected_question_id=expected_question_id,
            expected_question_turn=expected_question_turn,
        )
    now = time.time()
    should_launch = False
    should_cancel = False
    should_complete = False
    immediate: str | None = None

    def reply(job: CursorJob) -> CursorJob | None:
        nonlocal immediate, should_cancel, should_complete, should_launch
        if job.status != JobStatus.AWAITING_USER:
            return None
        question = questions.current(job)
        if question is None:
            return None
        try:
            validate_question_identity(
                question,
                QuestionIdentity(
                    job.id,
                    expected_question_id or question.id,
                    expected_question_turn or question.origin.turn_token,
                ),
            )
        except QuestionError:
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
            return job.mark_delivered(
                updated_at=now,
                voice_question=questions.envelope(
                    question, QuestionState.DEFERRED, job=job
                ),
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
                foreground_until=(now + foreground_seconds + FOREGROUND_GRACE_SECONDS),
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
                voice_question=questions.envelope(
                    question, QuestionState.CANCELLED, job=job
                ),
                job_changes=(
                    {"plan_approval_state": "rejected"}
                    if question.owner == "workflow_plan_approval"
                    else None
                ),
            )
        if transition.complete:
            should_complete = True
            should_launch = False
            immediate = transition.message
            return recovery.stage_terminal_intent(
                job,
                JobStatus.COMPLETED,
                now=now,
                result=job.result or "Cursor implementation completed.",
                voice_question=questions.envelope(
                    question,
                    QuestionState.RESOLVED,
                    job=job,
                    answer=resolution.answer,
                    trusted_answer=resolution.trusted_answer,
                    answered_at=now,
                ),
            )
        should_launch = transition.launch
        immediate = transition.message
        return transition.job

    try:
        updated = _job_store().update(job_id, reply)
    except JobValidationError as exc:
        if "resource is reserved" not in str(exc):
            raise
        return (
            "That ticket became active before this answer was applied. "
            "No duplicate ticket job was started."
        )
    if updated is None and immediate is None:
        raise HarnessError(f"Cursor job {job_id} is not waiting for a reply")
    if should_cancel:
        assert updated is not None
        if updated.target_release_pending:
            _cancel_target_and_release(
                job_id,
                updated.herdr_target or "",
                updated.target_release_token or "",
                integrations=integrations,
            )
        return None
    if should_complete:
        assert updated is not None
        if updated.target_release_pending:
            _cancel_target_and_release(
                job_id,
                updated.herdr_target or "",
                updated.target_release_token or "",
                integrations=integrations,
            )
        return immediate
    if should_launch:
        launched = _dispatch_waiting_jobs(
            integrations=integrations,
            requested_ids=(job_id,),
            capacity=concurrency,
        )
        if job_id in launched and on_started is not None:
            on_started()
    return immediate


def start_follow_up(
    parent_job_id: str,
    text: str,
    *,
    expected_parent_revision: int,
    expected_completed_at: float | None = None,
    utterance: str | None = None,
    on_created: Callable[[], None] | None = None,
    foreground_seconds: float = 5.0,
    integrations: IntegrationRegistry | None = None,
) -> str:
    """Create and launch a child job that reuses a completed parent's checkout."""
    now = time.time()
    child_id = uuid.uuid4().hex[:12]
    spoken = utterance if utterance is not None else text
    store = _job_store()
    registry = _integration_registry(integrations)

    def build(parent: CursorJob) -> CursorJob:
        require_issue_provider(parent.issue_provider, registry)
        active_issue_key = resolve_issue_reference(
            parent.issue_key,
            registry,
            provider=parent.issue_provider,
        )
        if active_issue_key:
            require_issue_capabilities(
                active_issue_key,
                registry,
                provider=parent.issue_provider,
            )
        return CursorJob.new(
            NewCursorJob(
                id=child_id,
                parent_job_id=parent.id,
                request=text,
                created_at=now,
                foreground_until=(now + foreground_seconds + FOREGROUND_GRACE_SECONDS),
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
                harness_kind=parent.harness_kind,
                issue_key=parent.issue_key,
                issue_provider=parent.issue_provider,
                speakable_label=parent.speakable_label,
            )
        )

    created = store.create_follow_up(
        parent_job_id,
        build,
        expected_parent_revision=expected_parent_revision,
        expected_completed_at=expected_completed_at,
    )
    if on_created is not None:
        on_created()
    _dispatch_waiting_jobs(
        integrations=integrations,
        requested_ids=(created.id,),
    )
    return created.id


def _cancel_target_and_release(
    job_id: str,
    target: str,
    release_token: str,
    *,
    worker_stopped: bool = True,
    integrations: IntegrationRegistry | None = None,
) -> None:
    herdr_factory = (
        provisioning.HerdrClient if integrations is None else integrations.herdr_client
    )
    recovery.cancel_target_and_release(
        _job_store(),
        job_id,
        target,
        release_token,
        worker_stopped=worker_stopped,
        herdr_factory=herdr_factory,
    )
    _dispatch_waiting_jobs(integrations=integrations)


def cancel_job(
    job_id: str,
    *,
    integrations: IntegrationRegistry | None = None,
) -> str:
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
        if (
            job.participant_admission_state in {"waiting", "released"}
            and job.herdr_target is None
            and job.worker_token is None
        ):
            return job.evolve_for_delivery(
                now=cancelled_at,
                status=JobStatus.CANCELLED,
                result=f"Cursor job {job_id} was cancelled.",
                completed_at=cancelled_at,
                participant_admission_state="released",
            )
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
                questions.envelope(question, QuestionState.CANCELLED, job=job)
                if question is not None
                else None
            ),
            job_changes=(
                {"plan_approval_state": "rejected"}
                if question is not None and question.owner == "workflow_plan_approval"
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
            integrations=integrations,
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
        label = inbox.speakable_label_for(job)
        if (
            job.manual_reconcile_operation == "pane"
            and job.participant_creation_state == "manual_required"
        ):
            participant = job.participant_creation_participant or "workflow"
            target = job.participant_creation_target or "unknown"
            return (
                f"{label} requires manual reconciliation for the "
                f"{participant} pane target {target}. Inspect Herdr, then resolve "
                "pane creation as materialized or confirmed absent using token "
                f"{job.manual_reconcile_token}."
            )
        workflow = (
            f", {job.workflow_tier.value} tier" if job.workflow_tier is not None else ""
        )
        return (
            f"{label} is {job.status.value.replace('_', ' ')}, "
            f"in {job.workflow_phase.value.replace('_', ' ')}{workflow}."
        )
    jobs = [job for job in _job_store().list() if job.status in ACTIVE_STATUSES]
    if not jobs:
        return "There are no active Cursor jobs."
    return (
        "Active Cursor jobs: "
        + "; ".join(
            f"{inbox.speakable_label_for(job)} is "
            f"{job.status.value.replace('_', ' ')} "
            f"({job.workflow_phase.value.replace('_', ' ')})"
            for job in jobs
        )
        + "."
    )


def acknowledge_worktree_quarantine(job_id: str) -> None:
    recovery.acknowledge_worktree_quarantine(_job_store(), job_id)


def relinquish_session_control(job_id: str) -> CursorJob:
    return recovery.relinquish_session_control(_job_store(), job_id)


def resume_session_control(
    job_id: str,
    *,
    integrations: IntegrationRegistry | None = None,
) -> CursorJob:
    herdr_factory = (
        provisioning.HerdrClient if integrations is None else integrations.herdr_client
    )
    return recovery.resume_session_control(
        _job_store(),
        job_id,
        herdr_factory=herdr_factory,
    )


def resolve_manual_reconciliation(
    job_id: str,
    operation: str,
    token: str,
    outcome: str,
    *,
    pane_id: str | None = None,
    workspace_id: str | None = None,
) -> CursorJob:
    return recovery.resolve_manual_reconciliation(
        _job_store(),
        job_id,
        operation,
        token,
        outcome,
        pane_id=pane_id,
        workspace_id=workspace_id,
    )


def mark_delivered(job_id: str) -> CursorJob:
    def deliver(job: CursorJob) -> CursorJob:
        return job.mark_delivered()

    delivered = _job_store().update(job_id, deliver)
    assert delivered is not None
    return delivered


def claim_delivery(
    job_id: str | None = None, *, foreground: bool = False
) -> DeliveryClaim | None:
    return delivery.claim_delivery(_job_store(), job_id, foreground=foreground)


def renew_delivery(job_id: str, token: str) -> bool:
    return delivery.renew_delivery(_job_store(), job_id, token)


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


def recover_jobs(
    *,
    integrations: IntegrationRegistry | None = None,
    outbox_handlers: Mapping[str, outbox.EffectHandler] | None = None,
) -> None:
    registry = _integration_registry(integrations)
    runtime = registry.platform or default_user_config().platform
    store = _job_store()
    if isinstance(store, JobStore):
        for job in store.list():
            if any(
                launch.get("state") == "pending"
                for launch in job.grouped_repository_launches
            ):
                _launch_grouped_children(
                    job.id,
                    concurrency=runtime.agent_job_start_concurrency,
                )
    herdr_factory = (
        provisioning.HerdrClient if integrations is None else integrations.herdr_client
    )
    github_factory = (
        provisioning.GitHubClient
        if integrations is None
        else integrations.github_client
    )

    def dispatch_recovered(job_id: str) -> None:
        _dispatch_waiting_jobs(
            integrations=integrations,
            requested_ids=(job_id,),
            capacity=runtime.agent_job_start_concurrency,
        )

    recovery.recover_jobs(
        store,
        launch_worker=dispatch_recovered,
        herdr_factory=herdr_factory,
        github_factory=github_factory,
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
        require_issue_provider=lambda name: require_issue_provider(name, registry),
        outbox_handlers=outbox_handlers,
    )
    _dispatch_waiting_jobs(
        integrations=integrations,
        capacity=runtime.agent_job_start_concurrency,
    )


def pending_results(
    *,
    limit: int = delivery.DELIVERY_WINDOW,
    integrations: IntegrationRegistry | None = None,
) -> list[DeliveryClaim]:
    recover_jobs(integrations=integrations)
    return delivery.pending_deliveries(_job_store(), limit=limit)


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


def missed_announcements(claims: DeliveryClaims | None) -> AssistantResponse:
    store = _job_store()
    if claims is None:
        jobs = announcements.inspect_missed_announcements(store)
    else:
        claimed = announcements.claim_missed_announcements(store)
        claims.extend(claimed)
        jobs = [claim.job for claim in claimed]
    return announcements.render_digest(
        jobs,
        missed=True,
        render_job=render_job_announcement,
    )


def count_jobs() -> int:
    """Return how many durable Cursor jobs currently exist."""
    return len(_job_store().list())


def list_quarantine_evidence(
    *, include_resolved: bool = False
) -> list[QuarantineEvidence]:
    """Return quarantine evidence for operator inspection."""
    return _job_store().list_quarantine_evidence(include_resolved=include_resolved)


def acknowledge_quarantine_reservations(job_id: str, *, reason: str) -> str:
    """Release reservations after an operator verifies quarantine evidence."""
    acknowledgement = _job_store().acknowledge_quarantine_reservations(
        job_id, reason=reason
    )
    count = len(acknowledgement.resolved_metadata)
    if not count:
        return f"Quarantine reservations for job {job_id} were already acknowledged."
    noun = "record" if count == 1 else "records"
    return (
        f"Acknowledged {count} quarantined {noun} for job {job_id}. "
        "The payload and metadata were preserved."
    )


def nuke_jobs(*, integrations: IntegrationRegistry | None = None) -> str:
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
                integrations=integrations,
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
        message = str(exc)
        if "quarantine evidence" in message:
            message += (
                ". Inspect it with 'voice-harness jobs quarantine list', then "
                "acknowledge each verified job before retrying"
            )
        raise HarnessError(message) from exc
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
        return job.dismiss_announcement(delivered_at=now)

    updated = _job_store().update(job_id, dismiss)
    label = inbox.speakable_label_for(updated or read_job(job_id))
    return f"Dismissed the update for {label}."


def repeat_announcement(job_id: str) -> str:
    now = time.time()

    def repeat(job: CursorJob) -> CursorJob | None:
        if job.status not in ANNOUNCEABLE_STATUSES:
            return None
        return job.repeat_announcement(now=now)

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
    github_issue_create_requested: bool = False,
    linear_team: str | None = None,
    linear_ticket_create_requested: bool = False,
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
    harness_kind: HarnessKind | None = None,
    delivery_claims: DeliveryClaims | None = None,
    platform: PlatformSettings | None = None,
    integrations: IntegrationRegistry | None = None,
) -> CursorTurnResult:
    registry = _integration_registry(integrations)
    runtime = platform or registry.platform or default_user_config().platform
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
        github_issue_create_requested = request.github_issue_create_requested
        linear_team = request.linear_team
        linear_ticket_create_requested = request.linear_ticket_create_requested
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
        harness_kind = request.harness_kind
        answer_provenance = request.answer_provenance
        expected_completed_at = request.expected_completed_at
        expected_parent_revision = request.expected_parent_revision
        on_follow_up_started = request.on_follow_up_started
        on_job_started = request.on_job_started
    else:
        text = request
        expected_completed_at = None
        expected_parent_revision = None
    if action == "list":
        return CursorTurnResult(list_jobs(), session_id)
    if action == "missed":
        return CursorTurnResult(
            missed_announcements(delivery_claims),
            session_id,
        )
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
        result = cancel_job(resolved.job_id, integrations=registry)
        _defer_or_acknowledge(resolved.job_id, delivery_claims)
        label = _speakable_label(resolved.job_id)
        return CursorTurnResult(
            AssistantResponse(
                spoken_text=f"Cursor cancelled {label}.",
                display_text=result,
            ),
            None,
            mutated=True,
        )
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
        return CursorTurnResult(message, session_id, mutated=True)
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
        before_reply = read_job(reply_id)
        immediate = reply_job(
            reply_id,
            text,
            trusted_utterance=utterance,
            expected_question_id=expected_question_id,
            expected_question_turn=expected_question_turn,
            answer_provenance=answer_provenance,
            on_started=on_job_started,
            foreground_seconds=runtime.cursor_foreground_seconds,
            concurrency=runtime.agent_job_start_concurrency,
            integrations=registry,
        )
        if immediate is not None:
            current = read_job(reply_id)
            pending = questions.current(current)
            next_session = (
                None
                if pending is not None and pending.state == QuestionState.DEFERRED
                else reply_id
            )
            mutated = current.revision != before_reply.revision
            return CursorTurnResult(immediate, next_session, mutated=mutated)
        job_id = reply_id
        return _await_foreground(
            job_id,
            delivery_claims,
            timeout=runtime.cursor_foreground_seconds,
            continuation=True,
        )
    elif action == "follow_up":
        if not job_id:
            return CursorTurnResult(
                "I don't have a recent completed Cursor job to follow up on.", None
            )
        try:
            if expected_parent_revision is None:
                raise FollowUpUnavailable(
                    "follow-up parent revision identity is required"
                )
            job_id = start_follow_up(
                job_id,
                text,
                expected_parent_revision=expected_parent_revision,
                expected_completed_at=expected_completed_at,
                utterance=utterance,
                on_created=on_follow_up_started,
                foreground_seconds=runtime.cursor_foreground_seconds,
                integrations=registry,
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
                github_issue_create_requested=github_issue_create_requested,
                linear_team=linear_team,
                linear_ticket_create_requested=linear_ticket_create_requested,
                fork_requested=fork_requested,
                github_pull_request=github_pull_request,
                agent=agent,
                utterance=utterance,
                context_repository=context_repository,
                issue_key=issue_key,
                foreground=not extraction.batch_requested,
                harness_kind=harness_kind,
            )
            outcomes = _submit_extracted_targets(
                extraction,
                base,
                foreground=not extraction.batch_requested,
                foreground_seconds=runtime.cursor_foreground_seconds,
                concurrency=runtime.agent_job_start_concurrency,
                integrations=registry,
            )
            accepted = tuple(
                outcome
                for outcome in outcomes
                if outcome.status in {"accepted", "queued"}
            )
            if accepted and on_job_started is not None:
                on_job_started()
            if (
                extraction.batch_requested
                or not accepted
                or accepted[0].status == "queued"
            ):
                return CursorTurnResult(
                    _ticket_start_summary(outcomes, utterance),
                    None,
                    mutated=bool(accepted),
                )
            assert len(accepted) == 1 and accepted[0].job_id is not None
            return _await_foreground(
                accepted[0].job_id,
                delivery_claims,
                timeout=runtime.cursor_foreground_seconds,
            )
        try:
            job_id = start_job(
                text,
                repository=repository,
                github_repository=github_repository,
                github_issue=github_issue,
                github_issue_context=github_issue_context,
                github_issue_create_requested=github_issue_create_requested,
                linear_team=linear_team,
                linear_ticket_create_requested=linear_ticket_create_requested,
                fork_requested=fork_requested,
                github_pull_request=github_pull_request,
                agent=agent,
                utterance=utterance,
                context_repository=context_repository,
                issue_key=issue_key,
                harness_kind=harness_kind,
                foreground_seconds=runtime.cursor_foreground_seconds,
                integrations=registry,
            )
        except ActiveTicketConflict as exc:
            label = _speakable_label(exc.active_job_id)
            return CursorTurnResult(
                AssistantResponse(
                    spoken_text=f"That ticket is already in progress as {label}.",
                    display_text=str(exc),
                ),
                None,
            )
        if on_job_started is not None:
            on_job_started()
        queued = read_job(job_id)
        if queued.participant_admission_state == "waiting":
            label = inbox.speakable_label_for(queued)
            return CursorTurnResult(
                _queued_accept_response(label, job_id, utterance),
                None,
                mutated=True,
            )
    return _await_foreground(
        job_id,
        delivery_claims,
        timeout=runtime.cursor_foreground_seconds,
    )


def _job_failure_stage(job: CursorJob) -> str:
    if job.worktree_provision_state not in {None, "ready"}:
        return "repository setup"
    if job.agent_dispatch_state not in {None, "ready"}:
        return "agent setup"
    phase = {
        "classifying": "workflow classification",
        "planning": "planning",
        "reviewing": "plan review",
        "revising": "plan revision",
        "implementing": "implementation",
    }.get(job.workflow_phase.value)
    return phase or "execution"


def _foreground_failure_response(job: CursorJob) -> AssistantResponse:
    stage = _job_failure_stage(job)
    log_path = JOB_LOGS_DIR / f"{job.id}.log"
    return AssistantResponse(
        spoken_text=f"The Cursor job failed during {stage}.",
        display_text=(
            f"Cursor job {job.id} failed during {stage}. "
            f"Inspect {log_path} for diagnostic details before retrying."
        ),
    )


def render_job_announcement(job: CursorJob) -> AssistantResponse:
    """Render one claimed job snapshot for background display and speech."""

    label = inbox.speakable_label_for(job)
    identity = f"Cursor job {job.id} ({label})"
    if job.status == JobStatus.COMPLETED:
        detail = str(job.result or "").strip()
        if job.github_issue_create_requested and job.github_issue_created_number:
            return AssistantResponse(
                spoken_text=(
                    f"Created GitHub issue {job.github_issue_created_number}."
                ),
                display_text=detail,
            )
        if job.linear_ticket_create_requested and job.linear_ticket_created_identifier:
            return AssistantResponse(
                spoken_text=(
                    f"Created Linear ticket {job.linear_ticket_created_identifier}."
                ),
                display_text=detail,
            )
        return AssistantResponse(
            spoken_text=f"Cursor finished {label}.",
            display_text=(
                f"{identity} completed: {detail}"
                if detail
                else f"{identity} completed."
            ),
        )
    if job.status == JobStatus.AWAITING_USER:
        pending = questions.current(job)
        question = (
            question_prompt(pending)
            if pending is not None
            else str(job.question or job.result or "").strip()
        )
        if job.clarification_kind == "github_issue_create_confirmation":
            return AssistantResponse(
                spoken_text=(
                    f"I drafted “{job.github_issue_create_title}” for "
                    f"{job.github_repository}. Should I create it?"
                ),
                display_text=question,
            )
        if job.clarification_kind == "linear_ticket_create_confirmation":
            return AssistantResponse(
                spoken_text=(
                    f"I drafted “{job.linear_ticket_create_title}” for Linear team "
                    f"{job.linear_ticket_create_team}. Should I create it?"
                ),
                display_text=question,
            )
        return AssistantResponse(
            spoken_text=f"Cursor needs clarification for {label}. {question}",
            display_text=f"{identity} needs clarification: {question}",
        )
    if job.status == JobStatus.BLOCKED:
        return AssistantResponse(
            spoken_text=f"Cursor needs attention for {label}.",
            display_text=(f"{identity} is blocked. Open Herdr for recovery guidance."),
        )
    if job.status == JobStatus.CANCELLED:
        return AssistantResponse(
            spoken_text=f"Cursor cancelled {label}.",
            display_text=f"{identity} was cancelled.",
        )
    if job.status == JobStatus.FAILED:
        detail = str(job.result or "").strip()
        if job.linear_ticket_create_requested and detail.startswith(
            "I couldn't find Linear team "
        ):
            return AssistantResponse(spoken_text=detail, display_text=detail)
        stage = _job_failure_stage(job)
        log_path = JOB_LOGS_DIR / f"{job.id}.log"
        return AssistantResponse(
            spoken_text=f"Cursor failed {label} during {stage}.",
            display_text=(
                f"{identity} failed during {stage}. "
                f"Inspect {log_path} for diagnostic details before retrying."
            ),
        )
    raise ValueError(f"{job.status.value} job cannot be announced")


def _await_foreground(
    job_id: str,
    delivery_claims: DeliveryClaims | None,
    *,
    timeout: float = 5.0,
    continuation: bool = False,
) -> CursorTurnResult:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = read_job(job_id)
        result = _foreground_delivery_result(job_id, job, delivery_claims)
        if result is not None:
            if job.status == JobStatus.COMPLETED:
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
            return result
        time.sleep(0.1)
    # The timestamp expires naturally. Persisting a cosmetic zero here would
    # advance the lifecycle revision while a worker is awaiting an external
    # callback, causing that correctly fenced callback to look stale.
    job = read_job(job_id)
    result = _foreground_delivery_result(job_id, job, delivery_claims)
    if result is not None:
        if job.status == JobStatus.COMPLETED:
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
        return result
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
    if continuation:
        return CursorTurnResult(
            AssistantResponse(
                spoken_text="Answer sent; Cursor is continuing.",
                display_text=(
                    f"Answer sent to Cursor job {job_id}; the same job is continuing "
                    "in the background."
                ),
            ),
            None,
            mutated=True,
        )
    label = inbox.speakable_label_for(job)
    return CursorTurnResult(
        AssistantResponse(
            spoken_text=(
                f"Cursor started {label}. I will report back when it finishes."
            ),
            display_text=(
                f"Cursor job {job_id} started for {label} and is continuing in "
                "the background. Its result will be announced when it finishes."
            ),
        ),
        None,
        mutated=True,
    )


def _foreground_delivery_result(
    job_id: str,
    job: CursorJob,
    delivery_claims: DeliveryClaims | None,
) -> CursorTurnResult | None:
    if job.status == JobStatus.COMPLETED:
        claimed = _defer_or_acknowledge(job_id, delivery_claims)
        completed = claimed if claimed is not None else job
        if (
            completed.github_issue_create_requested
            or completed.linear_ticket_create_requested
        ):
            return CursorTurnResult(
                render_job_announcement(completed), None, mutated=True
            )
        result = completed.result
        return CursorTurnResult(str(result or "").strip(), None, mutated=True)
    if job.status == JobStatus.AWAITING_USER:
        claimed = _defer_or_acknowledge(job_id, delivery_claims)
        awaiting = claimed if claimed is not None else job
        pending = questions.current(awaiting)
        rendered_question = (
            question_prompt(pending)
            if pending is not None
            else str(awaiting.question or awaiting.result or "").strip()
        )
        if awaiting.clarification_kind == "github_issue_create_confirmation":
            return CursorTurnResult(
                AssistantResponse(
                    spoken_text=(
                        f"I drafted “{awaiting.github_issue_create_title}” for "
                        f"{awaiting.github_repository}. Should I create it?"
                    ),
                    display_text=rendered_question,
                ),
                job_id,
                mutated=True,
            )
        if awaiting.clarification_kind == "linear_ticket_create_confirmation":
            return CursorTurnResult(
                AssistantResponse(
                    spoken_text=(
                        f"I drafted “{awaiting.linear_ticket_create_title}” for "
                        f"Linear team {awaiting.linear_ticket_create_team}. "
                        "Should I create it?"
                    ),
                    display_text=rendered_question,
                ),
                job_id,
                mutated=True,
            )
        return CursorTurnResult(rendered_question, job_id, mutated=True)
    if job.status == JobStatus.BLOCKED:
        claimed = _defer_or_acknowledge(job_id, delivery_claims)
        blocked = claimed if claimed is not None else job
        return CursorTurnResult(render_job_announcement(blocked), None, mutated=True)
    if job.status == JobStatus.FAILED:
        claimed = _defer_or_acknowledge(job_id, delivery_claims)
        failed = claimed if claimed is not None else job
        return CursorTurnResult(
            _foreground_failure_response(failed), None, mutated=True
        )
    if job.status == JobStatus.CANCELLED:
        claimed = _defer_or_acknowledge(job_id, delivery_claims)
        cancelled = claimed if claimed is not None else job
        return CursorTurnResult(render_job_announcement(cancelled), None, mutated=True)
    return None


# Compatibility aliases for callers migrating to the agent-neutral service API.
StartAgentJobRequest = StartJobRequest
AgentTurnRequest = CursorTurnRequest
AgentTurnResult = CursorTurnResult
agent_turn = cursor_turn
