from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from pathlib import Path

from ..errors import HarnessError
from ..integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubIssue,
    GitHubOperationAmbiguous,
    GitHubRepository,
)
from ..integrations.herdr import (
    SETTLED,
    AgentSelection,
    HerdrClient,
    HerdrError,
    agent_session_identity,
    extract_marker,
    normalize_name,
)
from ..integrations.registry import (
    prompt_instructions,
    require_issue_capabilities,
    resolve_issue_reference,
    route_issue_repository,
)
from ..questions import (
    PromptOperationState,
    QuestionError,
    QuestionSensitivity,
    QuestionSpec,
    QuestionState,
    parse_question_spec,
)
from ..user_config import (
    PlanApprovalMode,
    PlanApprovalPreferences,
    UserConfigurationError,
    load_plan_approval_preferences,
    record_explicit_plan_approval,
)
from . import questions as question_adapter
from . import recovery, worker_lifecycle
from .model import (
    ACTIVE_STATUSES as MODEL_ACTIVE_STATUSES,
)
from .model import (
    TERMINAL_STATUSES as MODEL_TERMINAL_STATUSES,
)
from .model import (
    WORKER_STATUSES as MODEL_WORKER_STATUSES,
)
from .model import (
    CursorJob,
    JobStatus,
    JobValidationError,
    WorkflowParticipant,
    WorkflowPhase,
    WorkflowTier,
    transition,
)
from .prompts import (
    classification_prompt,
    implementation_prompt,
    plan_approval_prompt,
    planning_prompt,
    review_prompt,
    revision_prompt,
)
from .store import JobStore

DELIVERY_RETRY_SECONDS = 5.0
FOREGROUND_GRACE_SECONDS = 2.0


WorkerCancelled = worker_lifecycle.WorkerCancelled


@dataclass(frozen=True, slots=True)
class ClientFactories:
    herdr: Callable[[], HerdrClient] = HerdrClient
    github: Callable[[], GitHubClient] = GitHubClient


class ReservationConflict(Exception):
    pass


def _agent_not_found(exc: HerdrError) -> bool:
    return exc.code in {"agent_not_found", "not_found"}


def reserved_targets(store: JobStore, exclude_job_id: str | None = None) -> set[str]:
    reserved: set[str] = set()
    for job in store.list():
        if job.id == exclude_job_id:
            continue
        if not (
            job.status in MODEL_ACTIVE_STATUSES
            or job.target_release_pending
            or job.has_uncertain_operation()
            or job.manual_reconcile_operation
        ):
            continue
        reserved.update(
            target
            for target in (
                job.herdr_target,
                job.participant_target(WorkflowParticipant.PLANNER),
                job.participant_target(WorkflowParticipant.REVIEWER),
                job.participant_target(WorkflowParticipant.IMPLEMENTER),
                job.participant_creation_target,
            )
            if target
        )
    return reserved


def repository_question(repositories: list[Path], reason: str = "") -> str:
    names = ", ".join(path.name for path in repositories)
    prefix = f"{reason.strip()} " if reason.strip() else ""
    return (
        f"{prefix}Which repository should Cursor use? Available repositories are {names}."
        if names
        else f"{prefix}I could not find an available local Git repository."
    )


def resolve_job_repository(
    client: HerdrClient,
    job: CursorJob,
    repositories: list[Path],
) -> tuple[Path | None, list[Path]]:
    hint = (job.repository_hint or "").strip() or None
    task = job.request
    utterance = job.utterance or task
    repository, candidates = client.resolve_repository(hint, utterance, repositories)
    context_hint = (job.context_repository or "").strip() or None
    if repository is None and not hint and context_hint:
        repository, context_candidates = client.resolve_repository(
            context_hint, "", repositories
        )
        candidates = context_candidates or candidates
    return repository, candidates


def complete_from_output(
    job: CursorJob, *, output: str, agent_status: str, now: float | None = None
) -> CursorJob:
    completed_at = time.time() if now is None else now
    token = job.turn_token or ""
    summary = extract_marker(output, "VOICE_SUMMARY", token)
    question = extract_marker(output, "VOICE_QUESTION", token)
    if agent_status not in SETTLED:
        summary = None
        question = None
    summary_position = output.rfind(f"VOICE_SUMMARY[{token}]")
    question_position = output.rfind(f"VOICE_QUESTION[{token}]")
    if question and question_position > summary_position:
        try:
            spec = parse_question_spec(question)
        except QuestionError as exc:
            return job.evolve_for_delivery(
                now=completed_at,
                status=JobStatus.BLOCKED,
                result=f"Cursor returned an invalid voice question: {exc}",
                completed_at=completed_at,
            )
        return question_adapter.ask(
            job,
            spec,
            owner="agent",
            turn_token=token,
            now=completed_at,
        )
    pending = question_adapter.current(job)
    resolved_question = (
        question_adapter.envelope(
            pending,
            QuestionState.RESOLVED,
            prompt_state=(
                PromptOperationState.RESOLVED
                if pending.prompt_state is not None
                else None
            ),
        )
        if pending is not None
        and pending.state in {QuestionState.ANSWERED, QuestionState.DISPATCHING}
        else job.voice_question
    )
    if summary and summary_position > question_position:
        return job.evolve_for_delivery(
            now=completed_at,
            status=JobStatus.COMPLETED,
            result=summary,
            completed_at=completed_at,
            voice_question=resolved_question,
        )
    return job.evolve_for_delivery(
        now=completed_at,
        status=JobStatus.BLOCKED,
        result=(
            f"Herdr agent {job.herdr_target or 'Cursor'} needs attention; "
            f"it settled as {agent_status} without a voice summary."
        ),
        completed_at=completed_at,
        voice_question=resolved_question,
    )


def read_agent_completion(
    client: HerdrClient,
    job: CursorJob,
    *,
    wait: bool,
    checkpoint: Callable[[], None] | None = None,
    active_marker: str | None = None,
) -> tuple[str, str]:
    target = job.herdr_target or ""
    if not target:
        raise HarnessError("Cursor job has no Herdr agent")
    if not wait:
        if checkpoint is not None:
            checkpoint()
        agent = client.get_agent(target)
        if checkpoint is not None:
            checkpoint()
        output = client.run_text(
            "agent", "read", target, "--source", "recent-unwrapped", "--lines", "160"
        )
        if checkpoint is not None:
            checkpoint()
        return output, str(agent.get("agent_status") or "unknown")
    outcome = client.wait_for_stable_completion(
        target,
        token=job.turn_token or "",
        checkpoint=checkpoint,
        active_marker=active_marker,
    )
    return outcome.output, outcome.status


def _worker_change(
    store: JobStore,
    job_id: str,
    token: str,
    allowed_statuses: Set[JobStatus],
    change: Callable[[CursorJob], CursorJob],
) -> CursorJob | None:
    def guarded(job: CursorJob) -> CursorJob | None:
        if (
            job.worker_token != token
            or job.status not in allowed_statuses
            or job.terminal_intent_status is not None
        ):
            return None
        return change(job)

    return store.update_unless_maintenance(job_id, guarded)


def _worker_question(
    store: JobStore,
    job_id: str,
    token: str,
    question: str,
    *,
    clarification_kind: str,
    job_changes: Mapping[str, object] | None = None,
) -> None:
    def ask(job: CursorJob) -> CursorJob:
        now = time.time()
        return question_adapter.ask(
            job,
            QuestionSpec(
                question,
                sensitivity=QuestionSensitivity.ROUTINE,
            ),
            owner=clarification_kind,
            turn_token=job.turn_token or f"{job.id}-routing-{job.turn}",
            now=now,
            clear_worker=True,
            remove_reconcile=True,
            job_changes=job_changes,
        )

    _worker_change(store, job_id, token, {JobStatus.ROUTING, JobStatus.RUNNING}, ask)


def _completion_preferences(
    job: CursorJob,
) -> tuple[PlanApprovalPreferences | None, bool]:
    approval_id = job.plan_approval_id
    if job.plan_approval_source != "explicit" or approval_id is None:
        return None, False
    try:
        preferences = (
            load_plan_approval_preferences()
            if job.plan_approval_counted
            else record_explicit_plan_approval(approval_id)
        )
    except (OSError, UserConfigurationError):
        return None, True
    return preferences, False


def _finish_completed_workflow(
    job: CursorJob,
    *,
    preferences: PlanApprovalPreferences | None,
    result: str,
    voice_question: dict[str, object] | None,
    now: float,
) -> CursorJob:
    approval_id = job.plan_approval_id
    counted = job.plan_approval_counted or bool(
        approval_id
        and preferences is not None
        and approval_id in preferences.explicit_approval_ids
    )
    completion_changes: Mapping[str, object] = {
        "result": result,
        "workflow_phase": WorkflowPhase.FINISHED.value,
        "active_participant": None,
        "prompt_operation_state": "none",
        "plan_approval_counted": counted,
        "plan_approval_completion_pending": False,
    }
    if (
        counted
        and job.plan_approval_source == "explicit"
        and preferences is not None
        and approval_id == preferences.offer_pending_id
    ):
        return question_adapter.ask(
            job,
            QuestionSpec(
                f"{result or 'Cursor implementation completed.'} "
                "You've explicitly approved three reviewed Cursor plans. "
                "Would you like me to automatically approve ordinary "
                "reviewed plans from now on? Say yes to enable automatic "
                "plan approval, or no to keep asking.",
                sensitivity=QuestionSensitivity.ARCHITECTURE,
            ),
            owner="workflow_plan_auto_offer",
            turn_token=job.turn_token or f"{job.id}-approval-offer",
            now=now,
            clear_worker=True,
            remove_reconcile=True,
            prompt_operation_state="none",
            job_changes=completion_changes,
        )
    return recovery.stage_terminal_intent(
        job,
        JobStatus.COMPLETED,
        now=now,
        result=result,
        voice_question=voice_question,
        job_changes=completion_changes,
    )


def _worker_complete(
    store: JobStore,
    job_id: str,
    token: str,
    *,
    output: str,
    agent_status: str,
    preserve_blocked_delivery: bool = False,
) -> None:
    snapshot = store.get(job_id)
    if (
        snapshot.worker_token != token
        or snapshot.status not in {JobStatus.RUNNING, JobStatus.RECONCILING}
        or snapshot.terminal_intent_status is not None
    ):
        return
    preferences, preference_update_failed = _completion_preferences(snapshot)

    def finish(job: CursorJob) -> CursorJob:
        now = time.time()
        outcome = complete_from_output(
            job, output=output, agent_status=agent_status, now=now
        )
        preserve = preserve_blocked_delivery and outcome.status == JobStatus.BLOCKED
        workflow_finished = (
            outcome.status == JobStatus.COMPLETED and job.workflow_tier is not None
        )
        if outcome.status == JobStatus.COMPLETED:
            if preference_update_failed and job.plan_approval_source == "explicit":
                return job.evolve(
                    status=JobStatus.QUEUED,
                    reconcile=True,
                    queued_at=now,
                    result=outcome.result or "Cursor implementation completed.",
                    voice_question=outcome.voice_question,
                    workflow_phase=WorkflowPhase.FINISHED.value,
                    active_participant=None,
                    prompt_operation_state="none",
                    plan_approval_completion_pending=True,
                    worker_pid=None,
                    worker_boot_id=None,
                    worker_process_start=None,
                    worker_token=None,
                )
            return _finish_completed_workflow(
                job,
                preferences=preferences,
                result=outcome.result or "",
                voice_question=outcome.voice_question,
                now=now,
            )
        return job.evolve(
            status=outcome.status,
            remove=frozenset({"reconcile", "continuation", "continuation_answer"}),
            result=outcome.result,
            error=outcome.error,
            question=outcome.question,
            clarification_kind=outcome.clarification_kind,
            voice_question=outcome.voice_question,
            completed_at=outcome.completed_at,
            updated_at=outcome.updated_at,
            delivered=job.delivered if preserve else outcome.delivered,
            delivery_generation=(
                job.delivery_generation if preserve else outcome.delivery_generation
            ),
            delivery_claim_token=(
                job.delivery_claim_token if preserve else outcome.delivery_claim_token
            ),
            delivery_claimed_at=(
                job.delivery_claimed_at if preserve else outcome.delivery_claimed_at
            ),
            delivery_retry_at=(
                job.delivery_retry_at if preserve else outcome.delivery_retry_at
            ),
            delivery_attempts=(
                job.delivery_attempts if preserve else outcome.delivery_attempts
            ),
            next_reconcile_at=now + DELIVERY_RETRY_SECONDS,
            pull_request_worktree_state=(
                "retained"
                if outcome.status in MODEL_TERMINAL_STATUSES
                and job.pull_request_worktree_state == "ready"
                else job.pull_request_worktree_state
            ),
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            workflow_phase=(
                WorkflowPhase.FINISHED.value
                if workflow_finished
                else job.workflow_phase.value
            ),
            active_participant=(
                None
                if workflow_finished
                else (
                    job.active_participant.value
                    if job.active_participant is not None
                    else None
                )
            ),
            prompt_operation_state="none",
        )

    _worker_change(
        store,
        job_id,
        token,
        {JobStatus.RUNNING, JobStatus.RECONCILING},
        finish,
    )


def _resume_plan_approval_completion(
    store: JobStore,
    job: CursorJob,
    worker_token: str,
) -> None:
    if (
        job.worker_token != worker_token
        or job.status not in MODEL_WORKER_STATUSES
        or job.terminal_intent_status is not None
    ):
        return
    preferences, preference_update_failed = _completion_preferences(job)
    if preference_update_failed:
        _worker_change(
            store,
            job.id,
            worker_token,
            MODEL_WORKER_STATUSES,
            lambda current: current.evolve(
                status=JobStatus.QUEUED,
                queued_at=time.time(),
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
            ),
        )
        return

    def finish(current: CursorJob) -> CursorJob:
        if not current.plan_approval_completion_pending:
            raise WorkerCancelled
        return _finish_completed_workflow(
            current,
            preferences=preferences,
            result=current.result or "Cursor implementation completed.",
            voice_question=current.voice_question,
            now=time.time(),
        )

    _worker_change(
        store,
        job.id,
        worker_token,
        MODEL_WORKER_STATUSES,
        finish,
    )


def _worker_fail(
    store: JobStore,
    job_id: str,
    token: str,
    exc: Exception,
    *,
    target_may_be_active: bool = False,
) -> None:
    del target_may_be_active  # Terminal intent always retains the cleanup fence.
    message = (str(exc) or type(exc).__name__)[:500]

    def fail(job: CursorJob) -> CursorJob:
        now = time.time()
        return recovery.stage_terminal_intent(
            job,
            JobStatus.FAILED,
            now=now,
            result=message,
            error=message,
        )

    def guarded(job: CursorJob) -> CursorJob | None:
        if (
            job.worker_token != token
            or job.status not in MODEL_WORKER_STATUSES
            or job.terminal_intent_status is not None
        ):
            return None
        return fail(job)

    store.update(job_id, guarded)


def _worker_block_interactive(
    store: JobStore,
    job_id: str,
    token: str,
    message: str,
) -> None:
    def block(job: CursorJob) -> CursorJob:
        now = time.time()
        return job.evolve_for_delivery(
            now=now,
            status=JobStatus.BLOCKED,
            result=message,
            completed_at=now,
            next_reconcile_at=now + DELIVERY_RETRY_SECONDS,
            interactive_questionnaire_blocked=True,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
        )

    _worker_change(store, job_id, token, MODEL_WORKER_STATUSES, block)


def _worker_error(
    store: JobStore,
    job_id: str,
    token: str,
    exc: Exception,
    *,
    prompt_may_be_active: bool,
    client: HerdrClient | None = None,
    target: str = "",
    checkpoint: Callable[[], None] | None = None,
) -> None:
    if isinstance(exc, HerdrError) and exc.code == "interactive_questionnaire":
        _worker_block_interactive(
            store,
            job_id,
            token,
            "Cursor opened an interactive questionnaire and is blocked for "
            "manual attention.",
        )
        return
    if (
        isinstance(exc, HerdrError)
        and exc.code == "agent_stalled"
        and prompt_may_be_active
        and client is not None
        and target
    ):
        try:
            if checkpoint is not None:
                checkpoint()
            client.cancel_agent(target)
            if checkpoint is not None:
                checkpoint()
        except Exception as cancel_exc:
            _worker_fail(
                store,
                job_id,
                token,
                cancel_exc,
                target_may_be_active=True,
            )
            return
        _worker_block(store, job_id, token, str(exc))
        return
    _worker_fail(
        store,
        job_id,
        token,
        exc,
        target_may_be_active=prompt_may_be_active,
    )


def _worker_block(
    store: JobStore,
    job_id: str,
    token: str,
    message: str,
) -> None:
    blocked_at = time.time()

    def block(job: CursorJob) -> CursorJob:
        return job.evolve_for_delivery(
            now=blocked_at,
            status=JobStatus.BLOCKED,
            result=message[:500],
            completed_at=blocked_at,
        )

    _worker_change(store, job_id, token, MODEL_WORKER_STATUSES, block)


def _pull_request_branch(job: CursorJob) -> str:
    configured = job.worktree_branch or ""
    if configured:
        return configured
    return f"voice/github-pr-{job.id}"


def _prepare_pull_request_checkout(
    store: JobStore,
    job_id: str,
    token: str,
    job: CursorJob,
    checkpoint: Callable[[], None] | None = None,
    *,
    github_factory: Callable[[], GitHubClient] | None = None,
) -> CursorJob | None:
    if not job.github_pull_request:
        return job
    if job.pull_request_worktree_state in {"ready", "retained"}:
        return job
    repository = Path(job.repository or "").resolve()
    checkout_value = job.worktree_path or ""
    checkout = Path(checkout_value).resolve() if checkout_value else repository
    branch = _pull_request_branch(job)
    number = job.github_pull_request
    try:
        if checkout == repository:
            raise HarnessError(
                "refusing to check out a pull request in the shared repository clone"
            )
        if not checkout.is_dir() or not (checkout / ".git").exists():
            raise HarnessError("pull-request worktree is missing or invalid")
        if checkpoint is not None:
            checkpoint()
        github = (github_factory or GitHubClient)()
        if checkpoint is None:
            checked_out_branch = github.checkout_pull_request(
                checkout, number, branch=branch
            )
        else:
            checked_out_branch = github.checkout_pull_request(
                checkout, number, branch=branch, checkpoint=checkpoint
            )
        if checkpoint is not None:
            checkpoint()
    except Exception as exc:
        message = (str(exc) or type(exc).__name__)[:500]

        def quarantine(current: CursorJob) -> CursorJob:
            return current.evolve(
                pull_request_worktree_state="quarantined",
                pull_request_worktree_error=message,
            )

        _worker_change(store, job_id, token, {JobStatus.ROUTING}, quarantine)
        raise

    def ready(current: CursorJob) -> CursorJob:
        return current.evolve(
            pull_request_branch=checked_out_branch or branch,
            pull_request_worktree_state="ready",
            pull_request_worktree_error=None,
        )

    return _worker_change(store, job_id, token, {JobStatus.ROUTING}, ready)


def _reserve_worker_target(
    store: JobStore,
    job_id: str,
    token: str,
    selection: AgentSelection,
    repository: Path,
    issue_key: str | None,
    *,
    dispatching: bool = False,
    participant: WorkflowParticipant | None = None,
) -> CursorJob | None:
    target = str(selection.target)

    def reserve(job: CursorJob) -> CursorJob | None:
        if (
            job.worker_token != token
            or job.status not in MODEL_WORKER_STATUSES
            or job.terminal_intent_status is not None
        ):
            return None
        repository_value = (
            job.repository
            if job.parent_job_id or (participant is not None and job.repository)
            else str(repository)
        )
        worktree_value = (
            job.worktree_path
            if job.parent_job_id or (participant is not None and job.worktree_path)
            else selection.worktree_path
        )
        participant_changes: Mapping[str, object] = {}
        if participant is not None:
            participant_changes = {
                "active_participant": participant.value,
                f"{participant.value}_target": target,
            }
        return transition(
            job,
            job.status,
            repository=repository_value,
            issue_key=issue_key,
            herdr_target=target,
            herdr_pane_id=selection.pane_id,
            herdr_workspace_id=selection.workspace_id,
            worktree_path=worktree_value,
            agent_name=selection.name,
            agent_dispatch_state="dispatching" if dispatching else "ready",
            worker_operation="agent_start" if dispatching else None,
            **participant_changes,
        )

    try:
        reserved = store.reserve_target(job_id, reserve)
    except JobValidationError as exc:
        if "reserved by both" in str(exc):
            return None
        raise
    return reserved


def _settle_worker_agent(
    store: JobStore,
    job_id: str,
    token: str,
    selection: AgentSelection,
) -> CursorJob | None:
    def settle(job: CursorJob) -> CursorJob | None:
        if (
            job.worker_token != token
            or job.herdr_target != selection.target
            or job.agent_dispatch_state != "dispatching"
            or job.status
            not in MODEL_WORKER_STATUSES | {JobStatus.CANCELLED, JobStatus.FAILED}
        ):
            return None
        return transition(
            job,
            job.status,
            herdr_pane_id=selection.pane_id,
            herdr_workspace_id=selection.workspace_id,
            worktree_path=selection.worktree_path,
            agent_name=selection.name,
            agent_dispatch_state="ready",
            worker_operation=None,
            participant_creation_state="none",
            participant_creation_participant=None,
            participant_creation_target=None,
            participant_creation_label=None,
            participant_creation_workspace_id=None,
            participant_creation_pane_id=None,
        )

    return store.update(job_id, settle)


def _failed_operation_state(exc: HerdrError) -> str:
    return (
        "ambiguous"
        if exc.code in {"operation_timeout", "operation_ambiguous"}
        else "failed_observing"
    )


def _fail_worker_agent_dispatch(
    store: JobStore, job_id: str, token: str, exc: HerdrError
) -> None:
    def fail(job: CursorJob) -> CursorJob | None:
        if job.worker_token != token or job.agent_dispatch_state != "dispatching":
            return None
        return job.evolve(
            agent_dispatch_state=_failed_operation_state(exc),
            agent_dispatch_exited=True,
            agent_reconcile_attempts=0,
            agent_next_reconcile_at=time.time(),
            worker_operation=None,
        )

    store.update(job_id, fail)


def _reserve_worker_worktree(
    store: JobStore,
    job_id: str,
    token: str,
    repository: Path,
    branch: str,
    checkout: Path,
    *,
    state: str,
) -> CursorJob | None:
    if state not in {"planned", "dispatching", "ready"}:
        raise HarnessError("invalid worktree provisioning state")
    checkout_value = str(checkout.resolve())

    def reserve(job: CursorJob) -> CursorJob | None:
        if (
            job.worker_token != token
            or job.status.value != "routing"
            or job.terminal_intent_status is not None
        ):
            return None
        return transition(
            job,
            job.status,
            repository=str(repository.resolve()),
            worktree_branch=branch,
            worktree_path=checkout_value,
            worktree_provision_state=state,
            worker_operation="worktree_create" if state == "dispatching" else None,
        )

    try:
        reserved = store.reserve_worktree(job_id, reserve)
    except JobValidationError as exc:
        if "reserved by both" in str(exc):
            return None
        raise
    return reserved


def _settle_worker_worktree(
    store: JobStore,
    job_id: str,
    token: str,
    checkout: Path,
    workspace_id: str | None,
    pane_id: str | None,
) -> CursorJob | None:
    checkout_value = str(checkout.resolve())

    def settle(job: CursorJob) -> CursorJob | None:
        if (
            job.worker_token != token
            or job.worktree_path != checkout_value
            or job.status.value not in {"routing", "cancelled", "failed"}
        ):
            return None
        return transition(
            job,
            job.status,
            worktree_provision_state=(
                "retained" if job.status.value == "cancelled" else "ready"
            ),
            worktree_workspace_id=workspace_id,
            worktree_root_pane_id=pane_id,
            worker_operation=(
                None
                if job.worker_operation == "worktree_create"
                else job.worker_operation
            ),
        )

    return store.update(job_id, settle)


def _fail_worker_worktree(
    store: JobStore, job_id: str, token: str, exc: HerdrError
) -> None:
    def fail(job: CursorJob) -> CursorJob | None:
        if job.worker_token != token or job.worktree_provision_state != "dispatching":
            return None
        return job.evolve(
            worktree_provision_state=_failed_operation_state(exc),
            worktree_dispatch_exited=True,
            worktree_reconcile_attempts=0,
            worktree_next_reconcile_at=time.time(),
            worker_operation=None,
        )

    store.update(job_id, fail)


def _begin_fork_operation(
    store: JobStore,
    job_id: str,
    token: str,
    source: GitHubRepository,
    login: str,
    target: str,
) -> CursorJob | None:
    def begin(job: CursorJob) -> CursorJob:
        return job.evolve(
            github_repository=source.name_with_owner,
            fork_operation_source=source.name_with_owner,
            fork_operation_source_url=source.url,
            fork_operation_source_parent=source.parent,
            fork_operation_source_default_branch=source.default_branch,
            fork_operation_source_private=source.is_private,
            fork_operation_login=login,
            fork_operation_target=target,
            fork_operation_state="planned",
            worker_operation=None,
        )

    return _worker_change(store, job_id, token, {JobStatus.ROUTING}, begin)


def _mark_fork_dispatching(store: JobStore, job_id: str, token: str) -> None:
    def dispatch(job: CursorJob) -> CursorJob:
        return job.evolve(
            fork_operation_state="submitted",
            fork_committed=True,
            fork_committed_at=time.time(),
            worker_operation="fork_create",
        )

    if _worker_change(store, job_id, token, {JobStatus.ROUTING}, dispatch) is None:
        raise WorkerCancelled


def _settle_fork_operation(
    store: JobStore,
    job_id: str,
    token: str,
    fork: GitHubRepository | None,
    *,
    ambiguous: bool = False,
    failed_observing: bool = False,
) -> CursorJob | None:
    def settle(job: CursorJob) -> CursorJob | None:
        if (
            job.worker_token != token
            or job.fork_operation_state not in {"planned", "submitted"}
            or job.status.value not in {"routing", "cancelled", "failed"}
        ):
            return None
        if fork is not None:
            return transition(
                job,
                job.status,
                fork_repository=fork.name_with_owner,
                fork_operation_state="exists",
                fork_exists=True,
                worker_operation=None,
            )
        if ambiguous or failed_observing:
            return transition(
                job,
                job.status,
                fork_operation_state=(
                    "failed_observing" if failed_observing else "ambiguous"
                ),
                fork_exists=None,
                fork_dispatch_exited=True,
                fork_reconcile_attempts=0,
                fork_next_reconcile_at=time.time(),
                worker_operation=None,
            )
        return transition(
            job,
            job.status,
            fork_operation_state="failed",
            fork_exists=False,
            worker_operation=None,
        )

    return store.update(job_id, settle)


def _validate_followup_checkout(
    client: HerdrClient, job: CursorJob
) -> tuple[Path, Path, str, dict[str, object]]:
    """Confirm a follow-up child's inherited checkout still exists in Herdr."""
    repository = Path(job.repository or "").resolve()
    checkout = Path(job.worktree_path or "").resolve()
    branch = job.worktree_branch or ""
    if not client.allowed_repository(repository):
        raise HarnessError("follow-up parent repository is not allowed")
    if checkout == repository:
        raise HarnessError("refusing to follow up in the shared repository clone")
    if not checkout.is_dir() or not (checkout / ".git").exists():
        raise HarnessError("follow-up worktree is missing")
    listing = client.run_json("worktree", "list", "--cwd", str(repository), "--json")
    match = next(
        (
            item
            for item in listing.get("worktrees") or []
            if item.get("branch") == branch
            and Path(str(item.get("path") or "")).resolve() == checkout
        ),
        None,
    )
    if match is None:
        raise HarnessError("follow-up worktree no longer matches Herdr")
    return repository, checkout, branch, match


def _validate_followup_agent_cwd(checkout: Path, cwd: object) -> None:
    """Require a follow-up agent to run in the retained checkout."""
    value = str(cwd or "").strip()
    if not value:
        raise HarnessError(
            "follow-up Cursor agent did not report its working directory"
        )
    try:
        agent_checkout = Path(value).resolve()
    except OSError as exc:
        raise HarnessError(
            "follow-up Cursor agent working directory is invalid"
        ) from exc
    if agent_checkout != checkout:
        raise HarnessError("follow-up Cursor agent is attached to a different checkout")


def _validate_followup_agent_binding(
    checkout: Path,
    *,
    cwd: object,
    pane_id: object,
    workspace_id: object,
    expected_pane_id: str | None,
    expected_workspace_id: str | None,
) -> None:
    _validate_followup_agent_cwd(checkout, cwd)
    if expected_pane_id and str(pane_id or "") != expected_pane_id:
        raise HarnessError("follow-up Cursor agent pane does not match its reservation")
    if expected_workspace_id and str(workspace_id or "") != expected_workspace_id:
        raise HarnessError(
            "follow-up Cursor agent workspace does not match its reservation"
        )


def _provision_followup_agent(
    store: JobStore,
    job_id: str,
    token: str,
    job: CursorJob,
    client: HerdrClient,
    reserved: set[str],
    checkpoint: Callable[[], None],
) -> tuple[CursorJob, str]:
    """Reserve a settled agent at the child's exact retained checkout."""
    repository, checkout, _branch, match = _validate_followup_checkout(client, job)
    checkpoint()
    label = job.worktree_label or repository.name
    workspace_id = job.worktree_workspace_id or ""
    if not workspace_id or not job.worktree_root_pane_id:
        raise HarnessError(
            "follow-up worktree has no retained Herdr pane for safe agent startup"
        )
    open_workspace_id = str(match.get("open_workspace_id") or "")
    if open_workspace_id and open_workspace_id != workspace_id:
        raise HarnessError("follow-up worktree is open in a different Herdr workspace")
    if (
        job.participant_creation_state == "planned"
        and job.participant_creation_participant == WorkflowParticipant.PLANNER
        and job.participant_creation_target
    ):
        name = job.participant_creation_target
        workspace_id = job.participant_creation_workspace_id or workspace_id
    else:
        name = f"voice-{normalize_name(label)[:15] or 'task'}-{uuid.uuid4().hex[:10]}"
        job = _plan_participant_creation(
            store,
            job,
            token,
            WorkflowParticipant.PLANNER,
            target=name,
            label=f"{label}-planner",
            workspace_id=workspace_id,
        )
    before_pane_submit, pane_accepted = _participant_pane_callbacks(
        store, job_id, token, name
    )
    try:
        before_pane_submit()
        pane, workspace_id = client.new_pane(
            checkout,
            f"{label}-planner",
            workspace_id,
            checkpoint=checkpoint,
        )
        pane_accepted(pane, workspace_id)
    except HerdrError as exc:
        _fence_participant_creation(store, job_id, token, exc)
        raise
    provisional = AgentSelection(
        target=name,
        pane_id=pane,
        workspace_id=workspace_id,
        cwd=str(checkout),
        name=name,
        worktree_path=job.worktree_path,
    )
    reserved_job = _reserve_worker_target(
        store,
        job_id,
        token,
        provisional,
        repository,
        job.issue_key,
        dispatching=True,
        participant=WorkflowParticipant.PLANNER,
    )
    if reserved_job is None:
        raise HarnessError("could not reserve a Cursor agent for the follow-up")
    checkpoint()
    try:
        selection = client.start_agent(
            checkout,
            label,
            pane,
            workspace_id,
            name=name,
            mode="plan",
            checkpoint=checkpoint,
        )
    except HerdrError as exc:
        _fail_worker_agent_dispatch(store, job_id, token, exc)
        raise
    if selection.target != name:
        error = HerdrError(
            "Herdr started a different follow-up agent than the reserved target",
            code="operation_ambiguous",
        )
        _fail_worker_agent_dispatch(store, job_id, token, error)
        raise error
    _validate_followup_agent_binding(
        checkout,
        cwd=selection.cwd,
        pane_id=selection.pane_id,
        workspace_id=selection.workspace_id,
        expected_pane_id=pane,
        expected_workspace_id=workspace_id,
    )
    reserved_job = _settle_worker_agent(store, job_id, token, selection)
    if reserved_job is None:
        raise HarnessError("could not reserve a Cursor agent for the follow-up")
    return reserved_job, selection.target


_HARD_RISK_TERMS = (
    "authentication",
    "authorization",
    "oauth",
    "permission",
    "credential",
    "secret",
    "privacy",
    "security",
    "migration",
    "schema",
    "database",
    "persistence",
    "recovery",
    "concurrency",
    "race condition",
    "thread",
    "locking",
    "lifecycle",
    "cancellation",
    "worker ownership",
    "worktree",
    "external write",
    "fork",
    "infrastructure",
    "systemd",
    "deployment",
    "terraform",
    "kubernetes",
    "public api",
    "destructive",
    "ambiguous",
    "unclear acceptance",
)
_MAX_RISK_EVIDENCE_BYTES = 16 * 1024
_NEGATED_REVIEW_RISK = re.compile(
    r"\b(?:no|without)\s+(?:known\s+)?"
    r"(?:[a-z-]+\s+){0,6}(?:concerns?|risks?|issues?)\b"
    r"|\b(?:does|did)\s+not\s+(?:raise|introduce|create)\s+"
    r"(?:[a-z-]+\s+){0,6}(?:concerns?|risks?|issues?)\b"
    r"|\b(?:[a-z-]+\s+){1,4}(?:is|are)\s+not\s+"
    r"(?:a\s+)?(?:concern|risk|issue)\b"
)


def _bounded_risk_text(value: str | None) -> str:
    if not value:
        return ""
    return value.encode()[:_MAX_RISK_EVIDENCE_BYTES].decode(errors="ignore").casefold()


def _hard_risk_evidence(
    request: str,
    reason: str,
    github_issue_context: str | None = None,
    *,
    plan: str | None = None,
    review: str | None = None,
) -> str | None:
    for source, value in (
        ("request", request),
        ("GitHub issue context", github_issue_context),
        ("classification or promotion reason", reason),
        ("approved plan", plan),
        ("review", review),
    ):
        evidence = _bounded_risk_text(value)
        if source == "review":
            evidence = _NEGATED_REVIEW_RISK.sub("", evidence)
        for term in _HARD_RISK_TERMS:
            if term in evidence:
                return f"{source} contains {term!r}"
    return None


def _classified_tier(
    value: str,
    request: str,
    reason: str,
    github_issue_context: str | None = None,
) -> WorkflowTier:
    normalized = value.strip().casefold().replace("_", "-")
    try:
        tier = WorkflowTier(normalized)
    except ValueError as exc:
        raise HarnessError("Cursor returned an invalid workflow tier") from exc
    if _hard_risk_evidence(request, reason, github_issue_context):
        return WorkflowTier.HIGH_RISK
    return tier


def _auto_plan_approval_allowed(
    job: CursorJob,
    *,
    plan: str,
    review: str,
    reviewer_approved: bool | None = None,
) -> bool:
    """Fail closed unless this is only the reviewed Plan Mode Build gate."""

    approved = (
        job.review_approved and job.review_decision == "approve"
        if reviewer_approved is None
        else reviewer_approved
    )
    if (
        job.plan_approval_state != "boundary"
        or not approved
        or job.workflow_tier not in {WorkflowTier.SIMPLE, WorkflowTier.MEDIUM}
    ):
        return False
    if _hard_risk_evidence(
        job.request,
        job.workflow_classification_reason or "",
        job.github_issue_context,
        plan=plan,
        review=review,
    ):
        return False
    try:
        return load_plan_approval_preferences().mode == PlanApprovalMode.AUTO
    except (OSError, UserConfigurationError):
        return False


def _workflow_question(
    store: JobStore,
    job_id: str,
    token: str,
    question: str,
    *,
    clarification_kind: str = "workflow",
    job_changes: Mapping[str, object] | None = None,
    sensitivity: QuestionSensitivity = QuestionSensitivity.ROUTINE,
) -> None:
    try:
        spec = parse_question_spec(question)
    except QuestionError:
        spec = QuestionSpec(question, sensitivity=sensitivity)
    if (
        sensitivity != QuestionSensitivity.ROUTINE
        and spec.sensitivity == QuestionSensitivity.UNSPECIFIED
    ):
        spec = QuestionSpec(
            spec.text,
            kind=spec.kind,
            choices=spec.choices,
            sensitivity=sensitivity,
        )

    def ask(job: CursorJob) -> CursorJob:
        now = time.time()
        asked = question_adapter.ask(
            job,
            spec,
            owner=clarification_kind,
            turn_token=job.turn_token or token,
            now=now,
            clear_worker=True,
            remove_reconcile=True,
            prompt_operation_state="none",
            job_changes=job_changes,
        )
        return asked

    _worker_change(store, job_id, token, MODEL_WORKER_STATUSES, ask)


def _workflow_block(store: JobStore, job_id: str, token: str, message: str) -> None:
    def block(job: CursorJob) -> CursorJob:
        now = time.time()
        return job.evolve_for_delivery(
            now=now,
            status=JobStatus.BLOCKED,
            remove=frozenset({"reconcile"}),
            result=message[:500],
            completed_at=now,
            prompt_operation_state="none",
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
        )

    _worker_change(store, job_id, token, MODEL_WORKER_STATUSES, block)


def _begin_phase_turn(
    store: JobStore, job_id: str, worker_token: str
) -> CursorJob | None:
    def begin(job: CursorJob) -> CursorJob:
        turn = job.turn + 1
        return job.evolve(
            turn=turn,
            turn_token=f"{job.id}-{turn}",
            workflow_turn_phase=job.workflow_phase.value,
            prompt_operation_state="none",
            prompt_operation_phase=None,
            prompt_operation_turn=None,
            prompt_operation_target=None,
            prompt_baseline_sequence=None,
        )

    return _worker_change(store, job_id, worker_token, MODEL_WORKER_STATUSES, begin)


def _record_plan_boundary(
    store: JobStore,
    job: CursorJob,
    worker_token: str,
    output: str,
    *,
    agent_status: str,
    agent_session: str | None,
    state_change_sequence: int | None,
    revision: int | None,
) -> CursorJob:
    token = job.turn_token or ""
    plan = extract_marker(output, "WORKFLOW_PLAN", token)
    if not plan:
        return job
    if agent_status not in {"idle", "done", "working", "blocked"}:
        raise HarnessError("Cursor Plan Mode boundary has an unsupported agent state")
    if agent_session is None or state_change_sequence is None:
        raise HarnessError(
            "Cursor Plan Mode boundary is missing durable Herdr session proof"
        )
    gate_material = (
        f"{job.id}\0{job.review_round}\0{token}\0{agent_session}\0"
        f"{state_change_sequence}\0{plan}"
    )
    gate_id = hashlib.sha256(gate_material.encode()).hexdigest()

    def record(current: CursorJob) -> CursorJob:
        if (
            current.workflow_phase != job.workflow_phase
            or current.turn_token != token
            or current.plan_approval_state not in {"none", "boundary"}
        ):
            raise WorkerCancelled
        return current.evolve(
            plan_approval_state="boundary",
            plan_approval_id=gate_id,
            plan_approval_source=None,
            plan_approval_agent_session=agent_session,
            plan_approval_state_change_sequence=state_change_sequence,
            plan_approval_revision=revision,
            plan_approval_counted=False,
        )

    recorded = _worker_change(
        store,
        job.id,
        worker_token,
        MODEL_WORKER_STATUSES,
        record,
    )
    if recorded is None:
        raise WorkerCancelled
    return recorded


def _accepted_explicit_plan_approval(job: CursorJob) -> bool:
    approval_id = job.plan_approval_id
    if job.plan_approval_source != "explicit" or approval_id is None:
        return False
    try:
        preferences = record_explicit_plan_approval(approval_id)
    except (OSError, UserConfigurationError):
        return False
    return approval_id in preferences.explicit_approval_ids


def _advance_workflow_output(
    store: JobStore,
    job: CursorJob,
    worker_token: str,
    output: str,
    agent_status: str,
) -> CursorJob | None:
    token = job.turn_token or ""
    question = extract_marker(output, "VOICE_QUESTION", token)
    if question:
        _workflow_question(store, job.id, worker_token, question)
        return None

    if job.workflow_phase == WorkflowPhase.CLASSIFYING:
        raw_tier = extract_marker(output, "WORKFLOW_TIER", token)
        reason = extract_marker(output, "WORKFLOW_REASON", token)
        if not raw_tier or not reason:
            _workflow_block(
                store,
                job.id,
                worker_token,
                "Cursor classification ended without valid tier and reason markers.",
            )
            return None
        try:
            tier = _classified_tier(
                raw_tier,
                job.request,
                reason,
                job.github_issue_context,
            )
        except HarnessError as exc:
            _workflow_block(store, job.id, worker_token, str(exc))
            return None
        hard_risk_evidence = _hard_risk_evidence(
            job.request,
            reason,
            job.github_issue_context,
        )
        classification_reason = reason
        if hard_risk_evidence:
            classification_reason = (
                f"{reason} Deterministic hard-risk floor: {hard_risk_evidence}."
            )
        phase = (
            WorkflowPhase.IMPLEMENTING
            if tier == WorkflowTier.SIMPLE
            else WorkflowPhase.PLANNING
        )
        return _worker_change(
            store,
            job.id,
            worker_token,
            MODEL_WORKER_STATUSES,
            lambda current: current.evolve(
                workflow_tier=tier.value,
                workflow_classification_reason=classification_reason[:500],
                workflow_phase=phase.value,
                prompt_operation_state="none",
                review_approved=False,
                review_decision=None,
                review_approval_source=None,
            ),
        )

    if job.workflow_phase in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING}:
        plan = extract_marker(output, "WORKFLOW_PLAN", token)
        if not plan:
            _workflow_block(
                store,
                job.id,
                worker_token,
                "Cursor planner ended without a valid plan marker.",
            )
            return None
        return store.publish_artifact(
            job.id,
            "plan",
            job.review_round,
            plan,
            expected_worker_token=worker_token,
            expected_turn_token=token,
            expected_phase=job.workflow_phase.value,
            expected_prior_reference=job.plan_artifact,
            change=lambda current, reference: current.evolve(
                workflow_phase=WorkflowPhase.REVIEWING.value,
                plan_artifact=reference,
                prompt_operation_state="none",
                review_approved=False,
                review_decision=None,
                review_approval_source=None,
            ),
        )

    if job.workflow_phase == WorkflowPhase.REVIEWING:
        decision = (
            extract_marker(output, "WORKFLOW_REVIEW_DECISION", token) or ""
        ).casefold()
        review = extract_marker(output, "WORKFLOW_REVIEW", token)
        if decision not in {"approve", "revise"} or not review:
            _workflow_block(
                store,
                job.id,
                worker_token,
                "Cursor reviewer ended without valid review markers.",
            )
            return None
        if job.plan_artifact is None:
            _workflow_block(
                store,
                job.id,
                worker_token,
                "Cursor reviewer has no current plan artifact.",
            )
            return None
        if decision == "approve":
            planner_target = job.participant_target(WorkflowParticipant.PLANNER)
            if not planner_target:
                _workflow_block(
                    store,
                    job.id,
                    worker_token,
                    "The approved plan has no live Plan Mode agent.",
                )
                return None
            plan = store.read_artifact(job.id, job.plan_artifact, kind="plan")
            auto_approval = _auto_plan_approval_allowed(
                job,
                plan=plan,
                review=review,
                reviewer_approved=True,
            )
            now = time.time()

            def approve_review(current: CursorJob, reference: str) -> CursorJob:
                if auto_approval:
                    return current.evolve(
                        workflow_phase=WorkflowPhase.IMPLEMENTING.value,
                        active_participant=WorkflowParticipant.PLANNER.value,
                        herdr_target=planner_target,
                        review_artifact=reference,
                        prompt_operation_state="none",
                        review_approved=True,
                        review_decision="approve",
                        review_approval_source="reviewer",
                        plan_approval_state="approved",
                        plan_approval_source="auto",
                        plan_approval_counted=False,
                    )
                return question_adapter.ask(
                    current,
                    QuestionSpec(
                        "The reviewed implementation plan is ready. Should I "
                        "approve Cursor's Build step and start implementation? "
                        "Say yes to implement it, or no to cancel this job.",
                        sensitivity=QuestionSensitivity.ARCHITECTURE,
                    ),
                    owner="workflow_plan_approval",
                    turn_token=current.turn_token or token,
                    now=now,
                    clear_worker=True,
                    remove_reconcile=True,
                    prompt_operation_state="none",
                    job_changes={
                        "review_artifact": reference,
                        "review_approved": True,
                        "review_decision": "approve",
                        "review_approval_source": "reviewer",
                        "plan_approval_state": "awaiting",
                    },
                )

            published = store.publish_artifact(
                job.id,
                "review",
                job.review_round,
                review,
                expected_worker_token=worker_token,
                expected_turn_token=token,
                expected_phase=job.workflow_phase.value,
                expected_prior_reference=job.review_artifact,
                expected_plan_reference=job.plan_artifact,
                change=approve_review,
            )
            if published is None:
                return None
            return published if auto_approval else None
        if job.workflow_tier == WorkflowTier.MEDIUM:
            published = store.publish_artifact(
                job.id,
                "review",
                job.review_round,
                review,
                expected_worker_token=worker_token,
                expected_turn_token=token,
                expected_phase=job.workflow_phase.value,
                expected_prior_reference=job.review_artifact,
                expected_plan_reference=job.plan_artifact,
                change=lambda current, reference: current.evolve(
                    review_artifact=reference,
                    review_approved=False,
                    review_decision="revise",
                    review_approval_source=None,
                ),
            )
            if published is None:
                return None
            _workflow_question(
                store,
                job.id,
                worker_token,
                f"The plan review needs your decision: {review}"[:500],
                clarification_kind="workflow_review",
            )
            return None
        if job.review_round >= 1:
            published = store.publish_artifact(
                job.id,
                "review",
                job.review_round,
                review,
                expected_worker_token=worker_token,
                expected_turn_token=token,
                expected_phase=job.workflow_phase.value,
                expected_prior_reference=job.review_artifact,
                expected_plan_reference=job.plan_artifact,
                change=lambda current, reference: current.evolve(
                    review_artifact=reference,
                    review_approved=False,
                    review_decision="revise",
                    review_approval_source=None,
                ),
            )
            if published is None:
                return None
            _workflow_question(
                store,
                job.id,
                worker_token,
                "The final plan review still has unresolved findings. "
                "Say approve to implement the reviewed plan anyway, or abort to "
                f"cancel the job: {review}"[:500],
                clarification_kind="workflow_review_exhausted",
            )
            return None
        planner_target = job.participant_target(WorkflowParticipant.PLANNER)
        if not planner_target:
            _workflow_block(
                store,
                job.id,
                worker_token,
                "The planner agent is unavailable for the required revision.",
            )
            return None
        return store.publish_artifact(
            job.id,
            "review",
            job.review_round,
            review,
            expected_worker_token=worker_token,
            expected_turn_token=token,
            expected_phase=job.workflow_phase.value,
            expected_prior_reference=job.review_artifact,
            expected_plan_reference=job.plan_artifact,
            change=lambda current, reference: current.evolve(
                workflow_phase=WorkflowPhase.REVISING.value,
                review_artifact=reference,
                review_round=current.review_round + 1,
                active_participant=WorkflowParticipant.PLANNER.value,
                herdr_target=planner_target,
                prompt_operation_state="none",
                review_approved=False,
                review_decision="revise",
                review_approval_source=None,
                plan_approval_state="none",
                plan_approval_id=None,
                plan_approval_source=None,
                plan_approval_agent_session=None,
                plan_approval_state_change_sequence=None,
                plan_approval_revision=None,
                plan_approval_counted=False,
            ),
        )

    if job.workflow_phase == WorkflowPhase.IMPLEMENTING:
        promotion = extract_marker(output, "WORKFLOW_PROMOTE", token)
        promotion_reason = extract_marker(output, "WORKFLOW_REASON", token)
        if promotion:
            if not promotion_reason:
                _workflow_block(
                    store,
                    job.id,
                    worker_token,
                    "Cursor requested promotion without a reason.",
                )
                return None
            try:
                promoted = _classified_tier(
                    promotion,
                    job.request,
                    promotion_reason,
                    job.github_issue_context,
                )
            except HarnessError as exc:
                _workflow_block(store, job.id, worker_token, str(exc))
                return None
            current_order = {
                WorkflowTier.SIMPLE: 0,
                WorkflowTier.MEDIUM: 1,
                WorkflowTier.HIGH_RISK: 2,
            }
            if (
                job.workflow_tier is None
                or current_order[promoted] <= current_order[job.workflow_tier]
            ):
                _workflow_block(
                    store,
                    job.id,
                    worker_token,
                    "Cursor requested a workflow promotion that did not increase risk.",
                )
                return None
            planner_target = job.participant_target(WorkflowParticipant.PLANNER)
            if not planner_target:
                _workflow_block(
                    store,
                    job.id,
                    worker_token,
                    "The planner agent is unavailable for workflow promotion.",
                )
                return None
            hard_risk_evidence = _hard_risk_evidence(
                job.request,
                promotion_reason,
                job.github_issue_context,
            )
            classification_reason = promotion_reason
            if hard_risk_evidence:
                classification_reason = (
                    f"{promotion_reason} Deterministic hard-risk floor: "
                    f"{hard_risk_evidence}."
                )
            return _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                lambda current: current.evolve(
                    request=_request_with_clarification(current),
                    remove=frozenset({"continuation", "continuation_answer"}),
                    workflow_tier=promoted.value,
                    workflow_classification_reason=classification_reason[:500],
                    workflow_phase=WorkflowPhase.PLANNING.value,
                    active_participant=WorkflowParticipant.PLANNER.value,
                    herdr_target=planner_target,
                    review_round=0,
                    plan_artifact=None,
                    review_artifact=None,
                    prompt_operation_state="none",
                    review_approved=False,
                    review_decision=None,
                    review_approval_source=None,
                    plan_approval_state="none",
                    plan_approval_id=None,
                    plan_approval_source=None,
                    plan_approval_agent_session=None,
                    plan_approval_state_change_sequence=None,
                    plan_approval_revision=None,
                    plan_approval_counted=False,
                ),
            )
        _worker_complete(
            store,
            job.id,
            worker_token,
            output=output,
            agent_status=agent_status,
        )
        return None

    _workflow_block(
        store,
        job.id,
        worker_token,
        f"Cursor returned output for unsupported workflow phase "
        f"{job.workflow_phase.value}.",
    )
    return None


def _plan_participant_creation(
    store: JobStore,
    job: CursorJob,
    worker_token: str,
    participant: WorkflowParticipant,
    *,
    target: str,
    label: str,
    workspace_id: str,
) -> CursorJob:
    planned = _worker_change(
        store,
        job.id,
        worker_token,
        MODEL_WORKER_STATUSES,
        lambda current: current.evolve(
            participant_creation_state="planned",
            participant_creation_participant=participant.value,
            participant_creation_target=target,
            participant_creation_label=label,
            participant_creation_workspace_id=workspace_id,
            participant_creation_pane_id=None,
        ),
    )
    if planned is None:
        raise WorkerCancelled
    return planned


def _participant_pane_callbacks(
    store: JobStore,
    job_id: str,
    worker_token: str,
    target: str,
) -> tuple[Callable[[], None], Callable[[str, str], None]]:
    def before_submit() -> None:
        updated = _worker_change(
            store,
            job_id,
            worker_token,
            MODEL_WORKER_STATUSES,
            lambda current: current.evolve(participant_creation_state="submitting"),
        )
        if updated is None:
            raise WorkerCancelled

    def accepted(pane_id: str, workspace_id: str) -> None:
        updated = _worker_change(
            store,
            job_id,
            worker_token,
            MODEL_WORKER_STATUSES,
            lambda current: current.evolve(
                participant_creation_state="created",
                participant_creation_pane_id=pane_id,
                participant_creation_workspace_id=workspace_id,
                herdr_pane_id=pane_id,
                herdr_workspace_id=workspace_id,
                agent_name=target,
            ),
        )
        if updated is None:
            raise WorkerCancelled

    return before_submit, accepted


def _fence_participant_creation(
    store: JobStore, job_id: str, worker_token: str, exc: HerdrError
) -> None:
    def fence(current: CursorJob) -> CursorJob | None:
        if current.participant_creation_state not in {"planned", "submitting"}:
            return None
        state = (
            "ambiguous"
            if current.participant_creation_state == "submitting"
            or exc.code in {"operation_timeout", "operation_ambiguous"}
            else "none"
        )
        return current.evolve(
            participant_creation_state=state,
            worker_operation=None,
        )

    store.update(job_id, fence)


def _ensure_workflow_participant(
    store: JobStore,
    job: CursorJob,
    worker_token: str,
    client: HerdrClient,
    participant: WorkflowParticipant,
    checkpoint: Callable[[], None],
) -> tuple[CursorJob, str]:
    target = job.participant_target(participant)
    if job.active_participant == participant and target == job.herdr_target and target:
        return job, target
    if participant == WorkflowParticipant.PLANNER and target:
        updated = _worker_change(
            store,
            job.id,
            worker_token,
            MODEL_WORKER_STATUSES,
            lambda current: current.evolve(
                active_participant=participant.value,
                herdr_target=target,
            ),
        )
        if updated is None:
            raise WorkerCancelled
        return updated, target

    # Non-planner roles must be fresh, but do not overwrite the only durable
    # handle to an earlier participant until that agent has been stopped.
    if target:
        try:
            client.cancel_agent(target)
        except HerdrError as exc:
            if not _agent_not_found(exc):
                raise HarnessError(
                    f"could not stop previous {participant.value} agent"
                ) from exc

    checkout_value = job.worktree_path or job.repository
    workspace = job.herdr_workspace_id or job.worktree_workspace_id or ""
    if not checkout_value or not workspace:
        raise HarnessError(
            "workflow participant requires a prepared checkout and workspace"
        )
    checkout = Path(checkout_value).resolve()
    selection_holder: dict[str, AgentSelection] = {}
    role_label = f"{job.worktree_label or checkout.name}-{participant.value}"
    if (
        job.participant_creation_state == "planned"
        and job.participant_creation_participant == participant
        and job.participant_creation_target
    ):
        name = job.participant_creation_target
        role_label = job.participant_creation_label or role_label
        workspace = job.participant_creation_workspace_id or workspace
    else:
        name = (
            f"voice-{normalize_name(participant.value)[:8] or 'agent'}-"
            f"{normalize_name(job.worktree_label or checkout.name)[:7] or 'task'}-"
            f"{uuid.uuid4().hex[:10]}"
        )[:32]
        job = _plan_participant_creation(
            store,
            job,
            worker_token,
            participant,
            target=name,
            label=role_label,
            workspace_id=workspace,
        )
    before_pane_submit, pane_accepted = _participant_pane_callbacks(
        store, job.id, worker_token, name
    )

    def reserve(selection: AgentSelection, dispatching: bool) -> None:
        reserved = _reserve_worker_target(
            store,
            job.id,
            worker_token,
            selection,
            Path(job.repository or checkout),
            job.issue_key,
            dispatching=dispatching,
            participant=participant,
        )
        if reserved is None:
            raise WorkerCancelled
        selection_holder["reserved"] = selection

    def settle(selection: AgentSelection) -> None:
        if _settle_worker_agent(store, job.id, worker_token, selection) is None:
            raise WorkerCancelled

    try:
        selection = client.start_fresh_agent(
            checkout,
            job.worktree_label or checkout.name,
            workspace,
            role=participant.value,
            mode=(
                "ask"
                if participant == WorkflowParticipant.REVIEWER
                else ("plan" if participant == WorkflowParticipant.PLANNER else None)
            ),
            name=name,
            checkpoint=checkpoint,
            reserve=reserve,
            settle=settle,
            fail_agent=lambda exc: _fail_worker_agent_dispatch(
                store, job.id, worker_token, exc
            ),
            before_pane_submit=before_pane_submit,
            pane_accepted=pane_accepted,
        )
    except HerdrError as exc:
        _fence_participant_creation(store, job.id, worker_token, exc)
        raise
    current = store.get(job.id)
    return current, selection.target


def _execute_phase_prompt(
    store: JobStore,
    job: CursorJob,
    worker_token: str,
    client: HerdrClient,
    checkpoint: Callable[[], None],
    *,
    target: str,
    prompt: str,
    token: str,
) -> tuple[str, str] | None:
    phase = job.workflow_phase
    state = job.prompt_operation_state
    if state == "none":
        checkpoint()
        baseline_agent = client.get_agent(target)
        checkpoint()
        raw_baseline = baseline_agent.get("state_change_seq")
        try:
            baseline = int(raw_baseline) if isinstance(raw_baseline, int | str) else 0
        except (TypeError, ValueError) as exc:
            raise HarnessError("Herdr returned an invalid prompt sequence") from exc
        if job.plan_approval_state == "approved" and (
            baseline != job.plan_approval_state_change_sequence
            or agent_session_identity(baseline_agent.get("agent_session"))
            != job.plan_approval_agent_session
        ):
            raise HarnessError(
                "Cursor Plan Mode boundary changed before approval submission"
            )

        def plan_prompt(current: CursorJob) -> CursorJob:
            if (
                current.workflow_phase != phase
                or current.turn_token != token
                or current.herdr_target != target
            ):
                raise WorkerCancelled
            return current.evolve(
                prompt_operation_state="planned",
                prompt_operation_phase=phase.value,
                prompt_operation_turn=current.turn,
                prompt_operation_target=target,
                prompt_baseline_sequence=baseline,
                continuation=(
                    False
                    if phase == WorkflowPhase.IMPLEMENTING
                    else current.continuation
                ),
            )

        planned = _worker_change(
            store,
            job.id,
            worker_token,
            MODEL_WORKER_STATUSES,
            plan_prompt,
        )
        if planned is None:
            return None
        job = planned
        state = "planned"

    if state == "submitting":
        checkpoint()
        observed = client.get_agent(target)
        checkpoint()
        sequence = int(observed.get("state_change_seq") or 0)
        approval_session_matches = (
            job.plan_approval_state != "approved"
            or agent_session_identity(observed.get("agent_session"))
            == job.plan_approval_agent_session
        )
        if (
            approval_session_matches
            and job.prompt_baseline_sequence is not None
            and job.prompt_baseline_sequence >= 0
            and sequence != job.prompt_baseline_sequence
        ):
            counted = _accepted_explicit_plan_approval(job)
            submitted = _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                lambda current: current.evolve(
                    prompt_operation_state="submitted",
                    plan_approval_counted=(current.plan_approval_counted or counted),
                ),
            )
            if submitted is None:
                return None
            job = submitted
            state = "submitted"
        else:
            _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                lambda current: current.evolve(
                    prompt_operation_state="ambiguous",
                    manual_reconcile_operation="prompt",
                    manual_reconcile_token=uuid.uuid4().hex,
                    manual_reconcile_required_at=time.time(),
                ),
            )
            raise HarnessError(
                "Cursor prompt submission is ambiguous and requires manual "
                "reconciliation"
            )

    if state in {"submitted", "ambiguous"}:
        if state == "ambiguous":
            raise HarnessError(
                "Cursor prompt submission requires manual reconciliation"
            )
        if (
            job.plan_approval_source == "explicit"
            and not job.plan_approval_counted
            and _accepted_explicit_plan_approval(job)
        ):
            counted = _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                lambda current: current.evolve(plan_approval_counted=True),
            )
            if counted is None:
                return None
            job = counted
        try:
            outcome = client.wait_for_stable_completion(
                target,
                token=job.turn_token or "",
                checkpoint=checkpoint,
                expected_agent_session=(
                    job.plan_approval_agent_session
                    if phase == WorkflowPhase.IMPLEMENTING
                    and job.plan_approval_source in {"auto", "explicit"}
                    else None
                ),
                active_marker=(
                    "WORKFLOW_PLAN"
                    if phase in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING}
                    else None
                ),
            )
            completion = (outcome.output, outcome.status)
            if phase in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING}:
                job = _record_plan_boundary(
                    store,
                    job,
                    worker_token,
                    outcome.output,
                    agent_status=outcome.status,
                    agent_session=outcome.agent_session,
                    state_change_sequence=outcome.state_change_sequence,
                    revision=outcome.revision,
                )
            if (
                phase == WorkflowPhase.IMPLEMENTING
                and job.plan_approval_state == "approved"
            ):
                observed = _worker_change(
                    store,
                    job.id,
                    worker_token,
                    MODEL_WORKER_STATUSES,
                    lambda current: current.evolve(plan_approval_state="observed"),
                )
                if observed is None:
                    return None
            return completion
        except HerdrError:
            _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                lambda current: current.evolve(
                    status=JobStatus.QUEUED,
                    reconcile=True,
                    queued_at=time.time(),
                    worker_pid=None,
                    worker_boot_id=None,
                    worker_process_start=None,
                    worker_token=None,
                ),
            )
            raise WorkerCancelled from None

    if state != "planned":
        raise HarnessError(f"invalid durable prompt state {state}")

    operation_baseline = job.prompt_baseline_sequence

    def before_agent(agent: dict[str, object]) -> None:
        if job.plan_approval_state != "approved":
            return
        if (
            agent_session_identity(agent.get("agent_session"))
            != job.plan_approval_agent_session
            or _state_change_sequence(agent) != job.plan_approval_state_change_sequence
        ):
            raise HarnessError(
                "Cursor Plan Mode boundary changed before approval submission"
            )

    def before_submit(observed_baseline: int) -> None:
        if observed_baseline != operation_baseline:
            raise HarnessError("Cursor prompt baseline changed before submission")

        def mark_submitting(current: CursorJob) -> CursorJob:
            if current.prompt_operation_state != "planned":
                raise WorkerCancelled
            return current.evolve(prompt_operation_state="submitting")

        if (
            _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                mark_submitting,
            )
            is None
        ):
            raise WorkerCancelled

    def accepted() -> None:
        counted = _accepted_explicit_plan_approval(job)

        def mark_submitted(current: CursorJob) -> CursorJob:
            if current.prompt_operation_state != "submitting":
                raise WorkerCancelled
            return current.evolve(
                prompt_operation_state="submitted",
                plan_approval_counted=(current.plan_approval_counted or counted),
            )

        if (
            _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                mark_submitted,
            )
            is None
        ):
            raise WorkerCancelled

    try:
        outcome = client.prompt_and_wait(
            target,
            prompt,
            token=token,
            checkpoint=checkpoint,
            baseline_sequence=job.prompt_baseline_sequence,
            before_submit=before_submit,
            accepted=accepted,
            before_agent=before_agent,
            expected_agent_session=(
                job.plan_approval_agent_session
                if job.plan_approval_state == "approved"
                else None
            ),
            active_marker=(
                "WORKFLOW_PLAN"
                if phase in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING}
                else None
            ),
            allow_enter_fallback=job.plan_approval_state != "approved",
        )
    except HerdrError as exc:
        if exc.code == "interactive_questionnaire":
            current = store.get(job.id)
            if current.prompt_operation_state == "planned":
                raise
        # Third-party/test clients that fail before invoking the callback still
        # crossed the call boundary. Conservatively fence the operation.
        store.update(
            job.id,
            lambda current: (
                current.evolve(
                    prompt_operation_state="ambiguous",
                    manual_reconcile_operation="prompt",
                    manual_reconcile_token=uuid.uuid4().hex,
                    manual_reconcile_required_at=time.time(),
                )
                if current.prompt_operation_state
                in {"planned", "submitting", "submitted"}
                else None
            ),
        )
        raise
    checkpoint()
    if phase in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING}:
        job = _record_plan_boundary(
            store,
            job,
            worker_token,
            outcome.output,
            agent_status=outcome.status,
            agent_session=outcome.agent_session,
            state_change_sequence=outcome.state_change_sequence,
            revision=outcome.revision,
        )
    if phase == WorkflowPhase.IMPLEMENTING and job.plan_approval_state == "approved":
        observed = _worker_change(
            store,
            job.id,
            worker_token,
            MODEL_WORKER_STATUSES,
            lambda current: current.evolve(plan_approval_state="observed"),
        )
        if observed is None:
            return None
    return outcome.output, outcome.status


def _run_tiered_workflow(
    store: JobStore,
    job: CursorJob,
    worker_token: str,
    client: HerdrClient,
    checkpoint: Callable[[], None],
    *,
    initial_output: tuple[str, str] | None = None,
) -> None:
    pending_output = initial_output
    while True:
        checkpoint()
        job = store.get(job.id)
        if job.status not in MODEL_WORKER_STATUSES:
            return
        if job.plan_approval_completion_pending:
            _resume_plan_approval_completion(store, job, worker_token)
            return
        if pending_output is not None:
            output, agent_status = pending_output
            pending_output = None
            advanced = _advance_workflow_output(
                store, job, worker_token, output, agent_status
            )
            if advanced is None:
                return
            job = advanced
            next_turn = _begin_phase_turn(store, job.id, worker_token)
            if next_turn is None:
                return
            job = next_turn

        phase = job.workflow_phase
        participant = (
            WorkflowParticipant.REVIEWER
            if phase == WorkflowPhase.REVIEWING
            else (
                (
                    WorkflowParticipant.PLANNER
                    if job.plan_approval_state in {"approved", "observed"}
                    else WorkflowParticipant.IMPLEMENTER
                )
                if phase == WorkflowPhase.IMPLEMENTING
                else WorkflowParticipant.PLANNER
            )
        )
        job, target = _ensure_workflow_participant(
            store, job, worker_token, client, participant, checkpoint
        )
        token = job.turn_token or ""
        if not token or job.workflow_turn_phase != phase:
            next_turn = _begin_phase_turn(store, job.id, worker_token)
            if next_turn is None:
                return
            job = next_turn
            token = job.turn_token or ""
        active_issue_key = resolve_issue_reference(job.issue_key)
        integration_instructions = prompt_instructions(active_issue_key)
        if phase == WorkflowPhase.CLASSIFYING:
            prompt = classification_prompt(
                job.request,
                token,
                github_issue_context=job.github_issue_context,
                integration_instructions=integration_instructions,
            )
        elif phase == WorkflowPhase.PLANNING:
            assert job.workflow_tier is not None
            prompt = planning_prompt(
                job.request,
                token,
                tier=job.workflow_tier.value,
                github_issue_context=job.github_issue_context,
                classification_reason=job.workflow_classification_reason,
                integration_instructions=integration_instructions,
            )
        elif phase == WorkflowPhase.REVIEWING:
            if not job.plan_artifact or job.workflow_tier is None:
                raise HarnessError("review phase requires a durable plan")
            plan = store.read_artifact(job.id, job.plan_artifact, kind="plan")
            prompt = review_prompt(
                job.request,
                plan,
                token,
                tier=job.workflow_tier.value,
                github_issue_context=job.github_issue_context,
                classification_reason=job.workflow_classification_reason,
                integration_instructions=integration_instructions,
            )
        elif phase == WorkflowPhase.REVISING:
            if not job.plan_artifact or not job.review_artifact:
                raise HarnessError("revision phase requires plan and review artifacts")
            plan = store.read_artifact(job.id, job.plan_artifact, kind="plan")
            review = store.read_artifact(job.id, job.review_artifact, kind="review")
            prompt = revision_prompt(
                job.request,
                plan,
                review,
                token,
                github_issue_context=job.github_issue_context,
                classification_reason=job.workflow_classification_reason,
                integration_instructions=integration_instructions,
            )
        elif phase == WorkflowPhase.IMPLEMENTING:
            plan = (
                store.read_artifact(job.id, job.plan_artifact, kind="plan")
                if job.plan_artifact
                else None
            )
            issue_reference = job.issue_key or (
                f"issue {job.github_issue}" if job.github_issue else None
            )
            if job.plan_approval_state in {"approved", "observed"}:
                if plan is None:
                    raise HarnessError("plan approval requires a durable plan")
                prompt = plan_approval_prompt(
                    _prompt_request(job),
                    token,
                    plan=plan,
                    github_issue_context=job.github_issue_context,
                    classification_reason=job.workflow_classification_reason,
                    issue_reference=issue_reference,
                    integration_instructions=integration_instructions,
                )
            else:
                prompt = implementation_prompt(
                    _prompt_request(job),
                    token,
                    plan=plan,
                    continuation=job.continuation or bool(job.continuation_answer),
                    github_issue_context=job.github_issue_context,
                    classification_reason=job.workflow_classification_reason,
                    issue_reference=issue_reference,
                    integration_instructions=integration_instructions,
                )
        else:
            raise HarnessError(f"unsupported workflow phase {phase.value}")
        pending_output = _execute_phase_prompt(
            store,
            job,
            worker_token,
            client,
            checkpoint,
            target=target,
            prompt=prompt,
            token=token,
        )
        if pending_output is None:
            return


def _begin_prompt_turn(job: CursorJob, turn: int, turn_token: str) -> CursorJob:
    question = question_adapter.current(job)
    question_envelope = job.voice_question
    if job.continuation:
        if (
            question is not None
            and question.state == QuestionState.DISPATCHING
            and question.dispatch_token == turn_token
        ):
            question_envelope = question.to_dict()
        elif (
            question is None
            or question.state != QuestionState.ANSWERED
            or question.origin.job_id != job.id
            or question.origin.turn_token != job.turn_token
        ):
            raise HarnessError(
                "Cursor clarification no longer matches its originating turn"
            )
        else:
            question_envelope = question_adapter.envelope(
                question,
                QuestionState.DISPATCHING,
                dispatch_token=turn_token,
                prompt_state=PromptOperationState.PLANNED,
                prompt_baseline_seq=None,
                prompt_submitted_at=None,
                prompt_absent_observations=0,
            )
    elif question is not None and question.state == QuestionState.ANSWERED:
        question_envelope = question_adapter.envelope(question, QuestionState.RESOLVED)
    return job.evolve(
        turn=turn,
        turn_token=turn_token,
        voice_question=question_envelope,
        workflow_turn_phase=job.workflow_phase.value,
    )


def _prompt_turn_identity(job: CursorJob) -> tuple[int, str]:
    question = question_adapter.current(job)
    if (
        job.continuation
        and question is not None
        and question.state == QuestionState.DISPATCHING
        and question.dispatch_token
    ):
        return job.turn, question.dispatch_token
    turn = job.turn + 1
    return turn, f"{job.id}-{turn}"


def _state_change_sequence(agent: dict[str, object]) -> int | None:
    value = agent.get("state_change_seq")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mark_prompt_boundary(
    store: JobStore,
    job_id: str,
    worker_token: str,
    turn_token: str,
    *,
    state: PromptOperationState,
    agent: dict[str, object],
) -> None:
    def mark(job: CursorJob) -> CursorJob:
        question = question_adapter.current(job)
        if (
            question is None
            or question.state != QuestionState.DISPATCHING
            or question.dispatch_token != turn_token
        ):
            raise WorkerCancelled
        baseline = (
            _state_change_sequence(agent)
            if state == PromptOperationState.SUBMITTED
            else question.prompt_baseline_seq
        )
        return job.evolve(
            voice_question=question_adapter.envelope(
                question,
                QuestionState.DISPATCHING,
                prompt_state=state,
                prompt_baseline_seq=baseline,
                prompt_submitted_at=(
                    time.time()
                    if state == PromptOperationState.SUBMITTED
                    else question.prompt_submitted_at
                ),
            )
        )

    updated = _worker_change(
        store,
        job_id,
        worker_token,
        {JobStatus.RUNNING},
        mark,
    )
    if updated is None:
        raise WorkerCancelled


def _prompt_request(job: CursorJob) -> str:
    return job.continuation_answer or job.request


def _request_with_clarification(job: CursorJob) -> str:
    answer = job.continuation_answer
    if not answer or answer in job.request:
        return job.request
    return f"{job.request}\n\nUser clarification:\n{answer}"


def run_claimed_worker(  # pyright: ignore[reportGeneralTypeIssues]
    context: worker_lifecycle.WorkerContext,
    factories: ClientFactories | None = None,
) -> None:
    clients = factories or ClientFactories(HerdrClient, GitHubClient)
    store = context.store
    job_id = context.job.id
    job = context.job
    worker_token = context.token
    client: HerdrClient | None = None
    target = ""

    def checkpoint() -> None:
        context.checkpoint()

    try:
        if job.plan_approval_completion_pending:
            checkpoint()
            _resume_plan_approval_completion(store, job, worker_token)
            return
        active_issue_key = resolve_issue_reference(job.issue_key)
        if active_issue_key:
            require_issue_capabilities(active_issue_key)
        client = clients.herdr()
        checkpoint()
        client.ensure_server()
        checkpoint()
        followup_checkout: Path | None = None
        followup_target_verified = False
        if job.parent_job_id:
            _repository, followup_checkout, _branch, _match = (
                _validate_followup_checkout(client, job)
            )
            checkpoint()
        if job.reconcile:
            job = store.get(job_id)
            if followup_checkout is not None and job.herdr_target:
                agent = client.get_agent(job.herdr_target)
                if not agent:
                    raise HarnessError("follow-up Cursor agent is no longer available")
                _validate_followup_agent_binding(
                    followup_checkout,
                    cwd=agent.get("cwd"),
                    pane_id=agent.get("pane_id"),
                    workspace_id=agent.get("workspace_id"),
                    expected_pane_id=job.herdr_pane_id,
                    expected_workspace_id=job.herdr_workspace_id,
                )
                checkpoint()
            _run_tiered_workflow(
                store,
                job,
                worker_token,
                client,
                checkpoint,
            )
            return

        turn, turn_token = _prompt_turn_identity(job)

        def begin_turn(current: CursorJob) -> CursorJob:
            return _begin_prompt_turn(current, turn, turn_token)

        updated = _worker_change(
            store, job_id, worker_token, {JobStatus.ROUTING}, begin_turn
        )
        if updated is None:
            return
        job = updated
        checkpoint()
        target = job.herdr_target or ""
        if job.parent_job_id:
            _repository, followup_checkout, _branch, _match = (
                _validate_followup_checkout(client, job)
            )
            checkpoint()
        if target and job.agent_dispatch_state == "dispatching":
            checkout_value = job.worktree_path or ""
            pane = job.herdr_pane_id or ""
            workspace = job.herdr_workspace_id or ""
            repository_value = job.repository or ""
            if checkout_value and pane and workspace and repository_value:
                checkpoint()
                try:
                    agent = client.get_agent(target)
                except HerdrError as exc:
                    if not _agent_not_found(exc):

                        def defer_dispatch(current: CursorJob) -> CursorJob:
                            return current.evolve(
                                status=JobStatus.QUEUED,
                                queued_at=time.time(),
                                worker_pid=None,
                                worker_boot_id=None,
                                worker_process_start=None,
                                worker_token=None,
                            )

                        _worker_change(
                            store,
                            job_id,
                            worker_token,
                            {JobStatus.ROUTING},
                            defer_dispatch,
                        )
                        return
                    agent = {}
                checkpoint()
                if agent:
                    if followup_checkout is not None:
                        _validate_followup_agent_binding(
                            followup_checkout,
                            cwd=agent.get("cwd"),
                            pane_id=agent.get("pane_id"),
                            workspace_id=agent.get("workspace_id"),
                            expected_pane_id=pane or None,
                            expected_workspace_id=workspace or None,
                        )
                    selection = AgentSelection(
                        target=target,
                        pane_id=str(agent.get("pane_id") or pane),
                        workspace_id=str(agent.get("workspace_id") or workspace),
                        cwd=str(agent.get("cwd") or checkout_value),
                        name=str(agent.get("name") or target),
                        worktree_path=checkout_value,
                    )
                else:
                    mode = (
                        "plan"
                        if job.active_participant == WorkflowParticipant.PLANNER
                        else (
                            "ask"
                            if job.active_participant == WorkflowParticipant.REVIEWER
                            else None
                        )
                    )
                    if mode is None:
                        selection = client.start_agent(
                            Path(checkout_value),
                            job.worktree_label or Path(repository_value).name,
                            pane,
                            workspace,
                            name=target,
                            checkpoint=checkpoint,
                        )
                    else:
                        selection = client.start_agent(
                            Path(checkout_value),
                            job.worktree_label or Path(repository_value).name,
                            pane,
                            workspace,
                            name=target,
                            mode=mode,
                            checkpoint=checkpoint,
                        )
                    if followup_checkout is not None:
                        _validate_followup_agent_binding(
                            followup_checkout,
                            cwd=selection.cwd,
                            pane_id=selection.pane_id,
                            workspace_id=selection.workspace_id,
                            expected_pane_id=pane or None,
                            expected_workspace_id=workspace or None,
                        )
                updated = _settle_worker_agent(store, job_id, worker_token, selection)
                if updated is None:
                    return
                job = updated
                target = selection.target
                followup_target_verified = followup_checkout is not None
                checkpoint()
            else:

                def retry_dispatch(current: CursorJob) -> CursorJob:
                    return current.evolve(
                        herdr_target=None,
                        active_participant=None,
                        herdr_pane_id=None,
                        herdr_workspace_id=None,
                        agent_name=None,
                        agent_dispatch_state=None,
                    )

                updated = _worker_change(
                    store,
                    job_id,
                    worker_token,
                    {JobStatus.ROUTING},
                    retry_dispatch,
                )
                if updated is None:
                    return
                job = updated
                target = ""
        if (
            not target
            and job.parent_job_id
            and job.worktree_path
            and job.repository
            and job.worktree_branch
        ):
            checkpoint()
            job, target = _provision_followup_agent(
                store,
                job_id,
                worker_token,
                job,
                client,
                reserved_targets(store, job_id),
                checkpoint,
            )
            followup_target_verified = True
            checkpoint()
        if target and followup_checkout is not None and not followup_target_verified:
            checkpoint()
            agent = client.get_agent(target)
            if not agent:
                raise HarnessError("follow-up Cursor agent is no longer available")
            _validate_followup_agent_binding(
                followup_checkout,
                cwd=agent.get("cwd"),
                pane_id=agent.get("pane_id"),
                workspace_id=agent.get("workspace_id"),
                expected_pane_id=job.herdr_pane_id,
                expected_workspace_id=job.herdr_workspace_id,
            )
            checkpoint()
        if not target:
            repository: Path | None = None
            repositories: list[Path] = []
            candidates: list[Path] = []
            hint = (job.repository_hint or "").strip() or None
            issue_key = active_issue_key
            reason = ""
            if job.github_pull_request:
                github_repository = (job.github_repository or "").strip()
                number = job.github_pull_request
                if not github_repository or number <= 0:
                    _worker_question(
                        store,
                        job_id,
                        worker_token,
                        "Which repository's pull request should I check out? "
                        "Please say its owner and repository name.",
                        clarification_kind="github_repository",
                    )
                    return
                checkpoint()
                provisioned_pr = clients.github().provision_pull_request(
                    github_repository, number, checkpoint=checkpoint
                )
                checkpoint()
                repository = provisioned_pr.checkout
                issue_key = None

                def record_pull_request(current: CursorJob) -> CursorJob:
                    return current.evolve(
                        github_repository=provisioned_pr.source.name_with_owner,
                        repository=str(provisioned_pr.checkout),
                        pull_request_worktree_state="provisioning",
                    )

                updated = _worker_change(
                    store,
                    job_id,
                    worker_token,
                    {JobStatus.ROUTING},
                    record_pull_request,
                )
                if updated is None:
                    return
                job = updated
            elif job.fork_requested:
                github_repository = (job.github_repository or "").strip()
                if not github_repository:
                    _worker_question(
                        store,
                        job_id,
                        worker_token,
                        "Which public GitHub repository should I fork? "
                        "Please say its owner and repository name.",
                        clarification_kind="github_repository",
                    )
                    return
                if not job.fork_confirmed:
                    _worker_question(
                        store,
                        job_id,
                        worker_token,
                        f"Please confirm: should I create a GitHub fork of "
                        f"{github_repository}? Say yes or no.",
                        clarification_kind="fork_confirmation",
                    )
                    return
                github = clients.github()
                checkpoint()
                source, login, fork_target = github.prepare_public_fork(
                    github_repository
                )
                checkpoint()
                if job.fork_operation_state == "exists" and job.fork_repository:
                    fork = github.reconcile_fork(source, fork_target)
                    if fork is None:
                        raise HarnessError(
                            "reconciled GitHub fork is no longer observable"
                        )
                else:
                    updated = _begin_fork_operation(
                        store, job_id, worker_token, source, login, fork_target
                    )
                    if updated is None:
                        return
                    job = updated
                    try:
                        fork = github.ensure_fork(
                            source,
                            login,
                            checkpoint=checkpoint,
                            before_submit=lambda: _mark_fork_dispatching(
                                store, job_id, worker_token
                            ),
                        )
                    except WorkerCancelled:
                        raise
                    except GitHubError as exc:
                        current = store.get(job_id)
                        was_submitted = current.fork_operation_state == "submitted"
                        visible = (
                            github.reconcile_fork(source, fork_target)
                            if was_submitted
                            else None
                        )
                        _settle_fork_operation(
                            store,
                            job_id,
                            worker_token,
                            visible,
                            ambiguous=(
                                was_submitted
                                and visible is None
                                and isinstance(exc, GitHubOperationAmbiguous)
                            ),
                            failed_observing=(
                                was_submitted
                                and visible is None
                                and not isinstance(exc, GitHubOperationAmbiguous)
                            ),
                        )
                        raise
                    updated = _settle_fork_operation(store, job_id, worker_token, fork)
                    if updated is None:
                        return
                    job = updated
                checkpoint()
                if job.status != JobStatus.ROUTING:
                    raise WorkerCancelled
                checkout = github.ensure_clone(source, fork, checkpoint=checkpoint)
                checkpoint()
                repository = checkout

                def record_provisioning(current: CursorJob) -> CursorJob:
                    return current.evolve(repository=str(checkout))

                updated = _worker_change(
                    store,
                    job_id,
                    worker_token,
                    {JobStatus.ROUTING},
                    record_provisioning,
                )
                if updated is None:
                    return
                job = updated
            elif job.github_issue:
                github_repository = (job.github_repository or "").strip()
                number = job.github_issue
                if not github_repository or number <= 0:
                    _worker_question(
                        store,
                        job_id,
                        worker_token,
                        "Which repository's GitHub issue should I work on? "
                        "Please say owner/repository and the issue number.",
                        clarification_kind="github_repository",
                    )
                    return
                owner, repository_name = github_repository.split("/", 1)
                checkpoint()
                repositories = client.repository_roots()
                checkpoint()
                provisioned_issue = clients.github().provision_issue(
                    GitHubIssue(owner, repository_name, number),
                    candidates=repositories,
                    checkpoint=checkpoint,
                )
                checkpoint()
                repository = provisioned_issue.checkout
                issue_key = None

                def record_issue(current: CursorJob) -> CursorJob:
                    return current.evolve(
                        github_repository=provisioned_issue.source.name_with_owner,
                        github_issue=provisioned_issue.issue.number,
                        github_issue_url=provisioned_issue.issue.url,
                        repository=str(provisioned_issue.checkout),
                    )

                updated = _worker_change(
                    store,
                    job_id,
                    worker_token,
                    {JobStatus.ROUTING},
                    record_issue,
                )
                if updated is None:
                    return
                job = updated
            else:
                checkpoint()
                repositories = client.repository_roots()
                checkpoint()
                repository, candidates = resolve_job_repository(
                    client, job, repositories
                )
                checkpoint()
            if (
                repository is None
                and issue_key
                and not hint
                and not job.fork_requested
                and not job.github_pull_request
                and not job.github_issue
            ):
                checkpoint()
                routed = route_issue_repository(
                    client,
                    issue_key,
                    repositories,
                    token=f"{job_id}-route",
                    reserved=reserved_targets(store, job_id),
                    checkpoint=checkpoint,
                )
                if routed is not None:
                    repository, _confidence, reason = routed
                checkpoint()
            if repository is None:
                checkpoint()
                repository, rofi_reason = client.choose_or_clone_repository(
                    candidates or repositories, checkpoint=checkpoint
                )
                checkpoint()
                reason = rofi_reason or reason
            if repository is None:
                question = repository_question(
                    candidates or repositories,
                    reason
                    or (
                        "The repository could not be determined confidently."
                        if hint or issue_key
                        else ""
                    ),
                )
                _worker_question(
                    store,
                    job_id,
                    worker_token,
                    question,
                    clarification_kind="repository",
                )
                return
            for _attempt in range(3):
                reservation: CursorJob | None = None
                agent_settled = False

                def reserve_selection(
                    selection: AgentSelection, dispatching: bool
                ) -> None:
                    nonlocal reservation, job, target
                    reservation = _reserve_worker_target(
                        store,
                        job_id,
                        worker_token,
                        selection,
                        repository,
                        issue_key,
                        dispatching=dispatching,
                        participant=WorkflowParticipant.PLANNER,
                    )
                    if reservation is None:
                        checkpoint()
                        raise ReservationConflict
                    job = reservation
                    target = selection.target

                def settle_selection(selection: AgentSelection) -> None:
                    nonlocal reservation, job, target, agent_settled
                    reservation = _settle_worker_agent(
                        store, job_id, worker_token, selection
                    )
                    if reservation is None:
                        raise ReservationConflict
                    job = reservation
                    target = selection.target
                    agent_settled = True

                def fail_selection(exc: HerdrError) -> None:
                    _fail_worker_agent_dispatch(store, job_id, worker_token, exc)

                def reserve_worktree(
                    worktree_repository: Path,
                    branch: str,
                    checkout: Path,
                    state: str,
                ) -> None:
                    nonlocal job
                    reserved_worktree = _reserve_worker_worktree(
                        store,
                        job_id,
                        worker_token,
                        worktree_repository,
                        branch,
                        checkout,
                        state=state,
                    )
                    if reserved_worktree is None:
                        checkpoint()
                        raise ReservationConflict
                    job = reserved_worktree

                def settle_worktree(
                    checkout: Path,
                    workspace_id: str | None,
                    pane_id: str | None,
                ) -> None:
                    nonlocal job
                    settled_worktree = _settle_worker_worktree(
                        store,
                        job_id,
                        worker_token,
                        checkout,
                        workspace_id,
                        pane_id,
                    )
                    if settled_worktree is None:
                        raise ReservationConflict
                    job = settled_worktree

                def fail_worktree(exc: HerdrError) -> None:
                    _fail_worker_worktree(store, job_id, worker_token, exc)

                planned_participant_target = [""]

                def plan_participant(
                    name: str,
                    label: str,
                    workspace_id: str | None,
                    target_holder: list[str] = planned_participant_target,
                ) -> None:
                    nonlocal job
                    target_holder[0] = name
                    job = _plan_participant_creation(
                        store,
                        job,
                        worker_token,
                        WorkflowParticipant.PLANNER,
                        target=name,
                        label=label,
                        workspace_id=workspace_id or "",
                    )

                def before_pane_submit(
                    target_holder: list[str] = planned_participant_target,
                ) -> None:
                    callback, _accepted = _participant_pane_callbacks(
                        store,
                        job_id,
                        worker_token,
                        target_holder[0],
                    )
                    callback()

                def pane_accepted(
                    pane_id: str,
                    workspace_id: str,
                    target_holder: list[str] = planned_participant_target,
                ) -> None:
                    _before, callback = _participant_pane_callbacks(
                        store,
                        job_id,
                        worker_token,
                        target_holder[0],
                    )
                    callback(pane_id, workspace_id)

                try:
                    checkpoint()
                    selection = client.ensure_agent(
                        repository,
                        issue_key=issue_key or None,
                        agent_hint=job.agent_hint,
                        reserved=reserved_targets(store, job_id),
                        worktree_branch=job.worktree_branch,
                        worktree_label=job.worktree_label,
                        mode="plan",
                        checkpoint=checkpoint,
                        reserve=reserve_selection,
                        settle=settle_selection,
                        fail_agent=fail_selection,
                        reserve_worktree=reserve_worktree,
                        settle_worktree=settle_worktree,
                        fail_worktree=fail_worktree,
                        plan_participant=plan_participant,
                        before_pane_submit=before_pane_submit,
                        pane_accepted=pane_accepted,
                        participant_name=(
                            job.participant_creation_target
                            if job.participant_creation_state == "planned"
                            else None
                        ),
                    )
                    if not agent_settled:
                        reserve_selection(selection, False)
                    checkpoint()
                    break
                except ReservationConflict:
                    continue
            else:
                raise HarnessError("could not reserve a Cursor agent")

        prepared = _prepare_pull_request_checkout(
            store,
            job_id,
            worker_token,
            job,
            checkpoint,
            github_factory=clients.github,
        )
        if prepared is None:
            return
        job = prepared
        checkpoint()

        def mark_running(current: CursorJob) -> CursorJob:
            return current.evolve(status=JobStatus.RUNNING)

        if (
            _worker_change(
                store,
                job_id,
                worker_token,
                {JobStatus.ROUTING},
                mark_running,
            )
            is None
        ):
            return
        checkpoint()
        _run_tiered_workflow(
            store,
            store.get(job_id),
            worker_token,
            client,
            checkpoint,
        )
    except WorkerCancelled:
        return
    except Exception as exc:
        _worker_error(
            store,
            job_id,
            worker_token,
            exc,
            prompt_may_be_active=bool(target),
            client=client,
            target=target,
            checkpoint=checkpoint,
        )
    finally:
        try:
            current = store.get(job_id)
            if (
                current.status == JobStatus.RECONCILING
                and current.terminal_intent_status is not None
                and current.terminal_intent_status != JobStatus.CANCELLED
                and current.target_release_token
            ):
                recovery.cancel_target_and_release(
                    store,
                    job_id,
                    current.herdr_target or "",
                    current.target_release_token,
                    worker_stopped=True,
                    herdr_factory=clients.herdr,
                )
        except Exception:
            # Durable terminal intent and release fences are recovered later.
            pass
