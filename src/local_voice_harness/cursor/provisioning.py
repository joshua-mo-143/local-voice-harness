from __future__ import annotations

import time
from collections.abc import Callable, Set
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
    AgentSelection,
    HerdrClient,
    HerdrError,
    extract_linear_issue,
    extract_marker,
)
from . import worker_lifecycle
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
    transition,
)
from .prompts import cursor_prompt
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
    return {
        job.herdr_target
        for job in store.list()
        if job.herdr_target
        and job.id != exclude_job_id
        and (job.status in MODEL_ACTIVE_STATUSES or job.target_release_pending)
    }


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
    summary_position = output.rfind(f"VOICE_SUMMARY[{token}]")
    question_position = output.rfind(f"VOICE_QUESTION[{token}]")
    if question and question_position > summary_position:
        return job.evolve_for_delivery(
            now=completed_at,
            status=JobStatus.AWAITING_USER,
            question=question,
            result=question,
            clarification_kind="agent",
        )
    if summary and summary_position > question_position:
        return job.evolve_for_delivery(
            now=completed_at,
            status=JobStatus.COMPLETED,
            result=summary,
            completed_at=completed_at,
        )
    return job.evolve_for_delivery(
        now=completed_at,
        status=JobStatus.BLOCKED,
        result=(
            f"Herdr agent {job.herdr_target or 'Cursor'} needs attention; "
            f"it settled as {agent_status} without a voice summary."
        ),
        completed_at=completed_at,
    )


def read_agent_completion(
    client: HerdrClient,
    job: CursorJob,
    *,
    wait: bool,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[str, str]:
    target = job.herdr_target or ""
    if not target:
        raise HarnessError("Cursor job has no Herdr agent")
    if checkpoint is not None:
        checkpoint()
    agent = client.get_agent(target)
    if checkpoint is not None:
        checkpoint()
    if wait and agent.get("agent_status") == "working":
        if checkpoint is not None:
            checkpoint()
        result = client.run_json(
            "agent", "wait", target, "--timeout", "900000", timeout=910
        )
        if checkpoint is not None:
            checkpoint()
        agent = dict(result.get("agent") or {})
    if checkpoint is not None:
        checkpoint()
    output = client.run_text(
        "agent", "read", target, "--source", "recent-unwrapped", "--lines", "160"
    )
    if checkpoint is not None:
        checkpoint()
    return output, str(agent.get("agent_status") or "unknown")


def _worker_change(
    store: JobStore,
    job_id: str,
    token: str,
    allowed_statuses: Set[JobStatus],
    change: Callable[[CursorJob], CursorJob],
) -> CursorJob | None:
    def guarded(job: CursorJob) -> CursorJob | None:
        if job.worker_token != token or job.status not in allowed_statuses:
            return None
        return change(job)

    return store.update(job_id, guarded)


def _worker_question(
    store: JobStore,
    job_id: str,
    token: str,
    question: str,
    *,
    clarification_kind: str,
) -> None:
    def ask(job: CursorJob) -> CursorJob:
        return job.evolve_for_delivery(
            now=time.time(),
            status=JobStatus.AWAITING_USER,
            remove=frozenset({"reconcile"}),
            question=question,
            result=question,
            clarification_kind=clarification_kind,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
        )

    _worker_change(store, job_id, token, {JobStatus.ROUTING, JobStatus.RUNNING}, ask)


def _worker_complete(
    store: JobStore,
    job_id: str,
    token: str,
    *,
    output: str,
    agent_status: str,
    preserve_blocked_delivery: bool = False,
) -> None:
    def finish(job: CursorJob) -> CursorJob:
        now = time.time()
        outcome = complete_from_output(
            job, output=output, agent_status=agent_status, now=now
        )
        preserve = preserve_blocked_delivery and outcome.status == JobStatus.BLOCKED
        return job.evolve(
            status=outcome.status,
            remove=frozenset({"reconcile"}),
            result=outcome.result,
            error=outcome.error,
            question=outcome.question,
            clarification_kind=outcome.clarification_kind,
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
        )

    _worker_change(
        store,
        job_id,
        token,
        {JobStatus.RUNNING, JobStatus.RECONCILING},
        finish,
    )


def _worker_fail(store: JobStore, job_id: str, token: str, exc: Exception) -> None:
    message = (str(exc) or type(exc).__name__)[:500]

    def fail(job: CursorJob) -> CursorJob:
        uncertain = job.has_uncertain_operation()
        if uncertain:
            message_with_fence = (
                f"{message}; external operation reconciliation is pending"
            )[:500]
        else:
            message_with_fence = message
        return job.evolve_for_delivery(
            now=time.time(),
            status=JobStatus.FAILED,
            remove=frozenset({"reconcile"}),
            error=message_with_fence,
            result=message_with_fence,
            reconciliation_base_error=message,
            completed_at=time.time(),
            target_release_pending=uncertain,
            target_release_token=None,
            target_release_owner_pid=None,
            target_release_owner_boot_id=None,
            target_release_owner_start=None,
            cancellation_reconciliation_pending=uncertain,
            worker_pid=job.worker_pid if uncertain else None,
            worker_boot_id=job.worker_boot_id if uncertain else None,
            worker_process_start=job.worker_process_start if uncertain else None,
            worker_token=job.worker_token if uncertain else None,
        )

    _worker_change(store, job_id, token, MODEL_WORKER_STATUSES, fail)


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
) -> CursorJob | None:
    target = str(selection.target)

    def reserve(job: CursorJob) -> CursorJob | None:
        if job.worker_token != token or job.status.value != "routing":
            return None
        return transition(
            job,
            job.status,
            repository=str(repository),
            issue_key=issue_key,
            herdr_target=target,
            herdr_pane_id=selection.pane_id,
            herdr_workspace_id=selection.workspace_id,
            worktree_path=selection.worktree_path,
            agent_name=selection.name,
            agent_dispatch_state="dispatching" if dispatching else "ready",
            worker_operation="agent_start" if dispatching else None,
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
            or job.status.value not in {"routing", "cancelled", "failed"}
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
        if job.worker_token != token or job.status.value != "routing":
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


def run_claimed_worker(
    context: worker_lifecycle.WorkerContext,
    factories: ClientFactories | None = None,
) -> None:
    clients = factories or ClientFactories(HerdrClient, GitHubClient)
    store = context.store
    job_id = context.job.id
    job = context.job
    worker_token = context.token

    def checkpoint() -> None:
        context.checkpoint()

    try:
        client = clients.herdr()
        checkpoint()
        client.ensure_server()
        checkpoint()
        if job.reconcile:
            output, agent_status = read_agent_completion(
                client, job, wait=True, checkpoint=checkpoint
            )
            _worker_complete(
                store,
                job_id,
                worker_token,
                output=output,
                agent_status=agent_status,
                preserve_blocked_delivery=job.delivered,
            )
            return

        turn = job.turn + 1
        turn_token = f"{job_id}-{turn}"

        def begin_turn(current: CursorJob) -> CursorJob:
            return current.evolve(turn=turn, turn_token=turn_token)

        updated = _worker_change(
            store, job_id, worker_token, {JobStatus.ROUTING}, begin_turn
        )
        if updated is None:
            return
        job = updated
        checkpoint()
        continuation = job.continuation
        target = job.herdr_target or ""
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
                    selection = AgentSelection(
                        target=target,
                        pane_id=str(agent.get("pane_id") or pane),
                        workspace_id=str(agent.get("workspace_id") or workspace),
                        cwd=str(agent.get("cwd") or checkout_value),
                        name=str(agent.get("name") or target),
                        worktree_path=checkout_value,
                    )
                else:
                    selection = client.start_agent(
                        Path(checkout_value),
                        job.worktree_label or Path(repository_value).name,
                        pane,
                        workspace,
                        name=target,
                        checkpoint=checkpoint,
                    )
                updated = _settle_worker_agent(store, job_id, worker_token, selection)
                if updated is None:
                    return
                job = updated
                target = selection.target
                checkpoint()
            else:

                def retry_dispatch(current: CursorJob) -> CursorJob:
                    return current.evolve(
                        herdr_target=None,
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
        if not target:
            repository: Path | None = None
            repositories: list[Path] = []
            candidates: list[Path] = []
            hint = (job.repository_hint or "").strip() or None
            task = job.request
            utterance = job.utterance or task
            issue_key = job.issue_key or extract_linear_issue(utterance)
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
                repository, _confidence, reason = client.infer_repository(
                    issue_key,
                    repositories,
                    token=f"{job_id}-route",
                    reserved=reserved_targets(store, job_id),
                    checkpoint=checkpoint,
                )
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

                try:
                    checkpoint()
                    selection = client.ensure_agent(
                        repository,
                        issue_key=issue_key or None,
                        agent_hint=job.agent_hint,
                        reserved=reserved_targets(store, job_id),
                        worktree_branch=job.worktree_branch,
                        worktree_label=job.worktree_label,
                        checkpoint=checkpoint,
                        reserve=reserve_selection,
                        settle=settle_selection,
                        fail_agent=fail_selection,
                        reserve_worktree=reserve_worktree,
                        settle_worktree=settle_worktree,
                        fail_worktree=fail_worktree,
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
            return current.evolve(
                status=JobStatus.RUNNING,
                remove=frozenset({"continuation"}),
            )

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
        outcome = client.prompt_and_wait(
            target,
            cursor_prompt(
                job.request,
                turn_token,
                continuation=continuation,
                github_issue_context=job.github_issue_context,
            ),
            token=turn_token,
            checkpoint=checkpoint,
        )
        checkpoint()
        _worker_complete(
            store,
            job_id,
            worker_token,
            output=outcome.output,
            agent_status=outcome.status,
        )
    except WorkerCancelled:
        return
    except Exception as exc:
        _worker_fail(store, job_id, worker_token, exc)
