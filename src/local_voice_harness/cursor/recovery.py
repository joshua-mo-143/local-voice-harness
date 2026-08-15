from __future__ import annotations

import os
import secrets
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from ..agents.harness import ReconciliationState
from ..diagnostic_safety import redact_diagnostic
from ..errors import HarnessError
from ..integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubForkPlan,
    GitHubProvider,
    GitHubRepository,
    github_repository_from_url,
)
from ..integrations.herdr import (
    HerdrClient,
    HerdrError,
    agent_session_identity,
    repository_name_from_url,
)
from ..integrations.linear import LinearError, LinearIntegration
from ..integrations.registry import build_integration_registry, issue_provider
from ..job_lifecycle import (
    CancellationEvent,
    JobLifecycleError,
    RecoveryEvent,
    SessionControlMode,
    SessionControlState,
)
from ..local_git import ExpectedRemote, LocalGitError
from ..prompt_operations import (
    AmbiguousPrompt,
    ObservedPrompt,
    PlannedPrompt,
    PromptOperationError,
    SubmittedPrompt,
    SubmittingPrompt,
    mark_prompt_ambiguous,
)
from ..questions import (
    PromptOperationState,
    QuestionState,
    observe_question_prompt,
    replan_question_prompt,
)
from ..user_config import default_user_config
from . import questions as question_adapter
from .agent_outbox import (
    AGENT_EFFECT_KINDS,
    CLARIFICATION_REPLY,
    TASK_SUBMIT,
    agent_effect_handlers,
    consume_agent_results,
)
from .coordinator import (
    CoordinatorCommand,
    CoordinatorDecision,
    OutboxLease,
)
from .lifecycle import (
    CleanupOwned,
    LifecycleTransitionError,
    MaterializedTerminalOutcome,
    TerminalIntent,
    abandon_cleanup_owner,
    begin_cleanup,
    claim_cleanup,
    cleanup_fields,
    finish_cleanup_reconciliation,
    settle_cleanup,
    take_over_cleanup,
)
from .model import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_BOOT_ID,
    TERMINAL_STATUSES,
    CursorJob,
    JobStatus,
    JobValidationError,
    WorkflowParticipant,
)
from .operations import (
    AgentSessionOperation,
    AgentSessionSpec,
    AgentSessionState,
    CheckoutState,
    OperationState,
    OperationTransitionError,
    SessionIdentity,
    agent_session_fields,
    checkout_blocks_reservation,
    checkout_fields,
)
from .outbox import EffectHandler, recover_outbox
from .store import JobStore, LegacyWorkerInspector
from .worker_lifecycle import (
    boot_identity,
    has_legacy_worker_claim,
    inspect_and_stop_legacy_worker,
    process_identity,
    stop_legacy_worker,
    stop_worker,
    worker_is_alive,
)

FAILED_RECONCILE_MAX_ATTEMPTS = 3
UNCERTAIN_RECONCILE_MAX_ATTEMPTS = 6
OPERATION_RECONCILE_BASE_SECONDS = 5.0
OPERATION_RECONCILE_MAX_SECONDS = 60.0
DELIVERY_RETRY_SECONDS = 5.0
PROMPT_ABSENT_OBSERVATIONS = 2

HerdrFactory = Callable[[], HerdrClient]
GitHubFactory = Callable[[], GitHubClient | GitHubProvider]
LinearFactory = Callable[[], LinearIntegration]
LaunchWorker = Callable[[str], None]
RequireIssueProvider = Callable[[str | None], None]


def _herdr_client_for_job(herdr_factory: HerdrFactory, job: CursorJob) -> HerdrClient:
    """Bind the persisted harness before any recovered session operation."""

    client = herdr_factory()
    client.bind_harness_kind(job.harness_kind.value)
    return client


def _recovery_agent_effect_handlers(
    herdr_factory: HerdrFactory,
    store: JobStore,
) -> dict[str, EffectHandler]:
    """Replay durable agent effects only through the persisted job harness."""

    def harness_for(lease: OutboxLease):
        return _herdr_client_for_job(
            herdr_factory,
            store.get(lease.job_id),
        ).harness

    handlers: dict[str, EffectHandler] = {}
    for kind in AGENT_EFFECT_KINDS:

        def handle(
            lease: OutboxLease,
            mark_dispatched: Callable[[], None],
            effect_kind: str = kind,
        ):
            return agent_effect_handlers(harness_for(lease))[effect_kind](
                lease,
                mark_dispatched,
            )

        handlers[kind] = handle
    return handlers


def _github_provider(factory: GitHubFactory) -> GitHubProvider:
    integration = factory()
    return (
        integration
        if isinstance(integration, GitHubProvider)
        else GitHubProvider(integration)
    )


def _linear_provider(factory: LinearFactory | None = None) -> LinearIntegration:
    if factory is not None:
        return factory()
    integration = issue_provider(
        "linear",
        build_integration_registry(default_user_config()),
    )
    if not isinstance(integration, LinearIntegration):
        raise HarnessError("selected Linear provider cannot create tickets")
    return integration


def _agent_not_found(exc: HerdrError) -> bool:
    return exc.code in {"agent_not_found", "not_found"}


def _reconciliation_due(job: CursorJob, prefix: str, now: float) -> bool:
    return now >= job.operation_reconcile_at(prefix)


def _record_reconciliation_observation(
    store: JobStore,
    job_id: str,
    prefix: str,
    state_key: str,
    expected_states: frozenset[str],
    *,
    now: float,
    observed_absent: bool,
) -> None:
    def observe(job: CursorJob) -> CursorJob | None:
        return job.record_operation_observation(
            prefix,
            state_key,
            expected_states,
            now=now,
            observed_absent=observed_absent,
            failed_max_attempts=FAILED_RECONCILE_MAX_ATTEMPTS,
            uncertain_max_attempts=UNCERTAIN_RECONCILE_MAX_ATTEMPTS,
            base_seconds=OPERATION_RECONCILE_BASE_SECONDS,
            max_seconds=OPERATION_RECONCILE_MAX_SECONDS,
        )

    store.update(job_id, observe)


def _followup_agent_cwd_matches(job: CursorJob, agent: dict[str, object]) -> bool:
    if job.parent_job_id is None:
        return True
    expected = str(job.worktree_path or "").strip()
    actual = str(agent.get("cwd") or "").strip()
    if not expected or not actual:
        return False
    if job.herdr_pane_id and str(agent.get("pane_id") or "") != job.herdr_pane_id:
        return False
    if (
        job.herdr_workspace_id
        and str(agent.get("workspace_id") or "") != job.herdr_workspace_id
    ):
        return False
    try:
        return Path(actual).resolve() == Path(expected).resolve()
    except OSError:
        return False


def _checkout_state_fields(
    job: CursorJob, state: CheckoutState, **transition_args: str | None
) -> dict[str, object]:
    operation = job.checkout_operation
    if operation is not None:
        return checkout_fields(operation.transition(state, **transition_args))
    return {"worktree_provision_state": state.value}


def _session_state_fields(
    job: CursorJob,
    state: AgentSessionState,
    *,
    session: SessionIdentity | None = None,
) -> dict[str, object]:
    operation = job.agent_session_operation
    if operation is not None:
        return agent_session_fields(operation.transition(state, session=session))
    if job.agent_dispatch_state is None and state == AgentSessionState.CONFIRMED_ABSENT:
        return {}
    return {"agent_dispatch_state": state.value}


def reconcile_uncertain_agent(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
) -> None:
    if job.session_control == SessionControlMode.USER_OWNED.value:
        return
    states = frozenset({"dispatching", "ambiguous", "failed_observing"})
    if job.agent_dispatch_state not in states or not _reconciliation_due(
        job, "agent", now
    ):
        return
    target = job.herdr_target or ""
    if not target:
        _record_reconciliation_observation(
            store,
            job.id,
            "agent",
            "agent_dispatch_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    try:
        operation = job.agent_session_operation
    except (JobValidationError, OperationTransitionError):
        operation = None
    identity: SessionIdentity | None = None
    agent: dict[str, object] = {}
    if operation is None or operation.session is None:
        if job.status in TERMINAL_STATUSES or job.terminal_intent_status is not None:
            try:
                client = _herdr_client_for_job(herdr_factory, job)
                client.ensure_server()
                observed_agent = client.get_agent(target)
            except HerdrError as exc:
                _record_reconciliation_observation(
                    store,
                    job.id,
                    "agent",
                    "agent_dispatch_state",
                    states,
                    now=now,
                    observed_absent=_agent_not_found(exc),
                )
                return
            if not observed_agent:
                _record_reconciliation_observation(
                    store,
                    job.id,
                    "agent",
                    "agent_dispatch_state",
                    states,
                    now=now,
                    observed_absent=True,
                )
                return
            agent = cast(dict[str, object], observed_agent)
        else:

            def require_manual(current: CursorJob) -> CursorJob | None:
                if (
                    current.agent_dispatch_state not in states
                    or current.herdr_target != target
                    or current.manual_reconcile_operation is not None
                ):
                    return None
                operation = current.agent_session_operation
                if operation is not None:
                    return current.evolve_recovery(
                        now=now,
                        agent_session_operation=operation.transition(
                            AgentSessionState.MANUAL_REQUIRED
                        ),
                        manual_reconcile_operation="agent",
                        manual_reconcile_token=uuid.uuid4().hex,
                        manual_reconcile_required_at=now,
                        worker_operation=None,
                    )
                return current.evolve_recovery(
                    now=now,
                    agent_dispatch_state="manual_required",
                    manual_reconcile_operation="agent",
                    manual_reconcile_token=uuid.uuid4().hex,
                    manual_reconcile_required_at=now,
                    worker_operation=None,
                )

            store.update(job.id, require_manual)
            return
    else:
        try:
            client = _herdr_client_for_job(herdr_factory, job)
            client.ensure_server()
            observation = client.reconcile_session(
                target,
                expected_session_id=operation.session.session_id,
            )
        except HerdrError as exc:
            _record_reconciliation_observation(
                store,
                job.id,
                "agent",
                "agent_dispatch_state",
                states,
                now=now,
                observed_absent=_agent_not_found(exc),
            )
            return
        if observation.state == ReconciliationState.MISSING:
            _record_reconciliation_observation(
                store,
                job.id,
                "agent",
                "agent_dispatch_state",
                states,
                now=now,
                observed_absent=True,
            )
            return
        observed = observation.session
        if observed is None or observation.state not in {
            ReconciliationState.ACTIVE,
            ReconciliationState.SETTLED,
        }:
            _record_reconciliation_observation(
                store,
                job.id,
                "agent",
                "agent_dispatch_state",
                states,
                now=now,
                observed_absent=False,
            )
            return
        identity = SessionIdentity(
            observed.provider,
            observed.session_id,
            observed.target,
            observed.state_sequence,
        )
        if not operation.accepts_observation(identity):
            return
        agent = cast(
            dict[str, object],
            {
                "name": observed.metadata.get("name", target),
                "pane_id": observed.metadata.get("pane_id", ""),
                "workspace_id": observed.metadata.get("workspace_id", ""),
                "cwd": observed.metadata.get("cwd", ""),
            },
        )
    if not _followup_agent_cwd_matches(job, agent):
        message = (
            "follow-up Cursor agent does not match its reserved checkout and pane; "
            "target cancellation is pending"
        )

        def reject_mismatch(current: CursorJob) -> CursorJob | None:
            if (
                current.agent_dispatch_state not in states
                or current.herdr_target != target
                or current.revision != job.revision
                or current.agent_session_operation != operation
            ):
                return None
            next_state = (
                AgentSessionState.READY
                if identity is not None
                else AgentSessionState.AMBIGUOUS
            )
            current_operation = current.agent_session_operation
            changes: dict[str, object] = {
                **(
                    {
                        "agent_session_operation": current_operation.transition(
                            next_state, session=identity
                        )
                    }
                    if current_operation is not None
                    else {"agent_dispatch_state": next_state.value}
                ),
                "agent_name": str(agent.get("name") or target),
                "herdr_pane_id": str(
                    agent.get("pane_id") or current.herdr_pane_id or ""
                ),
                "herdr_workspace_id": str(
                    agent.get("workspace_id") or current.herdr_workspace_id or ""
                ),
                "worker_operation": None,
                "agent_next_reconcile_at": None,
                "reconciliation_base_error": message,
            }
            if current.status in TERMINAL_STATUSES:
                return current.evolve_recovery(
                    changes,
                    now=now,
                    prepare_delivery=True,
                )
            if current.terminal_intent_status is not None:
                return current.evolve_recovery(changes, now=now)
            return stage_terminal_intent(
                current,
                JobStatus.FAILED,
                now=now,
                result=message,
                error=message,
                job_changes=changes,
            )

        store.update(job.id, reject_mismatch)
        return

    def visible(current: CursorJob) -> CursorJob | None:
        if (
            current.agent_dispatch_state not in states
            or current.herdr_target != target
            or current.revision != job.revision
            or current.agent_session_operation != operation
        ):
            return None
        next_state = (
            AgentSessionState.READY
            if identity is not None
            else AgentSessionState.AMBIGUOUS
        )
        current_operation = current.agent_session_operation
        changes: dict[str, object] = {
            **(
                {
                    "agent_session_operation": current_operation.transition(
                        next_state, session=identity
                    )
                }
                if current_operation is not None
                else {"agent_dispatch_state": next_state.value}
            ),
            "agent_name": str(agent.get("name") or target),
            "herdr_pane_id": str(agent.get("pane_id") or current.herdr_pane_id or ""),
            "herdr_workspace_id": str(
                agent.get("workspace_id") or current.herdr_workspace_id or ""
            ),
            "worker_operation": None,
            "agent_next_reconcile_at": None,
        }
        if identity is not None:
            changes.update(
                agent_provider=identity.provider,
                agent_provider_session_id=identity.session_id,
                agent_state_sequence=identity.state_sequence,
            )
        return current.evolve_recovery(changes, now=now)

    store.update(job.id, visible)


def _fork_source(job: CursorJob) -> GitHubRepository | None:
    name = job.fork_operation_source or ""
    url = job.fork_operation_source_url or ""
    if not name or not url:
        return None
    return GitHubRepository(
        name_with_owner=name,
        url=url,
        is_private=job.fork_operation_source_private,
        default_branch=job.fork_operation_source_default_branch or "",
        parent=job.fork_operation_source_parent,
    )


def reconcile_uncertain_fork(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    states = frozenset({"submitted", "ambiguous", "failed_observing"})
    if job.fork_operation_state not in states or not _reconciliation_due(
        job, "fork", now
    ):
        return
    source = _fork_source(job)
    target = job.fork_operation_target or ""
    if source is None or not target:
        _record_reconciliation_observation(
            store,
            job.id,
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    try:
        if job.loaded_schema_version < CURRENT_SCHEMA_VERSION:
            fork = cast(Any, github_factory()).reconcile_fork(source, target)
        else:
            plan = GitHubForkPlan(
                source=source,
                login=job.fork_operation_login or target.split("/", 1)[0],
                target=target,
            )
            fork = _github_provider(github_factory).observe_fork(plan)
    except GitHubError:
        _record_reconciliation_observation(
            store,
            job.id,
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    if fork is None:
        _record_reconciliation_observation(
            store,
            job.id,
            "fork",
            "fork_operation_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def visible(current: CursorJob) -> CursorJob | None:
        if current.fork_operation_state not in states:
            return None
        return current.evolve_recovery(
            now=now,
            fork_operation_state="exists",
            fork_exists=True,
            fork_repository=fork.name_with_owner,
            worker_operation=None,
            fork_next_reconcile_at=None,
        )

    store.update(job.id, visible)


def reconcile_uncertain_issue_creation(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    states = frozenset({"submitting", "submitted", "ambiguous"})
    if job.github_issue_create_operation_state not in states:
        return
    repository = job.github_repository or ""
    title = job.github_issue_create_title or ""
    body = job.github_issue_create_body or ""
    marker = job.github_issue_create_marker
    try:
        github = _github_provider(github_factory)
        plan = github.plan_issue_creation(
            repository,
            title,
            body,
            correlation_marker=marker,
        )
        result = github.observe_issue_creation(plan)
    except GitHubError:
        result = None

    def reconcile(current: CursorJob) -> CursorJob | None:
        if current.github_issue_create_operation_state not in states:
            return None
        if result is None:
            message = (
                "GitHub issue creation could not be reconciled automatically. "
                "Check the repository before trying again."
            )
            return current.evolve_recovery(
                now=now,
                status=JobStatus.BLOCKED,
                github_issue_create_operation_state="manual_required",
                result=message,
                completed_at=now,
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                prepare_delivery=True,
            )
        if current.github_repo_create_continue_workflow:
            return current.evolve_recovery(
                now=now,
                status=JobStatus.QUEUED,
                queued_at=now,
                issue_provider="github",
                github_issue=result.issue.number,
                github_issue_url=result.url,
                github_issue_create_operation_state="created",
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
            )
        return current.evolve_recovery(
            now=now,
            status=JobStatus.COMPLETED,
            github_issue=result.issue.number,
            github_issue_url=result.url,
            github_issue_create_operation_state="created",
            result=f"Created GitHub issue {result.issue.reference}: {result.url}",
            completed_at=now,
            reconcile=False,
            worker_operation=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            prepare_delivery=True,
        )

    store.update(job.id, reconcile)


def reconcile_uncertain_clone(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    """Observe an ambiguous clone destination without resubmitting the clone."""

    if job.clone_operation_state not in {"submitted", "ambiguous"}:
        return
    source = job.clone_source or ""
    github = _github_provider(github_factory)
    try:
        repository = github_repository_from_url(source)
        if repository is None:
            try:
                repository = GitHubClient.validate_repository(source)
            except GitHubError:
                repository = None
        if repository is not None:
            resolved = github.resolve_repository(repository)
            checkout = github.observe_repository_materialization(resolved)
        else:
            name = repository_name_from_url(source)
            if name is None:
                return
            checkout = github.local_git.observe_materialized(
                Path(name),
                expected=ExpectedRemote.from_url(source),
            )
    except (GitHubError, LocalGitError):
        return
    if checkout is None:
        return

    def reconciled(current: CursorJob) -> CursorJob | None:
        if current.clone_operation_state not in {"submitted", "ambiguous"}:
            return None
        return current.evolve_recovery(
            now=now,
            repository=str(checkout),
            clone_operation_state="cloned",
            reconcile=False,
            worker_operation=None,
        )

    store.update(job.id, reconciled)


def reconcile_uncertain_pr_creation(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    states = frozenset({"submitting", "submitted", "ambiguous"})
    if job.github_pr_create_operation_state not in states:
        return
    repository = job.github_repository or ""
    title = job.github_pr_create_title or ""
    body = job.github_pr_create_body or ""
    marker = job.github_pr_create_marker
    head_repository = job.github_pr_create_head_repository or ""
    head_owner = head_repository.split("/", 1)[0] if "/" in head_repository else ""
    head = f"{head_owner}:{job.worktree_branch or ''}"
    try:
        github = _github_provider(github_factory)
        plan = github.plan_pull_request_creation(
            repository,
            title,
            body,
            head,
            job.github_pr_create_base or "",
            job.github_pr_create_published_head_oid or "",
            head_repository,
            correlation_marker=marker,
        )
        result = github.observe_pull_request_creation(plan)
    except GitHubError:
        result = None

    def reconcile(current: CursorJob) -> CursorJob | None:
        if current.github_pr_create_operation_state not in states:
            return None
        if result is None:
            message = (
                "GitHub pull request creation could not be reconciled automatically. "
                "Check the checkout and GitHub before trying again."
            )
            return current.evolve_recovery(
                now=now,
                status=JobStatus.BLOCKED,
                github_pr_create_operation_state="manual_required",
                result=message,
                completed_at=now,
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                prepare_delivery=True,
            )
        return current.evolve_recovery(
            now=now,
            status=JobStatus.COMPLETED,
            github_pr_created_number=result.pull_request.number,
            github_pr_created_url=result.url,
            github_pr_create_operation_state="created",
            result=(
                f"Opened GitHub pull request {result.pull_request.number}: {result.url}"
            ),
            completed_at=now,
            reconcile=False,
            worker_operation=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            prepare_delivery=True,
        )

    store.update(job.id, reconcile)


def reconcile_uncertain_pr_merge(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    states = frozenset({"submitting", "submitted", "ambiguous"})
    if job.github_pr_merge_operation_state not in states:
        return
    repository = job.github_repository or ""
    number = job.github_pr_merge_number
    marker = job.github_pr_merge_marker
    try:
        github = _github_provider(github_factory)
        if not repository or number is None or not marker:
            result = None
        else:
            plan = github.plan_pull_request_merge(
                repository,
                number,
                correlation_marker=marker,
            )
            result = github.observe_pull_request_merge(plan)
    except GitHubError:
        result = None

    def reconcile(current: CursorJob) -> CursorJob | None:
        if current.github_pr_merge_operation_state not in states:
            return None
        if result is None:
            message = (
                "GitHub pull request merge could not be reconciled automatically. "
                "Check the pull request on GitHub before trying again."
            )
            return current.evolve_recovery(
                now=now,
                status=JobStatus.BLOCKED,
                github_pr_merge_operation_state="manual_required",
                result=message,
                completed_at=now,
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                prepare_delivery=True,
            )
        return current.evolve_recovery(
            now=now,
            status=JobStatus.COMPLETED,
            github_pr_merge_number=result.pull_request.number,
            github_pr_merge_url=result.url,
            github_pr_merge_operation_state="merged",
            result=(
                f"Merged GitHub pull request {result.pull_request.number}: {result.url}"
            ),
            completed_at=now,
            reconcile=False,
            worker_operation=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            prepare_delivery=True,
        )

    store.update(job.id, reconcile)


def reconcile_uncertain_repo_creation(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    states = frozenset({"submitting", "submitted", "ambiguous", "remote_created"})
    if job.github_repo_create_operation_state not in states:
        return
    repository = job.github_repository or ""
    visibility = job.github_repo_create_visibility or ""
    marker = job.github_repo_create_marker
    github: GitHubProvider | None = None
    try:
        github = _github_provider(github_factory)
        if "/" not in repository or not visibility or not marker:
            result = None
        else:
            owner, slug = repository.split("/", 1)
            plan = github.plan_repository_creation(
                owner,
                slug,
                visibility,
                correlation_marker=marker,
            )
            result = github.observe_repository_creation(plan)
    except GitHubError:
        result = None

    if result is None and job.github_repo_create_operation_state == "remote_created":
        return

    def record_remote(current: CursorJob) -> CursorJob | None:
        if current.github_repo_create_operation_state not in states:
            return None
        if result is None:
            message = (
                "GitHub repository creation could not be reconciled automatically. "
                "Check the account before trying again."
            )
            return current.evolve_recovery(
                now=now,
                status=JobStatus.BLOCKED,
                github_repo_create_operation_state="manual_required",
                result=message,
                completed_at=now,
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                prepare_delivery=True,
            )
        return current.evolve_recovery(
            now=now,
            github_repository=result.repository.name_with_owner,
            github_repo_created_url=result.url,
            github_repo_create_operation_state="remote_created",
            worker_operation=None,
        )

    recorded = store.update(job.id, record_remote)
    if result is None or recorded is None or github is None:
        return
    if recorded.github_repo_create_operation_state != "remote_created":
        return
    try:
        checkout = github.materialize_repository(result.repository)
    except GitHubError:
        return

    def clone_verified(current: CursorJob) -> CursorJob | None:
        if current.github_repo_create_operation_state != "remote_created":
            return None
        if current.github_repo_create_continue_workflow:
            return current.evolve_recovery(
                now=now,
                status=JobStatus.QUEUED,
                queued_at=now,
                repository=str(checkout),
                github_repository=result.repository.name_with_owner,
                github_repo_created_url=result.url,
                github_repo_create_operation_state="clone_verified",
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
            )
        return current.evolve_recovery(
            now=now,
            status=JobStatus.COMPLETED,
            repository=str(checkout),
            github_repository=result.repository.name_with_owner,
            github_repo_created_url=result.url,
            github_repo_create_operation_state="clone_verified",
            result=(
                f"Created GitHub repository {result.repository.name_with_owner}: "
                f"{result.url}"
            ),
            completed_at=now,
            reconcile=False,
            worker_operation=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            prepare_delivery=True,
        )

    store.update(job.id, clone_verified)


def reconcile_uncertain_clone(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    github_factory: GitHubFactory = GitHubClient,
) -> None:
    states = frozenset({"submitted", "ambiguous"})
    if job.clone_operation_state not in states:
        return
    source = (job.clone_source or "").strip()
    try:
        checkout = (
            _github_provider(github_factory).observe_clone(source) if source else None
        )
    except GitHubError:
        checkout = None

    def reconcile(current: CursorJob) -> CursorJob | None:
        if current.clone_operation_state not in states:
            return None
        if checkout is None:
            message = (
                "Repository clone could not be reconciled automatically. "
                "Check the configured GitHub root before trying again."
            )
            return current.evolve_recovery(
                now=now,
                status=JobStatus.BLOCKED,
                clone_operation_state="manual_required",
                result=message,
                completed_at=now,
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                prepare_delivery=True,
            )
        return current.evolve_recovery(
            now=now,
            status=JobStatus.QUEUED,
            queued_at=now,
            repository=str(checkout),
            clone_operation_state="cloned",
            reconcile=False,
            worker_operation=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
        )

    store.update(job.id, reconcile)


def reconcile_uncertain_linear_ticket_creation(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
    linear_factory: LinearFactory | None = None,
) -> None:
    states = frozenset({"submitted", "ambiguous"})
    if job.linear_ticket_create_operation_state not in states:
        return
    try:
        provider = _linear_provider(linear_factory)
        plan = provider.plan_ticket_creation(
            job.linear_ticket_create_team_id or "",
            job.linear_ticket_create_team or "",
            job.linear_ticket_create_title or "",
            job.linear_ticket_create_description or "",
            correlation_marker=job.linear_ticket_create_marker,
        )
        result = provider.observe_ticket_creation(herdr_factory(), plan)
    except (HarnessError, LinearError):
        # Incomplete observation is not proof of absence; retry on a later scan.
        return

    def reconcile(current: CursorJob) -> CursorJob | None:
        if current.linear_ticket_create_operation_state not in states:
            return None
        if result is None:
            message = (
                "Linear ticket creation could not be reconciled automatically. "
                "Check the team before trying again."
            )
            return current.evolve_recovery(
                now=now,
                status=JobStatus.BLOCKED,
                linear_ticket_create_operation_state="manual_required",
                result=message,
                completed_at=now,
                reconcile=False,
                worker_operation=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                prepare_delivery=True,
            )
        return current.evolve_recovery(
            now=now,
            status=JobStatus.COMPLETED,
            linear_ticket_created_identifier=result.issue.identifier,
            linear_ticket_created_url=result.url,
            linear_ticket_create_operation_state="created",
            result=f"Created Linear ticket {result.issue.identifier}: {result.url}",
            completed_at=now,
            reconcile=False,
            worker_operation=None,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            prepare_delivery=True,
        )

    store.update(job.id, reconcile)


def reconcile_uncertain_worktree(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
) -> None:
    states = frozenset({"dispatching", "ambiguous", "failed_observing"})
    if job.worktree_provision_state not in states or not _reconciliation_due(
        job, "worktree", now
    ):
        return
    try:
        operation = job.checkout_operation
    except JobValidationError:
        operation = None
    if operation is None:
        _record_reconciliation_observation(
            store,
            job.id,
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    repository = operation.spec.repository
    branch = operation.spec.branch
    checkout = Path(operation.spec.path).resolve()
    try:
        client = herdr_factory()
        client.ensure_server()
        listing = client.run_json("worktree", "list", "--cwd", repository, "--json")
    except HerdrError:
        _record_reconciliation_observation(
            store,
            job.id,
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=False,
        )
        return
    worktrees = cast(list[dict[str, object]], listing.get("worktrees") or [])
    match = next(
        (
            item
            for item in worktrees
            if item.get("branch") == branch
            and Path(str(item.get("path") or "")).resolve() == checkout
        ),
        None,
    )
    if match is None and not checkout.exists():
        _record_reconciliation_observation(
            store,
            job.id,
            "worktree",
            "worktree_provision_state",
            states,
            now=now,
            observed_absent=True,
        )
        return

    def reconciled(current: CursorJob) -> CursorJob | None:
        if current.worktree_provision_state not in states:
            return None
        if match is None:
            return current.evolve_recovery(
                {
                    **_checkout_state_fields(current, CheckoutState.QUARANTINED),
                    "worktree_provision_error": (
                        "reserved worktree path exists but Herdr did not report the "
                        "expected branch"
                    ),
                    "worktree_manual_inspection_required": True,
                    "worker_operation": None,
                    **cleanup_fields(
                        finish_cleanup_reconciliation(current.cleanup_state)
                    ),
                },
                now=now,
            )
        observed_workspace = str(match.get("open_workspace_id") or "") or None
        observed_root_pane = str(match.get("root_pane_id") or "") or None
        workspace_id = current.worktree_workspace_id or observed_workspace
        root_pane_id = current.worktree_root_pane_id or observed_root_pane
        if (
            not workspace_id
            or not root_pane_id
            or observed_workspace != workspace_id
            or (observed_root_pane is not None and observed_root_pane != root_pane_id)
        ):
            return current.evolve_recovery(
                {
                    **_checkout_state_fields(current, CheckoutState.QUARANTINED),
                    "worktree_provision_error": (
                        "worktree is visible without its exact workspace and root-pane "
                        "identity"
                    ),
                    "worktree_manual_inspection_required": True,
                    "worker_operation": None,
                    **cleanup_fields(
                        finish_cleanup_reconciliation(current.cleanup_state)
                    ),
                },
                now=now,
            )
        return current.evolve_recovery(
            {
                **_checkout_state_fields(
                    current,
                    CheckoutState.RETAINED,
                    workspace_id=workspace_id,
                    root_pane_id=root_pane_id,
                ),
                "worktree_workspace_id": workspace_id,
                "worktree_root_pane_id": root_pane_id,
                "worktree_provision_error": None,
                "worker_operation": None,
                "worktree_manual_inspection_required": False,
                "worktree_next_reconcile_at": None,
                **cleanup_fields(finish_cleanup_reconciliation(current.cleanup_state)),
            },
            now=now,
        )

    store.update(job.id, reconciled)


def reconcile_uncertain_operations(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
    github_factory: GitHubFactory = GitHubClient,
    linear_factory: LinearFactory | None = None,
) -> None:
    reconcile_uncertain_agent(store, job, now=now, herdr_factory=herdr_factory)
    current = store.get(job.id)
    reconcile_uncertain_issue_creation(
        store,
        current,
        now=now,
        github_factory=github_factory,
    )
    current = store.get(job.id)
    reconcile_uncertain_clone(
        store,
        current,
        now=now,
        github_factory=github_factory,
    )
    current = store.get(job.id)
    reconcile_uncertain_pr_creation(
        store,
        current,
        now=now,
        github_factory=github_factory,
    )
    current = store.get(job.id)
    reconcile_uncertain_pr_merge(
        store,
        current,
        now=now,
        github_factory=github_factory,
    )
    current = store.get(job.id)
    reconcile_uncertain_repo_creation(
        store,
        current,
        now=now,
        github_factory=github_factory,
    )
    current = store.get(job.id)
    reconcile_uncertain_linear_ticket_creation(
        store,
        current,
        now=now,
        herdr_factory=herdr_factory,
        linear_factory=linear_factory,
    )
    current = store.get(job.id)
    reconcile_uncertain_clone(store, current, now=now, github_factory=github_factory)
    current = store.get(job.id)
    reconcile_uncertain_fork(store, current, now=now, github_factory=github_factory)
    current = store.get(job.id)
    reconcile_uncertain_worktree(store, current, now=now, herdr_factory=herdr_factory)


def _observe_prompt_effect(
    store: JobStore,
    job: CursorJob,
    client: HerdrClient,
    operation: SubmittingPrompt,
    *,
    now: float,
) -> None:
    identity = operation.identity
    result = next(
        (
            candidate
            for kind in (TASK_SUBMIT, CLARIFICATION_REPLY)
            if (
                candidate := store.outbox_result(
                    f"{kind}:{job.id}:{identity.phase}:{identity.turn_token}"
                )
            )
            is not None
            and candidate.job_id == job.id
            and candidate.kind == kind
            and candidate.status == "unknown"
            and candidate.outcome.get("outcome") == "OutcomeUnknown"
            and candidate.payload.get("prompt_job_id") == identity.job_id
            and candidate.payload.get("phase") == identity.phase
            and candidate.payload.get("turn") == identity.turn
            and candidate.payload.get("turn_token") == identity.turn_token
            and candidate.payload.get("target") == identity.target
            and candidate.payload.get("session_id") == identity.agent_session
            and candidate.payload.get("baseline_sequence") == identity.baseline_sequence
        ),
        None,
    )
    if result is None:
        return

    try:
        observation = client.harness.reconcile(
            identity.target,
            expected_session_id=identity.agent_session,
        )
    except HerdrError:
        return
    observed_session = (
        observation.session
        if observation.state
        in {ReconciliationState.ACTIVE, ReconciliationState.SETTLED}
        else None
    )
    # Session sequence is shared by every operation targeting the session. An
    # advance proves that something happened, not that this exact correlation
    # was accepted. The reconciliation API cannot currently provide stronger
    # evidence, so an unknown outbox outcome must remain manual.
    resolved = mark_prompt_ambiguous(SubmittingPrompt(identity), identity)
    accepted = False
    sequence = (
        observed_session.state_sequence
        if observed_session is not None
        else identity.baseline_sequence
    )
    if (
        isinstance(job.prompt_operation, AmbiguousPrompt)
        and not accepted
        and job.manual_reconcile_operation == "prompt"
    ):
        return

    def decide(current: CursorJob) -> CoordinatorDecision | None:
        try:
            current_operation = current.prompt_operation
        except (PromptOperationError, JobValidationError):
            return None
        if (
            not isinstance(current_operation, SubmittingPrompt | AmbiguousPrompt)
            or current_operation.identity != identity
        ):
            return None
        return CoordinatorDecision(
            job=current.evolve_recovery(
                now=now,
                prompt_operation=resolved,
                manual_reconcile_operation=None if accepted else "prompt",
                manual_reconcile_token=None if accepted else uuid.uuid4().hex,
                manual_reconcile_required_at=None if accepted else now,
            ),
            event_kind=f"{result.kind}.reconciled",
            event_payload={
                "effect_id": result.effect_id,
                "idempotency_key": result.idempotency_key,
                "reconciliation_state": observation.state.value,
                "observed_sequence": sequence,
            },
        )

    store.apply(
        CoordinatorCommand(
            job_id=job.id,
            expected_revision=job.revision,
            command_id=(
                f"reconcile:{result.effect_id}:{observation.state.value}:{sequence}"
            ),
            kind=f"{result.kind}.reconcile",
        ),
        decide,
    )


def reconcile_prompt_and_pane_operations(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory = HerdrClient,
) -> None:
    """Resolve only externally observable acceptance; never replay a submit."""
    if job.session_control == SessionControlMode.USER_OWNED.value:
        return
    if job.prompt_operation_state in {"submitting", "ambiguous"}:
        try:
            operation = job.prompt_operation
        except (PromptOperationError, JobValidationError):
            # Older or manually repaired rows may satisfy the flat schema without
            # carrying the complete typed identity. Crossing the submit boundary
            # without that fence is permanently ambiguous and must never replay.
            def fence_incomplete_prompt(current: CursorJob) -> CursorJob | None:
                if current.prompt_operation_state not in {
                    "submitting",
                    "ambiguous",
                }:
                    return None
                return current.evolve_recovery(
                    now=now,
                    prompt_operation_state="ambiguous",
                    manual_reconcile_operation="prompt",
                    manual_reconcile_token=uuid.uuid4().hex,
                    manual_reconcile_required_at=now,
                )

            store.update(job.id, fence_incomplete_prompt)
            return
        if not isinstance(operation, SubmittingPrompt | AmbiguousPrompt):
            return
        try:
            client = _herdr_client_for_job(herdr_factory, job)
            client.ensure_server()
        except HerdrError:
            return
        _observe_prompt_effect(
            store,
            job,
            client,
            (
                operation
                if isinstance(operation, SubmittingPrompt)
                else SubmittingPrompt(operation.identity)
            ),
            now=now,
        )

    current = store.get(job.id)
    if current.participant_creation_state in {"submitting", "ambiguous"}:
        # Herdr does not expose an idempotency key for tab creation. Absence in
        # a listing cannot prove the create was not accepted, so retain every
        # deterministic identity and require an operator decision.
        token = uuid.uuid4().hex

        def require_pane_reconciliation(candidate: CursorJob) -> CursorJob | None:
            if candidate.participant_creation_state not in {
                "submitting",
                "ambiguous",
            }:
                return None
            action = _pane_manual_reconciliation_action(candidate, token)
            terminal_result = candidate.terminal_intent_result
            terminal_error = candidate.terminal_intent_error
            if candidate.terminal_intent_status is not None:
                terminal_result = _append_manual_action(terminal_result, action)
                if terminal_error is not None:
                    terminal_error = _append_manual_action(terminal_error, action)
            lifecycle = replace(
                candidate.participant_lifecycle,
                creation=candidate.participant_lifecycle.creation.require_manual(),
            )
            return candidate.evolve_recovery(
                now=now,
                participant_creation_state=lifecycle.creation.state.value,
                manual_reconcile_operation="pane",
                manual_reconcile_token=token,
                manual_reconcile_required_at=now,
                terminal_intent_result=terminal_result,
                terminal_intent_error=terminal_error,
            )

        store.update(
            job.id,
            require_pane_reconciliation,
        )


def _append_manual_action(message: str | None, action: str) -> str:
    base = (message or "Cursor job cleanup is waiting").rstrip(". ")
    suffix = f". {action}"
    return f"{base[: max(0, 500 - len(suffix))]}{suffix}"[:500]


def _pane_manual_reconciliation_action(job: CursorJob, token: str) -> str:
    participant = job.participant_creation_participant or "workflow"
    target = job.participant_creation_target or "unknown"
    return (
        f"Inspect Herdr for the {participant} pane target {target}, then manually "
        "reconcile pane creation as materialized or confirmed absent "
        f"with token {token}."
    )


def stage_terminal_intent(
    job: CursorJob,
    status: JobStatus,
    *,
    now: float,
    result: str,
    error: str | None = None,
    clear_worker: bool = False,
    preserve_worker_operation: bool = False,
    voice_question: dict[str, object] | None = None,
    job_changes: Mapping[str, object] | None = None,
) -> CursorJob:
    if status not in TERMINAL_STATUSES:
        raise HarnessError("terminal cleanup requires a terminal intent")
    if job.terminal_state is not None:
        raise HarnessError("terminal outcome or intent is already fixed")
    if (
        job.manual_reconcile_operation == "pane"
        and job.participant_creation_state == "manual_required"
        and job.manual_reconcile_token
    ):
        action = _pane_manual_reconciliation_action(job, job.manual_reconcile_token)
        result = _append_manual_action(result, action)
        if error is not None:
            error = _append_manual_action(error, action)
    intent = TerminalIntent(status.value, result, error, now)
    cleanup = begin_cleanup(uuid.uuid4().hex)
    changes: dict[str, object] = {
        "terminal_intent_status": intent.status,
        "terminal_intent_result": intent.result,
        "terminal_intent_error": intent.error,
        "terminal_intent_completed_at": intent.completed_at,
        "voice_question": (
            job.voice_question if voice_question is None else voice_question
        ),
        "worker_operation": (
            job.worker_operation if preserve_worker_operation else "target_cleanup"
        ),
        "pull_request_worktree_state": (
            "retained"
            if job.github_pull_request
            and job.worktree_path
            and job.pull_request_worktree_state != "quarantined"
            else job.pull_request_worktree_state
        ),
        "worker_pid": None if clear_worker else job.worker_pid,
        "worker_boot_id": None if clear_worker else job.worker_boot_id,
        "worker_process_start": None if clear_worker else job.worker_process_start,
        "worker_token": None if clear_worker else job.worker_token,
    }
    changes.update(cleanup_fields(cleanup))
    changes.update(job_changes or {})
    changes["status"] = JobStatus.RECONCILING.value
    updated = job._updated(**changes)
    event = (
        CancellationEvent(job.revision, updated.lifecycle)
        if status == JobStatus.CANCELLED
        else RecoveryEvent(job.revision, updated.lifecycle)
    )
    job.validate_lifecycle_event(updated, event)
    return replace(updated, _lifecycle_event=event)


def _infrastructure_uncertain_for_release(job: CursorJob) -> bool:
    if job.terminal_intent_status == JobStatus.CANCELLED:
        # User cancellation already chose teardown, but provisional dispatch and
        # materializing checkout/fork work still need reconciliation. Prompt,
        # agent, and pane ambiguity must not keep ticket fences up once those
        # stronger fences are clear.
        return bool(
            job.agent_dispatch_state
            in {"dispatching", "ambiguous", "failed_observing", "manual_required"}
            or job.fork_operation_state
            in {"submitted", "ambiguous", "failed_observing"}
            or job.worktree_provision_state
            in {"dispatching", "ambiguous", "failed_observing"}
            or job.participant_creation_state
            in {"submitting", "ambiguous", "manual_required"}
        )
    return bool(
        job.agent_dispatch_state
        in {"dispatching", "ambiguous", "failed_observing", "manual_required"}
        or job.prompt_operation_state in {"submitting", "ambiguous"}
        or job.fork_operation_state in {"submitted", "ambiguous", "failed_observing"}
        or job.worktree_provision_state
        in {"dispatching", "ambiguous", "failed_observing"}
        or job.participant_creation_state
        in {"submitting", "ambiguous", "manual_required"}
    )


def cancel_target_and_release(
    store: JobStore,
    job_id: str,
    target: str,
    release_token: str,
    *,
    worker_stopped: bool = True,
    herdr_factory: HerdrFactory = HerdrClient,
) -> None:
    current = store.get(job_id)
    staged = (
        current.status == JobStatus.RECONCILING
        and current.terminal_intent_status is not None
    )
    if (
        not staged and current.status not in {JobStatus.CANCELLED, JobStatus.FAILED}
    ) or current.target_release_token != release_token:
        return
    uncertain_pane_target = (
        current.participant_creation_target
        if current.participant_creation_state
        in {"submitting", "ambiguous", "manual_required"}
        else None
    )
    targets = list(
        dict.fromkeys(
            value
            for value in (
                target,
                current.participant_target(WorkflowParticipant.PLANNER),
                current.participant_target(WorkflowParticipant.REVIEWER),
                current.participant_target(WorkflowParticipant.IMPLEMENTER),
                (
                    current.participant_creation_target
                    if current.participant_creation_state != "failed"
                    else None
                ),
            )
            if value and value != uncertain_pane_target
        )
    )
    cleanup_confirmed = True
    unverified_targets: set[str] = (
        {uncertain_pane_target} if uncertain_pane_target else set()
    )
    if targets:
        try:
            client = _herdr_client_for_job(herdr_factory, current)
            client.ensure_server()
            for participant_target in targets:
                if (
                    current.agent_dispatch_state is None
                    or current.agent_provider is None
                    or current.agent_provider_session_id is None
                    or current.agent_state_sequence is None
                ):
                    expected_pane_id = ""
                    expected_binding_workspace = ""
                    creation_binding_replaced_current = bool(
                        current.participant_creation_state == "created"
                        and current.participant_creation_target
                        and current.participant_creation_target != current.herdr_target
                        and current.participant_creation_pane_id
                        == current.herdr_pane_id
                    )
                    if (
                        participant_target == current.herdr_target
                        and not creation_binding_replaced_current
                    ):
                        expected_pane_id = current.herdr_pane_id or ""
                        expected_binding_workspace = current.herdr_workspace_id or ""
                    if (
                        participant_target == current.participant_creation_target
                        and current.participant_creation_state == "created"
                        and current.participant_creation_pane_id
                        and current.participant_creation_workspace_id
                    ):
                        expected_pane_id = current.participant_creation_pane_id
                        expected_binding_workspace = (
                            current.participant_creation_workspace_id
                        )
                    expected_checkout = (
                        current.worktree_path or current.repository or ""
                    )
                    expected_workspace = current.worktree_workspace_id or ""
                    root_pane_id = current.worktree_root_pane_id or ""
                    evidence_complete = bool(
                        expected_pane_id
                        and expected_binding_workspace
                        and expected_checkout
                        and expected_workspace
                        and root_pane_id
                        and expected_workspace != LEGACY_BOOT_ID
                        and root_pane_id != LEGACY_BOOT_ID
                        and expected_binding_workspace == expected_workspace
                        and expected_pane_id != root_pane_id
                    )
                    try:
                        observed_agent = client.get_agent(participant_target)
                    except HerdrError as exc:
                        if _agent_not_found(exc) and evidence_complete:
                            continue
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    if not evidence_complete:
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    if not isinstance(observed_agent, dict):
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    if not observed_agent:
                        continue
                    pane_id = str(observed_agent.get("pane_id") or "")
                    workspace_id = str(observed_agent.get("workspace_id") or "")
                    actual_checkout = str(observed_agent.get("cwd") or "")
                    try:
                        binding_matches = bool(
                            pane_id == expected_pane_id
                            and workspace_id == expected_workspace
                            and actual_checkout
                            and Path(expected_checkout).resolve()
                            == Path(actual_checkout).resolve()
                        )
                    except OSError:
                        binding_matches = False
                    if not binding_matches:
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    try:
                        client.close_owned_pane(
                            participant_target,
                            pane_id,
                            workspace_id,
                        )
                    except HerdrError:
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                    continue
                try:
                    agent_operation = current.agent_session_operation
                    checkout_operation = current.checkout_operation
                except JobValidationError:
                    agent_operation = None
                    checkout_operation = None
                if (
                    agent_operation is None
                    or agent_operation.spec.target != participant_target
                ):
                    agent_operation = None
                    historical = next(
                        (
                            owner
                            for owner in current.participant_session_owners
                            if owner.get("target") == participant_target
                        ),
                        None,
                    )
                    if historical is not None:
                        agent_operation = AgentSessionOperation(
                            AgentSessionState.READY,
                            AgentSessionSpec(
                                str(historical["target"]),
                                str(historical["checkout"]),
                                str(historical["workspace_id"]),
                                str(historical["pane_id"]),
                            ),
                            SessionIdentity(
                                str(historical["provider"]),
                                str(historical["session_id"]),
                                str(historical["target"]),
                                cast(int, historical["state_sequence"]),
                            ),
                        )
                if agent_operation is None:
                    try:
                        creation_operation = current.participant_pane_operation
                    except JobValidationError:
                        creation_operation = None
                    root_pane_id = current.worktree_root_pane_id or ""
                    if (
                        creation_operation is None
                        or creation_operation.state != OperationState.SETTLED
                        or creation_operation.spec.target != participant_target
                        or creation_operation.pane_id is None
                        or not root_pane_id
                        or creation_operation.pane_id == root_pane_id
                    ):
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    try:
                        observed_agent = client.get_agent(participant_target)
                    except HerdrError as exc:
                        if _agent_not_found(exc):
                            continue
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    if not isinstance(observed_agent, dict):
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    if not observed_agent:
                        continue
                    pane_id = str(observed_agent.get("pane_id") or "")
                    workspace_id = str(observed_agent.get("workspace_id") or "")
                    actual_checkout = str(observed_agent.get("cwd") or "")
                    try:
                        binding_matches = bool(
                            pane_id == creation_operation.pane_id
                            and workspace_id == creation_operation.spec.workspace_id
                            and actual_checkout
                            and Path(actual_checkout).resolve()
                            == Path(creation_operation.spec.checkout).resolve()
                        )
                    except OSError:
                        binding_matches = False
                    if not binding_matches:
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                        continue
                    try:
                        client.close_owned_pane(
                            participant_target,
                            pane_id,
                            workspace_id,
                        )
                    except HerdrError:
                        cleanup_confirmed = False
                        unverified_targets.add(participant_target)
                    continue
                if (
                    agent_operation is None
                    or agent_operation.session is None
                    or participant_target != agent_operation.spec.target
                ):
                    cleanup_confirmed = False
                    unverified_targets.add(participant_target)
                    continue
                checkout_path = (
                    checkout_operation.spec.path
                    if checkout_operation is not None
                    else (current.worktree_path or current.repository or "")
                )
                checkout_workspace_id = (
                    checkout_operation.workspace_id
                    if checkout_operation is not None
                    else (
                        current.worktree_workspace_id
                        or agent_operation.spec.workspace_id
                    )
                )
                checkout_root_pane_id = (
                    checkout_operation.root_pane_id
                    if checkout_operation is not None
                    else current.worktree_root_pane_id
                )
                if (
                    not checkout_path
                    or checkout_workspace_id is None
                    or agent_operation.spec.checkout != checkout_path
                    or agent_operation.spec.workspace_id != checkout_workspace_id
                ):
                    cleanup_confirmed = False
                    unverified_targets.add(participant_target)
                    continue
                try:
                    observation = client.reconcile_session(
                        participant_target,
                        expected_session_id=agent_operation.session.session_id,
                    )
                except HerdrError:
                    cleanup_confirmed = False
                    unverified_targets.add(participant_target)
                    continue
                if observation.state in {
                    ReconciliationState.MISSING,
                    ReconciliationState.CHANGED,
                }:
                    # The owned session is gone. A replacement is not ours to close.
                    continue
                observed = observation.session
                if (
                    observation.state
                    not in {ReconciliationState.ACTIVE, ReconciliationState.SETTLED}
                    or observed is None
                ):
                    cleanup_confirmed = False
                    unverified_targets.add(participant_target)
                    continue
                identity = SessionIdentity(
                    observed.provider,
                    observed.session_id,
                    observed.target,
                    observed.state_sequence,
                )
                pane_id = str(observed.metadata.get("pane_id") or "")
                workspace_id = str(observed.metadata.get("workspace_id") or "")
                actual_checkout = str(observed.metadata.get("cwd") or "")
                try:
                    checkout_matches = (
                        Path(actual_checkout).resolve() == Path(checkout_path).resolve()
                    )
                except OSError:
                    checkout_matches = False
                try:
                    creation_operation = current.participant_pane_operation
                except JobValidationError:
                    creation_operation = None
                creation_proves_non_root = bool(
                    creation_operation is not None
                    and creation_operation.state == OperationState.SETTLED
                    and creation_operation.spec.target == participant_target
                    and creation_operation.pane_id == pane_id
                    and creation_operation.spec.workspace_id == workspace_id
                    and creation_operation.spec.checkout == checkout_path
                )
                if (
                    not agent_operation.accepts_observation(identity)
                    or pane_id != agent_operation.spec.pane_id
                    or workspace_id != agent_operation.spec.workspace_id
                    or not checkout_matches
                    or (
                        bool(checkout_root_pane_id) and pane_id == checkout_root_pane_id
                    )
                    or (not checkout_root_pane_id and not creation_proves_non_root)
                ):
                    cleanup_confirmed = False
                    unverified_targets.add(participant_target)
                    continue
                if (
                    creation_operation is not None
                    and creation_operation.spec.target == participant_target
                    and (
                        creation_operation.state != OperationState.SETTLED
                        or creation_operation.pane_id != pane_id
                        or creation_operation.spec.workspace_id != workspace_id
                        or creation_operation.spec.checkout != checkout_path
                    )
                ):
                    cleanup_confirmed = False
                    unverified_targets.add(participant_target)
                    continue
                try:
                    client.close_owned_pane(
                        participant_target,
                        pane_id,
                        workspace_id,
                    )
                except HerdrError:
                    cleanup_confirmed = False
                    unverified_targets.add(participant_target)
        except HerdrError:
            cleanup_confirmed = False
            unverified_targets.update(targets)

    def release(job: CursorJob) -> CursorJob | None:
        if (
            job.status
            not in {JobStatus.RECONCILING, JobStatus.CANCELLED, JobStatus.FAILED}
            or job.target_release_token != release_token
        ):
            return None
        if (
            cleanup_confirmed
            and worker_stopped
            and not _infrastructure_uncertain_for_release(job)
        ):
            if job.terminal_intent_status is not None:
                intent = job.terminal_state
                if not isinstance(intent, TerminalIntent):
                    return None
                try:
                    settled = settle_cleanup(job.cleanup_state, release_token)
                except LifecycleTransitionError:
                    return None
                outcome = MaterializedTerminalOutcome(
                    intent.status,
                    intent.result,
                    intent.error,
                    intent.completed_at,
                )
                delivery_changes: dict[str, Any] = {
                    **cleanup_fields(settled),
                    **_session_state_fields(job, AgentSessionState.CONFIRMED_ABSENT),
                    "result": outcome.result,
                    "error": outcome.error,
                    "completed_at": outcome.completed_at,
                    "terminal_intent_status": None,
                    "terminal_intent_result": None,
                    "terminal_intent_error": None,
                    "terminal_intent_completed_at": None,
                    "worker_operation": None,
                    "worker_pid": None,
                    "worker_boot_id": None,
                    "worker_process_start": None,
                    "worker_token": None,
                    "worker_claim_operation": None,
                    "worker_claimed_at": None,
                    "target_release_manual_required": False,
                    "target_release_unverified_targets": [],
                    "agent_operation_target": job.agent_operation_target
                    or job.herdr_target,
                    "agent_operation_workspace_id": job.agent_operation_workspace_id
                    or job.herdr_workspace_id,
                    "agent_operation_pane_id": job.agent_operation_pane_id
                    or job.herdr_pane_id,
                    "agent_operation_checkout": job.agent_operation_checkout
                    or job.worktree_path
                    or job.repository,
                    "herdr_target": None,
                    "herdr_pane_id": None,
                    "herdr_workspace_id": None,
                    "agent_name": None,
                    "active_participant": None,
                    "planner_target": None,
                    "reviewer_target": None,
                    "implementer_target": None,
                    "participant_admission_state": "released",
                    "prompt_operation_state": "none",
                    "prompt_operation_phase": None,
                    "prompt_operation_turn": None,
                    "prompt_operation_target": None,
                    "prompt_operation_agent_session": None,
                    "prompt_baseline_sequence": None,
                    "workflow_phase": (
                        "finished"
                        if outcome.status == JobStatus.COMPLETED.value
                        else job.workflow_phase.value
                    ),
                }
                return job.evolve_for_delivery(
                    now=outcome.completed_at,
                    status=JobStatus(outcome.status),
                    **delivery_changes,
                )
            try:
                settled = settle_cleanup(job.cleanup_state, release_token)
            except LifecycleTransitionError:
                return None
            release_changes: dict[str, Any] = {
                **cleanup_fields(settled),
                **_session_state_fields(job, AgentSessionState.CONFIRMED_ABSENT),
                "worker_pid": None,
                "worker_boot_id": None,
                "worker_process_start": None,
                "worker_token": None,
                "worker_claim_operation": None,
                "worker_claimed_at": None,
                "target_release_manual_required": False,
                "target_release_unverified_targets": [],
                "agent_operation_target": job.agent_operation_target
                or job.herdr_target,
                "agent_operation_workspace_id": job.agent_operation_workspace_id
                or job.herdr_workspace_id,
                "agent_operation_pane_id": job.agent_operation_pane_id
                or job.herdr_pane_id,
                "agent_operation_checkout": job.agent_operation_checkout
                or job.worktree_path
                or job.repository,
                "herdr_target": None,
                "herdr_pane_id": None,
                "herdr_workspace_id": None,
                "agent_name": None,
                "active_participant": None,
                "planner_target": None,
                "reviewer_target": None,
                "implementer_target": None,
                "participant_admission_state": (
                    "released"
                    if job.status in TERMINAL_STATUSES
                    else job.participant_admission_state
                ),
            }
            return job.evolve(**release_changes)
        cleanup = job.cleanup_state
        if not isinstance(cleanup, CleanupOwned):
            if not unverified_targets:
                return None
            return job.evolve(
                target_release_manual_required=True,
                target_release_unverified_targets=sorted(unverified_targets),
            )
        try:
            pending = abandon_cleanup_owner(cleanup, release_token)
        except LifecycleTransitionError:
            return None
        return job.evolve(
            **cleanup_fields(pending),
            target_release_manual_required=bool(unverified_targets),
            target_release_unverified_targets=sorted(unverified_targets),
        )

    store.update(job_id, release)


def acknowledge_worktree_quarantine(
    store: JobStore, job_id: str, *, now: float | None = None
) -> None:
    acknowledged_at = time.time() if now is None else now

    def acknowledge(job: CursorJob) -> CursorJob | None:
        checkout = job.checkout_operation
        if (
            checkout is None
            or checkout.state != CheckoutState.QUARANTINED
            or not job.worktree_manual_inspection_required
        ):
            return None
        acknowledged = checkout.transition(
            CheckoutState.RETAINED
            if job.worktree_workspace_id and job.worktree_root_pane_id
            else CheckoutState.AMBIGUOUS,
            workspace_id=job.worktree_workspace_id,
            root_pane_id=job.worktree_root_pane_id,
        )
        return job.evolve(
            checkout_operation=acknowledged,
            worktree_manual_inspection_required=False,
            worktree_quarantine_acknowledged_at=acknowledged_at,
        )

    if store.update(job_id, acknowledge) is None:
        raise HarnessError(
            f"Cursor job {job_id} has no worktree quarantine to acknowledge"
        )


def resolve_manual_reconciliation(
    store: JobStore,
    job_id: str,
    operation: str,
    token: str,
    outcome: str,
    *,
    now: float | None = None,
    pane_id: str | None = None,
    workspace_id: str | None = None,
) -> CursorJob:
    if operation not in {"agent", "fork", "worktree", "prompt", "pane"}:
        raise HarnessError("manual reconciliation operation is invalid")
    if outcome not in {"confirmed_absent", "materialized"}:
        raise HarnessError("manual reconciliation outcome is invalid")
    supplied_pane_identity = pane_id is not None or workspace_id is not None
    if supplied_pane_identity and (
        operation not in {"pane", "worktree"}
        or outcome != "materialized"
        or not pane_id
        or not workspace_id
    ):
        raise HarnessError(
            "pane and workspace identity require a materialized pane or worktree outcome"
        )
    resolved_at = time.time() if now is None else now

    def resolve(job: CursorJob) -> CursorJob | None:
        if (
            job.manual_reconcile_operation != operation
            or not secrets.compare_digest(job.manual_reconcile_token or "", token)
            or job.operation_state(operation)
            != ("ambiguous" if operation == "prompt" else "manual_required")
        ):
            return None
        followup_changes = (
            _agent_manual_reconciliation_changes(
                job,
                required_at=resolved_at,
            )
            if (
                operation == "prompt"
                and job.session_control == SessionControlMode.USER_OWNED.value
                and job.agent_dispatch_state
                in {"dispatching", "ambiguous", "failed_observing"}
            )
            else None
        )
        return job.resolve_manual_operation(
            operation,
            outcome,
            resolved_at=resolved_at,
            pane_id=pane_id,
            workspace_id=workspace_id,
            job_changes=followup_changes,
        )

    resolved = store.update(job_id, resolve)
    if resolved is None:
        raise HarnessError(f"Cursor job {job_id} manual reconciliation fence is stale")
    return resolved


_IN_FLIGHT_PROMPT_STATES = frozenset({"submitting", "observed", "ambiguous"})
_UNRESOLVED_SESSION_STATES = frozenset(
    {"dispatching", "ambiguous", "failed_observing", "manual_required"}
)


def _session_control_from(job: CursorJob) -> SessionControlState:
    return job.session_control_state()


def _confusable_harness_operation(job: CursorJob) -> bool:
    return (
        job.prompt_operation_state in _IN_FLIGHT_PROMPT_STATES
        or job.agent_dispatch_state in _UNRESOLVED_SESSION_STATES
        or job.manual_reconcile_operation is not None
    )


def _agent_manual_reconciliation_changes(
    job: CursorJob,
    *,
    required_at: float,
) -> dict[str, Any]:
    agent_operation = job.agent_session_operation
    changes: dict[str, Any] = {
        "manual_reconcile_operation": "agent",
        "manual_reconcile_token": uuid.uuid4().hex,
        "manual_reconcile_required_at": required_at,
    }
    if agent_operation is not None:
        changes["agent_session_operation"] = agent_operation.transition(
            AgentSessionState.MANUAL_REQUIRED
        )
    else:
        changes["agent_dispatch_state"] = "manual_required"
    return changes


def relinquish_session_control(
    store: JobStore,
    job_id: str,
    *,
    now: float | None = None,
    stop_owned_worker: Callable[[CursorJob], bool] = stop_worker,
) -> CursorJob:
    """Durably hand a nonterminal session to the user without cancelling it."""

    relinquished_at = time.time() if now is None else now
    snapshot = store.get(job_id)
    if snapshot.status in TERMINAL_STATUSES:
        raise HarnessError(f"Cursor job {job_id} is already {snapshot.status.value}")
    if snapshot.status not in {
        JobStatus.QUEUED,
        JobStatus.ROUTING,
        JobStatus.RUNNING,
        JobStatus.RECONCILING,
        JobStatus.AWAITING_USER,
        JobStatus.BLOCKED,
    }:
        raise HarnessError(f"Cursor job {job_id} cannot relinquish session control")
    try:
        _session_control_from(snapshot).relinquish()
    except JobLifecycleError as exc:
        raise HarnessError(str(exc)) from exc
    if any(
        value is not None
        for value in (
            snapshot.worker_pid,
            snapshot.worker_boot_id,
            snapshot.worker_process_start,
            snapshot.worker_token,
        )
    ) and not stop_owned_worker(snapshot):
        raise HarnessError(
            f"Cursor job {job_id} worker could not be stopped; "
            "session ownership was retained"
        )

    def relinquish(job: CursorJob) -> CursorJob | None:
        if job.revision != snapshot.revision:
            return None
        if job.status in TERMINAL_STATUSES:
            raise HarnessError(f"Cursor job {job_id} is already {job.status.value}")
        if job.status not in {
            JobStatus.QUEUED,
            JobStatus.ROUTING,
            JobStatus.RUNNING,
            JobStatus.RECONCILING,
            JobStatus.AWAITING_USER,
            JobStatus.BLOCKED,
        }:
            raise HarnessError(f"Cursor job {job_id} cannot relinquish session control")
        try:
            next_control = _session_control_from(job).relinquish()
        except JobLifecycleError as exc:
            raise HarnessError(str(exc)) from exc
        changes: dict[str, Any] = {
            "session_control": next_control.mode.value,
            "session_control_generation": next_control.generation,
            "worker_operation": None,
            "worker_pid": None,
            "worker_boot_id": None,
            "worker_process_start": None,
            "worker_token": None,
        }
        if job.worker_token is not None:
            changes["status"] = JobStatus.QUEUED
            changes["queued_at"] = relinquished_at
        try:
            operation = job.prompt_operation
        except (PromptOperationError, JobValidationError):
            operation = None
        if isinstance(
            operation,
            SubmittingPrompt | SubmittedPrompt | ObservedPrompt,
        ):
            changes["prompt_operation"] = mark_prompt_ambiguous(
                operation, operation.identity
            )
            if job.manual_reconcile_operation is None:
                changes["manual_reconcile_operation"] = "prompt"
                changes["manual_reconcile_token"] = uuid.uuid4().hex
                changes["manual_reconcile_required_at"] = relinquished_at
        elif (
            isinstance(operation, AmbiguousPrompt)
            and job.manual_reconcile_operation is None
        ):
            changes["manual_reconcile_operation"] = "prompt"
            changes["manual_reconcile_token"] = uuid.uuid4().hex
            changes["manual_reconcile_required_at"] = relinquished_at
        if (
            job.agent_dispatch_state in {"dispatching", "ambiguous", "failed_observing"}
            and job.manual_reconcile_operation is None
            and "manual_reconcile_operation" not in changes
        ):
            changes.update(
                _agent_manual_reconciliation_changes(
                    job,
                    required_at=relinquished_at,
                )
            )
        return job.evolve(**changes)

    updated = store.update(job_id, relinquish)
    if updated is None:
        raise HarnessError(f"Cursor job {job_id} session control was not updated")
    return updated


def resume_session_control(
    store: JobStore,
    job_id: str,
    *,
    now: float | None = None,
    herdr_factory: HerdrFactory = HerdrClient,
) -> CursorJob:
    """Hand a user-owned session back after reconciling identity and baseline."""

    job = store.get(job_id)
    if job.status in TERMINAL_STATUSES:
        raise HarnessError(f"Cursor job {job_id} is already {job.status.value}")
    if job.session_control != SessionControlMode.USER_OWNED.value:
        raise HarnessError(
            f"Cursor job {job_id} is not under user-owned session control"
        )
    if _confusable_harness_operation(job):
        raise HarnessError(
            f"Cursor job {job_id} has an unresolved harness operation that "
            "could be confused with manual activity; explicit reconciliation "
            "is required"
        )

    target = job.herdr_target
    expected_session = job.agent_provider_session_id
    observed_sequence: int | None = None
    if target and expected_session:
        try:
            client = herdr_factory()
            client.ensure_server()
            observation = client.harness.reconcile(
                target, expected_session_id=expected_session
            )
        except HerdrError as exc:
            raise HarnessError(
                f"could not reconcile session for hand-back: {exc}"
            ) from exc
        if observation.state != ReconciliationState.SETTLED:
            raise HarnessError(
                f"Cursor job {job_id} session is not settled and quiescent; "
                "hand-back cannot resume automation and explicit reconciliation "
                "is required"
            )
        if observation.session is None:
            raise HarnessError(
                f"Cursor job {job_id} session identity is unprovable; "
                "explicit reconciliation is required"
            )
        confirmation = client.harness.reconcile(
            target, expected_session_id=expected_session
        )
        observed = observation.session
        confirmed = confirmation.session
        if (
            confirmation.state != ReconciliationState.SETTLED
            or confirmed is None
            or confirmed.provider != observed.provider
            or confirmed.session_id != observed.session_id
            or confirmed.target != observed.target
            or confirmed.state_sequence != observed.state_sequence
        ):
            raise HarnessError(
                f"Cursor job {job_id} session changed during hand-back "
                "observation; automation remains paused"
            )
        observed_sequence = confirmed.state_sequence
    elif target or expected_session:
        raise HarnessError(
            f"Cursor job {job_id} session identity is incomplete; "
            "explicit reconciliation is required"
        )

    def resume(current: CursorJob) -> CursorJob | None:
        if (
            current.session_control != SessionControlMode.USER_OWNED.value
            or current.revision != job.revision
            or _confusable_harness_operation(current)
        ):
            return None
        next_control = _session_control_from(current).resume()
        changes: dict[str, Any] = {
            "session_control": next_control.mode.value,
            "session_control_generation": next_control.generation,
        }
        if observed_sequence is not None:
            changes["agent_state_sequence"] = observed_sequence
            agent_operation = current.agent_session_operation
            if agent_operation is not None and agent_operation.session is not None:
                changes["agent_session_operation"] = replace(
                    agent_operation,
                    session=replace(
                        agent_operation.session,
                        state_sequence=observed_sequence,
                    ),
                )
            try:
                prompt = current.prompt_operation
            except (PromptOperationError, JobValidationError):
                prompt = None
            if isinstance(prompt, PlannedPrompt | SubmittedPrompt):
                changes["prompt_operation"] = type(prompt)(
                    replace(prompt.identity, baseline_sequence=observed_sequence)
                )
        return current.evolve(**changes)

    updated = store.update(job_id, resume)
    if updated is None:
        raise HarnessError(f"Cursor job {job_id} session control hand-back is stale")
    return updated


def _target_release_owner_alive(
    job: CursorJob,
    *,
    get_boot_identity: Callable[[], str | None],
    get_process_identity: Callable[[int], str | None],
) -> bool:
    pid = job.target_release_owner_pid or 0
    expected = job.target_release_owner_start or ""
    expected_boot = job.target_release_owner_boot_id or ""
    return bool(
        pid
        and expected
        and expected_boot
        and get_boot_identity() == expected_boot
        and get_process_identity(pid) == expected
    )


def _reconcile_question_prompt(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory,
    is_worker_alive: Callable[[CursorJob], bool],
) -> None:
    question = question_adapter.current(job)
    if (
        job.session_control == SessionControlMode.USER_OWNED.value
        or question is None
        or question.state != QuestionState.DISPATCHING
        or question.prompt_state
        not in {PromptOperationState.SUBMITTED, PromptOperationState.OBSERVED}
        or is_worker_alive(job)
    ):
        return

    observed_started = question.prompt_state == PromptOperationState.OBSERVED
    current_sequence: int | None = None
    current_status = ""
    if not observed_started:
        try:
            prompt_identity = question_adapter.shared_prompt_identity(job, question)
        except PromptOperationError:
            return
        try:
            client = _herdr_client_for_job(herdr_factory, job)
            client.ensure_server()
            agent = client.get_agent(job.herdr_target or "")
        except (HarnessError, HerdrError):
            return
        if (
            agent_session_identity(agent.get("agent_session"))
            != prompt_identity.agent_session
        ):
            return
        value = agent.get("state_change_seq")
        current_sequence = (
            value if isinstance(value, int) and not isinstance(value, bool) else None
        )
        current_status = str(agent.get("agent_status") or "")
        observed_started = current_status == "working" or (
            current_sequence is not None
            and question.prompt_baseline_seq is not None
            and current_sequence != question.prompt_baseline_seq
        )

    def reconcile(current: CursorJob) -> CursorJob | None:
        current_question = question_adapter.current(current)
        if (
            current_question is None
            or current.revision != job.revision
            or current_question != question
            or is_worker_alive(current)
        ):
            return None
        if observed_started:
            observed = (
                current_question
                if current_question.prompt_state == PromptOperationState.OBSERVED
                else observe_question_prompt(
                    current_question,
                    question_adapter.shared_prompt_identity(current, current_question),
                )
            )
            return current.evolve(
                status=JobStatus.QUEUED,
                queued_at=now,
                reconcile=True,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                voice_question=question_adapter.envelope(
                    observed,
                    QuestionState.DISPATCHING,
                    job=current,
                    prompt_state=observed.prompt_state,
                ),
            )
        if (
            current_question.prompt_baseline_seq is None
            or current_sequence is None
            or current_status not in {"idle", "done"}
        ):
            message = (
                "Cursor clarification dispatch is ambiguous and requires "
                "manual attention."
            )
            return current.evolve_for_delivery(
                now=now,
                status=JobStatus.BLOCKED,
                result=message,
                completed_at=now,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
            )
        observations = current_question.prompt_absent_observations + 1
        if observations < PROMPT_ABSENT_OBSERVATIONS:
            return current.evolve(
                voice_question=question_adapter.envelope(
                    current_question,
                    QuestionState.DISPATCHING,
                    job=current,
                    prompt_absent_observations=observations,
                )
            )
        replanned = replan_question_prompt(
            current_question,
            question_adapter.shared_prompt_identity(current, current_question),
        )
        return current.evolve(
            status=JobStatus.QUEUED,
            queued_at=now,
            reconcile=False,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            voice_question=question_adapter.envelope(
                replanned,
                QuestionState.DISPATCHING,
                job=current,
                prompt_state=replanned.prompt_state,
                prompt_baseline_seq=None,
                prompt_submitted_at=None,
                prompt_absent_observations=0,
            ),
        )

    store.update(job.id, reconcile)


def _reconcile_interactive_questionnaire(
    store: JobStore,
    job: CursorJob,
    *,
    now: float,
    herdr_factory: HerdrFactory,
) -> None:
    if (
        job.status != JobStatus.BLOCKED
        or not job.interactive_questionnaire_blocked
        or not job.herdr_target
    ):
        return
    try:
        client = _herdr_client_for_job(herdr_factory, job)
        client.ensure_server()
        agent = client.get_agent(job.herdr_target)
    except (HarnessError, HerdrError):
        return
    if agent.get("interactive_ready") is False:
        return

    def resume(current: CursorJob) -> CursorJob | None:
        if (
            current.status != JobStatus.BLOCKED
            or not current.interactive_questionnaire_blocked
            or current.herdr_target != job.herdr_target
        ):
            return None
        if current.prompt_operation_state == "planned":
            return current.evolve(
                status=JobStatus.QUEUED,
                remove=frozenset({"interactive_questionnaire_blocked"}),
                queued_at=now,
                reconcile=True,
                prompt_operation_state="none",
                prompt_operation_phase=None,
                prompt_operation_turn=None,
                prompt_operation_target=None,
                prompt_operation_agent_session=None,
                prompt_baseline_sequence=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
            )
        return current.evolve(
            status=JobStatus.QUEUED,
            remove=frozenset({"interactive_questionnaire_blocked"}),
            queued_at=now,
            reconcile=True,
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
        )

    store.update(job.id, resume)


def recover_jobs(
    store: JobStore,
    *,
    launch_worker: LaunchWorker,
    herdr_factory: HerdrFactory = HerdrClient,
    github_factory: GitHubFactory = GitHubClient,
    is_worker_alive: Callable[[CursorJob], bool] = worker_is_alive,
    stop_owned_worker: Callable[[CursorJob], bool] = stop_worker,
    stop_unfenced_worker: Callable[[JobStore, str], bool] = stop_legacy_worker,
    get_boot_identity: Callable[[], str | None] = boot_identity,
    get_process_identity: Callable[[int], str | None] = process_identity,
    inspect_legacy_worker: LegacyWorkerInspector = inspect_and_stop_legacy_worker,
    require_issue_provider: RequireIssueProvider | None = None,
    outbox_handlers: Mapping[str, EffectHandler] | None = None,
    now: float | None = None,
) -> None:
    if store.maintenance_active() is True:
        return
    blocked_legacy_jobs = store.migrate_legacy(inspect_worker=inspect_legacy_worker)
    store.prune(now=now)
    recovered_at = time.time() if now is None else now
    if outbox_handlers is None:
        outbox_handlers = _recovery_agent_effect_handlers(herdr_factory, store)
    recover_outbox(store, handlers=outbox_handlers, now=recovered_at)
    from .provisioning import _reduce_agent_effect

    consume_agent_results(store, _reduce_agent_effect)

    for existing in store.list():
        if not has_legacy_worker_claim(existing):
            continue
        disposition = inspect_legacy_worker(existing)
        if disposition == "unsafe":
            blocked_legacy_jobs.add(existing.id)
            continue

        def clear_legacy_owner(job: CursorJob) -> CursorJob | None:
            if not has_legacy_worker_claim(job):
                return None
            requeue = job.terminal_intent_status is None and job.status in {
                JobStatus.QUEUED,
                JobStatus.ROUTING,
                JobStatus.RUNNING,
                JobStatus.RECONCILING,
            }
            return job.evolve(
                status=JobStatus.QUEUED if requeue else job.status,
                queued_at=(job.queued_at or recovered_at) if requeue else job.queued_at,
                reconcile=bool(
                    requeue and (job.herdr_target or job.has_uncertain_operation())
                ),
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
            )

        store.update(existing.id, clear_legacy_owner)

    for existing in store.list():
        if existing.id in blocked_legacy_jobs:
            continue
        reconcile_prompt_and_pane_operations(
            store,
            existing,
            now=recovered_at,
            herdr_factory=herdr_factory,
        )
        current = store.get(existing.id)
        if current.has_uncertain_operation():
            reconcile_uncertain_operations(
                store,
                current,
                now=recovered_at,
                herdr_factory=herdr_factory,
                github_factory=github_factory,
            )
            existing = store.get(existing.id)
        _reconcile_interactive_questionnaire(
            store,
            existing,
            now=recovered_at,
            herdr_factory=herdr_factory,
        )
        existing = store.get(existing.id)
        _reconcile_question_prompt(
            store,
            existing,
            now=recovered_at,
            herdr_factory=herdr_factory,
            is_worker_alive=is_worker_alive,
        )
    provider_errors: dict[str, str] = {}
    if require_issue_provider is not None:
        for existing in store.list():
            if (
                existing.status
                not in {
                    JobStatus.QUEUED,
                    JobStatus.ROUTING,
                    JobStatus.RUNNING,
                    JobStatus.RECONCILING,
                    JobStatus.BLOCKED,
                }
                or is_worker_alive(existing)
                or existing.has_uncertain_operation()
                or existing.manual_reconcile_operation
                or checkout_blocks_reservation(
                    None
                    if existing.checkout_operation is None
                    else existing.checkout_operation.state
                )
                or existing.pull_request_worktree_state == "quarantined"
            ):
                continue
            try:
                require_issue_provider(existing.issue_provider)
            except Exception as exc:  # noqa: BLE001 - persisted as a job failure
                detail = redact_diagnostic(
                    str(exc) or type(exc).__name__,
                    limit=360,
                )
                provider_errors[existing.id] = (
                    f"Selected issue provider {existing.issue_provider!r} is "
                    f"unavailable: {detail}"
                )[:500]
    for existing in store.list():
        if (
            existing.status
            in {JobStatus.ROUTING, JobStatus.RUNNING, JobStatus.RECONCILING}
            and not existing.worker_token
            and is_worker_alive(existing)
        ):
            stop_unfenced_worker(store, existing.id)
        elif (
            (
                existing.status == JobStatus.CANCELLED
                or existing.terminal_intent_status == JobStatus.CANCELLED
            )
            and existing.target_release_pending
            and existing.worker_token
            and is_worker_alive(existing)
        ):
            stop_owned_worker(existing)

    launches: list[str] = []
    releases: list[tuple[str, str, str]] = []
    for snapshot in store.list():
        if snapshot.id in blocked_legacy_jobs:
            continue
        should_launch = False
        release: tuple[str, str, str] | None = None

        def recover(job: CursorJob) -> CursorJob | None:
            nonlocal should_launch, release
            question = question_adapter.current(job)
            prompt_state = (
                question.prompt_state
                if question is not None and question.state == QuestionState.DISPATCHING
                else None
            )
            take_release = (
                job.target_release_pending
                and not _target_release_owner_alive(
                    job,
                    get_boot_identity=get_boot_identity,
                    get_process_identity=get_process_identity,
                )
                and not is_worker_alive(job)
                and not _infrastructure_uncertain_for_release(job)
            )
            release_token = job.target_release_token
            owner_pid = job.target_release_owner_pid
            owner_boot_id = job.target_release_owner_boot_id
            owner_start = job.target_release_owner_start
            claimed_cleanup: CleanupOwned | None = None
            if take_release:
                release_token = uuid.uuid4().hex
                owner_pid = os.getpid()
                owner_start = get_process_identity(owner_pid)
                owner_pid = owner_pid if owner_start else None
                owner_boot_id = get_boot_identity() if owner_start else None
                if owner_pid is None or owner_boot_id is None or owner_start is None:
                    take_release = False
                else:
                    try:
                        cleanup = job.cleanup_state
                        if isinstance(cleanup, CleanupOwned):
                            claimed_cleanup = take_over_cleanup(
                                cleanup,
                                job.target_release_token or "",
                                token=release_token,
                                owner_pid=owner_pid,
                                owner_boot_id=owner_boot_id,
                                owner_start=owner_start,
                            )
                        else:
                            claimed_cleanup = claim_cleanup(
                                cleanup,
                                job.target_release_token or "",
                                token=release_token,
                                owner_pid=owner_pid,
                                owner_boot_id=owner_boot_id,
                                owner_start=owner_start,
                            )
                    except LifecycleTransitionError:
                        take_release = False
                    else:
                        assert claimed_cleanup.token is not None
                        release_token = claimed_cleanup.token
                        release = (job.id, job.herdr_target or "", release_token)
            cleanup_updates = cleanup_fields(claimed_cleanup or job.cleanup_state)
            if job.terminal_intent_status is not None:
                if take_release and claimed_cleanup is not None:
                    return job.evolve(
                        **cleanup_fields(claimed_cleanup),
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
                if not is_worker_alive(job) and any(
                    value is not None
                    for value in (
                        job.worker_pid,
                        job.worker_boot_id,
                        job.worker_process_start,
                        job.worker_token,
                    )
                ):
                    return job.evolve(
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
                return None
            if job.session_control == SessionControlMode.USER_OWNED.value:
                if take_release:
                    return job.evolve(
                        **cleanup_updates,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
                return None
            if (
                job.has_uncertain_operation()
                or job.manual_reconcile_operation
                or checkout_blocks_reservation(
                    None
                    if job.checkout_operation is None
                    else job.checkout_operation.state
                )
                or job.pull_request_worktree_state == "quarantined"
            ):
                if take_release:
                    return job.evolve(
                        **cleanup_updates,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
                return None
            provider_error = provider_errors.get(job.id)
            if provider_error is not None:
                failed = stage_terminal_intent(
                    job,
                    JobStatus.FAILED,
                    now=recovered_at,
                    result=provider_error,
                    error=provider_error,
                    clear_worker=True,
                )
                release = (
                    failed.id,
                    failed.herdr_target or "",
                    failed.target_release_token or "",
                )
                return failed
            if job.status == JobStatus.BLOCKED:
                if (
                    job.delivered
                    and job.herdr_target
                    and recovered_at >= job.next_reconcile_at
                    and not job.interactive_questionnaire_blocked
                    and prompt_state != PromptOperationState.SUBMITTED
                ):
                    should_launch = True
                    return job.evolve(
                        status=JobStatus.QUEUED,
                        reconcile=True,
                        queued_at=recovered_at,
                        next_reconcile_at=recovered_at + DELIVERY_RETRY_SECONDS,
                        **cleanup_updates,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            elif job.status == JobStatus.QUEUED:
                if not job.worker_token or not is_worker_alive(job):
                    should_launch = True
                    return job.evolve(
                        queued_at=job.queued_at or recovered_at,
                        **cleanup_updates,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            elif job.status == JobStatus.ROUTING:
                if not is_worker_alive(job):
                    if prompt_state == PromptOperationState.SUBMITTED:
                        return None
                    should_launch = True
                    if prompt_state == PromptOperationState.OBSERVED:
                        return job.evolve(
                            status=JobStatus.QUEUED,
                            queued_at=recovered_at,
                            reconcile=True,
                            **cleanup_updates,
                            worker_pid=None,
                            worker_boot_id=None,
                            worker_process_start=None,
                            worker_token=None,
                        )
                    return job.evolve(
                        status=JobStatus.QUEUED,
                        remove=frozenset({"reconcile"}),
                        queued_at=recovered_at,
                        **cleanup_updates,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            elif job.status in {JobStatus.RUNNING, JobStatus.RECONCILING}:
                if not is_worker_alive(job):
                    if prompt_state == PromptOperationState.SUBMITTED:
                        return None
                    if job.herdr_target:
                        should_launch = True
                        if prompt_state == PromptOperationState.PLANNED:
                            return job.evolve(
                                status=JobStatus.QUEUED,
                                remove=frozenset({"reconcile"}),
                                queued_at=recovered_at,
                                **cleanup_updates,
                                worker_pid=None,
                                worker_boot_id=None,
                                worker_process_start=None,
                                worker_token=None,
                            )
                        return job.evolve(
                            status=JobStatus.QUEUED,
                            reconcile=True,
                            queued_at=recovered_at,
                            **cleanup_updates,
                            worker_pid=None,
                            worker_boot_id=None,
                            worker_process_start=None,
                            worker_token=None,
                        )
                    if job.prompt_operation_state in {"none", "planned"}:
                        should_launch = True
                        return job.evolve(
                            status=JobStatus.QUEUED,
                            remove=frozenset({"reconcile"}),
                            queued_at=recovered_at,
                            **cleanup_updates,
                            worker_pid=None,
                            worker_boot_id=None,
                            worker_process_start=None,
                            worker_token=None,
                        )
                    message = "Cursor job was interrupted before an agent started"
                    return job.evolve_recovery(
                        now=recovered_at,
                        status=JobStatus.FAILED,
                        remove=frozenset({"reconcile"}),
                        prepare_delivery=True,
                        error=message,
                        result=message,
                        completed_at=recovered_at,
                        **cleanup_updates,
                        worker_pid=None,
                        worker_boot_id=None,
                        worker_process_start=None,
                        worker_token=None,
                    )
            if take_release:
                return job.evolve(
                    **cleanup_updates,
                    worker_pid=None,
                    worker_boot_id=None,
                    worker_process_start=None,
                    worker_token=None,
                )
            return None

        updated = store.update(snapshot.id, recover)
        if updated is not None:
            if release is not None:
                releases.append(release)
            if should_launch:
                launches.append(updated.id)

    for job_id, target, token in releases:
        cancel_target_and_release(
            store,
            job_id,
            target,
            token,
            herdr_factory=herdr_factory,
        )
    for job_id in launches:
        try:
            launch_worker(job_id)
        except Exception:
            # The launch boundary persists a deliverable failure.
            continue
