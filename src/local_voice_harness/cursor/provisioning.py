from __future__ import annotations

import hashlib
import re
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, replace
from pathlib import Path

from ..diagnostic_safety import redact_diagnostic
from ..errors import HarnessError
from ..github_issue_creation import draft_github_issue
from ..integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubForkPlan,
    GitHubIssue,
    GitHubIssueCreationResult,
    GitHubIssueLookupError,
    GitHubOperationAmbiguous,
    GitHubProvider,
    GitHubPullRequestCheckoutInputs,
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
from ..integrations.linear import (
    LinearError,
    LinearIntegration,
    LinearOperationAmbiguous,
    LinearTicketCreationResult,
)
from ..integrations.registry import (
    IntegrationRegistry,
    build_integration_registry,
    issue_provider,
    prompt_instructions,
    require_issue_capabilities,
    require_issue_provider,
    resolve_issue_reference,
    route_issue_repository,
)
from ..linear_ticket_creation import draft_linear_ticket
from ..local_git import LocalGitRefChanged
from ..prompt_operations import (
    AmbiguousPrompt,
    PlannedPrompt,
    PromptIdentity,
    SubmittedPrompt,
    SubmittingPrompt,
    accept_prompt_submission,
    begin_prompt_submission,
    legacy_prompt_fields,
    mark_prompt_ambiguous,
    observe_prompt_submission,
)
from ..prompt_operations import (
    plan_prompt as transition_prompt_plan,
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
    default_user_config,
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
    PromptPayload,
    PromptSizeError,
    bounded_prompt_payload,
    classification_prompt,
    continuation_prompt,
    implementation_prompt,
    planning_prompt,
    review_prompt,
    revision_prompt,
)
from .store import JobStore
from .workflow import (
    ArtifactReference,
    ParticipantCreation,
    ParticipantCreationState,
    PlanApprovalProof,
    PlanApprovalSource,
    ReviewDecision,
    WorkflowTransitionError,
)

DELIVERY_RETRY_SECONDS = 5.0
FOREGROUND_GRACE_SECONDS = 2.0


WorkerCancelled = worker_lifecycle.WorkerCancelled


@dataclass(frozen=True, slots=True)
class ClientFactories:
    herdr: Callable[[], HerdrClient]
    github: Callable[[], GitHubClient | GitHubProvider]
    integrations: IntegrationRegistry | None = None


@dataclass(frozen=True, slots=True)
class _WorkflowPromptFactory:
    job: CursorJob
    phase: WorkflowPhase
    token: str
    integration_instructions: tuple[str, ...]
    plan: str | None = None
    review: str | None = None
    issue_reference: str | None = None

    def __call__(self, session_identity: str, full_rehydration: bool) -> PromptPayload:
        if not full_rehydration:
            clarification = self.job.continuation_answer
            if not clarification:
                raise HarnessError(
                    "same-session continuation has no clarification delta"
                )
            text = continuation_prompt(
                self.phase.value,
                clarification,
                self.token,
                issue_reference=self.issue_reference,
            )
            sections: dict[str, str | None] = {"clarification": clarification}
        else:
            text = self._full_prompt()
            issue_context = self._deduplicated_issue_context()
            sections = {
                "request": self.job.request,
                "issue_context": issue_context,
                "classification": self.job.workflow_classification_reason,
                "integration": " ".join(self.integration_instructions) or None,
                "plan": self.plan,
                "review": self.review,
                "clarification": self.job.continuation_answer,
            }
        return bounded_prompt_payload(
            text,
            phase=self.phase.value,
            session_identity=session_identity,
            full_rehydration=full_rehydration,
            sections=sections,
        )

    def _full_prompt(self) -> str:
        if self.phase == WorkflowPhase.CLASSIFYING:
            return self._with_current_clarification(
                classification_prompt(
                    self.job.request,
                    self.token,
                    github_issue_context=self._deduplicated_issue_context(),
                    integration_instructions=self.integration_instructions,
                )
            )
        if self.phase == WorkflowPhase.PLANNING:
            if self.job.workflow_tier is None:
                raise HarnessError("planning phase requires a workflow tier")
            return self._with_current_clarification(
                planning_prompt(
                    self.job.request,
                    self.token,
                    tier=self.job.workflow_tier.value,
                    github_issue_context=self._deduplicated_issue_context(),
                    classification_reason=self.job.workflow_classification_reason,
                    integration_instructions=self.integration_instructions,
                )
            )
        if self.phase == WorkflowPhase.REVIEWING:
            if self.plan is None or self.job.workflow_tier is None:
                raise HarnessError("review phase requires a durable plan and tier")
            return self._with_current_clarification(
                review_prompt(
                    self.job.request,
                    self.plan,
                    self.token,
                    tier=self.job.workflow_tier.value,
                    github_issue_context=self._deduplicated_issue_context(),
                    classification_reason=self.job.workflow_classification_reason,
                    integration_instructions=self.integration_instructions,
                )
            )
        if self.phase == WorkflowPhase.REVISING:
            if self.plan is None or self.review is None:
                raise HarnessError("revision phase requires plan and review artifacts")
            return self._with_current_clarification(
                revision_prompt(
                    self.job.request,
                    self.plan,
                    self.review,
                    self.token,
                    github_issue_context=self._deduplicated_issue_context(),
                    classification_reason=self.job.workflow_classification_reason,
                    integration_instructions=self.integration_instructions,
                )
            )
        if self.phase == WorkflowPhase.IMPLEMENTING:
            text = implementation_prompt(
                self.job.request,
                self.token,
                plan=self.plan,
                github_issue_context=self._deduplicated_issue_context(),
                classification_reason=self.job.workflow_classification_reason,
                issue_reference=self.issue_reference,
                integration_instructions=self.integration_instructions,
            )
            if self.review:
                text += f"\n\nApproved review findings:\n{self.review}"
            return self._with_current_clarification(text)
        raise HarnessError(f"unsupported workflow phase {self.phase.value}")

    def _with_current_clarification(self, text: str) -> str:
        if not self.job.continuation_answer:
            return text
        return f"{text}\n\nCurrent clarification:\n{self.job.continuation_answer}"

    def _deduplicated_issue_context(self) -> str | None:
        context = self.job.github_issue_context
        if not context:
            return None
        represented_by = (self.job.request, self.plan or "", self.review or "")
        return None if any(context in value for value in represented_by) else context


def _github_provider(
    factory: Callable[[], GitHubClient | GitHubProvider],
) -> GitHubProvider:
    integration = factory()
    return (
        integration
        if isinstance(integration, GitHubProvider)
        else GitHubProvider(integration)
    )


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
    job: CursorJob,
    *,
    output: str,
    agent_status: str,
    now: float | None = None,
    stage_terminal: bool = False,
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
            job=job,
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
        if stage_terminal or job.herdr_target is not None:
            return recovery.stage_terminal_intent(
                job,
                JobStatus.COMPLETED,
                now=completed_at,
                result=summary,
                voice_question=resolved_question,
            )
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
    sensitivity: QuestionSensitivity = QuestionSensitivity.ROUTINE,
) -> None:
    def ask(job: CursorJob) -> CursorJob:
        now = time.time()
        return question_adapter.ask(
            job,
            QuestionSpec(
                question,
                sensitivity=sensitivity,
            ),
            owner=clarification_kind,
            turn_token=job.turn_token or f"{job.id}-routing-{job.turn}",
            now=now,
            clear_worker=True,
            remove_reconcile=True,
            job_changes=job_changes,
        )

    _worker_change(store, job_id, token, {JobStatus.ROUTING, JobStatus.RUNNING}, ask)


def _finish_github_issue_creation(
    store: JobStore,
    job_id: str,
    token: str,
    result: GitHubIssueCreationResult,
) -> None:
    issue = result.issue
    url = str(result.url)

    def finish(job: CursorJob) -> CursorJob:
        now = time.time()
        return job.evolve_for_delivery(
            now=now,
            status=JobStatus.COMPLETED,
            issue_provider="github",
            result=f"Created GitHub issue {issue.reference}: {url}",
            completed_at=now,
            github_issue=issue.number,
            github_issue_url=url,
            github_issue_created_number=issue.number,
            github_issue_created_url=url,
            github_issue_create_operation_state="created",
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            worker_operation=None,
        )

    _worker_change(
        store,
        job_id,
        token,
        {JobStatus.ROUTING, JobStatus.RUNNING},
        finish,
    )


def _run_github_issue_creation(
    store: JobStore,
    job: CursorJob,
    token: str,
    clients: ClientFactories,
    checkpoint: Callable[[], None],
) -> None:
    repository = (job.github_repository or "").strip()
    if not repository:
        _worker_question(
            store,
            job.id,
            token,
            "Which GitHub repository should I create the issue in? "
            "Please say its owner and repository name.",
            clarification_kind="github_repository",
        )
        return
    github = _github_provider(clients.github)
    checkpoint()
    source = github.resolve_repository(repository)
    repository = source.name_with_owner
    checkpoint()
    if not job.github_issue_create_title or job.github_issue_create_body is None:
        config = default_user_config()
        draft = draft_github_issue(
            job.trusted_utterance or job.request,
            repository,
            settings=config.providers,
        )
        plan = github.plan_issue_creation(
            draft.repository,
            draft.title,
            draft.body,
            correlation_marker=uuid.uuid4().hex,
        )

        def persist_draft(current: CursorJob) -> CursorJob:
            return current.evolve(
                github_repository=plan.repository,
                github_issue_create_title=plan.title,
                github_issue_create_body=plan.body,
                github_issue_create_marker=plan.correlation_marker,
                github_issue_create_operation_state="planned",
            )

        updated = _worker_change(
            store, job.id, token, {JobStatus.ROUTING}, persist_draft
        )
        if updated is None:
            return
        job = updated
    plan = github.plan_issue_creation(
        job.github_repository or repository,
        job.github_issue_create_title or "",
        job.github_issue_create_body or "",
        correlation_marker=job.github_issue_create_marker,
    )
    if not job.github_issue_create_confirmed:
        preview = (
            f"Create this GitHub issue in {plan.repository}?\n\n"
            f"Title: {plan.title}\n\nBody:\n{plan.body}\n\n"
            "Say yes to create it or no to cancel."
        )
        _worker_question(
            store,
            job.id,
            token,
            preview,
            clarification_kind="github_issue_create_confirmation",
            sensitivity=QuestionSensitivity.DESTRUCTIVE,
        )
        return
    if job.github_issue_create_operation_state not in {None, "planned"}:
        raise HarnessError("GitHub issue creation requires reconciliation before retry")

    def mark_submitted(current: CursorJob) -> CursorJob:
        return current.evolve(
            status=JobStatus.RUNNING,
            github_issue_create_operation_state="submitted",
            worker_operation="github_issue_create",
        )

    submitted = _worker_change(
        store, job.id, token, {JobStatus.ROUTING}, mark_submitted
    )
    if submitted is None:
        return
    try:
        checkpoint()
        result = github.submit_issue_creation(plan, confirmed=True)
    except GitHubError as exc:
        checkpoint()
        observation_failed = False
        try:
            visible = github.observe_issue_creation(plan)
        except GitHubError:
            visible = None
            observation_failed = True
        if visible is not None:
            _finish_github_issue_creation(store, job.id, token, visible)
            return
        if isinstance(exc, GitHubOperationAmbiguous) or observation_failed:

            def ambiguous(current: CursorJob) -> CursorJob:
                return current.evolve(
                    status=JobStatus.QUEUED,
                    queued_at=time.time(),
                    github_issue_create_operation_state="ambiguous",
                    reconcile=True,
                    worker_pid=None,
                    worker_boot_id=None,
                    worker_process_start=None,
                    worker_token=None,
                    worker_operation=None,
                )

            _worker_change(store, job.id, token, {JobStatus.RUNNING}, ambiguous)
            return

        def failed_before_creation(current: CursorJob) -> CursorJob:
            return current.evolve(
                github_issue_create_operation_state="planned",
                worker_operation=None,
            )

        _worker_change(
            store,
            job.id,
            token,
            {JobStatus.RUNNING},
            failed_before_creation,
        )
        raise
    _finish_github_issue_creation(store, job.id, token, result)


def _finish_linear_ticket_creation(
    store: JobStore,
    job_id: str,
    token: str,
    result: LinearTicketCreationResult,
) -> None:
    identifier = result.issue.identifier
    url = result.url

    def finish(job: CursorJob) -> CursorJob:
        now = time.time()
        return job.evolve_for_delivery(
            now=now,
            status=JobStatus.COMPLETED,
            result=f"Created Linear ticket {identifier}: {url}",
            completed_at=now,
            linear_ticket_created_identifier=identifier,
            linear_ticket_created_url=url,
            linear_ticket_create_operation_state="created",
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            worker_operation=None,
        )

    _worker_change(
        store,
        job_id,
        token,
        {JobStatus.ROUTING, JobStatus.RUNNING},
        finish,
    )


def _run_linear_ticket_creation(
    store: JobStore,
    job: CursorJob,
    token: str,
    clients: ClientFactories,
    checkpoint: Callable[[], None],
) -> None:
    team = (job.linear_ticket_create_team or "").strip()
    if not team:
        _worker_question(
            store,
            job.id,
            token,
            "Which Linear team should I create the ticket in? Please say its team key.",
            clarification_kind="linear_team",
        )
        return
    registry = clients.integrations or build_integration_registry(default_user_config())
    provider = issue_provider("linear", registry)
    if not isinstance(provider, LinearIntegration):
        raise HarnessError("selected Linear provider cannot create tickets")
    team = provider.validate_team(team)
    client = clients.herdr()
    checkpoint()
    if not job.linear_ticket_create_team_id:
        try:
            resolved_team = provider.resolve_team(
                client,
                team,
                checkpoint=checkpoint,
            )
        except LinearError:
            message = f"I couldn't find Linear team {team}."

            def finish_missing_team(current: CursorJob) -> CursorJob:
                now = time.time()
                return recovery.stage_terminal_intent(
                    current,
                    JobStatus.FAILED,
                    now=now,
                    result=message,
                    error=message,
                )

            _worker_change(
                store,
                job.id,
                token,
                MODEL_WORKER_STATUSES,
                finish_missing_team,
            )
            return

        def persist_team(current: CursorJob) -> CursorJob:
            return current.evolve(
                linear_ticket_create_team=resolved_team.key,
                linear_ticket_create_team_id=resolved_team.id,
            )

        updated = _worker_change(
            store,
            job.id,
            token,
            {JobStatus.ROUTING},
            persist_team,
        )
        if updated is None:
            return
        job = updated
        team = resolved_team.key
    if (
        not job.linear_ticket_create_title
        or job.linear_ticket_create_description is None
    ):
        config = default_user_config()
        draft = draft_linear_ticket(
            job.trusted_utterance or job.request,
            team,
            settings=config.providers,
        )
        plan = provider.plan_ticket_creation(
            job.linear_ticket_create_team_id or "",
            draft.team,
            draft.title,
            draft.description,
            correlation_marker=uuid.uuid4().hex,
        )

        def persist_draft(current: CursorJob) -> CursorJob:
            return current.evolve(
                linear_ticket_create_team=plan.team_key,
                linear_ticket_create_team_id=plan.team_id,
                linear_ticket_create_title=plan.title,
                linear_ticket_create_description=plan.description,
                linear_ticket_create_marker=plan.correlation_marker,
                linear_ticket_create_operation_state="planned",
            )

        updated = _worker_change(
            store, job.id, token, {JobStatus.ROUTING}, persist_draft
        )
        if updated is None:
            return
        job = updated
    plan = provider.plan_ticket_creation(
        job.linear_ticket_create_team_id or "",
        job.linear_ticket_create_team or team,
        job.linear_ticket_create_title or "",
        job.linear_ticket_create_description or "",
        correlation_marker=job.linear_ticket_create_marker,
    )
    if not job.linear_ticket_create_confirmed:
        preview = (
            f"Create this Linear ticket in team {plan.team_key}?\n\n"
            f"Title: {plan.title}\n\nDescription:\n{plan.description}\n\n"
            "Say yes to create it or no to cancel."
        )
        _worker_question(
            store,
            job.id,
            token,
            preview,
            clarification_kind="linear_ticket_create_confirmation",
            sensitivity=QuestionSensitivity.DESTRUCTIVE,
        )
        return
    if job.linear_ticket_create_operation_state not in {None, "planned"}:
        raise HarnessError(
            "Linear ticket creation requires reconciliation before retry"
        )

    def mark_running(current: CursorJob) -> CursorJob:
        return current.evolve(
            status=JobStatus.RUNNING,
            worker_operation="linear_ticket_create",
        )

    running = _worker_change(store, job.id, token, {JobStatus.ROUTING}, mark_running)
    if running is None:
        return

    def persist_submit_fence(
        target: str,
        session: str,
        prompt_token: str,
        baseline: int,
    ) -> None:
        def fence(current: CursorJob) -> CursorJob:
            return current.evolve(
                linear_ticket_create_operation_state="submitting",
                linear_ticket_create_prompt_target=target,
                linear_ticket_create_prompt_session=session,
                linear_ticket_create_prompt_token=prompt_token,
                linear_ticket_create_baseline_sequence=baseline,
            )

        if (
            _worker_change(
                store,
                job.id,
                token,
                {JobStatus.RUNNING},
                fence,
            )
            is None
        ):
            raise WorkerCancelled

    def persist_prompt_acceptance() -> None:
        def accepted(current: CursorJob) -> CursorJob:
            if current.linear_ticket_create_operation_state != "submitting":
                raise JobValidationError(
                    "Linear ticket prompt acceptance requires a submit fence"
                )
            return current.evolve(
                linear_ticket_create_operation_state="submitted",
            )

        if (
            _worker_change(
                store,
                job.id,
                token,
                {JobStatus.RUNNING},
                accepted,
            )
            is None
        ):
            raise WorkerCancelled

    def queue_ambiguous() -> None:
        def ambiguous(current: CursorJob) -> CursorJob:
            return current.evolve(
                status=JobStatus.QUEUED,
                queued_at=time.time(),
                linear_ticket_create_operation_state="ambiguous",
                reconcile=True,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                worker_token=None,
                worker_operation=None,
            )

        _worker_change(store, job.id, token, {JobStatus.RUNNING}, ambiguous)

    try:
        checkpoint()
        provider.submit_ticket_creation(
            client,
            plan,
            confirmed=True,
            checkpoint=checkpoint,
            before_submit=persist_submit_fence,
            accepted=persist_prompt_acceptance,
        )
    except LinearError as exc:
        checkpoint()
        current = store.get(job.id)
        uncertain = isinstance(
            exc, LinearOperationAmbiguous
        ) or current.linear_ticket_create_operation_state in {"submitting", "submitted"}
        if uncertain:
            try:
                visible = provider.observe_ticket_creation(
                    client,
                    plan,
                    checkpoint=checkpoint,
                )
            except LinearError:
                visible = None
            if visible is not None:
                _finish_linear_ticket_creation(store, job.id, token, visible)
                return
            queue_ambiguous()
            return

        def failed_before_submit(current: CursorJob) -> CursorJob:
            return current.evolve(
                worker_operation=None,
            )

        _worker_change(
            store,
            job.id,
            token,
            {JobStatus.RUNNING},
            failed_before_submit,
        )
        raise
    current = store.get(job.id)
    if current.linear_ticket_create_operation_state != "submitted":
        queue_ambiguous()
        return
    try:
        visible = provider.observe_ticket_creation(
            client,
            plan,
            checkpoint=checkpoint,
        )
    except LinearError:
        visible = None
    if visible is None:
        queue_ambiguous()
        return
    _finish_linear_ticket_creation(store, job.id, token, visible)


def _completion_preferences(
    job: CursorJob,
) -> tuple[PlanApprovalPreferences | None, bool]:
    approval = job.plan_approval
    approval_id = approval.proof.gate_id if approval.proof is not None else None
    if approval.source != PlanApprovalSource.EXPLICIT or approval_id is None:
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
    approval = job.plan_approval
    if counted and not approval.counted:
        approval = approval.count()
    completion_changes: Mapping[str, object] = {
        "result": result,
        "workflow_phase": WorkflowPhase.FINISHED.value,
        "active_participant": None,
        "prompt_operation_state": "none",
        "plan_approval_counted": approval.counted,
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
            job,
            output=output,
            agent_status=agent_status,
            now=now,
            stage_terminal=True,
        )
        outcome_status = outcome.terminal_intent_status or outcome.status
        preserve = preserve_blocked_delivery and outcome.status == JobStatus.BLOCKED
        workflow_finished = (
            outcome_status == JobStatus.COMPLETED and job.workflow_tier is not None
        )
        if outcome_status == JobStatus.COMPLETED:
            outcome_result = outcome.terminal_intent_result or outcome.result or ""
            if preference_update_failed and job.plan_approval_source == "explicit":
                return job.evolve(
                    status=JobStatus.QUEUED,
                    reconcile=True,
                    queued_at=now,
                    result=outcome_result or "Cursor implementation completed.",
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
                result=outcome_result,
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
    diagnostic = redact_diagnostic(str(exc) or type(exc).__name__, limit=500)
    result = (
        exc.voice_message
        if isinstance(exc, GitHubIssueLookupError)
        else "Cursor job failed. Check the job log for diagnostic details."
    )

    def fail(job: CursorJob) -> CursorJob:
        now = time.time()
        return recovery.stage_terminal_intent(
            job,
            JobStatus.FAILED,
            now=now,
            result=result,
            error=diagnostic,
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
    diagnostic = redact_diagnostic(message, limit=500)

    def block(job: CursorJob) -> CursorJob:
        return job.evolve_for_delivery(
            now=blocked_at,
            status=JobStatus.BLOCKED,
            result="Cursor needs manual attention in Herdr.",
            error=diagnostic,
            completed_at=blocked_at,
        )

    _worker_change(store, job_id, token, MODEL_WORKER_STATUSES, block)


def _pull_request_branch(job: CursorJob) -> str:
    configured = job.worktree_branch or ""
    branch = configured or f"voice/github-pr-{job.id}"
    if not re.fullmatch(r"voice/[a-z0-9][a-z0-9._/-]{0,100}", branch):
        raise HarnessError("invalid voice pull-request branch")
    return branch


def _prepare_pull_request_checkout(
    store: JobStore,
    job_id: str,
    token: str,
    job: CursorJob,
    checkpoint: Callable[[], None] | None = None,
    *,
    github_factory: Callable[[], GitHubClient | GitHubProvider] | None = None,
) -> CursorJob | None:
    if not job.github_pull_request:
        return job
    if job.pull_request_worktree_state in {"ready", "retained"}:
        return job
    repository = Path(job.repository or "").resolve()
    checkout_value = job.worktree_path or ""
    checkout = Path(checkout_value).resolve() if checkout_value else repository
    number = job.github_pull_request
    try:
        branch = _pull_request_branch(job)
        if checkout == repository:
            raise HarnessError(
                "refusing to check out a pull request in the shared repository clone"
            )
        if not checkout.is_dir() or not (checkout / ".git").exists():
            raise HarnessError("pull-request worktree is missing or invalid")
        if checkpoint is not None:
            checkpoint()
        github = _github_provider(github_factory or GitHubClient)
        if not job.github_repository:
            raise HarnessError("pull-request repository metadata is missing")
        repository_name = job.github_repository
        checked_out_branch: str | None = None
        for attempt in range(2):
            plan = github.plan_pull_request(repository_name, number)
            source_name = plan.source.name_with_owner
            remote_url = plan.checkout.remote_url
            head_ref = plan.checkout.head_ref
            head_oid = plan.checkout.head_oid

            def refresh_checkout_inputs(
                current: CursorJob,
                source_name: str = source_name,
                remote_url: str = remote_url,
                head_ref: str = head_ref,
                head_oid: str = head_oid,
            ) -> CursorJob:
                return current.evolve(
                    github_repository=source_name,
                    pull_request_remote_url=remote_url,
                    pull_request_head_ref=head_ref,
                    pull_request_head_oid=head_oid,
                )

            refreshed = _worker_change(
                store,
                job_id,
                token,
                {JobStatus.ROUTING},
                refresh_checkout_inputs,
            )
            if refreshed is None:
                return None
            job = refreshed
            if checkpoint is not None:
                checkpoint()
            inputs = GitHubPullRequestCheckoutInputs(
                remote_url=job.pull_request_remote_url or "",
                head_ref=job.pull_request_head_ref or "",
                head_oid=job.pull_request_head_oid or "",
            )
            github.validate_pull_request_checkout_inputs(
                job.github_repository or "",
                number,
                inputs,
            )
            try:
                checked_out_branch = github.local_git.checkout_remote_ref(
                    checkout,
                    remote_url=inputs.remote_url,
                    remote_ref=inputs.head_ref,
                    branch=branch,
                    expected_oid=inputs.head_oid,
                    checkpoint=checkpoint,
                )
            except LocalGitRefChanged:
                if attempt == 0:
                    if checkpoint is not None:
                        checkpoint()
                    continue
                raise
            break
        if checkpoint is not None:
            checkpoint()
    except Exception as exc:
        message = redact_diagnostic(str(exc) or type(exc).__name__, limit=500)

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


def _persisted_fork_plan(job: CursorJob) -> GitHubForkPlan | None:
    if job.fork_operation_state is None:
        return None
    source_name = job.fork_operation_source or ""
    source_url = job.fork_operation_source_url or ""
    target = job.fork_operation_target or ""
    if not all((source_name, source_url, target)):
        raise HarnessError("persisted GitHub fork operation is incomplete")
    login = job.fork_operation_login or target.split("/", 1)[0]
    if (
        job.github_repository
        and job.github_repository.casefold() != source_name.casefold()
    ):
        raise HarnessError("persisted GitHub fork source does not match the job")
    return GitHubForkPlan(
        source=GitHubRepository(
            name_with_owner=source_name,
            url=source_url,
            is_private=job.fork_operation_source_private,
            default_branch=job.fork_operation_source_default_branch or "",
            parent=job.fork_operation_source_parent,
        ),
        login=login,
        target=target,
    )


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
_NEGATED_RISK = re.compile(
    r"\b(?:no|without)\s+(?:known\s+)?"
    r"(?:[a-z-]+\s*(?:,\s*|\s+(?:and|or)\s+|\s+)){0,12}"
    r"(?:concerns?|risks?|issues?)\b"
    r"|\b(?:does|did)\s+not\s+(?:raise|introduce|create)\s+"
    r"(?:[a-z-]+\s+){0,6}(?:concerns?|risks?|issues?)\b"
    r"|\b(?:[a-z-]+\s+){1,4}(?:is|are)\s+not\s+"
    r"(?:a\s+)?(?:concern|risk|issue)\b"
)
_NEGATION_WORD = re.compile(r"\b(?:no|without|not)\b")
_NEGATION_CONTRAST = re.compile(r"\b(?:but|however|yet)\b")
_NEGATED_RISK_SUFFIX = re.compile(
    r"^s?\s+(?:is|are|was|were)\s+not\s+"
    r"(?:required|involved|needed|modified|changed)\b"
)


def _bounded_risk_text(value: str | None) -> str:
    if not value:
        return ""
    return value.encode()[:_MAX_RISK_EVIDENCE_BYTES].decode(errors="ignore").casefold()


def _risk_occurrence_is_negated(evidence: str, start: int, end: int) -> bool:
    suffix = evidence[end : end + 64]
    if _NEGATED_RISK_SUFFIX.search(suffix) is not None:
        return True
    clause_start = max(
        evidence.rfind(separator, max(0, start - 160), start) for separator in ".;:!?\n"
    )
    prefix = evidence[clause_start + 1 : start]
    matches = list(_NEGATION_WORD.finditer(prefix))
    if not matches:
        return False
    negation = matches[-1]
    tail = prefix[negation.end() :]
    if _NEGATION_CONTRAST.search(tail) is not None:
        return False
    words = re.findall(r"[a-z-]+", tail)
    if len(words) > 8:
        return False
    if negation.group(0) == "not" and words[:1] == ["only"]:
        return False
    return True


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
        evidence = _NEGATED_RISK.sub("", evidence)
        for term in _HARD_RISK_TERMS:
            occurrences = tuple(re.finditer(re.escape(term), evidence))
            if any(
                not _risk_occurrence_is_negated(
                    evidence,
                    occurrence.start(),
                    occurrence.end(),
                )
                for occurrence in occurrences
            ):
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
            prompt_operation_agent_session=None,
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
    if agent_session is None or state_change_sequence is None or revision is None:
        raise HarnessError(
            "Cursor Plan Mode boundary is missing durable Herdr session/revision proof"
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
        approval = current.plan_approval.record_boundary(
            PlanApprovalProof(
                gate_id,
                agent_session,
                state_change_sequence,
                revision,
            )
        )
        return current.evolve_plan_approval(approval)

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
        return _worker_change(
            store,
            job.id,
            worker_token,
            MODEL_WORKER_STATUSES,
            lambda current: current.evolve_workflow(
                current.workflow_state.classify(
                    tier,
                    classification_reason[:500],
                ),
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
            change=lambda current, reference: current.evolve_review(
                current.review_state.publish_plan(
                    ArtifactReference.parse(
                        reference,
                        job_id=current.id,
                        kind="plan",
                    )
                ),
                workflow_phase=current.workflow_state.transition(
                    WorkflowPhase.REVIEWING
                ).phase.value,
                prompt_operation_state="none",
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
            plan = store.read_artifact(job.id, job.plan_artifact, kind="plan")
            auto_approval = _auto_plan_approval_allowed(
                job,
                plan=plan,
                review=review,
                reviewer_approved=True,
            )
            now = time.time()

            def approve_review(current: CursorJob, reference: str) -> CursorJob:
                approved_review = current.review_state.publish_review(
                    ArtifactReference.parse(
                        reference,
                        job_id=current.id,
                        kind="review",
                    ),
                    ReviewDecision.APPROVE,
                )
                if auto_approval:
                    approval = current.plan_approval.approve(
                        PlanApprovalSource.AUTO,
                        plan_reference=current.plan_artifact or "",
                        review_reference=reference,
                        review_accepted=approved_review.approved,
                    )
                    return current.evolve_review(
                        approved_review,
                        workflow_phase=current.workflow_state.transition(
                            WorkflowPhase.IMPLEMENTING
                        ).phase.value,
                        prompt_operation_state="none",
                        plan_approval_state=approval.state.value,
                        plan_approval_source=PlanApprovalSource.AUTO.value,
                        plan_approval_plan_artifact=approval.plan_reference,
                        plan_approval_review_artifact=approval.review_reference,
                        plan_approval_counted=approval.counted,
                    )
                return question_adapter.ask(
                    current,
                    QuestionSpec(
                        "The reviewed implementation plan is ready. Should I "
                        "approve it and start implementation in a fresh Agent-mode "
                        "session? "
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
                change=lambda current, reference: current.evolve_review(
                    current.review_state.publish_review(
                        ArtifactReference.parse(
                            reference,
                            job_id=current.id,
                            kind="review",
                        ),
                        ReviewDecision.REVISE,
                    )
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
        if job.review_round >= 2:
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
                change=lambda current, reference: current.evolve_review(
                    current.review_state.publish_review(
                        ArtifactReference.parse(
                            reference,
                            job_id=current.id,
                            kind="review",
                        ),
                        ReviewDecision.REVISE,
                    )
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
            change=lambda current, reference: current.evolve_review(
                current.review_state.publish_review(
                    ArtifactReference.parse(
                        reference,
                        job_id=current.id,
                        kind="review",
                    ),
                    ReviewDecision.REVISE,
                ).revise(),
                workflow_phase=current.workflow_state.transition(
                    WorkflowPhase.REVISING
                ).phase.value,
                prompt_operation_state="none",
                plan_approval_state="none",
                plan_approval_id=None,
                plan_approval_source=None,
                plan_approval_agent_session=None,
                plan_approval_plan_artifact=None,
                plan_approval_review_artifact=None,
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
            try:
                promoted_workflow = job.workflow_state.promote(
                    promoted,
                    promotion_reason,
                )
            except WorkflowTransitionError:
                _workflow_block(
                    store,
                    job.id,
                    worker_token,
                    "Cursor requested a workflow promotion that did not increase risk.",
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
            promoted_workflow = job.workflow_state.promote(
                promoted,
                classification_reason[:500],
            )
            return _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                lambda current: current.evolve_workflow(
                    promoted_workflow,
                    continuation=False,
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
                    plan_approval_plan_artifact=None,
                    plan_approval_review_artifact=None,
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
        lambda current: current.evolve_participant(
            replace(
                current.participant_lifecycle,
                creation=ParticipantCreation(
                    ParticipantCreationState.PLANNED,
                    participant,
                    target,
                    label,
                    workspace_id,
                ),
            )
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
            lambda current: current.evolve_participant(
                replace(
                    current.participant_lifecycle,
                    creation=current.participant_lifecycle.creation.begin(),
                )
            ),
        )
        if updated is None:
            raise WorkerCancelled

    def accepted(pane_id: str, workspace_id: str) -> None:
        def accept(current: CursorJob) -> CursorJob:
            if current.participant_creation_target != target:
                raise WorkerCancelled
            return current.evolve_participant(
                replace(
                    current.participant_lifecycle,
                    creation=current.participant_lifecycle.creation.created(
                        pane_id=pane_id,
                        workspace_id=workspace_id,
                    ),
                ),
                herdr_pane_id=pane_id,
                herdr_workspace_id=workspace_id,
                agent_name=target,
            )

        updated = _worker_change(
            store,
            job_id,
            worker_token,
            MODEL_WORKER_STATUSES,
            accept,
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
        uncertain = current.participant_creation_state == "submitting" or exc.code in {
            "operation_timeout",
            "operation_ambiguous",
        }
        if uncertain:
            return current.evolve_participant(
                replace(
                    current.participant_lifecycle,
                    creation=current.participant_lifecycle.creation.require_manual(),
                ),
                manual_reconcile_operation="pane",
                manual_reconcile_token=uuid.uuid4().hex,
                manual_reconcile_required_at=time.time(),
                worker_operation=None,
            )
        return current.evolve_participant(
            replace(
                current.participant_lifecycle,
                creation=ParticipantCreation(),
            ),
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

    checkout_value = job.worktree_path or job.repository
    workspace = job.worktree_workspace_id or job.herdr_workspace_id or ""
    if not checkout_value or not workspace:
        raise HarnessError(
            "workflow participant requires a prepared checkout and workspace"
        )
    checkout = Path(checkout_value).resolve()

    owned_targets = list(
        dict.fromkeys(
            candidate
            for candidate in (
                job.herdr_target,
                job.participant_target(WorkflowParticipant.PLANNER),
                job.participant_target(WorkflowParticipant.REVIEWER),
                job.participant_target(WorkflowParticipant.IMPLEMENTER),
            )
            if candidate
        )
    )
    for owned_target in owned_targets:
        pane_id = ""
        workspace_id = ""
        if owned_target == job.herdr_target:
            pane_id = job.herdr_pane_id or ""
            workspace_id = job.herdr_workspace_id or ""
        if not pane_id or not workspace_id:
            try:
                observed = client.get_agent(owned_target)
            except HerdrError as exc:
                if _agent_not_found(exc):
                    continue
                raise
            observed_cwd = str(observed.get("cwd") or "")
            observed_pane = str(observed.get("pane_id") or "")
            observed_workspace = str(observed.get("workspace_id") or "")
            try:
                cwd_matches = Path(observed_cwd).resolve() == checkout
            except OSError:
                cwd_matches = False
            if (
                not cwd_matches
                or observed_workspace != workspace
                or not observed_pane
                or observed_pane == job.worktree_root_pane_id
            ):
                raise HarnessError(
                    f"could not verify ownership of previous {owned_target} pane"
                )
            pane_id = observed_pane
            workspace_id = observed_workspace
        if pane_id == job.worktree_root_pane_id:
            raise HarnessError("refusing to close the retained workspace anchor")
        try:
            client.close_owned_pane(owned_target, pane_id, workspace_id)
        except HerdrError as exc:
            raise HarnessError("could not close previous workflow participant") from exc

    def clear_closed_participants(current: CursorJob) -> CursorJob:
        if (
            current.herdr_target != job.herdr_target
            or current.herdr_pane_id != job.herdr_pane_id
            or current.herdr_workspace_id != job.herdr_workspace_id
        ):
            raise WorkerCancelled
        return current.evolve(
            herdr_target=None,
            herdr_pane_id=None,
            herdr_workspace_id=None,
            agent_name=None,
            agent_dispatch_state="confirmed_absent",
            active_participant=None,
            planner_target=None,
            reviewer_target=None,
            implementer_target=None,
        )

    cleared = _worker_change(
        store,
        job.id,
        worker_token,
        MODEL_WORKER_STATUSES,
        clear_closed_participants,
    )
    if cleared is None:
        raise WorkerCancelled
    job = cleared
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
    token: str,
    prompt: str | None = None,
    prompt_factory: Callable[[str, bool], PromptPayload] | None = None,
) -> tuple[str, str] | None:
    phase = job.workflow_phase
    state = job.prompt_operation_state
    payload: PromptPayload | None = None
    full_rehydration = True
    build_prompt = prompt_factory
    if prompt_factory is None:
        if prompt is None:
            raise HarnessError("Cursor prompt content is missing")

        def legacy_factory(
            session_identity: str, full_rehydration: bool
        ) -> PromptPayload:
            assert prompt is not None
            return bounded_prompt_payload(
                prompt,
                phase=phase.value,
                session_identity=session_identity,
                full_rehydration=full_rehydration,
                sections={},
            )

        build_prompt = legacy_factory
    assert build_prompt is not None

    if state == "none":
        checkpoint()
        baseline_agent = client.get_agent(target)
        checkpoint()
        raw_baseline = baseline_agent.get("state_change_seq")
        try:
            baseline = int(raw_baseline) if isinstance(raw_baseline, int | str) else 0
        except (TypeError, ValueError) as exc:
            raise HarnessError("Herdr returned an invalid prompt sequence") from exc
        baseline_session = agent_session_identity(baseline_agent.get("agent_session"))
        if baseline_session is None:
            raise HarnessError("Herdr returned no agent session for prompt fencing")
        participant = job.active_participant
        prior_session = (
            job.prompt_context_sessions.get(participant.value)
            if participant is not None
            else None
        )
        full_rehydration = not (
            job.continuation
            and bool(job.continuation_answer)
            and prior_session == baseline_session
        )
        payload = build_prompt(baseline_session, full_rehydration)
        planned_manifest = payload.manifest

        def plan_prompt(current: CursorJob) -> CursorJob:
            if (
                current.workflow_phase != phase
                or current.turn_token != token
                or current.herdr_target != target
            ):
                raise WorkerCancelled
            operation = transition_prompt_plan(
                current.prompt_operation,
                PromptIdentity(
                    job_id=current.id,
                    phase=phase.value,
                    turn=current.turn,
                    turn_token=current.turn_token or "",
                    target=target,
                    agent_session=baseline_session,
                    baseline_sequence=baseline,
                ),
            )
            return current.evolve(
                **legacy_prompt_fields(operation),
                prompt_manifest=planned_manifest,
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
        observed_session = agent_session_identity(observed.get("agent_session"))
        operation = job.prompt_operation
        if not isinstance(operation, SubmittingPrompt):
            raise HarnessError(f"invalid durable prompt state {operation.state.value}")
        identity = operation.identity
        observed_operation = observe_prompt_submission(
            operation,
            identity,
            target=target,
            agent_session=observed_session,
            state_sequence=sequence,
        )
        if isinstance(observed_operation, SubmittedPrompt):
            assert observed_session is not None
            counted = _accepted_explicit_plan_approval(job)

            def mark_observed_submission(current: CursorJob) -> CursorJob:
                submitted_operation = observe_prompt_submission(
                    current.prompt_operation,
                    identity,
                    target=target,
                    agent_session=observed_session,
                    state_sequence=sequence,
                )
                sessions = current.prompt_context_sessions
                if current.active_participant is not None:
                    sessions[current.active_participant.value] = observed_session
                approval = current.plan_approval
                if counted and not approval.counted:
                    approval = approval.count()
                return current.evolve_plan_approval(
                    approval,
                    **legacy_prompt_fields(submitted_operation),
                    prompt_context_sessions=sessions,
                )

            submitted = _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                mark_observed_submission,
            )
            if submitted is None:
                return None
            job = submitted
            state = "submitted"
        else:
            assert isinstance(observed_operation, AmbiguousPrompt)
            _worker_change(
                store,
                job.id,
                worker_token,
                MODEL_WORKER_STATUSES,
                lambda current: current.evolve(
                    **legacy_prompt_fields(
                        observe_prompt_submission(
                            current.prompt_operation,
                            identity,
                            target=target,
                            agent_session=observed_session,
                            state_sequence=sequence,
                        )
                    ),
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
                lambda current: current.evolve_plan_approval(
                    current.plan_approval.count()
                ),
            )
            if counted is None:
                return None
            job = counted
        try:
            outcome = client.wait_for_stable_completion(
                target,
                token=job.turn_token or "",
                checkpoint=checkpoint,
                expected_agent_session=job.prompt_operation_agent_session,
                active_marker=(
                    "WORKFLOW_PLAN"
                    if phase in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING}
                    else None
                ),
                allow_interactive_plan_boundary=phase
                in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING},
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
                    lambda current: current.evolve_plan_approval(
                        current.plan_approval.observe()
                    ),
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
    planned_operation = job.prompt_operation
    if not isinstance(planned_operation, PlannedPrompt):
        raise HarnessError(
            f"invalid durable prompt state {planned_operation.state.value}"
        )
    prompt_identity = planned_operation.identity
    if payload is None:
        session_identity = job.prompt_operation_agent_session
        manifest = job.prompt_manifest
        if session_identity is None:
            raise HarnessError("planned Cursor prompt has no verified session identity")
        full_rehydration = not (
            manifest is not None
            and manifest.get("phase") == phase.value
            and manifest.get("session_identity") == session_identity
            and isinstance(manifest.get("full_rehydration"), bool)
        ) or bool(manifest and manifest.get("full_rehydration"))
        payload = build_prompt(session_identity, full_rehydration)
    prompt = payload.text

    def before_submit(observed_baseline: int) -> None:
        if observed_baseline != operation_baseline:
            raise HarnessError("Cursor prompt baseline changed before submission")

        def mark_submitting(current: CursorJob) -> CursorJob:
            operation = begin_prompt_submission(
                current.prompt_operation,
                prompt_identity,
            )
            return current.evolve(**legacy_prompt_fields(operation))

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
            operation = accept_prompt_submission(
                current.prompt_operation,
                prompt_identity,
            )
            sessions = current.prompt_context_sessions
            if (
                current.active_participant is not None
                and current.prompt_operation_agent_session is not None
            ):
                sessions[current.active_participant.value] = (
                    current.prompt_operation_agent_session
                )
            approval = current.plan_approval
            if counted and not approval.counted:
                approval = approval.count()
            return current.evolve_plan_approval(
                approval,
                **legacy_prompt_fields(operation),
                prompt_context_sessions=sessions,
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
            expected_agent_session=job.prompt_operation_agent_session,
            active_marker=(
                "WORKFLOW_PLAN"
                if phase in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING}
                else None
            ),
            allow_interactive_plan_boundary=phase
            in {WorkflowPhase.PLANNING, WorkflowPhase.REVISING},
            clarification_reply=bool(
                job.continuation and job.continuation_answer and not full_rehydration
            ),
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
                    **legacy_prompt_fields(
                        mark_prompt_ambiguous(
                            current.prompt_operation,
                            prompt_identity,
                        )
                    ),
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
            lambda current: current.evolve_plan_approval(
                current.plan_approval.observe()
            ),
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
    integrations: IntegrationRegistry | None = None,
) -> None:
    registry = integrations or build_integration_registry(default_user_config())
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
                WorkflowParticipant.IMPLEMENTER
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
        active_issue_key = resolve_issue_reference(
            job.issue_key,
            registry,
            provider=job.issue_provider,
        )
        integration_instructions = prompt_instructions(
            active_issue_key,
            registry,
            provider=job.issue_provider,
        )
        plan = (
            store.read_artifact(job.id, job.plan_artifact, kind="plan")
            if job.plan_artifact
            and phase
            in {
                WorkflowPhase.REVIEWING,
                WorkflowPhase.REVISING,
                WorkflowPhase.IMPLEMENTING,
            }
            else None
        )
        review = (
            store.read_artifact(job.id, job.review_artifact, kind="review")
            if job.review_artifact
            and phase in {WorkflowPhase.REVISING, WorkflowPhase.IMPLEMENTING}
            else None
        )
        prompt_factory = _WorkflowPromptFactory(
            job=job,
            phase=phase,
            token=token,
            integration_instructions=integration_instructions,
            plan=plan,
            review=review,
            issue_reference=job.issue_key
            or (f"issue {job.github_issue}" if job.github_issue else None),
        )
        try:
            pending_output = _execute_phase_prompt(
                store,
                job,
                worker_token,
                client,
                checkpoint,
                target=target,
                prompt_factory=prompt_factory,
                token=token,
            )
        except PromptSizeError as exc:
            _workflow_block(store, job.id, worker_token, str(exc))
            return
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
                job=job,
                dispatch_token=turn_token,
                prompt_state=PromptOperationState.PLANNED,
                prompt_baseline_seq=None,
                prompt_submitted_at=None,
                prompt_absent_observations=0,
            )
    elif question is not None and question.state == QuestionState.ANSWERED:
        question_envelope = question_adapter.envelope(
            question, QuestionState.RESOLVED, job=job
        )
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
                job=job,
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


def run_claimed_worker(  # pyright: ignore[reportGeneralTypeIssues]
    context: worker_lifecycle.WorkerContext,
    factories: ClientFactories | None = None,
) -> None:
    if factories is None:
        registry = build_integration_registry(default_user_config())
        clients = ClientFactories(HerdrClient, GitHubClient, registry)
    else:
        clients = factories
        registry = clients.integrations or build_integration_registry(
            default_user_config()
        )
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
        if job.github_issue_create_requested:
            _run_github_issue_creation(
                store,
                job,
                worker_token,
                clients,
                checkpoint,
            )
            return
        if job.linear_ticket_create_requested:
            _run_linear_ticket_creation(
                store,
                job,
                worker_token,
                clients,
                checkpoint,
            )
            return
        require_issue_provider(job.issue_provider, registry)
        active_issue_key = resolve_issue_reference(
            job.issue_key,
            registry,
            provider=job.issue_provider,
        )
        if active_issue_key:
            require_issue_capabilities(
                active_issue_key,
                registry,
                provider=job.issue_provider,
            )
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
                integrations=registry,
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
                github = _github_provider(clients.github)
                pull_request_plan = github.plan_pull_request(github_repository, number)
                checkpoint()
                repository = github.materialize_repository(
                    pull_request_plan.source,
                    checkpoint=checkpoint,
                )
                checkpoint()
                issue_key = None

                def record_pull_request(current: CursorJob) -> CursorJob:
                    return current.evolve(
                        github_repository=pull_request_plan.source.name_with_owner,
                        repository=str(repository),
                        pull_request_worktree_state="provisioning",
                        pull_request_remote_url=pull_request_plan.checkout.remote_url,
                        pull_request_head_ref=pull_request_plan.checkout.head_ref,
                        pull_request_head_oid=pull_request_plan.checkout.head_oid,
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
                github = _github_provider(clients.github)
                checkpoint()
                fork_plan = _persisted_fork_plan(job)
                if fork_plan is None:
                    fork_plan = github.plan_fork(github_repository)
                else:
                    fork_plan = github.refresh_fork_plan(fork_plan)
                    github.validate_fork_plan(
                        fork_plan,
                        materialized_repository=(
                            job.fork_repository
                            if job.fork_operation_state == "exists"
                            else None
                        ),
                    )
                source = fork_plan.source
                login = fork_plan.login
                fork_target = fork_plan.target
                checkpoint()
                if job.fork_operation_state == "exists" and job.fork_repository:
                    fork = github.observe_fork(fork_plan)
                    if fork is None:
                        raise HarnessError(
                            "reconciled GitHub fork is no longer observable"
                        )
                elif job.fork_operation_state not in {None, "planned"}:
                    raise HarnessError(
                        "GitHub fork operation requires reconciliation before retry"
                    )
                else:
                    updated = _begin_fork_operation(
                        store, job_id, worker_token, source, login, fork_target
                    )
                    if updated is None:
                        return
                    job = updated
                    try:
                        fork = github.submit_fork(
                            fork_plan,
                            confirmed=job.fork_confirmed,
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
                            github.observe_fork(fork_plan) if was_submitted else None
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
                checkout = github.materialize_fork(
                    fork_plan,
                    fork,
                    checkpoint=checkpoint,
                )
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
                github = _github_provider(clients.github)
                issue = GitHubIssue(owner, repository_name, number)
                # The preflight may have happened long ago; verify the issue
                # again so a vanished or newly inaccessible issue fails with
                # the same classified, voice-safe error.
                github.resolve_issue(issue)
                checkpoint()
                provisioned_issue = github.provision_issue(
                    issue,
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
                if issue_key and not hint:
                    routed = route_issue_repository(
                        client,
                        issue_key,
                        repositories,
                        token=f"{job_id}-route",
                        reserved=reserved_targets(store, job_id),
                        checkpoint=checkpoint,
                        integrations=registry,
                        provider=job.issue_provider,
                    )
                    if routed is not None:
                        repository, _confidence, reason = routed
                    checkpoint()
                if repository is None:
                    repository, candidates = resolve_job_repository(
                        client, job, repositories
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
            integrations=registry,
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
        except Exception as cleanup_exc:
            # Durable terminal intent and release fences are recovered later.
            print(
                "Cursor worker terminal cleanup failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
