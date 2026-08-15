from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ..integrations.github import (
    GITHUB_PROVIDER_STATE_FIELDS,
    GitHubError,
    dump_github_provider_state,
    github_issue_from_url,
    load_github_provider_state,
)
from ..job_lifecycle import (
    AwaitingUserJob,
    BlockedJob,
    ExecutionComponents,
    JobEvent,
    JobIdentity,
    JobLifecycle,
    JobLifecycleError,
    JobState,
    LifecycleEvent,
    QueuedJob,
    ReconcilingJob,
    RecoveryEvent,
    RoutingProvisioningJob,
    RunningJob,
    SessionControlMode,
    SessionControlState,
    TerminalJob,
    WorkerCallbackEvent,
    WorkerClaim,
    apply_event,
    legal_edges,
)
from ..prompt_operations import (
    AmbiguousPrompt,
    IdlePrompt,
    ObservedPrompt,
    PlannedPrompt,
    PromptOperation,
    PromptOperationError,
    ResolvedPrompt,
    SubmittedPrompt,
    SubmittingPrompt,
    legacy_prompt_fields,
    load_prompt_operation,
)
from ..questions import Question, QuestionError
from .lifecycle import (
    AnnouncementAck,
    CleanupOwned,
    CleanupSettled,
    CleanupState,
    Delivered,
    DeliveryState,
    LifecycleTransitionError,
    MaterializedTerminalOutcome,
    PendingDelivery,
    TerminalIntent,
    TerminalState,
    abandon_cleanup_owner,
    acknowledge_without_claim,
    cleanup_fields,
    delivery_fields,
    dismiss_announcement,
    finish_cleanup_reconciliation,
    load_cleanup_state,
    load_delivery_state,
    load_terminal_state,
    repeat_announcement,
)
from .lifecycle import (
    acknowledge_delivery as transition_delivery_acknowledgement,
)
from .lifecycle import (
    prepare_delivery as transition_prepare_delivery,
)
from .lifecycle import (
    release_delivery as transition_delivery_release,
)
from .lifecycle import (
    renew_delivery as transition_delivery_renewal,
)
from .operations import (
    AgentSessionOperation,
    AgentSessionState,
    CheckoutOperation,
    CheckoutState,
    ForkOperation,
    ForkSpec,
    OperationState,
    OperationTransitionError,
    ParticipantPaneOperation,
    ParticipantPaneSpec,
    WorkerOwnership,
    agent_session_fields,
    checkout_blocks_reservation,
    checkout_fields,
    load_agent_session_operation,
    load_checkout_operation,
    load_worker_ownership,
)
from .workflow import (
    ArtifactReference,
    LegacyPlanApprovalProof,
    ParticipantAdmissionState,
    ParticipantCreation,
    ParticipantCreationState,
    ParticipantLifecycle,
    PlanApproval,
    PlanApprovalProof,
    PlanApprovalSource,
    PlanApprovalState,
    ReviewApprovalSource,
    ReviewDecision,
    ReviewState,
    WorkflowClassification,
    WorkflowParticipant,
    WorkflowPhase,
    WorkflowState,
    WorkflowTier,
    WorkflowTransitionError,
)

CURRENT_SCHEMA_VERSION = 18
LEGACY_SCHEMA_VERSIONS = frozenset(range(CURRENT_SCHEMA_VERSION))
LEGACY_BOOT_ID = "legacy-unknown"


class JobValidationError(ValueError):
    """A persisted Cursor job does not satisfy the job schema."""


class JobStatus(StrEnum):
    QUEUED = "queued"
    ROUTING = "routing"
    RUNNING = "running"
    RECONCILING = "reconciling"
    AWAITING_USER = "awaiting_user"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ANNOUNCEMENT_ACK_STATES = frozenset(item.value for item in AnnouncementAck)
SPOKEN_ANNOUNCEMENT_ACKS = frozenset(
    {AnnouncementAck.SPOKEN.value, AnnouncementAck.DISMISSED.value}
)
SUPPRESSED_ANNOUNCEMENT_ACKS = frozenset(
    {AnnouncementAck.DEFERRED.value, AnnouncementAck.DESKTOP.value}
)


class HarnessKind(StrEnum):
    """Coding-agent harness responsible for a durable job."""

    CURSOR = "cursor"
    OPENCODE = "opencode"


ACTIVE_STATUSES = frozenset(
    {
        JobStatus.QUEUED,
        JobStatus.ROUTING,
        JobStatus.RUNNING,
        JobStatus.RECONCILING,
        JobStatus.AWAITING_USER,
        JobStatus.BLOCKED,
    }
)
WORKER_STATUSES = frozenset(
    {
        JobStatus.ROUTING,
        JobStatus.RUNNING,
        JobStatus.RECONCILING,
    }
)
TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)

_LEGAL_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus(source.value): frozenset(
        JobStatus(target.value) for target in legal_edges(source)
    )
    for source in JobState
}

_BOOL_FIELDS = frozenset(
    {
        "delivered",
        "continuation",
        "reconcile",
        "fork_requested",
        "fork_confirmed",
        "clone_confirmed",
        "fork_committed",
        "fork_exists",
        "fork_dispatch_exited",
        "github_issue_create_requested",
        "github_issue_create_confirmed",
        "github_repo_create_requested",
        "github_repo_create_org_requested",
        "github_repo_create_confirmed",
        "linear_ticket_create_requested",
        "linear_ticket_create_confirmed",
        "fork_operation_source_private",
        "agent_dispatch_exited",
        "worktree_dispatch_exited",
        "target_release_pending",
        "target_release_manual_required",
        "agent_identity_legacy_compatible",
        "cancellation_reconciliation_pending",
        "worktree_manual_inspection_required",
        "announcement_dismissed",
        "announcement_repeated",
        "phase_prompt_active",
        "review_approved",
        "interactive_questionnaire_blocked",
        "plan_approval_counted",
        "plan_approval_completion_pending",
    }
)
_INT_FIELDS = frozenset(
    {
        "schema_version",
        "migration_source_schema_version",
        "revision",
        "turn",
        "github_issue",
        "github_issue_created_number",
        "github_pull_request",
        "worker_pid",
        "target_release_owner_pid",
        "delivery_generation",
        "delivery_attempts",
        "agent_reconcile_attempts",
        "agent_absent_observations",
        "fork_reconcile_attempts",
        "fork_absent_observations",
        "worktree_reconcile_attempts",
        "worktree_absent_observations",
        "review_round",
        "prompt_operation_turn",
        "prompt_baseline_sequence",
        "plan_approval_state_change_sequence",
        "plan_approval_revision",
        "linear_ticket_create_baseline_sequence",
        "agent_state_sequence",
        "session_control_generation",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "created_at",
        "queued_at",
        "started_at",
        "attempt_started_at",
        "updated_at",
        "completed_at",
        "foreground_until",
        "next_reconcile_at",
        "delivery_claimed_at",
        "delivery_retry_at",
        "delivered_at",
        "fork_committed_at",
        "agent_next_reconcile_at",
        "agent_last_reconciled_at",
        "agent_confirmed_absent_at",
        "agent_automatic_reconcile_stopped_at",
        "agent_retained_at",
        "fork_next_reconcile_at",
        "fork_last_reconciled_at",
        "fork_confirmed_absent_at",
        "fork_automatic_reconcile_stopped_at",
        "fork_retained_at",
        "worktree_next_reconcile_at",
        "worktree_last_reconciled_at",
        "worktree_confirmed_absent_at",
        "worktree_automatic_reconcile_stopped_at",
        "worktree_retained_at",
        "pane_retained_at",
        "manual_reconcile_required_at",
        "manual_reconcile_resolved_at",
        "worktree_quarantine_acknowledged_at",
        "terminal_intent_completed_at",
        "worker_claimed_at",
    }
)
_STRING_FIELDS = frozenset(
    {
        "id",
        "parent_job_id",
        "request",
        "utterance",
        "trusted_utterance",
        "repository_hint",
        "context_repository",
        "grouped_repository_coordinator_id",
        "repository",
        "github_repository",
        "clone_source",
        "clone_operation_state",
        "github_issue_url",
        "github_issue_context",
        "github_issue_create_title",
        "github_issue_create_body",
        "github_issue_create_marker",
        "github_issue_create_operation_state",
        "github_issue_created_url",
        "github_repo_create_owner",
        "github_repo_create_visibility",
        "github_repo_create_marker",
        "github_repo_create_operation_state",
        "github_repo_created_url",
        "linear_ticket_create_team",
        "linear_ticket_create_team_id",
        "linear_ticket_create_title",
        "linear_ticket_create_description",
        "linear_ticket_create_marker",
        "linear_ticket_create_operation_state",
        "linear_ticket_create_prompt_target",
        "linear_ticket_create_prompt_session",
        "linear_ticket_create_prompt_token",
        "linear_ticket_created_identifier",
        "linear_ticket_created_url",
        "worktree_branch",
        "worktree_label",
        "worktree_path",
        "worktree_workspace_id",
        "worktree_root_pane_id",
        "worktree_provision_error",
        "pull_request_worktree_state",
        "pull_request_branch",
        "pull_request_worktree_error",
        "pull_request_remote_url",
        "pull_request_head_ref",
        "pull_request_head_oid",
        "agent_hint",
        "agent_name",
        "issue_key",
        "speakable_label",
        "status",
        "result",
        "error",
        "question",
        "clarification_kind",
        "continuation_answer",
        "turn_token",
        "worker_token",
        "worker_boot_id",
        "worker_process_start",
        "worker_operation",
        "worker_claim_operation",
        "herdr_target",
        "herdr_pane_id",
        "herdr_workspace_id",
        "announcement_ack",
        "delivery_claim_token",
        "target_release_token",
        "target_release_owner_boot_id",
        "target_release_owner_start",
        "agent_dispatch_state",
        "agent_provider",
        "agent_provider_session_id",
        "agent_operation_checkout",
        "agent_operation_target",
        "agent_operation_workspace_id",
        "agent_operation_pane_id",
        "fork_operation_state",
        "fork_operation_source",
        "fork_operation_source_url",
        "fork_operation_source_parent",
        "fork_operation_source_default_branch",
        "fork_operation_login",
        "fork_operation_target",
        "fork_repository",
        "worktree_provision_state",
        "manual_reconcile_operation",
        "manual_reconcile_token",
        "manual_reconcile_outcome",
        "reconciliation_base_error",
        "workflow_tier",
        "workflow_classification_reason",
        "workflow_phase",
        "plan_artifact",
        "review_artifact",
        "active_participant",
        "planner_target",
        "reviewer_target",
        "implementer_target",
        "review_decision",
        "review_approval_source",
        "plan_approval_state",
        "plan_approval_id",
        "plan_approval_source",
        "plan_approval_agent_session",
        "plan_approval_plan_artifact",
        "plan_approval_review_artifact",
        "workflow_turn_phase",
        "prompt_operation_state",
        "prompt_operation_phase",
        "prompt_operation_target",
        "prompt_operation_agent_session",
        "participant_creation_state",
        "participant_admission_state",
        "participant_creation_participant",
        "participant_creation_target",
        "participant_creation_label",
        "participant_creation_workspace_id",
        "participant_creation_pane_id",
        "participant_creation_checkout",
        "terminal_intent_status",
        "terminal_intent_result",
        "terminal_intent_error",
        "harness_kind",
        "issue_provider",
        "session_id",
        "session_control",
    }
)
_AGENT_OPERATION_STATES = frozenset(
    {
        "dispatching",
        "ready",
        "ambiguous",
        "failed_observing",
        "confirmed_absent",
        "manual_required",
        "retained",
    }
)
_FORK_OPERATION_STATES = frozenset(
    {
        "planned",
        "submitted",
        "exists",
        "failed",
        "ambiguous",
        "failed_observing",
        "confirmed_absent",
        "manual_required",
        "retained",
    }
)
_ISSUE_CREATE_OPERATION_STATES = frozenset(
    {
        "planned",
        "submitting",
        "submitted",
        "created",
        "ambiguous",
        "manual_required",
    }
)
_REPO_CREATE_OPERATION_STATES = _ISSUE_CREATE_OPERATION_STATES | {
    "remote_created",
    "clone_verified",
}
_CLONE_OPERATION_STATES = frozenset(
    {
        "planned",
        "submitted",
        "cloned",
        "ambiguous",
    }
)
_WORKTREE_OPERATION_STATES = frozenset(
    {
        "planned",
        "dispatching",
        "ready",
        "retained",
        "quarantined",
        "ambiguous",
        "failed_observing",
        "confirmed_absent",
        "manual_required",
    }
)
_UNCERTAIN_OPERATION_STATES = frozenset(
    {"dispatching", "submitted", "ambiguous", "failed_observing"}
)
_PROMPT_OPERATION_STATES = frozenset(
    {
        "none",
        "planned",
        "submitting",
        "submitted",
        "observed",
        "ambiguous",
        "resolved",
    }
)


@dataclass(frozen=True, slots=True)
class NewAgentJob:
    id: str
    request: str
    created_at: float
    foreground_until: float
    utterance: str | None = None
    trusted_utterance: str | None = None
    repository_hint: str | None = None
    context_repository: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_url: str | None = None
    github_issue_context: str | None = None
    github_issue_create_requested: bool = False
    github_issue_create_title: str | None = None
    github_issue_create_body: str | None = None
    github_issue_create_marker: str | None = None
    github_repo_create_requested: bool = False
    github_repo_create_org_requested: bool = False
    linear_ticket_create_requested: bool = False
    linear_ticket_create_team: str | None = None
    linear_ticket_create_team_id: str | None = None
    linear_ticket_create_title: str | None = None
    linear_ticket_create_description: str | None = None
    linear_ticket_create_marker: str | None = None
    fork_requested: bool = False
    github_pull_request: int | None = None
    worktree_branch: str | None = None
    worktree_label: str | None = None
    pull_request_worktree_state: str | None = None
    agent_hint: str | None = None
    issue_key: str | None = None
    speakable_label: str | None = None
    parent_job_id: str | None = None
    repository: str | None = None
    worktree_path: str | None = None
    worktree_workspace_id: str | None = None
    worktree_root_pane_id: str | None = None
    worktree_provision_state: str | None = None
    harness_kind: HarnessKind = HarnessKind.CURSOR
    issue_provider: str | None = None
    session_id: str | None = None


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise JobValidationError(f"{field} must be an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise JobValidationError(f"{field} must be an integer") from exc
    if isinstance(value, float) and value != parsed:
        raise JobValidationError(f"{field} must be an integer")
    return parsed


def _number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise JobValidationError(f"{field} must be a number")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise JobValidationError(f"{field} must be a number") from exc
    if not math.isfinite(parsed):
        raise JobValidationError(f"{field} must be finite")
    return parsed


def _boolean(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1, "0", "1"):
        return bool(int(value))
    raise JobValidationError(f"{field} must be a boolean")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise JobValidationError(f"{field} must be a string")
    return value


def _validate_json(value: object, field: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise JobValidationError(f"{field} must be finite")
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JobValidationError(f"{field} contains a non-string key")
            _validate_json(item, f"{field}.{key}")
        return
    raise JobValidationError(f"{field} is not JSON serializable")


_HARNESS_STATE_FIELDS = frozenset(
    {
        "agent_hint",
        "herdr_pane_id",
        "herdr_workspace_id",
        "agent_name",
        "agent_dispatch_state",
        "agent_provider",
        "agent_provider_session_id",
        "agent_operation_checkout",
        "agent_operation_target",
        "agent_operation_workspace_id",
        "agent_operation_pane_id",
        "agent_state_sequence",
        "agent_dispatch_exited",
        "agent_reconcile_attempts",
        "agent_absent_observations",
        "agent_next_reconcile_at",
        "agent_last_reconciled_at",
        "agent_confirmed_absent_at",
        "agent_automatic_reconcile_stopped_at",
        "agent_retained_at",
        "turn",
        "turn_token",
        "continuation",
        "next_reconcile_at",
        "target_release_pending",
        "target_release_token",
        "target_release_owner_pid",
        "target_release_owner_boot_id",
        "target_release_owner_start",
        "target_release_manual_required",
        "target_release_unverified_targets",
        "participant_session_owners",
        "session_control",
        "session_control_generation",
    }
)
_CHECKOUT_STATE_FIELDS = frozenset(
    {
        "repository",
        "worktree_branch",
        "worktree_label",
        "worktree_path",
        "worktree_workspace_id",
        "worktree_root_pane_id",
        "worktree_provision_state",
        "worktree_provision_error",
        "worktree_dispatch_exited",
        "worktree_reconcile_attempts",
        "worktree_absent_observations",
        "worktree_next_reconcile_at",
        "worktree_last_reconciled_at",
        "worktree_confirmed_absent_at",
        "worktree_automatic_reconcile_stopped_at",
        "worktree_retained_at",
        "worktree_manual_inspection_required",
        "worktree_quarantine_acknowledged_at",
    }
)
_GITHUB_STATE_FIELDS = GITHUB_PROVIDER_STATE_FIELDS
_LINEAR_STATE_FIELDS = frozenset(
    {
        "issue_key",
        "linear_ticket_create_requested",
        "linear_ticket_create_confirmed",
        "linear_ticket_create_team",
        "linear_ticket_create_team_id",
        "linear_ticket_create_title",
        "linear_ticket_create_description",
        "linear_ticket_create_marker",
        "linear_ticket_create_operation_state",
        "linear_ticket_create_prompt_target",
        "linear_ticket_create_prompt_session",
        "linear_ticket_create_prompt_token",
        "linear_ticket_create_baseline_sequence",
        "linear_ticket_created_identifier",
        "linear_ticket_created_url",
    }
)
_CHECKOUT_ALIASES = {
    "branch": "worktree_branch",
    "label": "worktree_label",
    "path": "worktree_path",
    "workspace_id": "worktree_workspace_id",
    "root_pane_id": "worktree_root_pane_id",
}


def _object_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise JobValidationError(f"{field} must be an object")
    return dict(value)


def _flatten_structured_state(values: dict[str, object]) -> None:
    """Expose schema-v9 structured state through the legacy typed properties."""

    harness_state = _object_mapping(values.pop("harness_state", {}), "harness_state")
    checkout_state = _object_mapping(values.pop("checkout_state", {}), "checkout_state")
    provider_state = _object_mapping(values.pop("provider_state", {}), "provider_state")
    raw_github_state = _object_mapping(
        provider_state.get("github", {}), "provider_state.github"
    )
    try:
        github_state = load_github_provider_state(raw_github_state)
    except GitHubError as exc:
        raise JobValidationError(str(exc)) from exc
    linear_state = _object_mapping(
        provider_state.get("linear", {}), "provider_state.linear"
    )
    provider_state.pop("github", None)
    provider_state.pop("linear", None)
    if provider_state:
        raise JobValidationError("provider_state contains an unsupported provider")

    aliases = {
        "pane_id": "herdr_pane_id",
        "workspace_id": "herdr_workspace_id",
    }
    for key, value in harness_state.items():
        field = aliases.get(key, key)
        if field not in _HARNESS_STATE_FIELDS:
            raise JobValidationError(f"harness_state contains unsupported field {key}")
        values.setdefault(field, value)
    for field, value in github_state.items():
        if field not in _GITHUB_STATE_FIELDS:
            raise JobValidationError(
                f"provider_state.github contains unsupported field {field}"
            )
        values.setdefault(field, value)
    for field, value in checkout_state.items():
        compatibility_field = _CHECKOUT_ALIASES.get(field, field)
        if compatibility_field not in _CHECKOUT_STATE_FIELDS:
            raise JobValidationError(
                f"checkout_state contains unsupported field {field}"
            )
        values.setdefault(compatibility_field, value)
    for field, value in linear_state.items():
        if field not in _LINEAR_STATE_FIELDS:
            raise JobValidationError(
                f"provider_state.linear contains unsupported field {field}"
            )
        values.setdefault(field, value)
    if values.get("session_id") is not None:
        values.setdefault("herdr_target", values["session_id"])


def _migrate_v17_typed_identities(values: dict[str, object]) -> None:
    """Convert v17's optional identity fields into safe v18 typed states."""

    worker_fields = (
        "worker_token",
        "worker_pid",
        "worker_boot_id",
        "worker_process_start",
    )
    worker_values = tuple(values.get(field) for field in worker_fields)
    if any(value is not None for value in worker_values):
        if all(value is not None for value in worker_values):
            if not values.get("worker_claim_operation"):
                values["worker_claim_operation"] = (
                    f"legacy:{values.get('status') or 'worker'}"
                )
            if values.get("worker_claimed_at") is None:
                values["worker_claimed_at"] = (
                    values.get("attempt_started_at")
                    or values.get("started_at")
                    or values.get("created_at")
                    or 0
                )

    worktree_state = values.get("worktree_provision_state")
    if worktree_state is not None:
        for field in ("repository", "worktree_branch", "worktree_path"):
            if not values.get(field):
                values[field] = LEGACY_BOOT_ID

    agent_state = values.get("agent_dispatch_state")
    if agent_state is not None:
        values["agent_identity_legacy_compatible"] = True
        values.setdefault("agent_operation_target", values.get("herdr_target"))
        values.setdefault(
            "agent_operation_workspace_id", values.get("herdr_workspace_id")
        )
        values.setdefault("agent_operation_pane_id", values.get("herdr_pane_id"))
        if not values.get("agent_operation_checkout"):
            values["agent_operation_checkout"] = (
                values.get("worktree_path")
                or values.get("repository")
                or LEGACY_BOOT_ID
            )
        for field in ("herdr_target", "herdr_pane_id", "herdr_workspace_id"):
            if not values.get(field):
                values[field] = LEGACY_BOOT_ID
                if (
                    field == "herdr_target"
                    and values.get("agent_dispatch_state") != "manual_required"
                ):
                    values["agent_dispatch_state"] = "ambiguous"
        values["agent_operation_target"] = (
            values.get("agent_operation_target") or values["herdr_target"]
        )
        values["agent_operation_workspace_id"] = (
            values.get("agent_operation_workspace_id") or values["herdr_workspace_id"]
        )
        values["agent_operation_pane_id"] = (
            values.get("agent_operation_pane_id") or values["herdr_pane_id"]
        )
        session_fields = (
            "agent_provider",
            "agent_provider_session_id",
            "agent_state_sequence",
        )
        session_values = tuple(values.get(field) for field in session_fields)
        session_complete = all(value is not None for value in session_values)
        if any(value is not None for value in session_values) and not session_complete:
            for field in session_fields:
                values[field] = None

    participant_state = values.get("participant_creation_state")
    if participant_state not in {None, "none"}:
        if not values.get("participant_creation_checkout"):
            values["participant_creation_checkout"] = (
                values.get("worktree_path")
                or values.get("repository")
                or LEGACY_BOOT_ID
            )

    fork_state = values.get("fork_operation_state")
    if fork_state is not None:
        target = values.get("fork_operation_target")
        if (
            not values.get("fork_operation_login")
            and isinstance(target, str)
            and "/" in target
        ):
            values["fork_operation_login"] = target.split("/", 1)[0]
        required_fork_fields = (
            "fork_operation_source",
            "fork_operation_source_url",
            "fork_operation_source_default_branch",
            "fork_operation_login",
            "fork_operation_target",
        )
        incomplete = any(not values.get(field) for field in required_fork_fields)
        for field in required_fork_fields:
            if not values.get(field):
                values[field] = LEGACY_BOOT_ID
        values.setdefault("fork_operation_source_private", False)
        if incomplete and fork_state not in {"failed", "confirmed_absent"}:
            values["fork_operation_state"] = "ambiguous"

    values.setdefault("target_release_manual_required", False)
    values.setdefault("agent_identity_legacy_compatible", False)
    values.setdefault("target_release_unverified_targets", [])
    values.setdefault("participant_session_owners", [])


def _advance_legacy_version(values: dict[str, object], version: int) -> None:
    """Apply one compatibility step and advance exactly one schema version."""

    if version == 5:
        if any(
            values.get(field) is not None
            for field in ("worker_token", "worker_pid", "worker_process_start")
        ):
            values.setdefault("worker_boot_id", LEGACY_BOOT_ID)
        if any(
            values.get(field) is not None
            for field in ("target_release_owner_pid", "target_release_owner_start")
        ):
            values.setdefault("target_release_owner_boot_id", LEGACY_BOOT_ID)
    elif version == 6:
        values.setdefault("announcement_dismissed", False)
        values.setdefault("announcement_repeated", False)
    elif version == 7:
        values.setdefault("parent_job_id", None)
    elif version == 8:
        values.setdefault("harness_kind", HarnessKind.CURSOR.value)
        if values.get("herdr_target") is not None:
            values.setdefault("session_id", values["herdr_target"])
    elif version == 10:
        values.setdefault("plan_approval_counted", False)
        if values.get("workflow_tier") in {
            WorkflowTier.MEDIUM.value,
            WorkflowTier.HIGH_RISK.value,
        } and values.get("workflow_phase") in {
            WorkflowPhase.IMPLEMENTING.value,
            WorkflowPhase.FINISHED.value,
        }:
            values.setdefault("plan_approval_state", "observed")
            values.setdefault("plan_approval_id", f"legacy-{values.get('id') or 'job'}")
            values.setdefault("plan_approval_source", "legacy")
            values.setdefault("plan_approval_agent_session", LEGACY_BOOT_ID)
            values.setdefault("plan_approval_state_change_sequence", -1)
            values.setdefault("plan_approval_revision", -1)
        else:
            values.setdefault("plan_approval_state", "none")
    elif version == 11:
        values.setdefault("plan_approval_completion_pending", False)
    elif version == 12:
        values.setdefault("issue_provider", _infer_legacy_issue_provider(values))
        # The GitHub provider-state serializer independently migrates its
        # legacy flat v12 payload to the nested provider-owned representation.
    elif version == 13:
        values.setdefault(
            "participant_admission_state",
            _default_participant_admission_state(values),
        )
    elif version == 14:
        values.setdefault("clarifications", [])
        values.setdefault("prompt_context_sessions", {})
    elif version == 15:
        _default_announcement_ack(values)
    elif version == 16:
        if values.get("plan_approval_state") in {
            PlanApprovalState.AWAITING.value,
            PlanApprovalState.APPROVED.value,
            PlanApprovalState.OBSERVED.value,
        }:
            values.setdefault(
                "plan_approval_plan_artifact",
                values.get("plan_artifact"),
            )
            values.setdefault(
                "plan_approval_review_artifact",
                values.get("review_artifact"),
            )
    elif version == 17:
        _migrate_v17_typed_identities(values)
    values["schema_version"] = version + 1


def _pair_announcement_ack(values: dict[str, object]) -> None:
    """Keep spoken/dismissed acks paired with delivered, and reopen pending otherwise."""

    ack = values.get("announcement_ack")
    if values.get("delivered"):
        if ack in {None, AnnouncementAck.PENDING.value}:
            values["announcement_ack"] = (
                AnnouncementAck.DISMISSED.value
                if values.get("announcement_dismissed")
                else AnnouncementAck.SPOKEN.value
            )
        return
    if ack in SPOKEN_ANNOUNCEMENT_ACKS:
        values["announcement_ack"] = AnnouncementAck.PENDING.value


def _legacy_announcement_ack_from_dismissal(values: dict[str, object]) -> str:
    return (
        AnnouncementAck.DISMISSED.value
        if values.get("announcement_dismissed")
        else AnnouncementAck.SPOKEN.value
    )


def _default_announcement_ack(values: dict[str, object]) -> None:
    ack = values.get("announcement_ack")
    if ack in ANNOUNCEMENT_ACK_STATES:
        if values.get("delivered") and ack == AnnouncementAck.PENDING.value:
            values["announcement_ack"] = _legacy_announcement_ack_from_dismissal(values)
        return
    if values.get("delivered"):
        values["announcement_ack"] = _legacy_announcement_ack_from_dismissal(values)
        return
    values["announcement_ack"] = AnnouncementAck.PENDING.value


def _canonicalize_announcement_dismissal(
    values: dict[str, object], *, legacy_record: bool
) -> None:
    """Translate legacy dismissal, fail closed on conflict, then drop the mirror."""

    raw_ack = values.get("announcement_ack")
    dismissed_present = "announcement_dismissed" in values
    dismissed = (
        bool(values.get("announcement_dismissed")) if dismissed_present else None
    )
    if (
        dismissed_present
        and raw_ack in ANNOUNCEMENT_ACK_STATES
        and dismissed != (raw_ack == AnnouncementAck.DISMISSED.value)
        and not (
            legacy_record
            and values.get("delivered")
            and raw_ack == AnnouncementAck.PENDING.value
        )
    ):
        raise JobValidationError("dismissal state and acknowledgement must match")
    _default_announcement_ack(values)
    values.pop("announcement_dismissed", None)


def _canonicalize_review_approval(
    values: dict[str, object], *, legacy_record: bool
) -> None:
    """Translate legacy approval boolean, fail closed on conflict, then drop it."""

    approved_present = "review_approved" in values
    source_present = "review_approval_source" in values
    approved = bool(values.get("review_approved")) if approved_present else None
    source = values.get("review_approval_source")
    if approved_present and source_present and approved != (source is not None):
        raise JobValidationError("review approval and approval source must be paired")
    if approved_present and not source_present and approved:
        if not legacy_record:
            raise JobValidationError(
                "current-schema review approval requires explicit approval source"
            )
        values["review_approval_source"] = "reviewer"
    values.pop("review_approved", None)


def _pair_worker_ownership(values: dict[str, object]) -> None:
    """Keep newly typed worker claims all-or-nothing without rewriting legacy rows."""
    if "worker_claimed_at" not in values:
        return
    if values.get("worker_token") is None:
        for field in (
            "worker_pid",
            "worker_boot_id",
            "worker_process_start",
            "worker_claim_operation",
            "worker_claimed_at",
        ):
            values[field] = None
        return
    if values.get("worker_claim_operation") is None:
        values["worker_claim_operation"] = str(values.get("status") or "worker")


def _infer_legacy_issue_provider(values: Mapping[str, object]) -> str | None:
    if values.get("issue_key") is not None:
        # Historically issue_key was exclusively owned by the Linear
        # integration. GitHub fields could also be present as captured context.
        return "linear"
    if (
        values.get("github_issue") is not None
        or values.get("github_issue_created_number") is not None
    ):
        return "github"
    return None


def _canonicalize_github_issue_creation_identity(values: dict[str, object]) -> None:
    """Translate legacy created identity, fail closed on conflict, then drop it."""

    created_number = values.get("github_issue_created_number")
    created_url = values.get("github_issue_created_url")
    issue = values.get("github_issue")
    url = values.get("github_issue_url")
    repository = values.get("github_repository")
    if created_number is not None:
        created_number = _integer(created_number, "github_issue_created_number")
    if issue is not None:
        issue = _integer(issue, "github_issue")
    parsed_urls = []
    for field, candidate in (
        ("github_issue_url", url),
        ("github_issue_created_url", created_url),
    ):
        if candidate is None:
            continue
        parsed = github_issue_from_url(str(candidate))
        if parsed is None:
            raise JobValidationError(f"{field} must be an exact GitHub issue URL")
        parsed_urls.append(parsed)
    repositories = {
        candidate.casefold()
        for candidate in (
            str(repository) if repository is not None else None,
            *(parsed.name_with_owner for parsed in parsed_urls),
        )
        if candidate is not None
    }
    numbers = {
        candidate
        for candidate in (
            issue,
            created_number,
            *(parsed.number for parsed in parsed_urls),
        )
        if candidate is not None
    }
    if len(repositories) > 1 or len(numbers) > 1:
        raise JobValidationError(
            "created GitHub issue identity must match canonical issue identity"
        )
    if created_number is not None and issue is None:
        values["github_issue"] = created_number
    if created_url is not None and url is None:
        values["github_issue_url"] = str(created_url)
    values.pop("github_issue_created_number", None)
    values.pop("github_issue_created_url", None)


def _canonicalize_pull_request_branch(values: dict[str, object]) -> None:
    """Translate legacy PR branch, fail closed on conflict, then drop the mirror."""

    stored = values.get("pull_request_branch")
    planned = values.get("worktree_branch")
    if stored is not None and planned is not None and str(stored) != str(planned):
        raise JobValidationError(
            "pull-request branch must match the planned worktree branch"
        )
    if stored is not None and planned is None:
        values["worktree_branch"] = str(stored)
    values.pop("pull_request_branch", None)


def _default_participant_admission_state(values: Mapping[str, object]) -> str:
    status = str(values.get("status") or "")
    if any(
        values.get(field)
        for field in (
            "herdr_target",
            "planner_target",
            "reviewer_target",
            "implementer_target",
            "participant_creation_target",
            "worker_token",
            "worker_pid",
            "target_release_pending",
        )
    ):
        return "held"
    if (
        status in {item.value for item in TERMINAL_STATUSES}
        or values.get("clarification_kind") == "grouped_repository"
    ):
        return "released"
    return "waiting"


def migrate_job_record(raw: Mapping[str, object]) -> tuple[dict[str, object], int]:
    """Migrate a persisted record to the current in-memory representation."""

    values = dict(raw)
    record_version = (
        _integer(values["schema_version"], "schema_version")
        if "schema_version" in values
        else 0
    )
    if record_version not in LEGACY_SCHEMA_VERSIONS | {CURRENT_SCHEMA_VERSION}:
        raise JobValidationError(
            f"unsupported agent job schema version {record_version}"
        )
    source_version = values.get("migration_source_schema_version")
    if source_version is None:
        loaded_version = record_version
    else:
        loaded_version = _integer(source_version, "migration_source_schema_version")
        if (
            record_version != CURRENT_SCHEMA_VERSION
            or loaded_version not in LEGACY_SCHEMA_VERSIONS
        ):
            raise JobValidationError("invalid migration source schema version")
    compatibility_input = not (
        "harness_state" in values
        or "checkout_state" in values
        or "provider_state" in values
    )
    _flatten_structured_state(values)
    if record_version < CURRENT_SCHEMA_VERSION:
        if compatibility_input:
            values.setdefault("harness_kind", HarnessKind.CURSOR.value)
            if values.get("herdr_target") is not None:
                values.setdefault("session_id", values["herdr_target"])
            elif values.get("session_id") is not None:
                values.setdefault("herdr_target", values["session_id"])
        _legacy_defaults(values)
        for version in range(record_version, CURRENT_SCHEMA_VERSION):
            _advance_legacy_version(values, version)
        values["migration_source_schema_version"] = record_version
    else:
        if compatibility_input:
            values.setdefault("harness_kind", HarnessKind.CURSOR.value)
            if values.get("herdr_target") is not None:
                values.setdefault("session_id", values["herdr_target"])
        values.setdefault("issue_provider", _infer_legacy_issue_provider(values))
        values.setdefault(
            "participant_admission_state",
            _default_participant_admission_state(values),
        )
    values.setdefault("session_control", SessionControlMode.AUTOMATED.value)
    values.setdefault("session_control_generation", 0)
    _canonicalize_announcement_dismissal(
        values,
        legacy_record=loaded_version < CURRENT_SCHEMA_VERSION,
    )
    _canonicalize_review_approval(
        values,
        legacy_record=loaded_version < CURRENT_SCHEMA_VERSION,
    )
    _canonicalize_github_issue_creation_identity(values)
    _canonicalize_pull_request_branch(values)
    values["schema_version"] = CURRENT_SCHEMA_VERSION
    return values, loaded_version


def _structured_record(values: Mapping[str, object]) -> dict[str, object]:
    """Serialize the compatibility representation as the agent-neutral v9 schema."""

    record = dict(values)
    record["schema_version"] = CURRENT_SCHEMA_VERSION
    record["harness_kind"] = str(record.get("harness_kind") or HarnessKind.CURSOR.value)
    session_id = record.pop("herdr_target", None)
    record["session_id"] = record.get("session_id") or session_id

    harness_state: dict[str, object] = {}
    for field in _HARNESS_STATE_FIELDS:
        if field not in record:
            continue
        value = record.pop(field)
        key = {
            "herdr_pane_id": "pane_id",
            "herdr_workspace_id": "workspace_id",
        }.get(field, field)
        harness_state[key] = value
    github_state: dict[str, object] = {}
    for field in _GITHUB_STATE_FIELDS:
        if field in record:
            github_state[field] = record.pop(field)
    checkout_state: dict[str, object] = {}
    checkout_names = {value: key for key, value in _CHECKOUT_ALIASES.items()}
    for field in _CHECKOUT_STATE_FIELDS:
        if field in record:
            checkout_state[checkout_names.get(field, field)] = record.pop(field)
    linear_state: dict[str, object] = {}
    for field in _LINEAR_STATE_FIELDS:
        if field in record:
            linear_state[field] = record.pop(field)
    record["harness_state"] = harness_state
    record["checkout_state"] = checkout_state
    provider_state: dict[str, object] = {}
    if github_state:
        provider_state["github"] = dump_github_provider_state(github_state)
    if linear_state:
        provider_state["linear"] = linear_state
    record["provider_state"] = provider_state
    return record


def _legacy_defaults(values: dict[str, object]) -> None:
    status = str(values.get("status") or "")
    job_id = str(values.get("id") or "")
    values.setdefault("revision", 0)
    values.setdefault("request", "")
    values.setdefault(
        "created_at",
        values.get("queued_at") or values.get("completed_at") or 0,
    )
    values.setdefault("delivered", False)
    values.setdefault("review_round", 0)
    if values.get("review_approved"):
        values.setdefault("review_approval_source", "reviewer")
    if (
        values.get("status") == JobStatus.AWAITING_USER.value
        and values.get("clarification_kind") == "workflow_review"
        and values.get("workflow_tier") == WorkflowTier.HIGH_RISK.value
        and _integer(values.get("review_round") or 0, "review_round") >= 2
        and values.get("review_decision") == "revise"
    ):
        values["clarification_kind"] = "workflow_review_exhausted"
    if status in {item.value for item in TERMINAL_STATUSES}:
        values.setdefault("workflow_phase", WorkflowPhase.FINISHED.value)
    elif values.get("herdr_target"):
        values.setdefault("workflow_phase", WorkflowPhase.IMPLEMENTING.value)
        values.setdefault("workflow_tier", WorkflowTier.SIMPLE.value)
        values.setdefault(
            "workflow_classification_reason", "Migrated active direct workflow."
        )
        participant = str(
            values.setdefault(
                "active_participant", WorkflowParticipant.IMPLEMENTER.value
            )
        )
        if participant in {item.value for item in WorkflowParticipant}:
            values.setdefault(f"{participant}_target", values.get("herdr_target"))
    else:
        values.setdefault("workflow_phase", WorkflowPhase.CLASSIFYING.value)
    if values.get("workflow_phase") != WorkflowPhase.CLASSIFYING.value:
        values.setdefault("workflow_tier", WorkflowTier.SIMPLE.value)
        values.setdefault(
            "workflow_classification_reason", "Migrated pre-workflow Cursor job."
        )
    if any(
        values.get(field) is not None
        for field in ("worker_token", "worker_pid", "worker_process_start")
    ):
        # A pre-v6 PID/start-time claim has no boot fence. Never attach the
        # current boot ID while loading it: after a reboot that could bless a
        # reused PID as the old owner.
        values.setdefault("worker_boot_id", LEGACY_BOOT_ID)
    if any(
        values.get(field) is not None
        for field in ("target_release_owner_pid", "target_release_owner_start")
    ):
        values.setdefault("target_release_owner_boot_id", LEGACY_BOOT_ID)
    if status == JobStatus.QUEUED:
        values.setdefault("queued_at", values["created_at"])
    elif status == JobStatus.AWAITING_USER:
        message = str(
            values.get("question")
            or values.get("result")
            or "Cursor job is awaiting user input."
        )
        values.setdefault("question", message)
        values.setdefault("result", message)
    elif status == JobStatus.BLOCKED:
        values.setdefault("result", "Cursor job needs attention.")
        values.setdefault("completed_at", values["created_at"])
    elif status == JobStatus.COMPLETED:
        values.setdefault("result", "")
        values.setdefault("completed_at", values["created_at"])
    elif status == JobStatus.FAILED:
        message = str(
            values.get("error") or values.get("result") or "Cursor job failed"
        )
        values.setdefault("error", message)
        values.setdefault("result", message)
        values.setdefault("completed_at", values["created_at"])
    elif status == JobStatus.CANCELLED:
        values.setdefault("result", f"Cursor job {job_id} was cancelled.")
        values.setdefault("completed_at", values["created_at"])


def _typed_prompt_operation(
    values: Mapping[str, object], job_id: str
) -> PromptOperation:
    try:
        turn = _integer(
            values.get("prompt_operation_turn") or 0, "prompt_operation_turn"
        )
        return load_prompt_operation(
            state=str(values.get("prompt_operation_state") or "none"),
            job_id=job_id,
            phase=(
                str(values["prompt_operation_phase"])
                if values.get("prompt_operation_phase") is not None
                else None
            ),
            turn=turn,
            turn_token=(
                str(values["turn_token"])
                if values.get("turn_token") is not None
                else (f"{job_id}-{turn}" if turn > 0 else None)
            ),
            target=(
                str(values["prompt_operation_target"])
                if values.get("prompt_operation_target") is not None
                else None
            ),
            agent_session=(
                str(values["prompt_operation_agent_session"])
                if values.get("prompt_operation_agent_session") is not None
                else None
            ),
            baseline_sequence=(
                _integer(values["prompt_baseline_sequence"], "prompt_baseline_sequence")
                if values.get("prompt_baseline_sequence") is not None
                else None
            ),
        )
    except PromptOperationError as exc:
        if str(values.get("prompt_operation_state") or "none") == "none":
            return IdlePrompt()
        raise JobValidationError(str(exc)) from exc


def _optional_typed_prompt_operation(
    values: Mapping[str, object], job_id: str, *, loaded_version: int
) -> PromptOperation | None:
    try:
        return _typed_prompt_operation(values, job_id)
    except JobValidationError:
        if (
            loaded_version < CURRENT_SCHEMA_VERSION
            or values.get("migration_source_schema_version") is not None
        ):
            return None
        raise


def _consume_typed_prompt_operation(changes: dict[str, object], *, job_id: str) -> None:
    operation = changes.pop("prompt_operation", None)
    if operation is None:
        return
    if not isinstance(
        operation,
        (
            IdlePrompt,
            PlannedPrompt,
            SubmittingPrompt,
            SubmittedPrompt,
            ObservedPrompt,
            AmbiguousPrompt,
            ResolvedPrompt,
        ),
    ):
        raise JobValidationError("prompt_operation must be a typed PromptOperation")
    if not isinstance(operation, IdlePrompt):
        if operation.identity.job_id != job_id:
            raise JobValidationError("prompt operation does not belong to this job")
        changes["turn"] = operation.identity.turn
        changes["turn_token"] = operation.identity.turn_token
    changes.update(legacy_prompt_fields(operation))


def _typed_checkout_operation(
    values: Mapping[str, object], *, loaded_version: int
) -> CheckoutOperation | None:
    state = values.get("worktree_provision_state")
    if state is None:
        return None
    parsed = CheckoutState(str(state))
    workspace_id = (
        str(values["worktree_workspace_id"])
        if values.get("worktree_workspace_id") is not None
        else None
    )
    root_pane_id = (
        str(values["worktree_root_pane_id"])
        if values.get("worktree_root_pane_id") is not None
        else None
    )
    if (
        loaded_version < CURRENT_SCHEMA_VERSION
        and parsed in {CheckoutState.READY, CheckoutState.RETAINED}
        and (
            not workspace_id
            or not root_pane_id
            or LEGACY_BOOT_ID
            in {
                values.get("repository"),
                values.get("worktree_branch"),
                values.get("worktree_path"),
            }
        )
    ):
        parsed = CheckoutState.AMBIGUOUS
    if loaded_version < CURRENT_SCHEMA_VERSION and bool(workspace_id) != bool(
        root_pane_id
    ):
        workspace_id = None
        root_pane_id = None
    try:
        return load_checkout_operation(
            state=parsed.value,
            repository=(
                str(values["repository"])
                if values.get("repository") is not None
                else None
            ),
            branch=(
                str(values["worktree_branch"])
                if values.get("worktree_branch") is not None
                else None
            ),
            path=(
                str(values["worktree_path"])
                if values.get("worktree_path") is not None
                else None
            ),
            workspace_id=workspace_id,
            root_pane_id=root_pane_id,
        )
    except OperationTransitionError as exc:
        raise JobValidationError(str(exc)) from exc


def _optional_typed_checkout_operation(
    values: Mapping[str, object], *, loaded_version: int
) -> CheckoutOperation | None:
    try:
        return _typed_checkout_operation(values, loaded_version=loaded_version)
    except (JobValidationError, ValueError):
        return None


def _consume_typed_checkout_operation(changes: dict[str, object]) -> None:
    operation = changes.pop("checkout_operation", None)
    if operation is None:
        return
    if not isinstance(operation, CheckoutOperation):
        raise JobValidationError("checkout_operation must be a typed CheckoutOperation")
    changes.update(checkout_fields(operation))


def _typed_agent_session_operation(
    values: Mapping[str, object], *, loaded_version: int
) -> AgentSessionOperation | None:
    state = values.get("agent_dispatch_state")
    if state is None:
        return None
    parsed = AgentSessionState(str(state))
    session_values = (
        values.get("agent_provider"),
        values.get("agent_provider_session_id"),
        values.get("agent_state_sequence"),
    )
    clear_legacy_session = False
    if (
        parsed in {AgentSessionState.READY, AgentSessionState.RETAINED}
        and loaded_version < CURRENT_SCHEMA_VERSION
        and not all(value is not None for value in session_values)
    ):
        parsed = AgentSessionState.AMBIGUOUS
        clear_legacy_session = True
    try:
        return load_agent_session_operation(
            state=parsed.value,
            target=(
                str(values["agent_operation_target"])
                if values.get("agent_operation_target") is not None
                else (
                    str(values["herdr_target"])
                    if values.get("herdr_target") is not None
                    else None
                )
            ),
            checkout=(
                str(values["agent_operation_checkout"])
                if values.get("agent_operation_checkout") is not None
                else (
                    str(values["worktree_path"])
                    if values.get("worktree_path") is not None
                    else (
                        str(values["repository"])
                        if values.get("repository") is not None
                        else None
                    )
                )
            ),
            workspace_id=(
                str(values["agent_operation_workspace_id"])
                if values.get("agent_operation_workspace_id") is not None
                else (
                    str(values["herdr_workspace_id"])
                    if values.get("herdr_workspace_id") is not None
                    else None
                )
            ),
            pane_id=(
                str(values["agent_operation_pane_id"])
                if values.get("agent_operation_pane_id") is not None
                else (
                    str(values["herdr_pane_id"])
                    if values.get("herdr_pane_id") is not None
                    else None
                )
            ),
            provider=(
                str(values["agent_provider"])
                if values.get("agent_provider") is not None and not clear_legacy_session
                else None
            ),
            session_id=(
                str(values["agent_provider_session_id"])
                if values.get("agent_provider_session_id") is not None
                and not clear_legacy_session
                else None
            ),
            state_sequence=(
                _integer(values["agent_state_sequence"], "agent_state_sequence")
                if values.get("agent_state_sequence") is not None
                and not clear_legacy_session
                else None
            ),
        )
    except OperationTransitionError as exc:
        raise JobValidationError(str(exc)) from exc


def _optional_typed_agent_session_operation(
    values: Mapping[str, object], *, loaded_version: int
) -> AgentSessionOperation | None:
    try:
        return _typed_agent_session_operation(values, loaded_version=loaded_version)
    except (JobValidationError, ValueError):
        return None


def _consume_typed_agent_session_operation(changes: dict[str, object]) -> None:
    operation = changes.pop("agent_session_operation", None)
    if operation is None:
        return
    if not isinstance(operation, AgentSessionOperation):
        raise JobValidationError(
            "agent_session_operation must be a typed AgentSessionOperation"
        )
    changes.update(agent_session_fields(operation))


def _prompt_operation_defaults(values: dict[str, object]) -> None:
    """Translate the schema-v10 boolean prompt fence without a schema bump."""
    if "prompt_operation_state" in values:
        return
    active = bool(values.get("phase_prompt_active", False))
    values["prompt_operation_state"] = "ambiguous" if active else "none"
    if active:
        values.setdefault(
            "prompt_operation_phase",
            values.get("workflow_turn_phase")
            or values.get("workflow_phase")
            or WorkflowPhase.CLASSIFYING.value,
        )
        values.setdefault("prompt_operation_turn", values.get("turn") or 1)
        values.setdefault("prompt_operation_target", values.get("herdr_target"))
        # The old boolean recorded no observation baseline. A negative sentinel
        # preserves that fact and prevents recovery from claiming acceptance.
        values.setdefault("prompt_baseline_sequence", -1)
        values.setdefault("manual_reconcile_operation", "prompt")
        values.setdefault(
            "manual_reconcile_token",
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "local-voice-harness:prompt:"
                f"{values.get('id') or 'unknown'}:{values.get('turn') or 1}",
            ).hex,
        )
        values.setdefault("manual_reconcile_required_at", values.get("updated_at") or 0)
    values.setdefault("participant_creation_state", "none")


@dataclass(frozen=True, slots=True)
class AgentJob:
    schema_version: int
    loaded_schema_version: int
    harness_kind: HarnessKind
    issue_provider: str | None
    session_id: str | None
    id: str
    revision: int
    request: str
    status: JobStatus
    created_at: float
    queued_at: float | None
    completed_at: float | None
    delivered: bool
    worker_token: str | None
    worker_pid: int | None
    worker_boot_id: str | None
    worker_process_start: str | None
    worker_operation: str | None
    delivery_claim_token: str | None
    delivery_claimed_at: float | None
    herdr_target: str | None
    worktree_path: str | None
    target_release_pending: bool
    target_release_manual_required: bool
    target_release_unverified_targets: tuple[str, ...]
    participant_session_owners: tuple[dict[str, object], ...]
    cancellation_reconciliation_pending: bool
    agent_dispatch_state: str | None
    fork_operation_state: str | None
    worktree_provision_state: str | None
    manual_reconcile_operation: str | None
    manual_reconcile_token: str | None
    workflow_tier: WorkflowTier | None
    workflow_phase: WorkflowPhase
    review_round: int
    active_participant: WorkflowParticipant | None
    _compatibility_layout: bool
    _values: dict[str, object]
    _prompt_operation: PromptOperation | None
    _checkout_operation: CheckoutOperation | None
    _agent_session_operation: AgentSessionOperation | None
    _lifecycle_event: JobEvent | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CursorJob:
        if not isinstance(raw, dict):
            raise JobValidationError("job must be a JSON object")
        raw_version = (
            _integer(raw["schema_version"], "schema_version")
            if "schema_version" in raw
            else 0
        )
        compatibility_layout = (
            raw_version in {17, CURRENT_SCHEMA_VERSION}
            and "harness_kind" not in raw
            and "harness_state" not in raw
            and "provider_state" not in raw
        )
        values, loaded_version = migrate_job_record(raw)
        _prompt_operation_defaults(values)

        for field in _INT_FIELDS:
            if field in values and values[field] is not None:
                values[field] = _integer(values[field], field)
        for field in _FLOAT_FIELDS:
            if field in values and values[field] is not None:
                values[field] = _number(values[field], field)
        for field in _BOOL_FIELDS:
            if field in values and values[field] is not None:
                values[field] = _boolean(values[field], field)
        for field in _STRING_FIELDS:
            if field in values and values[field] is not None:
                values[field] = _string(values[field], field)
        for key, value in values.items():
            _validate_json(value, key)
        if values.get("voice_question") is not None:
            try:
                Question.from_dict(values["voice_question"])
            except QuestionError as exc:
                raise JobValidationError(f"invalid voice_question: {exc}") from exc
        clarifications = values.setdefault("clarifications", [])
        if not isinstance(clarifications, list):
            raise JobValidationError("clarifications must be an array")
        for clarification in clarifications:
            if not isinstance(clarification, dict):
                raise JobValidationError("clarification records must be objects")
            if not all(
                isinstance(clarification.get(field), str)
                and bool(str(clarification[field]).strip())
                for field in ("question_id", "question", "answer", "turn_token")
            ):
                raise JobValidationError(
                    "clarification records require question, answer, and turn identity"
                )
        unverified_targets = values.setdefault("target_release_unverified_targets", [])
        if not isinstance(unverified_targets, list) or not all(
            isinstance(target, str) and bool(target) for target in unverified_targets
        ):
            raise JobValidationError(
                "target_release_unverified_targets must contain non-empty strings"
            )
        participant_session_owners = values.setdefault("participant_session_owners", [])
        if not isinstance(participant_session_owners, list):
            raise JobValidationError("participant_session_owners must be an array")
        required_owner_fields = {
            "provider",
            "session_id",
            "target",
            "state_sequence",
            "checkout",
            "workspace_id",
            "pane_id",
        }
        for owner in participant_session_owners:
            if (
                not isinstance(owner, dict)
                or set(owner) != required_owner_fields
                or not all(
                    isinstance(owner.get(field), str) and bool(owner[field])
                    for field in required_owner_fields - {"state_sequence"}
                )
                or not isinstance(owner.get("state_sequence"), int)
                or isinstance(owner.get("state_sequence"), bool)
                or int(owner["state_sequence"]) < 0
            ):
                raise JobValidationError("participant_session_owners is invalid")
        context_sessions = values.setdefault("prompt_context_sessions", {})
        if not isinstance(context_sessions, dict) or not all(
            key in {participant.value for participant in WorkflowParticipant}
            and isinstance(value, str)
            and bool(value)
            for key, value in context_sessions.items()
        ):
            raise JobValidationError("prompt_context_sessions is invalid")
        prompt_manifest = values.get("prompt_manifest")
        if prompt_manifest is not None and (
            not isinstance(prompt_manifest, dict)
            or prompt_manifest.get("version") != 1
            or not isinstance(prompt_manifest.get("phase"), str)
            or not isinstance(prompt_manifest.get("session_identity"), str)
            or not isinstance(prompt_manifest.get("full_rehydration"), bool)
            or not isinstance(prompt_manifest.get("total_chars"), int)
            or not isinstance(prompt_manifest.get("sha256"), str)
            or not isinstance(prompt_manifest.get("sections"), dict)
        ):
            raise JobValidationError("prompt_manifest is invalid")

        job_id = str(values.get("id") or "")
        if not re.fullmatch(r"[0-9a-f]{12}", job_id):
            raise JobValidationError("id must be 12 lowercase hexadecimal characters")
        parent_job_id = values.get("parent_job_id")
        if parent_job_id is not None:
            if not re.fullmatch(r"[0-9a-f]{12}", str(parent_job_id)):
                raise JobValidationError(
                    "parent_job_id must be 12 lowercase hexadecimal characters"
                )
            if parent_job_id == job_id:
                raise JobValidationError("parent_job_id must not reference the child")
        try:
            status = JobStatus(str(values.get("status") or ""))
        except ValueError as exc:
            raise JobValidationError("status has invalid value") from exc
        try:
            harness_kind = HarnessKind(str(values.get("harness_kind") or ""))
        except ValueError as exc:
            raise JobValidationError("harness_kind has invalid value") from exc
        issue_provider = values.get("issue_provider")
        if issue_provider is not None and not re.fullmatch(
            r"[a-z][a-z0-9-]*", str(issue_provider)
        ):
            raise JobValidationError("issue_provider has invalid value")
        try:
            workflow_phase = WorkflowPhase(
                str(values.get("workflow_phase") or WorkflowPhase.CLASSIFYING)
            )
        except ValueError as exc:
            raise JobValidationError("workflow_phase has invalid value") from exc
        try:
            workflow_tier = (
                WorkflowTier(str(values["workflow_tier"]))
                if values.get("workflow_tier") is not None
                else None
            )
        except ValueError as exc:
            raise JobValidationError("workflow_tier has invalid value") from exc
        try:
            active_participant = (
                WorkflowParticipant(str(values["active_participant"]))
                if values.get("active_participant") is not None
                else None
            )
        except ValueError as exc:
            raise JobValidationError("active_participant has invalid value") from exc

        required = ("revision", "request", "created_at", "delivered")
        missing = [field for field in required if field not in values]
        if missing:
            raise JobValidationError(f"job requires {', '.join(missing)}")
        if status == JobStatus.QUEUED and "queued_at" not in values:
            raise JobValidationError("queued job requires queued_at")
        if status == JobStatus.AWAITING_USER:
            for field in ("question", "result"):
                if not values.get(field):
                    raise JobValidationError(
                        f"awaiting_user job requires non-empty {field}"
                    )
        if status == JobStatus.BLOCKED and not values.get("result"):
            raise JobValidationError("blocked job requires result")
        if status == JobStatus.COMPLETED and "result" not in values:
            raise JobValidationError("completed job requires result")
        if status == JobStatus.FAILED:
            for field in ("error", "result"):
                if not values.get(field):
                    raise JobValidationError(f"failed job requires non-empty {field}")
        if status == JobStatus.CANCELLED and not values.get("result"):
            raise JobValidationError("cancelled job requires result")
        if status in TERMINAL_STATUSES | {JobStatus.BLOCKED} and (
            "completed_at" not in values
        ):
            raise JobValidationError(f"{status.value} job requires completed_at")

        prompt_operation = _optional_typed_prompt_operation(
            values,
            str(values["id"]),
            loaded_version=loaded_version,
        )
        checkout_operation = _optional_typed_checkout_operation(
            values, loaded_version=loaded_version
        )
        agent_session_operation = _optional_typed_agent_session_operation(
            values, loaded_version=loaded_version
        )
        if loaded_version < CURRENT_SCHEMA_VERSION:
            if checkout_operation is not None:
                values.update(checkout_fields(checkout_operation))
            if agent_session_operation is not None:
                values.update(agent_session_fields(agent_session_operation))
                if (
                    agent_session_operation.state == AgentSessionState.AMBIGUOUS
                    and values.get("agent_dispatch_state") == "ambiguous"
                    and agent_session_operation.session is None
                ):
                    values.update(
                        agent_provider=None,
                        agent_provider_session_id=None,
                        agent_state_sequence=None,
                    )

        job = cls(
            schema_version=CURRENT_SCHEMA_VERSION,
            loaded_schema_version=loaded_version,
            harness_kind=harness_kind,
            issue_provider=str(issue_provider) if issue_provider is not None else None,
            session_id=(
                str(values["session_id"])
                if values.get("session_id") is not None
                else None
            ),
            id=job_id,
            revision=_integer(values["revision"], "revision"),
            request=str(values["request"]),
            status=status,
            created_at=_number(values["created_at"], "created_at"),
            queued_at=(
                _number(values["queued_at"], "queued_at")
                if values.get("queued_at") is not None
                else None
            ),
            completed_at=(
                _number(values["completed_at"], "completed_at")
                if values.get("completed_at") is not None
                else None
            ),
            delivered=bool(values["delivered"]),
            worker_token=(
                str(values["worker_token"])
                if values.get("worker_token") is not None
                else None
            ),
            worker_pid=(
                _integer(values["worker_pid"], "worker_pid")
                if values.get("worker_pid") is not None
                else None
            ),
            worker_boot_id=(
                str(values["worker_boot_id"])
                if values.get("worker_boot_id") is not None
                else None
            ),
            worker_process_start=(
                str(values["worker_process_start"])
                if values.get("worker_process_start") is not None
                else None
            ),
            worker_operation=(
                str(values["worker_operation"])
                if values.get("worker_operation") is not None
                else None
            ),
            delivery_claim_token=(
                str(values["delivery_claim_token"])
                if values.get("delivery_claim_token") is not None
                else None
            ),
            delivery_claimed_at=(
                _number(values["delivery_claimed_at"], "delivery_claimed_at")
                if values.get("delivery_claimed_at") is not None
                else None
            ),
            herdr_target=(
                str(values["herdr_target"])
                if values.get("herdr_target") is not None
                else None
            ),
            worktree_path=(
                str(values["worktree_path"])
                if values.get("worktree_path") is not None
                else None
            ),
            target_release_pending=bool(values.get("target_release_pending", False)),
            target_release_manual_required=bool(
                values.get("target_release_manual_required", False)
            ),
            target_release_unverified_targets=tuple(unverified_targets),
            participant_session_owners=tuple(
                dict(owner) for owner in participant_session_owners
            ),
            cancellation_reconciliation_pending=bool(
                values.get("cancellation_reconciliation_pending", False)
            ),
            agent_dispatch_state=(
                str(values["agent_dispatch_state"])
                if values.get("agent_dispatch_state") is not None
                else None
            ),
            fork_operation_state=(
                str(values["fork_operation_state"])
                if values.get("fork_operation_state") is not None
                else None
            ),
            worktree_provision_state=(
                str(values["worktree_provision_state"])
                if values.get("worktree_provision_state") is not None
                else None
            ),
            manual_reconcile_operation=(
                str(values["manual_reconcile_operation"])
                if values.get("manual_reconcile_operation") is not None
                else None
            ),
            manual_reconcile_token=(
                str(values["manual_reconcile_token"])
                if values.get("manual_reconcile_token") is not None
                else None
            ),
            workflow_tier=workflow_tier,
            workflow_phase=workflow_phase,
            review_round=_integer(values.get("review_round") or 0, "review_round"),
            active_participant=active_participant,
            _compatibility_layout=compatibility_layout,
            _values=values,
            _prompt_operation=prompt_operation,
            _checkout_operation=checkout_operation,
            _agent_session_operation=agent_session_operation,
            _lifecycle_event=None,
        )
        job.validate_invariants(
            require_worker_owner=loaded_version == CURRENT_SCHEMA_VERSION
        )
        return job

    @classmethod
    def new(cls, spec: NewCursorJob) -> CursorJob:
        return cls.from_dict(
            {
                "id": spec.id,
                "parent_job_id": spec.parent_job_id,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "harness_kind": spec.harness_kind.value,
                "issue_provider": spec.issue_provider,
                "session_id": spec.session_id,
                "revision": 0,
                "request": spec.request,
                "utterance": spec.utterance,
                "trusted_utterance": spec.trusted_utterance,
                "repository_hint": spec.repository_hint,
                "context_repository": spec.context_repository,
                "repository": spec.repository,
                "github_repository": spec.github_repository,
                "github_issue": spec.github_issue,
                "github_issue_url": spec.github_issue_url,
                "github_issue_context": spec.github_issue_context,
                "github_issue_create_requested": spec.github_issue_create_requested,
                "github_issue_create_title": spec.github_issue_create_title,
                "github_issue_create_body": spec.github_issue_create_body,
                "github_issue_create_marker": spec.github_issue_create_marker,
                "github_repo_create_requested": spec.github_repo_create_requested,
                "github_repo_create_org_requested": (
                    spec.github_repo_create_org_requested
                ),
                "linear_ticket_create_requested": spec.linear_ticket_create_requested,
                "linear_ticket_create_team": spec.linear_ticket_create_team,
                "linear_ticket_create_team_id": spec.linear_ticket_create_team_id,
                "linear_ticket_create_title": spec.linear_ticket_create_title,
                "linear_ticket_create_description": (
                    spec.linear_ticket_create_description
                ),
                "linear_ticket_create_marker": spec.linear_ticket_create_marker,
                "fork_requested": spec.fork_requested,
                "github_pull_request": spec.github_pull_request,
                "worktree_branch": spec.worktree_branch,
                "worktree_label": spec.worktree_label,
                "worktree_path": spec.worktree_path,
                "worktree_workspace_id": spec.worktree_workspace_id,
                "worktree_root_pane_id": spec.worktree_root_pane_id,
                "worktree_provision_state": spec.worktree_provision_state,
                "pull_request_worktree_state": spec.pull_request_worktree_state,
                "agent_hint": spec.agent_hint,
                "issue_key": spec.issue_key,
                "speakable_label": spec.speakable_label,
                "status": JobStatus.QUEUED.value,
                "delivered": False,
                "created_at": spec.created_at,
                "queued_at": spec.created_at,
                "foreground_until": spec.foreground_until,
                "workflow_phase": WorkflowPhase.CLASSIFYING.value,
                "review_round": 0,
                "participant_admission_state": "waiting",
            }
        )

    def to_dict(self, *, preserve_loaded_version: bool = False) -> dict[str, object]:
        values = dict(self._values)
        if (
            preserve_loaded_version
            and self.loaded_schema_version < CURRENT_SCHEMA_VERSION
        ):
            values.pop("migration_source_schema_version", None)
            if self.loaded_schema_version == 0:
                values.pop("schema_version", None)
            else:
                values["schema_version"] = self.loaded_schema_version
        return values

    def to_record(self) -> dict[str, object]:
        """Return the structured, agent-neutral durable representation."""

        if self._compatibility_layout:
            record = dict(self._values)
            record.pop("harness_kind", None)
            record.pop("session_id", None)
            return record
        return _structured_record(self._values)

    def evolve(
        self,
        *,
        status: JobStatus | None = None,
        remove: frozenset[str] = frozenset(),
        **changes: object,
    ) -> CursorJob:
        values = self.to_dict()
        for field in remove:
            values.pop(field, None)
        _consume_typed_prompt_operation(changes, job_id=self.id)
        _consume_typed_checkout_operation(changes)
        _consume_typed_agent_session_operation(changes)
        values.update(changes)
        _pair_announcement_ack(values)
        _pair_worker_ownership(values)
        if "prompt_operation_state" in changes:
            values.pop("phase_prompt_active", None)
        if (
            values.get("prompt_operation_state") == "none"
            and values.get("manual_reconcile_operation") == "prompt"
        ) or (
            values.get("participant_creation_state") == "none"
            and values.get("manual_reconcile_operation") == "pane"
        ):
            values["manual_reconcile_operation"] = None
            values["manual_reconcile_token"] = None
        if "herdr_target" in changes and "session_id" not in changes:
            values["session_id"] = changes["herdr_target"]
        elif "session_id" in changes and "herdr_target" not in changes:
            values["herdr_target"] = changes["session_id"]
        values["status"] = (status or self.status).value
        values["schema_version"] = CURRENT_SCHEMA_VERSION
        values["revision"] = self.revision + 1
        updated = CursorJob.from_dict(values)
        validate_transition(self, updated)
        return updated

    def clear_worker(self) -> CursorJob:
        return self.evolve(
            worker_pid=None,
            worker_boot_id=None,
            worker_process_start=None,
            worker_token=None,
            worker_operation=None,
            worker_claim_operation=None,
            worker_claimed_at=None,
        )

    @property
    def worker_ownership(self) -> WorkerOwnership | None:
        try:
            return load_worker_ownership(
                token=self.worker_token,
                pid=self.worker_pid,
                boot_id=self.worker_boot_id,
                process_start=self.worker_process_start,
                operation=self._optional_string("worker_claim_operation"),
                claimed_at=self._optional_float("worker_claimed_at"),
            )
        except OperationTransitionError as exc:
            raise JobValidationError(str(exc)) from exc

    def prepare_delivery(self, *, now: float) -> CursorJob:
        return self.evolve_for_delivery(now=now)

    def evolve_for_delivery(
        self,
        *,
        now: float,
        status: JobStatus | None = None,
        remove: frozenset[str] = frozenset(),
        **changes: object,
    ) -> CursorJob:
        prepared = delivery_fields(transition_prepare_delivery(self.delivery_state))
        return self.evolve(
            status=status,
            remove=remove,
            **prepared,
            updated_at=now,
            **changes,
        )

    def evolve_recovery(
        self,
        dynamic_changes: Mapping[str, object] | None = None,
        *,
        now: float,
        status: JobStatus | None = None,
        remove: frozenset[str] = frozenset(),
        prepare_delivery: bool = False,
        **changes: object,
    ) -> CursorJob:
        values = self.to_dict()
        for field in remove:
            values.pop(field, None)
        if dynamic_changes is not None:
            mutable_dynamic = dict(dynamic_changes)
            _consume_typed_prompt_operation(mutable_dynamic, job_id=self.id)
            _consume_typed_checkout_operation(mutable_dynamic)
            _consume_typed_agent_session_operation(mutable_dynamic)
            values.update(mutable_dynamic)
            dynamic_changes = mutable_dynamic
        _consume_typed_prompt_operation(changes, job_id=self.id)
        _consume_typed_checkout_operation(changes)
        _consume_typed_agent_session_operation(changes)
        values.update(changes)
        _pair_worker_ownership(values)
        if "prompt_operation_state" in values:
            values.pop("phase_prompt_active", None)
        combined_changes = dict(dynamic_changes or {})
        combined_changes.update(changes)
        if "herdr_target" in combined_changes and "session_id" not in combined_changes:
            values["session_id"] = combined_changes["herdr_target"]
        elif (
            "session_id" in combined_changes and "herdr_target" not in combined_changes
        ):
            values["herdr_target"] = combined_changes["session_id"]
        final_status = status or self.status
        values["status"] = final_status.value
        operation = str(values.get("manual_reconcile_operation") or "")
        materialized = isinstance(self.terminal_state, MaterializedTerminalOutcome)
        if materialized:
            if operation or final_status == JobStatus.FAILED:
                values["reconciliation_base_error"] = str(
                    self.reconciliation_base_error
                    or self.error
                    or self.result
                    or "Cursor job failed"
                ).split("; external operation reconciliation", 1)[0]
        elif final_status in TERMINAL_STATUSES and operation:
            base = str(
                values.get("reconciliation_base_error")
                or (
                    values.get("error")
                    if final_status == JobStatus.FAILED
                    else values.get("result")
                )
                or "Cursor job failed"
            )
            base = base.split("; external operation reconciliation", 1)[0]
            base = base.split("; manual reconciliation required", 1)[0]
            if operation == "agent":
                resource = f"Herdr agent {values.get('herdr_target') or 'unknown'}"
            elif operation == "fork":
                resource = (
                    f"GitHub fork {values.get('fork_operation_target') or 'unknown'}"
                )
            elif operation == "pane":
                role = values.get("participant_creation_participant") or "workflow"
                target = values.get("participant_creation_target") or "unknown"
                resource = f"{role} pane target {target}"
            else:
                resource = f"worktree {values.get('worktree_path') or 'unknown'}"
            message = (f"{base}; manual reconciliation required for {resource}")[:500]
            prepare_delivery = prepare_delivery or (
                values.get("result") != message
                or (final_status == JobStatus.FAILED and values.get("error") != message)
            )
            values.update(
                reconciliation_base_error=base,
                result=message,
            )
            if final_status == JobStatus.FAILED:
                values["error"] = message
        elif final_status == JobStatus.FAILED:
            base = str(
                values.get("reconciliation_base_error")
                or values.get("error")
                or "Cursor job failed"
            )
            base = base.split("; external operation reconciliation", 1)[0]
            if any(
                values.get(field)
                in {"dispatching", "submitted", "ambiguous", "failed_observing"}
                for field in (
                    "agent_dispatch_state",
                    "fork_operation_state",
                    "worktree_provision_state",
                )
            ):
                message = (f"{base}; external operation reconciliation is pending")[
                    :500
                ]
            else:
                message = base[:500]
            prepare_delivery = prepare_delivery or (
                values.get("error") != message or values.get("result") != message
            )
            values.update(
                reconciliation_base_error=base,
                error=message,
                result=message,
            )
        if materialized:
            values["status"] = self.status.value
            for field in ("result", "error", "completed_at"):
                if field in self._values:
                    values[field] = self._values[field]
                else:
                    values.pop(field, None)
        if prepare_delivery:
            values.update(
                delivery_fields(transition_prepare_delivery(self.delivery_state))
            )
            values["updated_at"] = now
        values["schema_version"] = CURRENT_SCHEMA_VERSION
        values["revision"] = self.revision + 1
        updated = CursorJob.from_dict(values)
        updated = replace(
            updated,
            _lifecycle_event=RecoveryEvent(self.revision, updated.lifecycle),
        )
        validate_transition(self, updated)
        return updated

    def record_operation_observation(
        self,
        operation: str,
        state_key: str,
        expected_states: frozenset[str],
        *,
        now: float,
        observed_absent: bool,
        failed_max_attempts: int,
        uncertain_max_attempts: int,
        base_seconds: float,
        max_seconds: float,
    ) -> CursorJob | None:
        values = self.to_dict()
        state = str(values.get(state_key) or "")
        if state not in expected_states:
            return None
        attempts_key = f"{operation}_reconcile_attempts"
        absent_key = f"{operation}_absent_observations"
        attempts = _integer(values.get(attempts_key) or 0, attempts_key) + 1
        absent = _integer(values.get(absent_key) or 0, absent_key)
        changes: dict[str, object] = {
            attempts_key: attempts,
            f"{operation}_last_reconciled_at": now,
        }
        if observed_absent:
            absent += 1
            changes[absent_key] = absent
        can_confirm_absent = not (operation == "fork" and self.fork_committed)
        if (
            state == "failed_observing"
            and absent >= failed_max_attempts
            and can_confirm_absent
        ):
            if operation == "worktree" and self.checkout_operation is not None:
                changes.update(
                    checkout_fields(
                        self.checkout_operation.transition(
                            CheckoutState.CONFIRMED_ABSENT
                        )
                    )
                )
            elif operation == "agent" and self.agent_session_operation is not None:
                changes.update(
                    agent_session_fields(
                        self.agent_session_operation.transition(
                            AgentSessionState.CONFIRMED_ABSENT
                        )
                    )
                )
            else:
                changes[state_key] = "confirmed_absent"
            changes.update(
                {
                    f"{operation}_confirmed_absent_at": now,
                    f"{operation}_next_reconcile_at": None,
                    "worker_operation": None,
                }
            )
            changes.update(
                cleanup_fields(finish_cleanup_reconciliation(self.cleanup_state))
            )
            if operation == "agent":
                participant = self.active_participant
                changes.update(
                    herdr_target=None,
                    active_participant=None,
                    herdr_pane_id=None,
                    herdr_workspace_id=None,
                    agent_name=None,
                    agent_dispatch_exited=None,
                )
                if participant is not None:
                    changes[f"{participant.value}_target"] = None
        elif attempts >= uncertain_max_attempts:
            if operation == "worktree" and self.checkout_operation is not None:
                changes.update(
                    checkout_fields(
                        self.checkout_operation.transition(
                            CheckoutState.MANUAL_REQUIRED
                        )
                    )
                )
            elif operation == "agent" and self.agent_session_operation is not None:
                changes.update(
                    agent_session_fields(
                        self.agent_session_operation.transition(
                            AgentSessionState.MANUAL_REQUIRED
                        )
                    )
                )
            else:
                changes[state_key] = "manual_required"
            changes.update(
                {
                    f"{operation}_next_reconcile_at": None,
                    f"{operation}_automatic_reconcile_stopped_at": now,
                    "manual_reconcile_operation": operation,
                    "manual_reconcile_token": uuid.uuid4().hex,
                    "manual_reconcile_required_at": now,
                    "worker_operation": None,
                }
            )
            changes.update(
                cleanup_fields(finish_cleanup_reconciliation(self.cleanup_state))
            )
        else:
            changes[f"{operation}_next_reconcile_at"] = now + min(
                max_seconds, base_seconds * (2 ** (attempts - 1))
            )
        return self.evolve_recovery(
            changes,
            now=now,
            prepare_delivery=(
                changes.get(state_key) == "manual_required"
                and self.terminal_intent_status is None
            ),
        )

    def _optional_string(self, field: str) -> str | None:
        value = self._values.get(field)
        return str(value) if value is not None else None

    def _optional_int(self, field: str) -> int | None:
        value = self._values.get(field)
        return _integer(value, field) if value is not None else None

    def _optional_float(self, field: str) -> float | None:
        value = self._values.get(field)
        return _number(value, field) if value is not None else None

    def _boolean_field(self, field: str) -> bool:
        return bool(self._values.get(field, False))

    @property
    def result(self) -> str | None:
        value = self._values.get("result")
        return str(value) if value is not None else None

    @property
    def error(self) -> str | None:
        value = self._values.get("error")
        return str(value) if value is not None else None

    @property
    def question(self) -> str | None:
        value = self._values.get("question")
        return str(value) if value is not None else None

    @property
    def voice_question(self) -> dict[str, object] | None:
        value = self._values.get("voice_question")
        return dict(value) if isinstance(value, dict) else None

    @property
    def continuation_answer(self) -> str | None:
        return self._optional_string("continuation_answer")

    @property
    def clarifications(self) -> tuple[dict[str, object], ...]:
        value = self._values.get("clarifications")
        if not isinstance(value, list):
            return ()
        return tuple(dict(item) for item in value if isinstance(item, dict))

    @property
    def prompt_context_sessions(self) -> dict[str, str]:
        value = self._values.get("prompt_context_sessions")
        if not isinstance(value, dict):
            return {}
        return {
            str(key): str(session)
            for key, session in value.items()
            if isinstance(key, str) and isinstance(session, str)
        }

    @property
    def prompt_manifest(self) -> dict[str, object] | None:
        value = self._values.get("prompt_manifest")
        return dict(value) if isinstance(value, dict) else None

    @property
    def interactive_questionnaire_blocked(self) -> bool:
        return self._boolean_field("interactive_questionnaire_blocked")

    @property
    def utterance(self) -> str | None:
        return self._optional_string("utterance")

    @property
    def trusted_utterance(self) -> str | None:
        return self._optional_string("trusted_utterance")

    @property
    def repository_hint(self) -> str | None:
        return self._optional_string("repository_hint")

    @property
    def grouped_repository_targets(self) -> list[dict[str, object]] | None:
        value = self._values.get("grouped_repository_targets")
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            return None
        return [dict(item) for item in value]

    @property
    def grouped_repository_candidates(self) -> tuple[str, ...] | None:
        value = self._values.get("grouped_repository_candidates")
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            return None
        return tuple(value)

    @property
    def grouped_repository_launches(self) -> list[dict[str, object]]:
        value = self._values.get("grouped_repository_launches", [])
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            return []
        return [dict(item) for item in value]

    @property
    def context_repository(self) -> str | None:
        return self._optional_string("context_repository")

    @property
    def parent_job_id(self) -> str | None:
        return self._optional_string("parent_job_id")

    @property
    def repository(self) -> str | None:
        return self._optional_string("repository")

    @property
    def github_repository(self) -> str | None:
        return self._optional_string("github_repository")

    @property
    def github_issue(self) -> int | None:
        return self._optional_int("github_issue")

    @property
    def github_issue_url(self) -> str | None:
        return self._optional_string("github_issue_url")

    @property
    def github_issue_context(self) -> str | None:
        return self._optional_string("github_issue_context")

    @property
    def github_issue_create_requested(self) -> bool:
        return self._boolean_field("github_issue_create_requested")

    @property
    def github_issue_create_confirmed(self) -> bool:
        return self._boolean_field("github_issue_create_confirmed")

    @property
    def github_issue_create_title(self) -> str | None:
        return self._optional_string("github_issue_create_title")

    @property
    def github_issue_create_body(self) -> str | None:
        return self._optional_string("github_issue_create_body")

    @property
    def github_issue_create_marker(self) -> str | None:
        return self._optional_string("github_issue_create_marker")

    @property
    def github_issue_create_operation_state(self) -> str | None:
        return self._optional_string("github_issue_create_operation_state")

    @property
    def github_issue_created_number(self) -> int | None:
        if not self.github_issue_create_requested:
            return None
        return self.github_issue

    @property
    def github_issue_created_url(self) -> str | None:
        if not self.github_issue_create_requested:
            return None
        return self.github_issue_url

    @property
    def github_repo_create_requested(self) -> bool:
        return self._boolean_field("github_repo_create_requested")

    @property
    def github_repo_create_org_requested(self) -> bool:
        return self._boolean_field("github_repo_create_org_requested")

    @property
    def github_repo_create_owner(self) -> str | None:
        return self._optional_string("github_repo_create_owner")

    @property
    def github_repo_create_confirmed(self) -> bool:
        return self._boolean_field("github_repo_create_confirmed")

    @property
    def github_repo_create_visibility(self) -> str | None:
        return self._optional_string("github_repo_create_visibility")

    @property
    def github_repo_create_marker(self) -> str | None:
        return self._optional_string("github_repo_create_marker")

    @property
    def github_repo_create_operation_state(self) -> str | None:
        return self._optional_string("github_repo_create_operation_state")

    @property
    def github_repo_created_url(self) -> str | None:
        return self._optional_string("github_repo_created_url")

    @property
    def linear_ticket_create_requested(self) -> bool:
        return self._boolean_field("linear_ticket_create_requested")

    @property
    def linear_ticket_create_confirmed(self) -> bool:
        return self._boolean_field("linear_ticket_create_confirmed")

    @property
    def linear_ticket_create_team(self) -> str | None:
        return self._optional_string("linear_ticket_create_team")

    @property
    def linear_ticket_create_team_id(self) -> str | None:
        return self._optional_string("linear_ticket_create_team_id")

    @property
    def linear_ticket_create_title(self) -> str | None:
        return self._optional_string("linear_ticket_create_title")

    @property
    def linear_ticket_create_description(self) -> str | None:
        return self._optional_string("linear_ticket_create_description")

    @property
    def linear_ticket_create_marker(self) -> str | None:
        return self._optional_string("linear_ticket_create_marker")

    @property
    def linear_ticket_create_operation_state(self) -> str | None:
        return self._optional_string("linear_ticket_create_operation_state")

    @property
    def linear_ticket_create_prompt_target(self) -> str | None:
        return self._optional_string("linear_ticket_create_prompt_target")

    @property
    def linear_ticket_create_prompt_session(self) -> str | None:
        return self._optional_string("linear_ticket_create_prompt_session")

    @property
    def linear_ticket_create_prompt_token(self) -> str | None:
        return self._optional_string("linear_ticket_create_prompt_token")

    @property
    def linear_ticket_create_baseline_sequence(self) -> int | None:
        return self._optional_int("linear_ticket_create_baseline_sequence")

    @property
    def linear_ticket_created_identifier(self) -> str | None:
        return self._optional_string("linear_ticket_created_identifier")

    @property
    def linear_ticket_created_url(self) -> str | None:
        return self._optional_string("linear_ticket_created_url")

    @property
    def github_pull_request(self) -> int | None:
        return self._optional_int("github_pull_request")

    @property
    def fork_requested(self) -> bool:
        return self._boolean_field("fork_requested")

    @property
    def fork_confirmed(self) -> bool:
        return self._boolean_field("fork_confirmed")

    @property
    def clone_confirmed(self) -> bool:
        return self._boolean_field("clone_confirmed")

    @property
    def clone_source(self) -> str | None:
        return self._optional_string("clone_source")

    @property
    def clone_operation_state(self) -> str | None:
        return self._optional_string("clone_operation_state")

    @property
    def fork_committed(self) -> bool:
        return self._boolean_field("fork_committed")

    @property
    def fork_exists(self) -> bool | None:
        value = self._values.get("fork_exists")
        return bool(value) if value is not None else None

    @property
    def worktree_branch(self) -> str | None:
        return self._optional_string("worktree_branch")

    @property
    def worktree_label(self) -> str | None:
        return self._optional_string("worktree_label")

    @property
    def worktree_workspace_id(self) -> str | None:
        return self._optional_string("worktree_workspace_id")

    @property
    def worktree_root_pane_id(self) -> str | None:
        return self._optional_string("worktree_root_pane_id")

    @property
    def checkout_operation(self) -> CheckoutOperation | None:
        if self._checkout_operation is not None:
            return self._checkout_operation
        if self.worktree_provision_state is None:
            return None
        return _typed_checkout_operation(
            self._values, loaded_version=self.loaded_schema_version
        )

    @property
    def worktree_provision_error(self) -> str | None:
        return self._optional_string("worktree_provision_error")

    @property
    def worktree_manual_inspection_required(self) -> bool:
        return self._boolean_field("worktree_manual_inspection_required")

    @property
    def pull_request_worktree_state(self) -> str | None:
        return self._optional_string("pull_request_worktree_state")

    @property
    def pull_request_branch(self) -> str | None:
        if self.github_pull_request is None:
            return None
        if self.pull_request_worktree_state not in {"ready", "retained"}:
            return None
        return self.worktree_branch

    @property
    def pull_request_worktree_error(self) -> str | None:
        return self._optional_string("pull_request_worktree_error")

    @property
    def pull_request_remote_url(self) -> str | None:
        return self._optional_string("pull_request_remote_url")

    @property
    def pull_request_head_ref(self) -> str | None:
        return self._optional_string("pull_request_head_ref")

    @property
    def pull_request_head_oid(self) -> str | None:
        return self._optional_string("pull_request_head_oid")

    @property
    def agent_hint(self) -> str | None:
        return self._optional_string("agent_hint")

    @property
    def agent_name(self) -> str | None:
        return self._optional_string("agent_name")

    @property
    def issue_key(self) -> str | None:
        return self._optional_string("issue_key")

    @property
    def speakable_label(self) -> str | None:
        return self._optional_string("speakable_label")

    @property
    def announcement_ack(self) -> str:
        value = self._optional_string("announcement_ack")
        if value in ANNOUNCEMENT_ACK_STATES:
            return value
        if self.delivered:
            return (
                AnnouncementAck.DISMISSED.value
                if self._values.get("announcement_dismissed")
                else AnnouncementAck.SPOKEN.value
            )
        return AnnouncementAck.PENDING.value

    @property
    def announcement_dismissed(self) -> bool:
        return self.announcement_ack == AnnouncementAck.DISMISSED.value

    @property
    def announcement_repeated(self) -> bool:
        return self._boolean_field("announcement_repeated")

    @property
    def delivered_at(self) -> float | None:
        return self._optional_float("delivered_at")

    @property
    def clarification_kind(self) -> str | None:
        return self._optional_string("clarification_kind")

    @property
    def turn(self) -> int:
        return _integer(self._values.get("turn") or 0, "turn")

    @property
    def turn_token(self) -> str | None:
        return self._optional_string("turn_token")

    @property
    def continuation(self) -> bool:
        return self._boolean_field("continuation")

    @property
    def reconcile(self) -> bool:
        return self._boolean_field("reconcile")

    @property
    def herdr_pane_id(self) -> str | None:
        return self._optional_string("herdr_pane_id")

    @property
    def herdr_workspace_id(self) -> str | None:
        return self._optional_string("herdr_workspace_id")

    @property
    def agent_provider(self) -> str | None:
        return self._optional_string("agent_provider")

    @property
    def agent_provider_session_id(self) -> str | None:
        return self._optional_string("agent_provider_session_id")

    @property
    def agent_state_sequence(self) -> int | None:
        return self._optional_int("agent_state_sequence")

    @property
    def session_control(self) -> str:
        return (
            self._optional_string("session_control")
            or SessionControlMode.AUTOMATED.value
        )

    @property
    def session_control_generation(self) -> int:
        return self._optional_int("session_control_generation") or 0

    def session_control_state(self) -> SessionControlState:
        try:
            mode = SessionControlMode(self.session_control)
        except ValueError as exc:
            raise JobValidationError("session_control has invalid value") from exc
        return SessionControlState(mode, self.session_control_generation)

    @property
    def agent_operation_checkout(self) -> str | None:
        return self._optional_string("agent_operation_checkout")

    @property
    def agent_operation_target(self) -> str | None:
        return self._optional_string("agent_operation_target")

    @property
    def agent_operation_workspace_id(self) -> str | None:
        return self._optional_string("agent_operation_workspace_id")

    @property
    def agent_operation_pane_id(self) -> str | None:
        return self._optional_string("agent_operation_pane_id")

    @property
    def agent_identity_legacy_compatible(self) -> bool:
        return bool(self._values.get("agent_identity_legacy_compatible", False))

    @property
    def agent_session_operation(self) -> AgentSessionOperation | None:
        if self._agent_session_operation is not None:
            return self._agent_session_operation
        if self.agent_dispatch_state is None:
            return None
        return _typed_agent_session_operation(
            self._values, loaded_version=self.loaded_schema_version
        )

    @property
    def delivery_generation(self) -> int:
        return _integer(
            self._values.get("delivery_generation") or 0, "delivery_generation"
        )

    @property
    def target_release_token(self) -> str | None:
        return self._optional_string("target_release_token")

    @property
    def target_release_owner_pid(self) -> int | None:
        return self._optional_int("target_release_owner_pid")

    @property
    def target_release_owner_boot_id(self) -> str | None:
        return self._optional_string("target_release_owner_boot_id")

    @property
    def target_release_owner_start(self) -> str | None:
        return self._optional_string("target_release_owner_start")

    @property
    def reconciliation_base_error(self) -> str | None:
        return self._optional_string("reconciliation_base_error")

    @property
    def workflow_classification_reason(self) -> str | None:
        return self._optional_string("workflow_classification_reason")

    @property
    def workflow_state(self) -> WorkflowState:
        try:
            classification = (
                WorkflowClassification(
                    self.workflow_tier,
                    self.workflow_classification_reason or "",
                )
                if self.workflow_tier is not None
                else None
            )
            return WorkflowState(self.workflow_phase, classification)
        except WorkflowTransitionError as exc:
            raise JobValidationError(str(exc)) from exc

    def evolve_workflow(
        self,
        state: WorkflowState,
        **changes: Any,
    ) -> CursorJob:
        classification = state.classification
        return self.evolve(
            workflow_phase=state.phase.value,
            workflow_tier=(
                classification.tier.value if classification is not None else None
            ),
            workflow_classification_reason=(
                classification.reason if classification is not None else None
            ),
            **changes,
        )

    @property
    def plan_artifact(self) -> str | None:
        return self._optional_string("plan_artifact")

    @property
    def review_artifact(self) -> str | None:
        return self._optional_string("review_artifact")

    @property
    def prompt_operation_state(self) -> str:
        return self._optional_string("prompt_operation_state") or "none"

    @property
    def prompt_operation(self) -> PromptOperation:
        if self._prompt_operation is not None:
            return self._prompt_operation
        return _typed_prompt_operation(self._values, self.id)

    @property
    def prompt_operation_phase(self) -> WorkflowPhase | None:
        value = self._optional_string("prompt_operation_phase")
        if value is None:
            return None
        try:
            return WorkflowPhase(value)
        except ValueError as exc:
            raise JobValidationError(
                "prompt_operation_phase has invalid value"
            ) from exc

    @property
    def prompt_operation_turn(self) -> int:
        return _integer(
            self._values.get("prompt_operation_turn") or 0,
            "prompt_operation_turn",
        )

    @property
    def prompt_baseline_sequence(self) -> int | None:
        return self._optional_int("prompt_baseline_sequence")

    @property
    def prompt_operation_target(self) -> str | None:
        return self._optional_string("prompt_operation_target")

    @property
    def prompt_operation_agent_session(self) -> str | None:
        return self._optional_string("prompt_operation_agent_session")

    @property
    def participant_creation_state(self) -> str:
        return self._optional_string("participant_creation_state") or "none"

    @property
    def participant_admission_state(self) -> str:
        return self._optional_string("participant_admission_state") or "waiting"

    @property
    def participant_creation_target(self) -> str | None:
        return self._optional_string("participant_creation_target")

    @property
    def participant_creation_participant(self) -> WorkflowParticipant | None:
        value = self._optional_string("participant_creation_participant")
        if value is None:
            return None
        try:
            return WorkflowParticipant(value)
        except ValueError as exc:
            raise JobValidationError(
                "participant_creation_participant has invalid value"
            ) from exc

    @property
    def participant_creation_label(self) -> str | None:
        return self._optional_string("participant_creation_label")

    @property
    def participant_creation_workspace_id(self) -> str | None:
        return self._optional_string("participant_creation_workspace_id")

    @property
    def participant_creation_pane_id(self) -> str | None:
        return self._optional_string("participant_creation_pane_id")

    @property
    def participant_creation_checkout(self) -> str | None:
        return self._optional_string("participant_creation_checkout")

    @property
    def participant_pane_operation(self) -> ParticipantPaneOperation | None:
        if self.participant_creation_state == "none":
            return None
        state = {
            "planned": OperationState.PLANNED,
            "submitting": OperationState.ACTIVE,
            "created": OperationState.SETTLED,
            "ambiguous": OperationState.UNKNOWN,
            "failed": OperationState.FAILED,
            "manual_required": OperationState.MANUAL,
        }[self.participant_creation_state]
        participant = self.participant_creation_participant
        try:
            return ParticipantPaneOperation(
                state,
                ParticipantPaneSpec(
                    participant.value if participant is not None else "",
                    self.participant_creation_target or "",
                    self.participant_creation_label or "",
                    self.participant_creation_checkout
                    or self.worktree_path
                    or self.repository
                    or "",
                    self.participant_creation_workspace_id or "",
                ),
                self.participant_creation_pane_id,
            )
        except OperationTransitionError as exc:
            raise JobValidationError(str(exc)) from exc

    @property
    def terminal_intent_status(self) -> JobStatus | None:
        value = self._optional_string("terminal_intent_status")
        if value is None:
            return None
        try:
            return JobStatus(value)
        except ValueError as exc:
            raise JobValidationError(
                "terminal_intent_status has invalid value"
            ) from exc

    @property
    def terminal_intent_result(self) -> str | None:
        return self._optional_string("terminal_intent_result")

    @property
    def terminal_intent_error(self) -> str | None:
        return self._optional_string("terminal_intent_error")

    @property
    def terminal_intent_completed_at(self) -> float | None:
        return self._optional_float("terminal_intent_completed_at")

    @property
    def terminal_state(self) -> TerminalState:
        try:
            return load_terminal_state(
                status=self.status.value,
                result=self.result,
                error=self.error,
                completed_at=self.completed_at,
                intent_status=(
                    self.terminal_intent_status.value
                    if self.terminal_intent_status is not None
                    else None
                ),
                intent_result=self.terminal_intent_result,
                intent_error=self.terminal_intent_error,
                intent_completed_at=self.terminal_intent_completed_at,
            )
        except LifecycleTransitionError as exc:
            raise JobValidationError(str(exc)) from exc

    @property
    def lifecycle(self) -> JobLifecycle:
        """Adapt the flat compatibility record to the active typed lifecycle."""

        try:
            identity = JobIdentity(
                self.id,
                self.created_at,
                self.parent_job_id,
                self.request,
                self.harness_kind.value,
                self.issue_provider,
                self.revision,
            )
            worker = (
                WorkerClaim(
                    ownership.token,
                    ownership.pid,
                    ownership.boot_id,
                    ownership.process_start,
                    ownership.operation,
                    ownership.claimed_at,
                )
                if (ownership := self.worker_ownership) is not None
                else None
            )
            try:
                prompt_operation = self.prompt_operation
            except JobValidationError:
                # The flat adapter still accepts a narrow set of historical
                # current-schema records whose prompt session was not stored.
                # They remain compatibility records, not typed prompt states.
                prompt_operation = None
            execution = ExecutionComponents(
                prompt_operation,
                self.workflow_state,
            )
            state = JobState(self.status.value)
            if state == JobState.QUEUED:
                if self.queued_at is None:
                    raise JobLifecycleError("queued job requires a finite queued time")
                return QueuedJob(identity, self.queued_at, execution, worker)
            if state == JobState.ROUTING:
                if worker is None:
                    raise JobLifecycleError("routing job requires worker identity")
                return RoutingProvisioningJob(identity, worker, execution)
            if state == JobState.RUNNING:
                if worker is None:
                    raise JobLifecycleError("running job requires worker identity")
                return RunningJob(identity, worker, execution)
            if state == JobState.AWAITING_USER:
                return AwaitingUserJob(
                    identity, self.question or "", self.result or "", execution
                )
            if state == JobState.BLOCKED:
                if self.completed_at is None:
                    raise JobLifecycleError("blocked job requires blocked time")
                return BlockedJob(
                    identity, self.result or "", self.completed_at, execution
                )
            if state == JobState.RECONCILING:
                terminal_state = self.terminal_state
                intent = (
                    terminal_state
                    if isinstance(terminal_state, TerminalIntent)
                    else None
                )
                return ReconcilingJob(
                    identity,
                    execution,
                    self.cleanup_state,
                    intent,
                    worker,
                )
            if self.completed_at is None:
                raise JobLifecycleError("terminal job requires completion time")
            terminal_state = self.terminal_state
            if not isinstance(terminal_state, MaterializedTerminalOutcome):
                raise JobLifecycleError("terminal job requires materialized outcome")
            return TerminalJob(identity, execution, terminal_state)
        except (JobLifecycleError, OperationTransitionError, ValueError) as exc:
            raise JobValidationError(str(exc)) from exc

    def validate_lifecycle_event(
        self,
        updated: CursorJob,
        event: JobEvent,
    ) -> None:
        """Validate a typed event against this exact persisted revision."""

        if (
            self.loaded_schema_version != CURRENT_SCHEMA_VERSION
            or updated.loaded_schema_version != CURRENT_SCHEMA_VERSION
        ):
            # Compatibility imports may lack worker/session evidence that the
            # active v18 constructors require. Existing transition validation
            # remains the fail-closed boundary until such a row is normalized.
            return
        try:
            if event.next_state != updated.lifecycle:
                raise JobLifecycleError("lifecycle event payload does not match update")
            apply_event(self.lifecycle, event)
        except JobLifecycleError as exc:
            raise JobValidationError(str(exc)) from exc

    @property
    def cleanup_state(self) -> CleanupState:
        try:
            return load_cleanup_state(
                pending=self.target_release_pending,
                token=self.target_release_token,
                owner_pid=self.target_release_owner_pid,
                owner_boot_id=self.target_release_owner_boot_id,
                owner_start=self.target_release_owner_start,
                reconciliation_pending=self.cancellation_reconciliation_pending,
            )
        except LifecycleTransitionError as exc:
            raise JobValidationError(str(exc)) from exc

    @property
    def review_approved(self) -> bool:
        return self.review_approval_source is not None

    @property
    def review_decision(self) -> str | None:
        return self._optional_string("review_decision")

    @property
    def review_approval_source(self) -> str | None:
        return self._optional_string("review_approval_source")

    @property
    def plan_approval_state(self) -> str:
        return self._optional_string("plan_approval_state") or "none"

    @property
    def plan_approval_id(self) -> str | None:
        return self._optional_string("plan_approval_id")

    @property
    def plan_approval_source(self) -> str | None:
        return self._optional_string("plan_approval_source")

    @property
    def plan_approval_agent_session(self) -> str | None:
        return self._optional_string("plan_approval_agent_session")

    @property
    def plan_approval_plan_artifact(self) -> str | None:
        return self._optional_string("plan_approval_plan_artifact")

    @property
    def plan_approval_review_artifact(self) -> str | None:
        return self._optional_string("plan_approval_review_artifact")

    @property
    def plan_approval_state_change_sequence(self) -> int | None:
        return self._optional_int("plan_approval_state_change_sequence")

    @property
    def plan_approval_revision(self) -> int | None:
        return self._optional_int("plan_approval_revision")

    @property
    def plan_approval_counted(self) -> bool:
        return self._boolean_field("plan_approval_counted")

    @property
    def plan_approval_completion_pending(self) -> bool:
        return self._boolean_field("plan_approval_completion_pending")

    @property
    def review_state(self) -> ReviewState:
        try:
            plan = (
                ArtifactReference.parse(
                    self.plan_artifact,
                    job_id=self.id,
                    kind="plan",
                )
                if self.plan_artifact is not None
                else None
            )
            review = (
                ArtifactReference.parse(
                    self.review_artifact,
                    job_id=self.id,
                    kind="review",
                )
                if self.review_artifact is not None
                else None
            )
            return ReviewState(
                self.workflow_tier,
                self.review_round,
                plan,
                review,
                (
                    ReviewDecision(self.review_decision)
                    if self.review_decision is not None
                    else None
                ),
                (
                    ReviewApprovalSource(self.review_approval_source)
                    if self.review_approval_source is not None
                    else None
                ),
            )
        except (ValueError, WorkflowTransitionError) as exc:
            raise JobValidationError(str(exc)) from exc

    def evolve_review(
        self,
        review: ReviewState,
        **changes: Any,
    ) -> CursorJob:
        return self.evolve(
            review_round=review.round,
            plan_artifact=review.plan.value if review.plan is not None else None,
            review_artifact=(
                review.review.value if review.review is not None else None
            ),
            review_decision=(
                review.decision.value if review.decision is not None else None
            ),
            review_approval_source=(
                review.approval_source.value
                if review.approval_source is not None
                else None
            ),
            **changes,
        )

    @property
    def plan_approval(self) -> PlanApproval:
        try:
            state = PlanApprovalState(self.plan_approval_state)
            source = (
                PlanApprovalSource(self.plan_approval_source)
                if self.plan_approval_source is not None
                else None
            )
            proof_values = (
                self.plan_approval_id,
                self.plan_approval_agent_session,
                self.plan_approval_state_change_sequence,
                self.plan_approval_revision,
            )
            if state != PlanApprovalState.NONE and not all(
                value is not None for value in proof_values
            ):
                raise WorkflowTransitionError(
                    "active plan approval requires gate ID, agent session, and "
                    "sequence; revision proof is also required"
                )
            proof = None
            if state != PlanApprovalState.NONE:
                assert self.plan_approval_id is not None
                assert self.plan_approval_agent_session is not None
                assert self.plan_approval_state_change_sequence is not None
                assert self.plan_approval_revision is not None
                if source == PlanApprovalSource.LEGACY:
                    proof = LegacyPlanApprovalProof(
                        self.plan_approval_id,
                        self.plan_approval_agent_session,
                        self.plan_approval_state_change_sequence,
                        self.plan_approval_revision,
                    )
                else:
                    proof = PlanApprovalProof(
                        self.plan_approval_id,
                        self.plan_approval_agent_session,
                        self.plan_approval_state_change_sequence,
                        self.plan_approval_revision,
                    )
            question = (
                Question.from_dict(self.voice_question)
                if state == PlanApprovalState.AWAITING
                and self.voice_question is not None
                else None
            )
            approval = PlanApproval(
                state=state,
                proof=proof,
                source=source,
                counted=self.plan_approval_counted,
                plan_reference=self.plan_approval_plan_artifact,
                review_reference=self.plan_approval_review_artifact,
                review_accepted=self.review_approved,
                question_id=question.id if question is not None else None,
                question_turn_token=(
                    question.origin.turn_token if question is not None else None
                ),
            )
            if state in {
                PlanApprovalState.AWAITING,
                PlanApprovalState.APPROVED,
                PlanApprovalState.OBSERVED,
            } and (
                approval.plan_reference != self.plan_artifact
                or approval.review_reference != self.review_artifact
            ):
                raise WorkflowTransitionError(
                    "accepted plan approval artifacts cannot be replaced"
                )
            return approval
        except (ValueError, WorkflowTransitionError, QuestionError) as exc:
            raise JobValidationError(str(exc)) from exc

    def evolve_plan_approval(
        self,
        approval: PlanApproval,
        **changes: Any,
    ) -> CursorJob:
        proof = approval.proof
        return self.evolve(
            plan_approval_state=approval.state.value,
            plan_approval_id=proof.gate_id if proof is not None else None,
            plan_approval_source=(
                approval.source.value if approval.source is not None else None
            ),
            plan_approval_agent_session=(
                proof.agent_session if proof is not None else None
            ),
            plan_approval_plan_artifact=approval.plan_reference,
            plan_approval_review_artifact=approval.review_reference,
            plan_approval_state_change_sequence=(
                proof.state_change_sequence if proof is not None else None
            ),
            plan_approval_revision=proof.revision if proof is not None else None,
            plan_approval_counted=approval.counted,
            **changes,
        )

    @property
    def participant_lifecycle(self) -> ParticipantLifecycle:
        try:
            creation = ParticipantCreation(
                ParticipantCreationState(self.participant_creation_state),
                self.participant_creation_participant,
                self.participant_creation_target,
                self.participant_creation_label,
                self.participant_creation_workspace_id,
                self.participant_creation_pane_id,
            )
            return ParticipantLifecycle(
                ParticipantAdmissionState(self.participant_admission_state),
                creation,
            )
        except (ValueError, WorkflowTransitionError) as exc:
            raise JobValidationError(str(exc)) from exc

    def evolve_participant(
        self,
        lifecycle: ParticipantLifecycle,
        **changes: Any,
    ) -> CursorJob:
        creation = lifecycle.creation
        return self.evolve(
            participant_admission_state=lifecycle.admission.value,
            participant_creation_state=creation.state.value,
            participant_creation_participant=(
                creation.participant.value if creation.participant is not None else None
            ),
            participant_creation_target=creation.target,
            participant_creation_label=creation.label,
            participant_creation_workspace_id=creation.workspace_id,
            participant_creation_pane_id=creation.pane_id,
            **changes,
        )

    @property
    def workflow_turn_phase(self) -> WorkflowPhase | None:
        value = self._optional_string("workflow_turn_phase")
        if value is None:
            return None
        try:
            return WorkflowPhase(value)
        except ValueError as exc:
            raise JobValidationError("workflow_turn_phase has invalid value") from exc

    def participant_target(self, participant: WorkflowParticipant) -> str | None:
        return self._optional_string(f"{participant.value}_target")

    @property
    def fork_operation_source(self) -> str | None:
        return self._optional_string("fork_operation_source")

    @property
    def fork_operation_source_url(self) -> str | None:
        return self._optional_string("fork_operation_source_url")

    @property
    def fork_operation_source_parent(self) -> str | None:
        return self._optional_string("fork_operation_source_parent")

    @property
    def fork_operation_source_default_branch(self) -> str | None:
        return self._optional_string("fork_operation_source_default_branch")

    @property
    def fork_operation_login(self) -> str | None:
        return self._optional_string("fork_operation_login")

    @property
    def fork_operation_source_private(self) -> bool:
        return self._boolean_field("fork_operation_source_private")

    @property
    def fork_operation_target(self) -> str | None:
        return self._optional_string("fork_operation_target")

    @property
    def fork_repository(self) -> str | None:
        return self._optional_string("fork_repository")

    @property
    def fork_operation(self) -> ForkOperation | None:
        if self.fork_operation_state is None:
            return None
        state = {
            "planned": OperationState.PLANNED,
            "submitted": OperationState.ACTIVE,
            "exists": OperationState.SETTLED,
            "retained": OperationState.SETTLED,
            "ambiguous": OperationState.UNKNOWN,
            "failed_observing": OperationState.UNKNOWN,
            "failed": OperationState.FAILED,
            "confirmed_absent": OperationState.FAILED,
            "manual_required": OperationState.MANUAL,
        }[self.fork_operation_state]
        try:
            return ForkOperation(
                state,
                ForkSpec(
                    self.fork_operation_source or "",
                    self.fork_operation_source_url or "",
                    self.fork_operation_source_default_branch or "",
                    self.fork_operation_source_private,
                    self.fork_operation_login
                    or (self.fork_operation_target or "").split("/", 1)[0],
                    self.fork_operation_target or "",
                    self.fork_operation_source_parent,
                ),
            )
        except OperationTransitionError as exc:
            raise JobValidationError(str(exc)) from exc

    @property
    def next_reconcile_at(self) -> float:
        return _number(self._values.get("next_reconcile_at") or 0, "next_reconcile_at")

    def operation_reconcile_at(self, operation: str) -> float:
        field = f"{operation}_next_reconcile_at"
        return _number(self._values.get(field) or 0, field)

    def operation_reconcile_attempts(self, operation: str) -> int:
        field = f"{operation}_reconcile_attempts"
        return _integer(self._values.get(field) or 0, field)

    def operation_absent_observations(self, operation: str) -> int:
        field = f"{operation}_absent_observations"
        return _integer(self._values.get(field) or 0, field)

    def operation_state(self, operation: str) -> str | None:
        states = {
            "agent": self.agent_dispatch_state,
            "fork": self.fork_operation_state,
            "issue_create": self.github_issue_create_operation_state,
            "linear_ticket_create": self.linear_ticket_create_operation_state,
            "worktree": self.worktree_provision_state,
            "prompt": self.prompt_operation_state,
            "pane": self.participant_creation_state,
        }
        return states.get(operation)

    def resolve_manual_operation(
        self,
        operation: str,
        outcome: str,
        *,
        resolved_at: float,
        pane_id: str | None = None,
        workspace_id: str | None = None,
        job_changes: Mapping[str, object] | None = None,
    ) -> CursorJob | None:
        retain_terminal_release = self.terminal_intent_status is not None
        state_key = {
            "agent": "agent_dispatch_state",
            "fork": "fork_operation_state",
            "worktree": "worktree_provision_state",
            "prompt": "prompt_operation_state",
            "pane": "participant_creation_state",
        }[operation]
        changes: dict[str, object] = {
            state_key: "confirmed_absent"
            if outcome == "confirmed_absent"
            else "retained",
            "manual_reconcile_operation": None,
            "manual_reconcile_token": None,
            "manual_reconcile_resolved_at": resolved_at,
            "manual_reconcile_outcome": outcome,
            "worker_operation": None,
            "worker_pid": None,
            "worker_boot_id": None,
            "worker_process_start": None,
            "worker_token": None,
        }
        if operation != "prompt":
            changes[
                f"{operation}_{'confirmed_absent' if outcome == 'confirmed_absent' else 'retained'}_at"
            ] = resolved_at
        cleanup = self.cleanup_state
        if isinstance(cleanup, CleanupOwned):
            cleanup = abandon_cleanup_owner(cleanup, cleanup.token or "")
        reconciled_cleanup = finish_cleanup_reconciliation(cleanup)
        preserve_release = (
            retain_terminal_release or self.target_release_manual_required
        )
        if not preserve_release:
            reconciled_cleanup = CleanupSettled()
        changes.update(cleanup_fields(reconciled_cleanup))
        if retain_terminal_release and not self.target_release_pending:
            raise JobValidationError("terminal reconciliation requires release cleanup")
        if operation == "prompt":
            changes[state_key] = "submitted" if outcome == "materialized" else "none"
            if outcome == "confirmed_absent":
                changes.update(
                    prompt_operation_phase=None,
                    prompt_operation_turn=None,
                    prompt_operation_target=None,
                    prompt_baseline_sequence=None,
                )
        elif operation == "pane":
            resolved_pane = pane_id or self.participant_creation_pane_id
            resolved_workspace = workspace_id or self.participant_creation_workspace_id
            if outcome == "materialized":
                if not resolved_pane or not resolved_workspace:
                    return None
                try:
                    creation = self.participant_lifecycle.creation.created(
                        pane_id=resolved_pane,
                        workspace_id=resolved_workspace,
                    )
                except WorkflowTransitionError:
                    return None
                changes.update(
                    participant_creation_state=creation.state.value,
                    participant_creation_pane_id=creation.pane_id,
                    participant_creation_workspace_id=creation.workspace_id,
                    herdr_pane_id=resolved_pane,
                    herdr_workspace_id=resolved_workspace,
                    agent_name=self.participant_creation_target,
                )
            else:
                changes[state_key] = "none"
            if outcome == "confirmed_absent":
                changes.update(
                    participant_creation_participant=None,
                    participant_creation_target=None,
                    participant_creation_label=None,
                    participant_creation_workspace_id=None,
                    participant_creation_pane_id=None,
                )
        elif operation == "worktree" and outcome == "materialized":
            checkout = self.checkout_operation
            resolved_pane = pane_id or self.worktree_root_pane_id
            resolved_workspace = workspace_id or self.worktree_workspace_id
            if checkout is None or not resolved_pane or not resolved_workspace:
                return None
            try:
                changes["checkout_operation"] = checkout.transition(
                    CheckoutState.RETAINED,
                    workspace_id=resolved_workspace,
                    root_pane_id=resolved_pane,
                )
            except OperationTransitionError:
                return None
        elif outcome == "materialized":
            if operation == "fork":
                if not self.fork_operation_target:
                    return None
                changes.update(
                    fork_exists=True,
                    fork_repository=self.fork_operation_target,
                )
            elif operation == "agent":
                if not self.herdr_target:
                    return None
                if (
                    self.agent_provider is None
                    or self.agent_provider_session_id is None
                    or self.agent_state_sequence is None
                ):
                    return None
            elif operation == "worktree" and not self.worktree_path:
                return None
        elif operation == "agent":
            participant = self.active_participant
            changes.update(
                agent_operation_target=self.agent_operation_target or self.herdr_target,
                agent_operation_workspace_id=self.agent_operation_workspace_id
                or self.herdr_workspace_id,
                agent_operation_pane_id=self.agent_operation_pane_id
                or self.herdr_pane_id,
                agent_operation_checkout=self.agent_operation_checkout
                or self.worktree_path
                or self.repository,
                herdr_target=None,
                active_participant=None,
                herdr_pane_id=None,
                herdr_workspace_id=None,
                agent_name=None,
                agent_dispatch_exited=None,
                agent_next_reconcile_at=None,
            )
            if participant is not None:
                changes[f"{participant.value}_target"] = None
        changes.update(job_changes or {})
        return self.evolve_recovery(
            changes,
            now=resolved_at,
            prepare_delivery=True,
        )

    @property
    def foreground_until(self) -> float:
        return _number(self._values.get("foreground_until") or 0, "foreground_until")

    @property
    def updated_at(self) -> float | None:
        return self._optional_float("updated_at")

    @property
    def delivery_retry_at(self) -> float:
        return _number(self._values.get("delivery_retry_at") or 0, "delivery_retry_at")

    @property
    def delivery_attempts(self) -> int:
        return _integer(self._values.get("delivery_attempts") or 0, "delivery_attempts")

    @property
    def delivery_state(self) -> DeliveryState:
        try:
            return load_delivery_state(
                delivered=self.delivered,
                generation=self.delivery_generation,
                claim_token=self.delivery_claim_token,
                claimed_at=self.delivery_claimed_at,
                retry_at=self.delivery_retry_at,
                attempts=self.delivery_attempts,
                delivered_at=self.delivered_at,
                acknowledgement=self.announcement_ack,
                repeated=self.announcement_repeated,
            )
        except LifecycleTransitionError as exc:
            raise JobValidationError(str(exc)) from exc

    def claim_delivery(
        self, token: str, *, claimed_at: float, lease_seconds: float
    ) -> CursorJob:
        from .lifecycle import claim_delivery

        state = claim_delivery(
            self.delivery_state,
            token,
            claimed_at,
            lease_seconds=lease_seconds,
        )
        return self._updated(**delivery_fields(state))

    def renew_delivery(
        self, token: str, *, claimed_at: float, lease_seconds: float
    ) -> CursorJob:
        state = transition_delivery_renewal(
            self.delivery_state,
            token,
            claimed_at,
            lease_seconds=lease_seconds,
        )
        return self._updated(**delivery_fields(state))

    def acknowledge_delivery(
        self, token: str, *, delivered_at: float, lease_seconds: float
    ) -> CursorJob:
        state = transition_delivery_acknowledgement(
            self.delivery_state,
            token,
            delivered_at,
            AnnouncementAck.SPOKEN,
            lease_seconds=lease_seconds,
        )
        return self._updated(**delivery_fields(state))

    def acknowledge_desktop_delivery(
        self, token: str, *, acknowledged_at: float, lease_seconds: float
    ) -> CursorJob:
        state = transition_delivery_acknowledgement(
            self.delivery_state,
            token,
            acknowledged_at,
            AnnouncementAck.DESKTOP,
            lease_seconds=lease_seconds,
        )
        return self._updated(**delivery_fields(state))

    def acknowledge_deferred_delivery(
        self, token: str, *, acknowledged_at: float, lease_seconds: float
    ) -> CursorJob:
        state = transition_delivery_acknowledgement(
            self.delivery_state,
            token,
            acknowledged_at,
            AnnouncementAck.DEFERRED,
            lease_seconds=lease_seconds,
        )
        return self._updated(**delivery_fields(state))

    def release_delivery(self, token: str, *, retry_at: float) -> CursorJob:
        state = transition_delivery_release(
            self.delivery_state,
            token,
            retry_at=retry_at,
        )
        return self._updated(**delivery_fields(state))

    def mark_delivered(
        self,
        *,
        delivered_at: float | None = None,
        status: JobStatus | None = None,
        **changes: object,
    ) -> CursorJob:
        state = acknowledge_without_claim(self.delivery_state, delivered_at)
        values = dict(changes)
        values.update(delivery_fields(state))
        if status is not None:
            values["status"] = status.value
        return self._updated(**values)

    def dismiss_announcement(self, *, delivered_at: float) -> CursorJob:
        current = self.delivery_state
        announcement = dismiss_announcement(current.announcement)
        state = Delivered(
            current.generation,
            delivered_at,
            current.attempts,
            announcement,
        )
        return self._updated(**delivery_fields(state))

    def repeat_announcement(self, *, now: float) -> CursorJob:
        state = transition_prepare_delivery(self.delivery_state)
        announcement = repeat_announcement(state.announcement)
        return self._updated(
            **delivery_fields(
                PendingDelivery(
                    state.generation,
                    state.retry_at,
                    state.attempts,
                    announcement,
                )
            ),
            updated_at=now,
        )

    def _updated(self, **changes: object) -> CursorJob:
        values = dict(self._values)
        mutable = dict(changes)
        _consume_typed_prompt_operation(mutable, job_id=self.id)
        _consume_typed_checkout_operation(mutable)
        _consume_typed_agent_session_operation(mutable)
        values.update(mutable)
        _pair_worker_ownership(values)
        values["schema_version"] = CURRENT_SCHEMA_VERSION
        values["revision"] = self.revision + 1
        updated = CursorJob.from_dict(values)
        validate_transition(self, updated)
        return updated

    def validate_invariants(self, *, require_worker_owner: bool = False) -> None:
        # Parse every compatibility field group through its typed adapter. Flat
        # JSON remains the durable format, but invalid cross-field states fail
        # at the model boundary.
        _ = (
            self.terminal_state,
            self.cleanup_state,
            self.delivery_state,
            self.workflow_state,
            self.review_state,
            self.plan_approval,
            self.participant_lifecycle,
        )
        if self.loaded_schema_version == CURRENT_SCHEMA_VERSION and (
            self._values.get("worker_claimed_at") is not None
            or self._values.get("worker_claim_operation") is not None
        ):
            _ = self.worker_ownership
        if (
            self.agent_provider is not None
            or self.agent_provider_session_id is not None
            or self.agent_state_sequence is not None
        ):
            _ = self.agent_session_operation
        if self.participant_creation_checkout is not None:
            _ = self.participant_pane_operation
        if self.participant_admission_state not in {"waiting", "held", "released"}:
            raise JobValidationError("participant_admission_state has invalid value")
        if (
            self.clone_operation_state is not None
            and self.clone_operation_state not in _CLONE_OPERATION_STATES
        ):
            raise JobValidationError("clone_operation_state is invalid")
        if self.clone_operation_state is not None and not self.clone_source:
            raise JobValidationError("clone operation requires a clone source")
        if (
            self.github_issue_create_operation_state is not None
            and self.github_issue_create_operation_state
            not in _ISSUE_CREATE_OPERATION_STATES
        ):
            raise JobValidationError("github_issue_create_operation_state is invalid")
        if (
            self.github_issue_create_confirmed
            and not self.github_issue_create_requested
        ):
            raise JobValidationError(
                "GitHub issue creation confirmation requires a creation request"
            )
        if (
            self.github_repo_create_operation_state is not None
            and self.github_repo_create_operation_state
            not in _REPO_CREATE_OPERATION_STATES
        ):
            raise JobValidationError("github_repo_create_operation_state is invalid")
        if self.github_repo_create_confirmed and not self.github_repo_create_requested:
            raise JobValidationError(
                "GitHub repository creation confirmation requires a creation request"
            )
        if (
            self.github_repo_create_org_requested
            and not self.github_repo_create_requested
        ):
            raise JobValidationError(
                "GitHub organization repository creation requires a creation request"
            )
        if self.github_repo_create_visibility is not None and (
            self.github_repo_create_visibility not in {"private", "public"}
        ):
            raise JobValidationError("github_repo_create_visibility is invalid")
        if self.github_repo_create_operation_state is not None and not all(
            (
                self.github_repository,
                self.github_repo_create_visibility,
                self.github_repo_create_marker,
            )
        ):
            raise JobValidationError(
                "GitHub repository creation operation requires repository, "
                "visibility, and marker"
            )
        if self.github_issue_create_operation_state is not None and not all(
            (
                self.github_repository,
                self.github_issue_create_title,
                self.github_issue_create_marker,
            )
        ):
            raise JobValidationError(
                "GitHub issue creation operation requires repository, title, and marker"
            )
        if (
            self.linear_ticket_create_operation_state is not None
            and self.linear_ticket_create_operation_state
            not in _ISSUE_CREATE_OPERATION_STATES
        ):
            raise JobValidationError("linear_ticket_create_operation_state is invalid")
        if (
            self.linear_ticket_create_confirmed
            and not self.linear_ticket_create_requested
        ):
            raise JobValidationError(
                "Linear ticket creation confirmation requires a creation request"
            )
        if self.linear_ticket_create_operation_state is not None and not all(
            (
                self.linear_ticket_create_team,
                self.linear_ticket_create_team_id,
                self.linear_ticket_create_title,
                self.linear_ticket_create_marker,
            )
        ):
            raise JobValidationError(
                "Linear ticket creation operation requires team, title, and marker"
            )
        if self.linear_ticket_create_operation_state in {"submitting", "submitted"}:
            if (
                not all(
                    (
                        self.linear_ticket_create_prompt_target,
                        self.linear_ticket_create_prompt_session,
                        self.linear_ticket_create_prompt_token,
                    )
                )
                or self.linear_ticket_create_baseline_sequence is None
            ):
                raise JobValidationError(
                    "Linear ticket submission requires a durable prompt fence"
                )
        if (
            self.github_issue is not None
            and self.issue_key is None
            and self.issue_provider != "github"
        ):
            raise JobValidationError("GitHub issue job requires github issue_provider")
        if self.issue_key is not None and self.issue_provider is None:
            raise JobValidationError("issue-key job requires issue_provider")
        if (
            self.issue_provider == "github"
            and self.github_issue is None
            and not self.github_issue_create_requested
            and not self.github_repo_create_requested
        ):
            raise JobValidationError(
                "github issue_provider requires a GitHub issue identity"
            )
        if (
            self.issue_provider not in {None, "github"}
            and self.issue_key is None
            and not (
                self.issue_provider == "linear" and self.linear_ticket_create_requested
            )
        ):
            raise JobValidationError("selected issue_provider requires an issue key")
        if self.revision < 0:
            raise JobValidationError("revision must not be negative")
        if self.worker_pid is not None and self.worker_pid <= 0:
            raise JobValidationError("worker_pid must be positive")
        worker_claim = (
            self.worker_token,
            self.worker_pid,
            self.worker_boot_id,
            self.worker_process_start,
            self._optional_string("worker_claim_operation"),
            self._optional_float("worker_claimed_at"),
        )
        if (
            require_worker_owner
            and any(item is not None for item in worker_claim)
            and not all(item is not None for item in worker_claim)
        ):
            raise JobValidationError(
                f"{self.status.value} job requires complete worker ownership"
            )
        if (
            require_worker_owner
            and self.terminal_intent_status is None
            and self.status in WORKER_STATUSES
            and not all(item is not None for item in worker_claim)
        ):
            raise JobValidationError(
                f"{self.status.value} job requires complete worker ownership"
            )
        if bool(self.delivery_claim_token) != (self.delivery_claimed_at is not None):
            raise JobValidationError(
                "delivery claim token and timestamp must be paired"
            )
        if self.delivered and self.delivery_claim_token:
            raise JobValidationError("delivered job cannot retain a delivery claim")
        ack = self.announcement_ack
        if ack not in ANNOUNCEMENT_ACK_STATES:
            raise JobValidationError("announcement_ack has invalid value")
        if ack in SPOKEN_ANNOUNCEMENT_ACKS and not self.delivered:
            raise JobValidationError(
                "spoken or dismissed announcement requires delivered"
            )
        if ack in SUPPRESSED_ANNOUNCEMENT_ACKS and self.delivered:
            raise JobValidationError(
                "desktop or deferred announcement cannot be marked delivered"
            )
        if self.review_round < 0 or self.review_round > 2:
            raise JobValidationError("review_round must be between zero and two")
        if (
            self.workflow_phase != WorkflowPhase.CLASSIFYING
            and self.workflow_tier is None
            and self.loaded_schema_version == CURRENT_SCHEMA_VERSION
        ):
            raise JobValidationError(
                f"{self.workflow_phase.value} workflow requires workflow_tier"
            )
        if self.workflow_tier is not None and not self.workflow_classification_reason:
            raise JobValidationError(
                "classified workflow requires workflow_classification_reason"
            )
        if self.workflow_tier == WorkflowTier.SIMPLE and self.workflow_phase in {
            WorkflowPhase.PLANNING,
            WorkflowPhase.REVIEWING,
            WorkflowPhase.REVISING,
        }:
            raise JobValidationError("simple workflow cannot enter planning or review")
        if self.workflow_tier == WorkflowTier.MEDIUM and self.review_round > 1:
            raise JobValidationError("medium workflow allows only one review")
        review_decision = self._optional_string("review_decision")
        if review_decision not in {None, "approve", "revise"}:
            raise JobValidationError("review_decision has invalid value")
        approval_source = self.review_approval_source
        if approval_source not in {None, "reviewer", "user"}:
            raise JobValidationError("review_approval_source has invalid value")
        if self.review_approved != (approval_source is not None):
            raise JobValidationError(
                "review approval and approval source must be paired"
            )
        if self.review_approved and not self.review_artifact:
            raise JobValidationError("approved review requires a review artifact")
        if approval_source == "reviewer" and review_decision != "approve":
            raise JobValidationError(
                "reviewer approval requires an approving review decision"
            )
        if approval_source == "user" and (
            self.workflow_tier != WorkflowTier.HIGH_RISK
            or review_decision != "revise"
            or self.review_round < 1
        ):
            raise JobValidationError(
                "user approval requires an exhausted high-risk rejected review"
            )
        plan_approval_state = self.plan_approval_state
        if plan_approval_state not in {state.value for state in PlanApprovalState}:
            raise JobValidationError("plan_approval_state has invalid value")
        plan_approval_source = self.plan_approval_source
        if plan_approval_source not in {None, "explicit", "auto", "legacy"}:
            raise JobValidationError("plan_approval_source has invalid value")
        approval_proof = (
            self.plan_approval_id,
            self.plan_approval_agent_session,
            self.plan_approval_state_change_sequence,
        )
        if plan_approval_state == "none":
            if any(value is not None for value in approval_proof) or (
                plan_approval_source is not None
            ):
                raise JobValidationError(
                    "inactive plan approval cannot retain gate proof or source"
                )
        elif not all(value is not None for value in approval_proof):
            raise JobValidationError(
                "active plan approval requires gate ID, agent session, and sequence"
            )
        if plan_approval_state == "boundary" and (
            self.workflow_phase
            not in {
                WorkflowPhase.PLANNING,
                WorkflowPhase.REVISING,
                WorkflowPhase.REVIEWING,
            }
            or plan_approval_source is not None
        ):
            raise JobValidationError(
                "plan approval boundary requires a planning or review phase"
            )
        if plan_approval_state == "awaiting" and (
            self.status != JobStatus.AWAITING_USER
            or self.clarification_kind != "workflow_plan_approval"
            or self.workflow_phase != WorkflowPhase.REVIEWING
            or plan_approval_source is not None
        ):
            raise JobValidationError(
                "awaiting plan approval requires the reviewed-plan question"
            )
        if plan_approval_state in {"approved", "observed"} and (
            plan_approval_source is None
            or self.workflow_phase
            not in {WorkflowPhase.IMPLEMENTING, WorkflowPhase.FINISHED}
        ):
            raise JobValidationError(
                "approved plan requires a source and implementation phase"
            )
        if plan_approval_state == "rejected" and (
            (
                self.status != JobStatus.CANCELLED
                and not (
                    self.status == JobStatus.RECONCILING
                    and self.terminal_intent_status == JobStatus.CANCELLED
                )
            )
            or plan_approval_source is not None
        ):
            raise JobValidationError(
                "rejected plan approval requires a cancelled job without a source"
            )
        if self.plan_approval_counted and (
            plan_approval_source != "explicit"
            or plan_approval_state not in {"approved", "observed"}
        ):
            raise JobValidationError(
                "counted plan approval requires an accepted explicit approval"
            )
        if self.plan_approval_completion_pending and (
            plan_approval_source != "explicit"
            or plan_approval_state != "observed"
            or self.workflow_phase != WorkflowPhase.FINISHED
            or self.status not in {JobStatus.QUEUED, JobStatus.RECONCILING}
            or not self.reconcile
            or not self.result
            or self.active_participant is not None
            or self.prompt_operation_state != "none"
        ):
            raise JobValidationError(
                "pending plan approval completion requires durable finished output"
            )
        if (
            self.workflow_tier in {WorkflowTier.MEDIUM, WorkflowTier.HIGH_RISK}
            and self.workflow_phase
            in {WorkflowPhase.IMPLEMENTING, WorkflowPhase.FINISHED}
            and (
                not self.plan_artifact
                or not self.review_artifact
                or not self.review_approved
            )
        ):
            raise JobValidationError(
                "planned workflow cannot implement without approved plan and review"
            )
        if (
            self.workflow_tier in {WorkflowTier.MEDIUM, WorkflowTier.HIGH_RISK}
            and self.workflow_phase
            in {WorkflowPhase.IMPLEMENTING, WorkflowPhase.FINISHED}
            and plan_approval_state not in {"approved", "observed"}
        ):
            raise JobValidationError(
                "planned workflow cannot implement without plan approval"
            )
        turn_phase = self._optional_string("workflow_turn_phase")
        if turn_phase is not None and turn_phase not in {
            item.value for item in WorkflowPhase
        }:
            raise JobValidationError("workflow_turn_phase has invalid value")
        if self.active_participant is not None:
            target = self.participant_target(self.active_participant)
            if not target or target != self.herdr_target:
                raise JobValidationError(
                    "active participant target must match herdr_target"
                )
        self._validate_operation_state(
            "prompt_operation_state",
            self.prompt_operation_state,
            _PROMPT_OPERATION_STATES,
        )
        self._validate_operation_state(
            "participant_creation_state",
            self.participant_creation_state,
            frozenset(state.value for state in ParticipantCreationState),
        )
        if self.prompt_operation_state != "none":
            if (
                self.prompt_operation_phase is None
                or self.prompt_operation_turn <= 0
                or not self.prompt_operation_target
            ):
                raise JobValidationError(
                    "durable prompt operation requires phase, turn, and target"
                )
        if self.prompt_operation_state in {"submitting", "submitted", "ambiguous"}:
            if self.prompt_baseline_sequence is None:
                raise JobValidationError(
                    f"{self.prompt_operation_state} prompt requires baseline sequence"
                )
        if self.participant_creation_state != "none" and not all(
            self._values.get(field)
            for field in (
                "participant_creation_participant",
                "participant_creation_target",
                "participant_creation_label",
            )
        ):
            raise JobValidationError(
                "participant creation requires role, target, and label"
            )
        if self.participant_creation_state == "created" and not all(
            self._values.get(field)
            for field in (
                "participant_creation_workspace_id",
                "participant_creation_pane_id",
            )
        ):
            raise JobValidationError("created participant requires pane and workspace")
        if self.terminal_intent_status is not None:
            if (
                self.terminal_intent_status not in TERMINAL_STATUSES
                or self.status != JobStatus.RECONCILING
                or not self.target_release_pending
            ):
                raise JobValidationError(
                    "terminal intent requires reconciling status and release fence"
                )
        try:
            self.session_control_state()
        except JobLifecycleError as exc:
            raise JobValidationError(str(exc)) from exc
        self._validate_operation_state(
            "agent_dispatch_state",
            self.agent_dispatch_state,
            _AGENT_OPERATION_STATES,
        )
        self._validate_operation_state(
            "fork_operation_state",
            self.fork_operation_state,
            _FORK_OPERATION_STATES,
        )
        self._validate_operation_state(
            "worktree_provision_state",
            self.worktree_provision_state,
            _WORKTREE_OPERATION_STATES,
        )
        if (
            self.agent_dispatch_state in _UNCERTAIN_OPERATION_STATES
            and not self.herdr_target
        ):
            raise JobValidationError(
                f"{self.agent_dispatch_state} agent operation requires herdr_target"
            )
        if self.fork_operation_state in {
            "planned",
            "submitted",
            "ambiguous",
            "failed_observing",
            "manual_required",
        } and not self._values.get("fork_operation_target"):
            raise JobValidationError(
                f"{self.fork_operation_state} fork operation requires target"
            )
        if self.worktree_provision_state in {
            "planned",
            "dispatching",
            "ambiguous",
            "failed_observing",
            "manual_required",
        } and not all(
            self._values.get(field)
            for field in ("repository", "worktree_branch", "worktree_path")
        ):
            raise JobValidationError(
                f"{self.worktree_provision_state} worktree operation requires "
                "repository, branch, and path"
            )

        manual_states = {
            "agent": self.agent_dispatch_state,
            "fork": self.fork_operation_state,
            "worktree": self.worktree_provision_state,
            "prompt": self.prompt_operation_state,
            "pane": self.participant_creation_state,
        }
        manual_required = any(
            state == "manual_required" for state in manual_states.values()
        )
        if manual_required and (
            not self.manual_reconcile_operation or not self.manual_reconcile_token
        ):
            raise JobValidationError(
                "manual reconciliation requires operation and token"
            )
        if self.manual_reconcile_operation:
            if self.manual_reconcile_operation not in manual_states:
                raise JobValidationError("manual reconciliation operation is invalid")
            expected_manual_state = (
                "ambiguous"
                if self.manual_reconcile_operation == "prompt"
                else "manual_required"
            )
            if (
                manual_states[self.manual_reconcile_operation] != expected_manual_state
                or not self.manual_reconcile_token
            ):
                raise JobValidationError(
                    "manual reconciliation fields do not match operation fence"
                )
        elif self.manual_reconcile_token:
            raise JobValidationError(
                "manual reconciliation requires operation and token"
            )
        if require_worker_owner and self.worktree_provision_state is not None:
            _ = self.checkout_operation
        if require_worker_owner and self.agent_dispatch_state is not None:
            _ = self.agent_session_operation
        if require_worker_owner and self.participant_creation_state != "none":
            _ = self.participant_pane_operation
        if require_worker_owner and self.fork_operation_state is not None:
            _ = self.fork_operation
        if (
            self.cancellation_reconciliation_pending
            and not self.has_uncertain_operation()
            and not self.target_release_pending
        ):
            raise JobValidationError(
                "cancellation reconciliation requires an operation or release fence"
            )
        if self.cancellation_reconciliation_pending and self.status not in {
            JobStatus.RECONCILING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }:
            raise JobValidationError(
                "cancellation reconciliation is only valid for cancelled or failed jobs"
            )
        release_owner = (
            self._values.get("target_release_owner_pid"),
            self._values.get("target_release_owner_boot_id"),
            self._values.get("target_release_owner_start"),
        )
        if any(item is not None for item in release_owner) and not all(
            item is not None for item in release_owner
        ):
            raise JobValidationError(
                "target release owner PID, boot ID, and process start must be paired"
            )
        if not self.target_release_pending and any(
            self._values.get(field)
            for field in (
                "target_release_token",
                "target_release_owner_pid",
                "target_release_owner_boot_id",
                "target_release_owner_start",
                "target_release_manual_required",
                "target_release_unverified_targets",
            )
        ):
            raise JobValidationError(
                "released target cannot retain release ownership fields"
            )
        if self.target_release_manual_required != bool(
            self.target_release_unverified_targets
        ):
            raise JobValidationError(
                "manual target release requires unverified target identities"
            )

    def has_uncertain_operation(self) -> bool:
        return (
            self.agent_dispatch_state in _UNCERTAIN_OPERATION_STATES
            or self.fork_operation_state in _UNCERTAIN_OPERATION_STATES
            or self.worktree_provision_state in _UNCERTAIN_OPERATION_STATES
            or self.github_issue_create_operation_state in {"submitted", "ambiguous"}
            or self.clone_operation_state in {"submitted", "ambiguous"}
            or self.github_repo_create_operation_state in {"submitted", "ambiguous"}
            or self.linear_ticket_create_operation_state
            in {"submitting", "submitted", "ambiguous"}
            or self.prompt_operation_state in {"submitting", "ambiguous"}
            or self.participant_creation_state
            in {"submitting", "ambiguous", "manual_required"}
        )

    @staticmethod
    def _validate_operation_state(
        field: str, value: str | None, allowed: frozenset[str]
    ) -> None:
        if value is not None and value not in allowed:
            raise JobValidationError(f"{field} has invalid value")


def _validate_operation_transitions(before: CursorJob, after: CursorJob) -> None:
    operations = (
        ("checkout", before.checkout_operation, after.checkout_operation),
        ("fork", before.fork_operation, after.fork_operation),
        (
            "agent session",
            before.agent_session_operation,
            after.agent_session_operation,
        ),
        (
            "participant pane",
            before.participant_pane_operation,
            after.participant_pane_operation,
        ),
    )
    for name, previous, current in operations:
        if previous is None:
            continue
        if current is None:
            if name == "participant pane":
                continue
            raise JobValidationError(f"{name} operation identity cannot be discarded")
        if previous.spec != current.spec:
            archived_agent = (
                name == "agent session"
                and isinstance(previous, AgentSessionOperation)
                and previous.session is not None
                and any(
                    owner.get("provider") == previous.session.provider
                    and owner.get("session_id") == previous.session.session_id
                    and owner.get("target") == previous.session.target
                    and owner.get("state_sequence") == previous.session.state_sequence
                    and owner.get("checkout") == previous.spec.checkout
                    and owner.get("workspace_id") == previous.spec.workspace_id
                    and owner.get("pane_id") == previous.spec.pane_id
                    for owner in after.participant_session_owners
                )
            )
            replacement = (
                (
                    name == "agent session"
                    and previous.state == AgentSessionState.CONFIRMED_ABSENT
                    and current.state
                    in {
                        AgentSessionState.DISPATCHING,
                        AgentSessionState.READY,
                        AgentSessionState.RETAINED,
                    }
                )
                or (
                    name == "participant pane"
                    and previous.state == OperationState.FAILED
                    and current.state == OperationState.PLANNED
                )
                or (
                    LEGACY_BOOT_ID in repr(previous.spec)
                    and (
                        (
                            isinstance(previous, CheckoutOperation)
                            and isinstance(current, CheckoutOperation)
                            and previous.state
                            in {
                                CheckoutState.QUARANTINED,
                                CheckoutState.AMBIGUOUS,
                                CheckoutState.FAILED_OBSERVING,
                                CheckoutState.MANUAL_REQUIRED,
                            }
                            and current.state
                            in {CheckoutState.READY, CheckoutState.RETAINED}
                        )
                        or (
                            isinstance(previous, AgentSessionOperation)
                            and isinstance(current, AgentSessionOperation)
                            and previous.state
                            in {
                                AgentSessionState.AMBIGUOUS,
                                AgentSessionState.FAILED_OBSERVING,
                                AgentSessionState.MANUAL_REQUIRED,
                            }
                            and current.state
                            in {AgentSessionState.READY, AgentSessionState.RETAINED}
                        )
                        or (
                            not isinstance(
                                previous, CheckoutOperation | AgentSessionOperation
                            )
                            and previous.state
                            in {OperationState.UNKNOWN, OperationState.MANUAL}
                            and current.state == OperationState.SETTLED
                        )
                    )
                )
                or archived_agent
            )
            if not replacement:
                raise JobValidationError(
                    f"{name} operation identity cannot be rewritten"
                )
            continue
        try:
            if previous.state != current.state:
                legacy_agent_materialized = bool(
                    name == "agent session"
                    and before.loaded_schema_version < CURRENT_SCHEMA_VERSION
                    and before.agent_dispatch_state == "manual_required"
                    and after.agent_dispatch_state == "retained"
                    and previous.state == AgentSessionState.MANUAL_REQUIRED
                    and current.state == AgentSessionState.AMBIGUOUS
                    and isinstance(current, AgentSessionOperation)
                    and current.session is None
                )
                if legacy_agent_materialized:
                    continue
                if isinstance(previous, CheckoutOperation) and isinstance(
                    current, CheckoutOperation
                ):
                    previous.transition(
                        current.state,
                        workspace_id=current.workspace_id,
                        root_pane_id=current.root_pane_id,
                    )
                elif isinstance(previous, AgentSessionOperation) and isinstance(
                    current, AgentSessionOperation
                ):
                    previous.transition(current.state, session=current.session)
                elif isinstance(previous, ParticipantPaneOperation) and isinstance(
                    current, ParticipantPaneOperation
                ):
                    previous.transition(current.state, pane_id=current.pane_id)
                else:
                    assert isinstance(previous, ForkOperation)
                    assert isinstance(current, ForkOperation)
                    previous.transition(current.state)
            if isinstance(previous, AgentSessionOperation) and isinstance(
                current, AgentSessionOperation
            ):
                if previous.session is not None and current.session is None:
                    raise OperationTransitionError(
                        "agent session identity cannot be discarded"
                    )
                if (
                    previous.session is not None
                    and current.session is not None
                    and not previous.accepts_observation(current.session)
                ):
                    raise OperationTransitionError(
                        "agent session identity or sequence is stale"
                    )
        except OperationTransitionError as exc:
            raise JobValidationError(str(exc)) from exc


def validate_transition(before: CursorJob, after: CursorJob) -> None:
    if before.id != after.id:
        raise JobValidationError("Cursor job transition cannot change id")
    if before.created_at != after.created_at:
        raise JobValidationError("Cursor job transition cannot change created_at")
    if before.parent_job_id != after.parent_job_id:
        raise JobValidationError("Cursor job transition cannot change parent_job_id")
    if before.request != after.request:
        raise JobValidationError(
            "Cursor job transition cannot change the original request"
        )
    if before.harness_kind != after.harness_kind:
        raise JobValidationError("agent job transition cannot change harness_kind")
    if before.issue_provider != after.issue_provider:
        raise JobValidationError("agent job transition cannot change issue_provider")
    _validate_operation_transitions(before, after)
    before_terminal = before.terminal_state
    after_terminal = after.terminal_state
    if (
        before.terminal_intent_status is not None
        and after.terminal_intent_status is not None
        and before.terminal_intent_status != after.terminal_intent_status
    ):
        raise JobValidationError("terminal intent status cannot be rewritten")
    if (
        isinstance(after_terminal, MaterializedTerminalOutcome)
        and not isinstance(before_terminal, MaterializedTerminalOutcome)
        and (
            after.target_release_pending
            or after.cancellation_reconciliation_pending
            or any(
                after._values.get(field)
                for field in (
                    "herdr_target",
                    "planner_target",
                    "reviewer_target",
                    "implementer_target",
                )
            )
        )
    ):
        raise JobValidationError(
            "terminal outcome cannot materialize before cleanup settles"
        )
    if after.parent_job_id is not None:
        # A follow-up child inherits its parent's exact checkout. That identity
        # must never be substituted, removed, or reconstructed by recovery.
        for field in (
            "repository",
            "worktree_branch",
            "worktree_path",
            "worktree_workspace_id",
            "worktree_root_pane_id",
        ):
            old = before._values.get(field)
            new = after._values.get(field)
            if old != new:
                raise JobValidationError(
                    f"follow-up child cannot change inherited {field}"
                )
    if after.revision != before.revision + 1:
        raise JobValidationError(
            "Cursor job revision must increase by exactly one "
            f"({before.revision} -> {after.revision})"
        )
    recovery_event = isinstance(after._lifecycle_event, RecoveryEvent)
    if (
        before.status != after.status
        and after.status not in _LEGAL_TRANSITIONS[before.status]
        and not recovery_event
    ):
        raise JobValidationError(
            "illegal Cursor job transition "
            f"{before.status.value} -> {after.status.value}"
        )
    if isinstance(before_terminal, MaterializedTerminalOutcome) and (
        after_terminal != before_terminal
    ):
        raise JobValidationError("materialized terminal outcome is immutable")
    try:
        if (
            before.participant_lifecycle.admission
            != after.participant_lifecycle.admission
        ):
            if after.participant_lifecycle.admission == ParticipantAdmissionState.HELD:
                before.participant_lifecycle.admit()
            elif (
                after.participant_lifecycle.admission
                == ParticipantAdmissionState.WAITING
            ):
                resource_fields = (
                    "herdr_target",
                    "planner_target",
                    "reviewer_target",
                    "implementer_target",
                    "participant_creation_target",
                    "worktree_path",
                )
                if (
                    before.status not in {JobStatus.ROUTING, JobStatus.RUNNING}
                    or after.status != JobStatus.AWAITING_USER
                    or after.clarification_kind != "repository"
                    or after.worker_token is not None
                    or after.active_participant is not None
                    or any(after._values.get(field) for field in resource_fields)
                ):
                    raise WorkflowTransitionError(
                        "held participant capacity can only yield for a "
                        "resource-free repository clarification"
                    )
                before.participant_lifecycle.yield_capacity()
            elif (
                after.participant_lifecycle.admission
                == ParticipantAdmissionState.RELEASED
            ):
                before.participant_lifecycle.release(
                    cleanup_confirmed=bool(
                        after.status in TERMINAL_STATUSES
                        and not after.target_release_pending
                        and not any(
                            after._values.get(field)
                            for field in (
                                "herdr_target",
                                "planner_target",
                                "reviewer_target",
                                "implementer_target",
                            )
                        )
                    )
                )
            else:
                raise WorkflowTransitionError(
                    "participant admission state cannot move backward"
                )
        before.participant_lifecycle.creation.validate_transition(
            after.participant_lifecycle.creation
        )
        before.plan_approval.validate_transition(after.plan_approval)
    except WorkflowTransitionError as exc:
        raise JobValidationError(str(exc)) from exc
    if before.participant_admission_state == "held" and (
        after.participant_admission_state == "released"
    ):
        participant_fields = (
            "herdr_target",
            "planner_target",
            "reviewer_target",
            "implementer_target",
        )
        if (
            after.status not in TERMINAL_STATUSES
            or after.target_release_pending
            or any(after._values.get(field) for field in participant_fields)
        ):
            raise JobValidationError(
                "held participant capacity requires confirmed terminal cleanup"
            )
    try:
        before_workflow = before.workflow_state
        after_workflow = after.workflow_state
        if before_workflow.classification is None:
            if after_workflow.classification is not None:
                expected = before_workflow.classify(
                    after_workflow.classification.tier,
                    after_workflow.classification.reason,
                )
                if expected != after_workflow:
                    raise WorkflowTransitionError(
                        "classification selected an invalid workflow phase"
                    )
            elif before_workflow.phase != after_workflow.phase:
                before_workflow.transition(after_workflow.phase)
        else:
            classification = before_workflow.classification
            if after_workflow.classification != classification:
                if after_workflow.classification is None:
                    raise WorkflowTransitionError(
                        "classified workflow cannot discard classification"
                    )
                classification = classification.promote(
                    after_workflow.classification.tier,
                    after_workflow.classification.reason,
                )
            WorkflowState(before_workflow.phase, classification).transition(
                after_workflow.phase
            )
    except WorkflowTransitionError as exc:
        raise JobValidationError(str(exc)) from exc
    after.validate_invariants(require_worker_owner=True)
    lifecycle_event = after._lifecycle_event or LifecycleEvent(
        before.revision, after.lifecycle
    )
    before.validate_lifecycle_event(after, lifecycle_event)


def transition(
    job: CursorJob,
    status: JobStatus,
    **changes: Any,
) -> CursorJob:
    if status != job.status and status not in _LEGAL_TRANSITIONS[job.status]:
        raise JobValidationError(
            f"illegal Cursor job transition {job.status.value} -> {status.value}"
        )
    values = job.to_dict()
    _consume_typed_prompt_operation(changes, job_id=job.id)
    _consume_typed_checkout_operation(changes)
    _consume_typed_agent_session_operation(changes)
    values.update(changes)
    _pair_worker_ownership(values)
    if "herdr_target" in changes and "session_id" not in changes:
        values["session_id"] = changes["herdr_target"]
    elif "session_id" in changes and "herdr_target" not in changes:
        values["herdr_target"] = changes["session_id"]
    values["status"] = status.value
    values["schema_version"] = CURRENT_SCHEMA_VERSION
    values.setdefault("revision", job.revision + 1)
    if "revision" not in changes:
        values["revision"] = job.revision + 1
    updated = CursorJob.from_dict(values)
    validate_transition(job, updated)
    return updated


def worker_callback_transition(
    job: CursorJob,
    expected_revision: int,
    expected_worker: WorkerOwnership,
    status: JobStatus,
    **changes: Any,
) -> CursorJob:
    """Apply a callback only to the exact worker revision and ownership."""

    updated = transition(job, status, **changes)
    event = WorkerCallbackEvent(
        expected_revision,
        updated.lifecycle,
        expected_worker,
    )
    job.validate_lifecycle_event(
        updated,
        event,
    )
    return replace(
        updated,
        _lifecycle_event=event,
    )


def legal_transitions(status: JobStatus) -> frozenset[JobStatus]:
    return _LEGAL_TRANSITIONS[status]


def validate_reservations(jobs: list[CursorJob]) -> None:
    targets: dict[str, str] = {}
    worktrees: dict[str, str] = {}
    for job in jobs:
        reserves = (
            job.status in ACTIVE_STATUSES
            or job.target_release_pending
            or job.cancellation_reconciliation_pending
            or job.has_uncertain_operation()
            or job.manual_reconcile_operation is not None
            or job.worktree_manual_inspection_required
            or checkout_blocks_reservation(
                None if job.checkout_operation is None else job.checkout_operation.state
            )
            or job.pull_request_worktree_state == "quarantined"
        )
        job_targets = {
            target
            for target in (
                job.herdr_target,
                job.participant_target(WorkflowParticipant.PLANNER),
                job.participant_target(WorkflowParticipant.REVIEWER),
                job.participant_target(WorkflowParticipant.IMPLEMENTER),
                job._optional_string("participant_creation_target"),
            )
            if target
        }
        job_targets.update(
            str(owner["target"]) for owner in job.participant_session_owners
        )
        if reserves:
            for target in job_targets:
                owner = targets.setdefault(target, job.id)
                if owner != job.id:
                    raise JobValidationError(
                        f"Herdr target {target} is reserved by both "
                        f"{owner} and {job.id}"
                    )
        worktree_blocked = reserves or checkout_blocks_reservation(
            None if job.checkout_operation is None else job.checkout_operation.state
        )
        if worktree_blocked and job.worktree_path:
            owner = worktrees.setdefault(job.worktree_path, job.id)
            if owner != job.id:
                raise JobValidationError(
                    f"worktree {job.worktree_path} is reserved by both "
                    f"{owner} and {job.id}"
                )


# Agent-neutral names are canonical for new integrations. Cursor names remain
# aliases so existing callers and persisted-job recovery continue to work.
CursorJob = AgentJob
NewCursorJob = NewAgentJob
AgentJobValidationError = JobValidationError
