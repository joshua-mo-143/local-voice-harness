from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..integrations.github import (
    GITHUB_PROVIDER_STATE_FIELDS,
    GitHubError,
    dump_github_provider_state,
    load_github_provider_state,
)
from ..questions import Question, QuestionError

CURRENT_SCHEMA_VERSION = 13
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


class HarnessKind(StrEnum):
    """Coding-agent harness responsible for a durable job."""

    CURSOR = "cursor"


class WorkflowTier(StrEnum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH_RISK = "high-risk"


class WorkflowPhase(StrEnum):
    CLASSIFYING = "classifying"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    REVISING = "revising"
    IMPLEMENTING = "implementing"
    FINISHED = "finished"


class WorkflowParticipant(StrEnum):
    PLANNER = "planner"
    REVIEWER = "reviewer"
    IMPLEMENTER = "implementer"


_LEGAL_WORKFLOW_TRANSITIONS: dict[WorkflowPhase, frozenset[WorkflowPhase]] = {
    WorkflowPhase.CLASSIFYING: frozenset(
        {WorkflowPhase.PLANNING, WorkflowPhase.IMPLEMENTING}
    ),
    WorkflowPhase.PLANNING: frozenset({WorkflowPhase.REVIEWING}),
    WorkflowPhase.REVIEWING: frozenset(
        {WorkflowPhase.REVISING, WorkflowPhase.IMPLEMENTING}
    ),
    WorkflowPhase.REVISING: frozenset({WorkflowPhase.REVIEWING}),
    WorkflowPhase.IMPLEMENTING: frozenset(
        {WorkflowPhase.PLANNING, WorkflowPhase.FINISHED}
    ),
    WorkflowPhase.FINISHED: frozenset(),
}


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
    JobStatus.QUEUED: frozenset(
        {
            JobStatus.ROUTING,
            JobStatus.RECONCILING,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.ROUTING: frozenset(
        {
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.RECONCILING,
            JobStatus.AWAITING_USER,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.QUEUED,
            JobStatus.AWAITING_USER,
            JobStatus.BLOCKED,
            JobStatus.RECONCILING,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.RECONCILING: frozenset(
        {
            JobStatus.QUEUED,
            JobStatus.AWAITING_USER,
            JobStatus.BLOCKED,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.AWAITING_USER: frozenset(
        {
            JobStatus.QUEUED,
            JobStatus.RECONCILING,
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.BLOCKED: frozenset(
        {JobStatus.QUEUED, JobStatus.RECONCILING, JobStatus.CANCELLED}
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}

_BOOL_FIELDS = frozenset(
    {
        "delivered",
        "continuation",
        "reconcile",
        "fork_requested",
        "fork_confirmed",
        "fork_committed",
        "fork_exists",
        "fork_dispatch_exited",
        "fork_operation_source_private",
        "agent_dispatch_exited",
        "worktree_dispatch_exited",
        "target_release_pending",
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
        "revision",
        "turn",
        "github_issue",
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
        "manual_reconcile_required_at",
        "manual_reconcile_resolved_at",
        "worktree_quarantine_acknowledged_at",
        "terminal_intent_completed_at",
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
        "repository",
        "github_repository",
        "github_issue_url",
        "github_issue_context",
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
        "herdr_target",
        "herdr_pane_id",
        "herdr_workspace_id",
        "delivery_claim_token",
        "target_release_token",
        "target_release_owner_boot_id",
        "target_release_owner_start",
        "agent_dispatch_state",
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
        "workflow_turn_phase",
        "prompt_operation_state",
        "prompt_operation_phase",
        "prompt_operation_target",
        "prompt_operation_agent_session",
        "participant_creation_state",
        "participant_creation_participant",
        "participant_creation_target",
        "participant_creation_label",
        "participant_creation_workspace_id",
        "participant_creation_pane_id",
        "terminal_intent_status",
        "terminal_intent_result",
        "terminal_intent_error",
        "harness_kind",
        "issue_provider",
        "session_id",
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
    {"none", "planned", "submitting", "submitted", "ambiguous"}
)
_PLAN_APPROVAL_STATES = frozenset(
    {"none", "boundary", "awaiting", "approved", "observed", "rejected"}
)
_PARTICIPANT_CREATION_STATES = frozenset(
    {"none", "planned", "submitting", "created", "ambiguous", "manual_required"}
)
_WORKFLOW_ARTIFACT_REFERENCE = re.compile(
    r"^\.artifacts/(?P<job>[0-9a-f]{12})/"
    r"(?P<kind>plan|review)-(?P<round>[0-2])"
    r"(?:-(?P<digest>[0-9a-f]{64}))?\.json$"
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
_LINEAR_STATE_FIELDS = frozenset({"issue_key"})
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
        else:
            values.setdefault("plan_approval_state", "none")
    elif version == 11:
        values.setdefault("plan_approval_completion_pending", False)
    elif version == 12:
        values.setdefault("issue_provider", _infer_legacy_issue_provider(values))
        # The GitHub provider-state serializer independently migrates its
        # legacy flat v12 payload to the nested provider-owned representation.
    values["schema_version"] = version + 1


def _infer_legacy_issue_provider(values: Mapping[str, object]) -> str | None:
    if values.get("issue_key") is not None:
        # Historically issue_key was exclusively owned by the Linear
        # integration. GitHub fields could also be present as captured context.
        return "linear"
    if values.get("github_issue") is not None:
        return "github"
    return None


def migrate_job_record(raw: Mapping[str, object]) -> tuple[dict[str, object], int]:
    """Migrate a persisted record to the current in-memory representation."""

    values = dict(raw)
    loaded_version = (
        _integer(values["schema_version"], "schema_version")
        if "schema_version" in values
        else 0
    )
    if loaded_version not in LEGACY_SCHEMA_VERSIONS | {CURRENT_SCHEMA_VERSION}:
        raise JobValidationError(
            f"unsupported agent job schema version {loaded_version}"
        )
    compatibility_input = not (
        "harness_state" in values
        or "checkout_state" in values
        or "provider_state" in values
    )
    _flatten_structured_state(values)
    if loaded_version < CURRENT_SCHEMA_VERSION:
        if compatibility_input:
            values.setdefault("harness_kind", HarnessKind.CURSOR.value)
            if values.get("herdr_target") is not None:
                values.setdefault("session_id", values["herdr_target"])
            elif values.get("session_id") is not None:
                values.setdefault("herdr_target", values["session_id"])
        _legacy_defaults(values)
        for version in range(loaded_version, CURRENT_SCHEMA_VERSION):
            _advance_legacy_version(values, version)
    else:
        if compatibility_input:
            values.setdefault("harness_kind", HarnessKind.CURSOR.value)
            if values.get("herdr_target") is not None:
                values.setdefault("session_id", values["herdr_target"])
        values.setdefault("issue_provider", _infer_legacy_issue_provider(values))
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
            raw_version == CURRENT_SCHEMA_VERSION
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
            }
        )

    def to_dict(self, *, preserve_loaded_version: bool = False) -> dict[str, object]:
        values = dict(self._values)
        if (
            preserve_loaded_version
            and self.loaded_schema_version < CURRENT_SCHEMA_VERSION
        ):
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
        values.update(changes)
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
        )

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
        return self.evolve(
            status=status,
            remove=remove,
            delivered=False,
            delivery_generation=self.delivery_generation + 1,
            delivery_claim_token=None,
            delivery_claimed_at=None,
            delivery_retry_at=0,
            delivery_attempts=0,
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
            values.update(dynamic_changes)
        values.update(changes)
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
        if final_status in TERMINAL_STATUSES and operation:
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
        if prepare_delivery:
            values.update(
                delivered=False,
                delivery_generation=self.delivery_generation + 1,
                delivery_claim_token=None,
                delivery_claimed_at=None,
                delivery_retry_at=0,
                delivery_attempts=0,
                updated_at=now,
            )
        values["schema_version"] = CURRENT_SCHEMA_VERSION
        values["revision"] = self.revision + 1
        updated = CursorJob.from_dict(values)
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
            changes.update(
                {
                    state_key: "confirmed_absent",
                    f"{operation}_confirmed_absent_at": now,
                    f"{operation}_next_reconcile_at": None,
                    "worker_operation": None,
                    "cancellation_reconciliation_pending": False,
                }
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
            elif operation == "worktree":
                if self.parent_job_id is None:
                    changes.update(
                        worktree_path=None,
                        worktree_workspace_id=None,
                        worktree_root_pane_id=None,
                    )
        elif attempts >= uncertain_max_attempts:
            changes.update(
                {
                    state_key: "manual_required",
                    f"{operation}_next_reconcile_at": None,
                    f"{operation}_automatic_reconcile_stopped_at": now,
                    "manual_reconcile_operation": operation,
                    "manual_reconcile_token": uuid.uuid4().hex,
                    "manual_reconcile_required_at": now,
                    "worker_operation": None,
                    "cancellation_reconciliation_pending": False,
                }
            )
        else:
            changes[f"{operation}_next_reconcile_at"] = now + min(
                max_seconds, base_seconds * (2 ** (attempts - 1))
            )
        return self.evolve_recovery(
            changes,
            now=now,
            prepare_delivery=changes.get(state_key) == "manual_required",
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
    def github_pull_request(self) -> int | None:
        return self._optional_int("github_pull_request")

    @property
    def fork_requested(self) -> bool:
        return self._boolean_field("fork_requested")

    @property
    def fork_confirmed(self) -> bool:
        return self._boolean_field("fork_confirmed")

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
        return self._optional_string("pull_request_branch")

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
    def announcement_dismissed(self) -> bool:
        return self._boolean_field("announcement_dismissed")

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
    def plan_artifact(self) -> str | None:
        return self._optional_string("plan_artifact")

    @property
    def review_artifact(self) -> str | None:
        return self._optional_string("review_artifact")

    @property
    def prompt_operation_state(self) -> str:
        return self._optional_string("prompt_operation_state") or "none"

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
    def review_approved(self) -> bool:
        return self._boolean_field("review_approved")

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
            f"{operation}_{'confirmed_absent' if outcome == 'confirmed_absent' else 'retained'}_at": resolved_at,
            "manual_reconcile_operation": None,
            "manual_reconcile_token": None,
            "manual_reconcile_resolved_at": resolved_at,
            "manual_reconcile_outcome": outcome,
            "cancellation_reconciliation_pending": False,
            "target_release_pending": retain_terminal_release,
            "target_release_token": (
                self.target_release_token if retain_terminal_release else None
            ),
            "target_release_owner_pid": None,
            "target_release_owner_boot_id": None,
            "target_release_owner_start": None,
            "worker_operation": None,
            "worker_pid": None,
            "worker_boot_id": None,
            "worker_process_start": None,
            "worker_token": None,
        }
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
                changes.update(
                    participant_creation_state="created",
                    participant_creation_pane_id=resolved_pane,
                    participant_creation_workspace_id=resolved_workspace,
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
        elif outcome == "materialized":
            if operation == "fork":
                if not self.fork_operation_target:
                    return None
                changes.update(
                    fork_exists=True,
                    fork_repository=self.fork_operation_target,
                )
            elif operation == "agent" and not self.herdr_target:
                return None
            elif operation == "worktree" and not self.worktree_path:
                return None
        elif operation == "agent":
            participant = self.active_participant
            changes.update(
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

    def claim_delivery(self, token: str, *, claimed_at: float) -> CursorJob:
        return self._updated(
            delivery_claim_token=token,
            delivery_claimed_at=claimed_at,
            delivery_attempts=self.delivery_attempts + 1,
        )

    def renew_delivery(self, *, claimed_at: float) -> CursorJob:
        return self._updated(delivery_claimed_at=claimed_at)

    def acknowledge_delivery(self, *, delivered_at: float) -> CursorJob:
        return self._updated(
            delivered=True,
            delivery_claim_token=None,
            delivery_claimed_at=None,
            delivery_retry_at=0,
            delivered_at=delivered_at,
        )

    def release_delivery(self, *, retry_at: float) -> CursorJob:
        return self._updated(
            delivery_claim_token=None,
            delivery_claimed_at=None,
            delivery_retry_at=retry_at,
        )

    def _updated(self, **changes: object) -> CursorJob:
        values = dict(self._values)
        values.update(changes)
        values["schema_version"] = CURRENT_SCHEMA_VERSION
        values["revision"] = self.revision + 1
        updated = CursorJob.from_dict(values)
        validate_transition(self, updated)
        return updated

    def validate_invariants(self, *, require_worker_owner: bool = False) -> None:
        if (
            self.github_issue is not None
            and self.issue_key is None
            and self.issue_provider != "github"
        ):
            raise JobValidationError("GitHub issue job requires github issue_provider")
        if self.issue_key is not None and self.issue_provider is None:
            raise JobValidationError("issue-key job requires issue_provider")
        if self.issue_provider == "github" and self.github_issue is None:
            raise JobValidationError(
                "github issue_provider requires a GitHub issue identity"
            )
        if self.issue_provider not in {None, "github"} and self.issue_key is None:
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
        )
        if (
            require_worker_owner
            and self.terminal_intent_status is None
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
        if self.workflow_phase == WorkflowPhase.REVISING and self.review_round >= 2:
            raise JobValidationError(
                "round-two workflow cannot remain in revising phase"
            )
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
        if plan_approval_state not in _PLAN_APPROVAL_STATES:
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
            _PARTICIPANT_CREATION_STATES,
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
        artifact_rounds: dict[str, int] = {}
        for field, kind in (
            ("plan_artifact", "plan"),
            ("review_artifact", "review"),
        ):
            reference = self._optional_string(field)
            if reference is None:
                continue
            match = _WORKFLOW_ARTIFACT_REFERENCE.fullmatch(reference)
            if (
                match is None
                or match.group("job") != self.id
                or match.group("kind") != kind
            ):
                raise JobValidationError(
                    f"{field} has invalid workflow artifact reference"
                )
            artifact_rounds[kind] = int(match.group("round"))
        if self.review_approved and (
            artifact_rounds.get("plan") != self.review_round
            or artifact_rounds.get("review") != self.review_round
        ):
            raise JobValidationError(
                "approved review artifacts must match the current review round"
            )

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
            )
        ):
            raise JobValidationError(
                "released target cannot retain release ownership fields"
            )

    def has_uncertain_operation(self) -> bool:
        return (
            self.agent_dispatch_state in _UNCERTAIN_OPERATION_STATES
            or self.fork_operation_state in _UNCERTAIN_OPERATION_STATES
            or self.worktree_provision_state in _UNCERTAIN_OPERATION_STATES
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


def validate_transition(before: CursorJob, after: CursorJob) -> None:
    if before.id != after.id:
        raise JobValidationError("Cursor job transition cannot change id")
    if before.created_at != after.created_at:
        raise JobValidationError("Cursor job transition cannot change created_at")
    if before.parent_job_id != after.parent_job_id:
        raise JobValidationError("Cursor job transition cannot change parent_job_id")
    if before.harness_kind != after.harness_kind:
        raise JobValidationError("agent job transition cannot change harness_kind")
    if before.issue_provider != after.issue_provider:
        raise JobValidationError("agent job transition cannot change issue_provider")
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
    if (
        before.status != after.status
        and after.status not in _LEGAL_TRANSITIONS[before.status]
    ):
        raise JobValidationError(
            "illegal Cursor job transition "
            f"{before.status.value} -> {after.status.value}"
        )
    if (
        before.workflow_phase != after.workflow_phase
        and after.workflow_phase
        not in _LEGAL_WORKFLOW_TRANSITIONS[before.workflow_phase]
    ):
        raise JobValidationError(
            "illegal Cursor workflow transition "
            f"{before.workflow_phase.value} -> {after.workflow_phase.value}"
        )
    if before.workflow_tier is not None and after.workflow_tier is not None:
        tier_order = {
            WorkflowTier.SIMPLE: 0,
            WorkflowTier.MEDIUM: 1,
            WorkflowTier.HIGH_RISK: 2,
        }
        if tier_order[after.workflow_tier] < tier_order[before.workflow_tier]:
            raise JobValidationError("Cursor workflow tier cannot be downgraded")
    after.validate_invariants(require_worker_owner=True)


def transition(
    job: CursorJob,
    status: JobStatus,
    **changes: Any,
) -> CursorJob:
    values = job.to_dict()
    values.update(changes)
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
            or job.worktree_provision_state in {"quarantined", "manual_required"}
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
        if reserves:
            for target in job_targets:
                owner = targets.setdefault(target, job.id)
                if owner != job.id:
                    raise JobValidationError(
                        f"Herdr target {target} is reserved by both "
                        f"{owner} and {job.id}"
                    )
        worktree_blocked = reserves or job.worktree_provision_state in {
            "quarantined",
            "manual_required",
        }
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
