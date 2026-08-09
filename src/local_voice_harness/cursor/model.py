from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

CURRENT_SCHEMA_VERSION = 9
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
        {JobStatus.QUEUED, JobStatus.COMPLETED, JobStatus.CANCELLED}
    ),
    JobStatus.BLOCKED: frozenset({JobStatus.QUEUED, JobStatus.CANCELLED}),
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
        "agent_hint",
        "agent_name",
        "issue_key",
        "speakable_label",
        "status",
        "result",
        "error",
        "question",
        "clarification_kind",
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
        "harness_kind",
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
_GITHUB_STATE_FIELDS = frozenset(
    {
        "github_repository",
        "github_issue",
        "github_issue_url",
        "github_issue_context",
        "github_pull_request",
        "fork_requested",
        "fork_confirmed",
        "fork_committed",
        "fork_exists",
        "fork_dispatch_exited",
        "fork_committed_at",
        "fork_operation_state",
        "fork_operation_source",
        "fork_operation_source_url",
        "fork_operation_source_parent",
        "fork_operation_source_default_branch",
        "fork_operation_source_private",
        "fork_operation_login",
        "fork_operation_target",
        "fork_repository",
        "fork_reconcile_attempts",
        "fork_absent_observations",
        "fork_next_reconcile_at",
        "fork_last_reconciled_at",
        "fork_confirmed_absent_at",
        "fork_automatic_reconcile_stopped_at",
        "fork_retained_at",
        "pull_request_worktree_state",
        "pull_request_branch",
        "pull_request_worktree_error",
    }
)
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
    github_state = _object_mapping(
        provider_state.get("github", {}), "provider_state.github"
    )
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
    values["schema_version"] = version + 1


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
    if loaded_version < CURRENT_SCHEMA_VERSION:
        _legacy_defaults(values)
        for version in range(loaded_version, CURRENT_SCHEMA_VERSION):
            _advance_legacy_version(values, version)
    else:
        compatibility_input = not (
            "harness_state" in values
            or "checkout_state" in values
            or "provider_state" in values
        )
        _flatten_structured_state(values)
        if compatibility_input:
            values.setdefault("harness_kind", HarnessKind.CURSOR.value)
            if values.get("herdr_target") is not None:
                values.setdefault("session_id", values["herdr_target"])
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
        provider_state["github"] = github_state
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


@dataclass(frozen=True, slots=True)
class AgentJob:
    schema_version: int
    loaded_schema_version: int
    harness_kind: HarnessKind
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
                changes.update(
                    herdr_target=None,
                    herdr_pane_id=None,
                    herdr_workspace_id=None,
                    agent_name=None,
                    agent_dispatch_exited=None,
                )
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
    def utterance(self) -> str | None:
        return self._optional_string("utterance")

    @property
    def trusted_utterance(self) -> str | None:
        return self._optional_string("trusted_utterance")

    @property
    def repository_hint(self) -> str | None:
        return self._optional_string("repository_hint")

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
        }
        return states.get(operation)

    def resolve_manual_operation(
        self, operation: str, outcome: str, *, resolved_at: float
    ) -> CursorJob | None:
        state_key = {
            "agent": "agent_dispatch_state",
            "fork": "fork_operation_state",
            "worktree": "worktree_provision_state",
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
            "target_release_pending": False,
            "target_release_token": None,
            "target_release_owner_pid": None,
            "target_release_owner_boot_id": None,
            "target_release_owner_start": None,
            "worker_operation": None,
            "worker_pid": None,
            "worker_boot_id": None,
            "worker_process_start": None,
            "worker_token": None,
        }
        if outcome == "materialized":
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
            changes.update(
                herdr_target=None,
                herdr_pane_id=None,
                herdr_workspace_id=None,
                agent_name=None,
                agent_dispatch_exited=None,
                agent_next_reconcile_at=None,
            )
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
            and any(item is not None for item in worker_claim)
            and not all(item is not None for item in worker_claim)
        ):
            raise JobValidationError(
                f"{self.status.value} job requires complete worker ownership"
            )
        if (
            require_worker_owner
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
            if (
                manual_states[self.manual_reconcile_operation] != "manual_required"
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
    after.validate_invariants(require_worker_owner=True)


def transition(
    job: CursorJob,
    status: JobStatus,
    **changes: Any,
) -> CursorJob:
    values = job.to_dict()
    values.update(changes)
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
        if reserves and job.herdr_target:
            owner = targets.setdefault(job.herdr_target, job.id)
            if owner != job.id:
                raise JobValidationError(
                    f"Herdr target {job.herdr_target} is reserved by both "
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
