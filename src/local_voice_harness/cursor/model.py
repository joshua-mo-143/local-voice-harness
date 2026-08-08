from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

CURRENT_SCHEMA_VERSION = 5
LEGACY_SCHEMA_VERSIONS = frozenset(range(CURRENT_SCHEMA_VERSION))


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
        "status",
        "result",
        "error",
        "question",
        "clarification_kind",
        "turn_token",
        "worker_token",
        "worker_process_start",
        "worker_operation",
        "herdr_target",
        "herdr_pane_id",
        "herdr_workspace_id",
        "delivery_claim_token",
        "target_release_token",
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
class CursorJob:
    schema_version: int
    loaded_schema_version: int
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
    _values: dict[str, object]

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CursorJob:
        if not isinstance(raw, dict):
            raise JobValidationError("job must be a JSON object")
        values = dict(raw)
        if "schema_version" not in values:
            loaded_version = 0
        else:
            loaded_version = _integer(values["schema_version"], "schema_version")
        if loaded_version not in LEGACY_SCHEMA_VERSIONS | {CURRENT_SCHEMA_VERSION}:
            raise JobValidationError(
                f"unsupported Cursor job schema version {loaded_version}"
            )
        if loaded_version < CURRENT_SCHEMA_VERSION:
            _legacy_defaults(values)
        values["schema_version"] = CURRENT_SCHEMA_VERSION

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
        try:
            status = JobStatus(str(values.get("status") or ""))
        except ValueError as exc:
            raise JobValidationError("status has invalid value") from exc

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
            _values=values,
        )
        job.validate_invariants(
            require_worker_owner=loaded_version == CURRENT_SCHEMA_VERSION
        )
        return job

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

    def validate_invariants(self, *, require_worker_owner: bool = False) -> None:
        if self.revision < 0:
            raise JobValidationError("revision must not be negative")
        if self.worker_pid is not None and self.worker_pid <= 0:
            raise JobValidationError("worker_pid must be positive")
        worker_claim = (
            self.worker_token,
            self.worker_pid,
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
            and not self.has_uncertain_operation
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
            self._values.get("target_release_owner_start"),
        )
        if bool(release_owner[0]) != bool(release_owner[1]):
            raise JobValidationError(
                "target release owner PID and process start must be paired"
            )
        if not self.target_release_pending and any(
            self._values.get(field)
            for field in (
                "target_release_token",
                "target_release_owner_pid",
                "target_release_owner_start",
            )
        ):
            raise JobValidationError(
                "released target cannot retain release ownership fields"
            )

    @property
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
        reserves = job.status in ACTIVE_STATUSES or job.target_release_pending
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
