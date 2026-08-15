"""Executable ownership inventory for durable lifecycle state.

This module is the mechanically checkable contract for issue #357. It records
which representation owns each persisted fact and which module may authorize a
transition. Tests fail when a schema-v18 field or public transition entry point
is added without an inventory row. The inventory does not change runtime
behavior.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
from typing import Any

from . import lifecycle as lifecycle_mod
from . import model as model_mod
from . import operations as operations_mod
from .model import AgentJob
from .sqlite_store import _IMPORT_ONLY_FIELDS, _NAMED_TABLE_FIELDS


class CrashKind(StrEnum):
    IDENTITY = "identity"
    REVISION = "revision"
    TOKEN = "token"
    TIMESTAMP = "timestamp"
    COUNTER = "counter"
    UNCERTAINTY = "uncertainty"
    RECONCILIATION = "reconciliation"
    CONTENT = "content"
    IMPORT = "import"


class AuthorityKind(StrEnum):
    CANONICAL = "canonical"
    CALLER_INTERPRETS = "caller-interprets"
    WRITE_BOUNDARY = "write-boundary"
    COMPATIBILITY = "compatibility"
    ADAPTER = "adapter"


_CRASH = {
    CrashKind.IDENTITY: (
        "Identity fence; a missing, substituted, or unpaired value fails closed "
        "and cannot be reconstructed from a sibling field."
    ),
    CrashKind.REVISION: (
        "CAS fence; a writer must observe the exact current revision and publish "
        "revision+1 or the update is rejected."
    ),
    CrashKind.TOKEN: (
        "Claim or lease fence; a mismatched token cannot advance, release, or "
        "replay the operation."
    ),
    CrashKind.TIMESTAMP: (
        "Ordering, lease, or timeout evidence; required to distinguish in-flight "
        "work from stale or completed work after a crash."
    ),
    CrashKind.COUNTER: (
        "Bounded retry or observation count; the count alone never authorizes "
        "success or blind replay."
    ),
    CrashKind.UNCERTAINTY: (
        "Explicit unknown, ambiguous, quarantined, or manual state; forbids "
        "blessing an incomplete observation as current."
    ),
    CrashKind.RECONCILIATION: (
        "Recovery schedule or retained/absent evidence; only reconciliation may "
        "consume it, and never by retrying a non-idempotent effect."
    ),
    CrashKind.CONTENT: (
        "Product or workflow content. Not a crash-recovery fence; retained for "
        "lossless durability and operator-visible history."
    ),
    CrashKind.IMPORT: (
        "Compatibility-only provenance. Native schema-v18 runtime must not treat "
        "it as a transition authority."
    ),
}


@dataclass(frozen=True, slots=True)
class FieldOwnership:
    name: str
    submachine: str
    persisted: str
    typed_runtime: str
    transition_owner: str
    compatibility_adapter: str
    production_callers: tuple[str, ...]
    crash_kind: CrashKind
    duplicate_authority: str | None = None

    @property
    def crash_boundary(self) -> str:
        return _CRASH[self.crash_kind]


@dataclass(frozen=True, slots=True)
class TransitionEntry:
    qualname: str
    owner: str
    decides: str
    authority: AuthorityKind


@dataclass(frozen=True, slots=True)
class DuplicateAuthority:
    name: str
    representations: tuple[str, ...]
    overlapping_files: tuple[str, ...]
    crash_risk: str
    child_issue: int


@dataclass(frozen=True, slots=True)
class ChildSequence:
    issue: int
    title: str
    blocked_by: tuple[int, ...]
    overlapping_files: tuple[str, ...]
    overlapping_contracts: tuple[str, ...]


def persisted_field_names() -> frozenset[str]:
    named = {field for fields in _NAMED_TABLE_FIELDS.values() for field in fields}
    return frozenset(named | _IMPORT_ONLY_FIELDS | {"id"})


def named_table_for(field: str) -> str:
    if field == "id":
        return "jobs"
    if field in _IMPORT_ONLY_FIELDS:
        return "import_only"
    for table, fields in _NAMED_TABLE_FIELDS.items():
        if field in fields:
            return table
    raise KeyError(field)


def cursor_job_public_properties() -> frozenset[str]:
    return frozenset(
        name for name, value in vars(AgentJob).items() if isinstance(value, property)
    )


def _field(
    name: str,
    *,
    submachine: str,
    typed_runtime: str,
    transition_owner: str,
    adapter: str,
    callers: tuple[str, ...],
    crash: CrashKind,
    duplicate: str | None = None,
) -> FieldOwnership:
    return FieldOwnership(
        name=name,
        submachine=submachine,
        persisted=named_table_for(name),
        typed_runtime=typed_runtime,
        transition_owner=transition_owner,
        compatibility_adapter=adapter,
        production_callers=callers,
        crash_kind=crash,
        duplicate_authority=duplicate,
    )


def _fields(
    names: Mapping[str, CrashKind],
    **shared: Any,
) -> tuple[FieldOwnership, ...]:
    return tuple(_field(name, crash=kind, **shared) for name, kind in names.items())


_STORE_CALLERS = (
    "cursor.store.JobStore.update",
    "cursor.store.JobStore.create",
    "cursor.service",
)
_PROMPT_CALLERS = (
    "cursor.provisioning._execute_phase_prompt",
    "cursor.provisioning._begin_prompt_turn",
    "cursor.recovery.reconcile_prompt_and_pane_operations",
)
_QUESTION_CALLERS = (
    "cursor.questions.ask",
    "cursor.provisioning._mark_prompt_boundary",
    "cursor.recovery._reconcile_question_prompt",
)
_CHECKOUT_CALLERS = (
    "cursor.provisioning._settle_worker_worktree",
    "cursor.provisioning._fail_worker_worktree",
    "cursor.recovery.reconcile_uncertain_worktree",
)
_SESSION_CALLERS = (
    "cursor.provisioning._settle_worker_agent",
    "cursor.provisioning._fail_worker_agent_dispatch",
    "cursor.recovery.reconcile_uncertain_agent",
)


def _build_field_ownership() -> tuple[FieldOwnership, ...]:
    rows: list[FieldOwnership] = []
    rows.extend(
        _fields(
            {
                "id": CrashKind.IDENTITY,
                "parent_job_id": CrashKind.IDENTITY,
                "revision": CrashKind.REVISION,
                "status": CrashKind.IDENTITY,
                "harness_kind": CrashKind.IDENTITY,
                "issue_provider": CrashKind.IDENTITY,
                "speakable_label": CrashKind.IDENTITY,
                "grouped_repository_coordinator_id": CrashKind.IDENTITY,
                "created_at": CrashKind.TIMESTAMP,
                "updated_at": CrashKind.TIMESTAMP,
                "queued_at": CrashKind.TIMESTAMP,
                "started_at": CrashKind.TIMESTAMP,
                "completed_at": CrashKind.TIMESTAMP,
                "foreground_until": CrashKind.TIMESTAMP,
                "reconcile": CrashKind.RECONCILIATION,
                "request": CrashKind.CONTENT,
                "utterance": CrashKind.CONTENT,
                "trusted_utterance": CrashKind.CONTENT,
                "continuation": CrashKind.CONTENT,
                "continuation_answer": CrashKind.CONTENT,
                "context_repository": CrashKind.CONTENT,
                "repository_hint": CrashKind.CONTENT,
                "grouped_repository_targets": CrashKind.CONTENT,
                "grouped_repository_candidates": CrashKind.CONTENT,
                "grouped_repository_launches": CrashKind.CONTENT,
            },
            submachine="identity",
            typed_runtime="JobIdentity / JobState / CursorJob.status",
            transition_owner="cursor.model.transition / job_lifecycle.apply_event",
            adapter="CursorJob.from_dict",
            callers=_STORE_CALLERS,
        )
    )
    rows.extend(
        _fields(
            {
                "prompt_operation_state": CrashKind.UNCERTAINTY,
                "prompt_operation_phase": CrashKind.IDENTITY,
                "prompt_operation_turn": CrashKind.COUNTER,
                "prompt_operation_target": CrashKind.IDENTITY,
                "prompt_operation_agent_session": CrashKind.IDENTITY,
                "prompt_baseline_sequence": CrashKind.COUNTER,
                "turn": CrashKind.COUNTER,
                "turn_token": CrashKind.TOKEN,
                "prompt_context_sessions": CrashKind.CONTENT,
                "prompt_manifest": CrashKind.CONTENT,
            },
            submachine="prompt",
            typed_runtime="prompt_operations.PromptOperation",
            transition_owner="prompt_operations.plan_prompt and submit transitions",
            adapter="prompt_operations.load_prompt_operation",
            callers=_PROMPT_CALLERS,
            duplicate=(
                "Ordinary runtime still reconstructs PromptOperation from flat "
                "job columns; question JSON prompt_state is a persistence-edge "
                "copy of the same shared enum."
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "voice_question": CrashKind.IDENTITY,
                "question": CrashKind.CONTENT,
                "clarification_kind": CrashKind.CONTENT,
                "clarifications": CrashKind.CONTENT,
                "interactive_questionnaire_blocked": CrashKind.UNCERTAINTY,
            },
            submachine="question",
            typed_runtime="questions.Question / QuestionState",
            transition_owner="questions.transition_question",
            adapter="questions.Question.from_dict / cursor.questions.current",
            callers=_QUESTION_CALLERS,
            duplicate=(
                "Question.prompt_state persists the shared PromptOperationState "
                "inside voice_question JSON rather than the job-level typed "
                "operation. Answer policy remains question-owned."
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "terminal_intent_status": CrashKind.UNCERTAINTY,
                "terminal_intent_result": CrashKind.CONTENT,
                "terminal_intent_error": CrashKind.UNCERTAINTY,
                "terminal_intent_completed_at": CrashKind.TIMESTAMP,
                "result": CrashKind.CONTENT,
                "error": CrashKind.UNCERTAINTY,
            },
            submachine="terminal",
            typed_runtime="TerminalIntent / MaterializedTerminalOutcome",
            transition_owner="cursor.lifecycle.load_terminal_state / CursorJob.evolve",
            adapter="cursor.lifecycle.load_terminal_state",
            callers=(
                "cursor.recovery.stage_terminal_intent",
                "cursor.provisioning._worker_complete",
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "delivered": CrashKind.IDENTITY,
                "delivered_at": CrashKind.TIMESTAMP,
                "delivery_attempts": CrashKind.COUNTER,
                "delivery_generation": CrashKind.COUNTER,
                "delivery_claim_token": CrashKind.TOKEN,
                "delivery_claimed_at": CrashKind.TIMESTAMP,
                "delivery_retry_at": CrashKind.TIMESTAMP,
                "announcement_ack": CrashKind.IDENTITY,
                "announcement_dismissed": CrashKind.CONTENT,
                "announcement_repeated": CrashKind.CONTENT,
            },
            submachine="delivery",
            typed_runtime="cursor.lifecycle.DeliveryState / AnnouncementState",
            transition_owner="cursor.lifecycle.claim_delivery and announcement transitions",
            adapter="cursor.lifecycle.load_delivery_state",
            callers=(
                "cursor.delivery.claim_delivery",
                "cursor.delivery.acknowledge_delivery",
                "cursor.delivery.release_delivery",
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "workflow_tier": CrashKind.CONTENT,
                "workflow_phase": CrashKind.IDENTITY,
                "workflow_classification_reason": CrashKind.CONTENT,
                "workflow_turn_phase": CrashKind.IDENTITY,
                "active_participant": CrashKind.IDENTITY,
                "planner_target": CrashKind.IDENTITY,
                "reviewer_target": CrashKind.IDENTITY,
                "implementer_target": CrashKind.IDENTITY,
                "review_round": CrashKind.COUNTER,
                "review_approved": CrashKind.CONTENT,
                "review_approval_source": CrashKind.IDENTITY,
                "review_decision": CrashKind.CONTENT,
                "review_artifact": CrashKind.IDENTITY,
                "plan_artifact": CrashKind.IDENTITY,
                "plan_approval_state": CrashKind.UNCERTAINTY,
                "plan_approval_id": CrashKind.IDENTITY,
                "plan_approval_source": CrashKind.IDENTITY,
                "plan_approval_agent_session": CrashKind.IDENTITY,
                "plan_approval_plan_artifact": CrashKind.IDENTITY,
                "plan_approval_review_artifact": CrashKind.IDENTITY,
                "plan_approval_state_change_sequence": CrashKind.COUNTER,
                "plan_approval_revision": CrashKind.REVISION,
                "plan_approval_counted": CrashKind.COUNTER,
                "plan_approval_completion_pending": CrashKind.UNCERTAINTY,
                "participant_admission_state": CrashKind.UNCERTAINTY,
                "participant_creation_state": CrashKind.UNCERTAINTY,
                "participant_creation_participant": CrashKind.IDENTITY,
                "participant_creation_target": CrashKind.IDENTITY,
                "participant_creation_label": CrashKind.IDENTITY,
                "participant_creation_workspace_id": CrashKind.IDENTITY,
                "participant_creation_pane_id": CrashKind.IDENTITY,
                "participant_creation_checkout": CrashKind.IDENTITY,
                "participant_session_owners": CrashKind.IDENTITY,
            },
            submachine="workflow",
            typed_runtime="cursor.workflow.WorkflowState / PlanApproval / ParticipantLifecycle",
            transition_owner="CursorJob.evolve_workflow / evolve_review / evolve_plan_approval / evolve_participant",
            adapter="CursorJob.workflow_state / plan_approval / participant_lifecycle",
            callers=(
                "cursor.provisioning._run_tiered_workflow",
                "cursor.provisioning._plan_participant_creation",
                "cursor.questions.ask",
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "repository": CrashKind.IDENTITY,
                "worktree_branch": CrashKind.IDENTITY,
                "worktree_path": CrashKind.IDENTITY,
                "worktree_label": CrashKind.IDENTITY,
                "worktree_workspace_id": CrashKind.IDENTITY,
                "worktree_root_pane_id": CrashKind.IDENTITY,
                "worktree_provision_state": CrashKind.UNCERTAINTY,
                "worktree_provision_error": CrashKind.UNCERTAINTY,
                "worktree_dispatch_exited": CrashKind.UNCERTAINTY,
                "worktree_absent_observations": CrashKind.COUNTER,
                "worktree_reconcile_attempts": CrashKind.COUNTER,
                "worktree_last_reconciled_at": CrashKind.TIMESTAMP,
                "worktree_next_reconcile_at": CrashKind.TIMESTAMP,
                "worktree_automatic_reconcile_stopped_at": CrashKind.TIMESTAMP,
                "worktree_confirmed_absent_at": CrashKind.TIMESTAMP,
                "worktree_retained_at": CrashKind.TIMESTAMP,
                "worktree_manual_inspection_required": CrashKind.UNCERTAINTY,
                "worktree_quarantine_acknowledged_at": CrashKind.TIMESTAMP,
                "pull_request_worktree_state": CrashKind.UNCERTAINTY,
                "pull_request_worktree_error": CrashKind.UNCERTAINTY,
                "pull_request_branch": CrashKind.IDENTITY,
                "pull_request_remote_url": CrashKind.IDENTITY,
                "pull_request_head_ref": CrashKind.IDENTITY,
                "pull_request_head_oid": CrashKind.IDENTITY,
                "fork_repository": CrashKind.IDENTITY,
                "fork_requested": CrashKind.CONTENT,
                "fork_confirmed": CrashKind.UNCERTAINTY,
                "fork_exists": CrashKind.UNCERTAINTY,
                "fork_committed": CrashKind.UNCERTAINTY,
                "fork_committed_at": CrashKind.TIMESTAMP,
                "fork_operation_state": CrashKind.UNCERTAINTY,
                "fork_operation_source": CrashKind.IDENTITY,
                "fork_operation_source_url": CrashKind.IDENTITY,
                "fork_operation_source_parent": CrashKind.IDENTITY,
                "fork_operation_source_default_branch": CrashKind.IDENTITY,
                "fork_operation_source_private": CrashKind.CONTENT,
                "fork_operation_login": CrashKind.IDENTITY,
                "fork_operation_target": CrashKind.IDENTITY,
                "fork_dispatch_exited": CrashKind.UNCERTAINTY,
                "fork_absent_observations": CrashKind.COUNTER,
                "fork_reconcile_attempts": CrashKind.COUNTER,
                "fork_last_reconciled_at": CrashKind.TIMESTAMP,
                "fork_next_reconcile_at": CrashKind.TIMESTAMP,
                "fork_automatic_reconcile_stopped_at": CrashKind.TIMESTAMP,
                "fork_confirmed_absent_at": CrashKind.TIMESTAMP,
                "fork_retained_at": CrashKind.TIMESTAMP,
            },
            submachine="checkout",
            typed_runtime="CheckoutOperation / ForkOperation",
            transition_owner="cursor.operations.CheckoutOperation.transition / ForkOperation.transition",
            adapter="CursorJob.checkout_operation / fork_operation",
            callers=_CHECKOUT_CALLERS + ("cursor.recovery.reconcile_uncertain_fork",),
            duplicate=(
                "Fork labels such as retained, quarantined, confirmed_absent, "
                "and failed_observing still collapse into generic "
                "SETTLED/UNKNOWN/FAILED. Checkout now uses CheckoutState."
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "github_repository": CrashKind.IDENTITY,
                "github_issue": CrashKind.IDENTITY,
                "github_issue_url": CrashKind.IDENTITY,
                "github_issue_context": CrashKind.CONTENT,
                "github_pull_request": CrashKind.IDENTITY,
                "issue_key": CrashKind.IDENTITY,
                "github_issue_create_requested": CrashKind.CONTENT,
                "github_issue_create_confirmed": CrashKind.UNCERTAINTY,
                "github_issue_create_title": CrashKind.CONTENT,
                "github_issue_create_body": CrashKind.CONTENT,
                "github_issue_create_marker": CrashKind.TOKEN,
                "github_issue_create_operation_state": CrashKind.UNCERTAINTY,
                "github_issue_created_number": CrashKind.IDENTITY,
                "github_issue_created_url": CrashKind.IDENTITY,
                "linear_ticket_create_requested": CrashKind.CONTENT,
                "linear_ticket_create_confirmed": CrashKind.UNCERTAINTY,
                "linear_ticket_create_team": CrashKind.IDENTITY,
                "linear_ticket_create_team_id": CrashKind.IDENTITY,
                "linear_ticket_create_title": CrashKind.CONTENT,
                "linear_ticket_create_description": CrashKind.CONTENT,
                "linear_ticket_create_marker": CrashKind.TOKEN,
                "linear_ticket_create_operation_state": CrashKind.UNCERTAINTY,
                "linear_ticket_create_prompt_target": CrashKind.IDENTITY,
                "linear_ticket_create_prompt_session": CrashKind.IDENTITY,
                "linear_ticket_create_prompt_token": CrashKind.TOKEN,
                "linear_ticket_create_baseline_sequence": CrashKind.COUNTER,
                "linear_ticket_created_identifier": CrashKind.IDENTITY,
                "linear_ticket_created_url": CrashKind.IDENTITY,
            },
            submachine="ticket",
            typed_runtime="CursorJob provider ticket fields",
            transition_owner="cursor.provisioning ticket create helpers / recovery issue-creation reconcile",
            adapter="CursorJob.from_dict",
            callers=(
                "cursor.provisioning._run_github_issue_creation",
                "cursor.provisioning._run_linear_ticket_creation",
                "cursor.recovery.reconcile_uncertain_issue_creation",
                "cursor.recovery.reconcile_uncertain_linear_ticket_creation",
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "session_id": CrashKind.IDENTITY,
                "herdr_target": CrashKind.IDENTITY,
                "herdr_pane_id": CrashKind.IDENTITY,
                "herdr_workspace_id": CrashKind.IDENTITY,
                "agent_name": CrashKind.CONTENT,
                "agent_hint": CrashKind.CONTENT,
                "agent_dispatch_state": CrashKind.UNCERTAINTY,
                "agent_dispatch_exited": CrashKind.UNCERTAINTY,
                "agent_provider": CrashKind.IDENTITY,
                "agent_provider_session_id": CrashKind.IDENTITY,
                "agent_state_sequence": CrashKind.COUNTER,
                "agent_operation_checkout": CrashKind.IDENTITY,
                "agent_operation_target": CrashKind.IDENTITY,
                "agent_operation_workspace_id": CrashKind.IDENTITY,
                "agent_operation_pane_id": CrashKind.IDENTITY,
                "agent_absent_observations": CrashKind.COUNTER,
                "agent_reconcile_attempts": CrashKind.COUNTER,
                "agent_last_reconciled_at": CrashKind.TIMESTAMP,
                "agent_next_reconcile_at": CrashKind.TIMESTAMP,
                "agent_automatic_reconcile_stopped_at": CrashKind.TIMESTAMP,
                "agent_confirmed_absent_at": CrashKind.TIMESTAMP,
                "agent_retained_at": CrashKind.TIMESTAMP,
                "attempt_started_at": CrashKind.TIMESTAMP,
                "pane_retained_at": CrashKind.TIMESTAMP,
                "next_reconcile_at": CrashKind.TIMESTAMP,
                "cancellation_reconciliation_pending": CrashKind.RECONCILIATION,
                "manual_reconcile_operation": CrashKind.UNCERTAINTY,
                "manual_reconcile_outcome": CrashKind.UNCERTAINTY,
                "manual_reconcile_token": CrashKind.TOKEN,
                "manual_reconcile_required_at": CrashKind.TIMESTAMP,
                "manual_reconcile_resolved_at": CrashKind.TIMESTAMP,
                "reconciliation_base_error": CrashKind.UNCERTAINTY,
                "target_release_pending": CrashKind.RECONCILIATION,
                "target_release_token": CrashKind.TOKEN,
                "target_release_owner_pid": CrashKind.IDENTITY,
                "target_release_owner_boot_id": CrashKind.IDENTITY,
                "target_release_owner_start": CrashKind.IDENTITY,
                "target_release_manual_required": CrashKind.UNCERTAINTY,
                "target_release_unverified_targets": CrashKind.UNCERTAINTY,
            },
            submachine="session",
            typed_runtime="AgentSessionOperation / CleanupState / SessionIdentity",
            transition_owner="cursor.operations.AgentSessionOperation.transition / cursor.lifecycle cleanup transitions",
            adapter="CursorJob.agent_session_operation / load_cleanup_state",
            callers=_SESSION_CALLERS
            + (
                "cursor.recovery.cancel_target_and_release",
                "cursor.recovery.resolve_manual_reconciliation",
            ),
            duplicate=(
                "session_id aliases herdr_target. Agent session labels now use "
                "AgentSessionState; herdr_target remains a live-binding alias."
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "worker_token": CrashKind.TOKEN,
                "worker_pid": CrashKind.IDENTITY,
                "worker_boot_id": CrashKind.IDENTITY,
                "worker_process_start": CrashKind.IDENTITY,
                "worker_operation": CrashKind.IDENTITY,
                "worker_claim_operation": CrashKind.IDENTITY,
                "worker_claimed_at": CrashKind.TIMESTAMP,
            },
            submachine="worker",
            typed_runtime="WorkerOwnership",
            transition_owner="cursor.model.worker_callback_transition / CursorJob.clear_worker",
            adapter="cursor.operations.load_worker_ownership",
            callers=(
                "cursor.worker_lifecycle.begin_worker",
                "cursor.provisioning._worker_change",
                "cursor.recovery.recover_jobs",
            ),
        )
    )
    rows.extend(
        _fields(
            {
                "schema_version": CrashKind.IMPORT,
                "migration_source_schema_version": CrashKind.IMPORT,
                "phase_prompt_active": CrashKind.IMPORT,
                "agent_identity_legacy_compatible": CrashKind.IMPORT,
            },
            submachine="import",
            typed_runtime="CursorJob.loaded_schema_version / import flags",
            transition_owner="cursor.model.migrate_job_record",
            adapter="migrate_job_record / sqlite_store relational reader",
            callers=("cursor.store.JobStore", "cursor.sqlite_store.SQLiteJobDatabase"),
        )
    )
    return tuple(rows)


FIELD_OWNERSHIP: tuple[FieldOwnership, ...] = _build_field_ownership()
FIELD_OWNERSHIP_BY_NAME: dict[str, FieldOwnership] = {
    row.name: row for row in FIELD_OWNERSHIP
}

SUBMACHINES: frozenset[str] = frozenset(row.submachine for row in FIELD_OWNERSHIP)

TOP_LEVEL_STATES: frozenset[str] = frozenset(
    status.value for status in model_mod.JobStatus
)

DUPLICATE_AUTHORITIES: tuple[DuplicateAuthority, ...] = ()

CHILD_SEQUENCE: tuple[ChildSequence, ...] = (
    ChildSequence(
        issue=358,
        title="Consolidate prompt and clarification submission lifecycle state",
        blocked_by=(357,),
        overlapping_files=(
            "src/local_voice_harness/prompt_operations.py",
            "src/local_voice_harness/questions/__init__.py",
            "src/local_voice_harness/cursor/model.py",
            "src/local_voice_harness/cursor/questions.py",
            "src/local_voice_harness/cursor/provisioning.py",
            "src/local_voice_harness/cursor/recovery.py",
        ),
        overlapping_contracts=(
            "PromptOperationState",
            "Question.prompt_state",
            "prompt_operation_* columns",
        ),
    ),
    ChildSequence(
        issue=359,
        title="Separate canonical runtime lifecycle state from import compatibility",
        blocked_by=(357, 358),
        overlapping_files=(
            "src/local_voice_harness/cursor/model.py",
            "src/local_voice_harness/prompt_operations.py",
            "src/local_voice_harness/cursor/provisioning.py",
            "src/local_voice_harness/cursor/recovery.py",
        ),
        overlapping_contracts=(
            "CursorJob.prompt_operation",
            "load_prompt_operation",
            "legacy_prompt_fields",
        ),
    ),
    ChildSequence(
        issue=360,
        title="Consolidate checkout and session operation lifecycle ownership",
        blocked_by=(357, 359),
        overlapping_files=(
            "src/local_voice_harness/cursor/operations.py",
            "src/local_voice_harness/cursor/model.py",
            "src/local_voice_harness/cursor/provisioning.py",
            "src/local_voice_harness/cursor/recovery.py",
        ),
        overlapping_contracts=(
            "CheckoutOperation",
            "AgentSessionOperation",
            "worktree_provision_state",
            "agent_dispatch_state",
        ),
    ),
)

_TYPED_PREFIXES: dict[str, tuple[str, ...]] = {
    "local_voice_harness.prompt_operations": (
        "plan_",
        "begin_",
        "accept_",
        "observe_",
        "mark_",
        "record_",
        "resolve_",
        "replan_",
    ),
    "local_voice_harness.questions": ("transition_question",),
    "local_voice_harness.job_lifecycle": ("apply_event", "apply_follow_up"),
    "local_voice_harness.cursor.model": (
        "transition",
        "worker_callback_transition",
        "validate_transition",
    ),
    "local_voice_harness.cursor.recovery": (
        "reconcile_",
        "recover_jobs",
        "cancel_target_and_release",
        "stage_terminal_intent",
        "acknowledge_worktree_quarantine",
        "resolve_manual_reconciliation",
    ),
    "local_voice_harness.cursor.delivery": (
        "claim_delivery",
        "renew_delivery",
        "acknowledge_delivery",
        "acknowledge_desktop_delivery",
        "acknowledge_deferred_delivery",
        "release_delivery",
        "acknowledge_deliveries",
        "release_deliveries",
    ),
    "local_voice_harness.cursor.lifecycle": (
        "begin_cleanup",
        "finish_cleanup_reconciliation",
        "claim_cleanup",
        "take_over_cleanup",
        "abandon_cleanup_owner",
        "settle_cleanup",
        "acknowledge_announcement",
        "dismiss_announcement",
        "repeat_announcement",
        "prepare_delivery",
        "claim_delivery",
        "renew_delivery",
        "acknowledge_delivery",
        "acknowledge_without_claim",
        "release_delivery",
    ),
}

_AGENT_JOB_TRANSITION_METHODS = frozenset(
    {
        "evolve",
        "evolve_for_delivery",
        "evolve_recovery",
        "evolve_workflow",
        "evolve_review",
        "evolve_plan_approval",
        "evolve_participant",
        "clear_worker",
        "prepare_delivery",
    }
)

_OPERATION_TRANSITION_TYPES = (
    operations_mod.CheckoutOperation,
    operations_mod.ForkOperation,
    operations_mod.AgentSessionOperation,
    operations_mod.ParticipantPaneOperation,
)

_STORE_WRITE_BOUNDARIES = frozenset(
    {
        "local_voice_harness.cursor.store.JobStore.create",
        "local_voice_harness.cursor.store.JobStore.update",
        "local_voice_harness.cursor.store.JobStore.apply",
        "local_voice_harness.cursor.store.JobStore.create_follow_up",
        "local_voice_harness.cursor.store.JobStore.reserve_target",
        "local_voice_harness.cursor.store.JobStore.reserve_worktree",
        "local_voice_harness.cursor.store.JobStore.update_unless_maintenance",
    }
)

_PROVISIONING_PUBLIC_ENTRIES = frozenset(
    {
        "local_voice_harness.cursor.provisioning.run_claimed_worker",
        "local_voice_harness.cursor.provisioning.complete_from_output",
    }
)


def _public_functions(module: ModuleType) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(module).items()
        if inspect.isfunction(value) and not name.startswith("_")
    }


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        name == prefix or (prefix.endswith("_") and name.startswith(prefix))
        for prefix in prefixes
        if prefix
    )


def discover_transition_entry_points() -> frozenset[str]:
    """Public production APIs that authorize or persist a lifecycle transition."""

    found: set[str] = set()
    modules = {
        "local_voice_harness.prompt_operations": __import__(
            "local_voice_harness.prompt_operations", fromlist=["*"]
        ),
        "local_voice_harness.questions": __import__(
            "local_voice_harness.questions", fromlist=["*"]
        ),
        "local_voice_harness.job_lifecycle": __import__(
            "local_voice_harness.job_lifecycle", fromlist=["*"]
        ),
        "local_voice_harness.cursor.model": model_mod,
        "local_voice_harness.cursor.recovery": __import__(
            "local_voice_harness.cursor.recovery", fromlist=["*"]
        ),
        "local_voice_harness.cursor.delivery": __import__(
            "local_voice_harness.cursor.delivery", fromlist=["*"]
        ),
        "local_voice_harness.cursor.lifecycle": lifecycle_mod,
        "local_voice_harness.cursor.provisioning": __import__(
            "local_voice_harness.cursor.provisioning", fromlist=["*"]
        ),
        "local_voice_harness.cursor.store": __import__(
            "local_voice_harness.cursor.store", fromlist=["*"]
        ),
    }
    for module_name, prefixes in _TYPED_PREFIXES.items():
        for name, value in _public_functions(modules[module_name]).items():
            if _matches(name, prefixes):
                found.add(f"{value.__module__}.{name}")
    for name in _AGENT_JOB_TRANSITION_METHODS:
        method = getattr(AgentJob, name)
        found.add(f"{method.__module__}.AgentJob.{name}")
    for operation_type in _OPERATION_TRANSITION_TYPES:
        found.add(
            f"{operation_type.transition.__module__}.{operation_type.__name__}.transition"
        )
    found.update(_STORE_WRITE_BOUNDARIES)
    found.update(_PROVISIONING_PUBLIC_ENTRIES)
    return frozenset(found)


def _entry(
    qualname: str,
    owner: str,
    decides: str,
    authority: AuthorityKind,
) -> TransitionEntry:
    return TransitionEntry(qualname, owner, decides, authority)


TRANSITION_ENTRY_POINTS: tuple[TransitionEntry, ...] = (
    _entry(
        "local_voice_harness.prompt_operations.plan_prompt",
        "prompt_operations",
        "idle to planned ordinary prompt",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.begin_prompt_submission",
        "prompt_operations",
        "planned to submitting dispatch fence",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.accept_prompt_submission",
        "prompt_operations",
        "submitting to submitted acceptance",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.observe_prompt_submission",
        "prompt_operations",
        "submitting to submitted or ambiguous from evidence",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.mark_prompt_ambiguous",
        "prompt_operations",
        "in-flight prompt to ambiguous",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.record_prompt_submitted",
        "prompt_operations",
        "planned or submitting to submitted fence",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.observe_accepted_prompt",
        "prompt_operations",
        "submitted to observed",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.resolve_prompt",
        "prompt_operations",
        "planned, submitted, or observed to resolved",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.prompt_operations.replan_unobserved_prompt",
        "prompt_operations",
        "submitted to planned after confirmed absence",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.questions.transition_question",
        "questions",
        "question answer-policy state",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.job_lifecycle.apply_event",
        "job_lifecycle",
        "top-level JobLifecycle variant",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.job_lifecycle.apply_follow_up",
        "job_lifecycle",
        "completed parent to queued child",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.transition",
        "cursor.model",
        "flat CursorJob status and field mutation",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.model.worker_callback_transition",
        "cursor.model",
        "worker-fenced CursorJob mutation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.validate_transition",
        "cursor.model",
        "cross-field and typed operation validation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.evolve",
        "cursor.model",
        "generic flat-field mutation used by most callers",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.evolve_for_delivery",
        "cursor.model",
        "delivery-prepared mutation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.evolve_recovery",
        "cursor.model",
        "recovery-tagged mutation",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.evolve_workflow",
        "cursor.model",
        "workflow phase mutation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.evolve_review",
        "cursor.model",
        "review mutation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.evolve_plan_approval",
        "cursor.model",
        "plan-approval mutation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.evolve_participant",
        "cursor.model",
        "participant mutation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.clear_worker",
        "cursor.model",
        "worker ownership release",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.model.AgentJob.prepare_delivery",
        "cursor.model",
        "terminal delivery preparation",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.operations.CheckoutOperation.transition",
        "cursor.operations",
        "typed checkout state",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.operations.ForkOperation.transition",
        "cursor.operations",
        "typed fork state",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.operations.AgentSessionOperation.transition",
        "cursor.operations",
        "typed session state",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.operations.ParticipantPaneOperation.transition",
        "cursor.operations",
        "typed participant-pane state",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.begin_cleanup",
        "cursor.lifecycle",
        "cleanup pending",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.finish_cleanup_reconciliation",
        "cursor.lifecycle",
        "cleanup reconciliation bit",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.claim_cleanup",
        "cursor.lifecycle",
        "cleanup ownership claim",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.take_over_cleanup",
        "cursor.lifecycle",
        "cleanup ownership takeover",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.abandon_cleanup_owner",
        "cursor.lifecycle",
        "cleanup owner release",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.settle_cleanup",
        "cursor.lifecycle",
        "cleanup settlement",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.acknowledge_announcement",
        "cursor.lifecycle",
        "announcement ack",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.dismiss_announcement",
        "cursor.lifecycle",
        "announcement dismiss",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.repeat_announcement",
        "cursor.lifecycle",
        "announcement repeat",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.prepare_delivery",
        "cursor.lifecycle",
        "typed delivery prepare",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.claim_delivery",
        "cursor.lifecycle",
        "typed delivery claim",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.renew_delivery",
        "cursor.lifecycle",
        "typed delivery renew",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.acknowledge_delivery",
        "cursor.lifecycle",
        "typed delivery ack",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.acknowledge_without_claim",
        "cursor.lifecycle",
        "typed delivery ack without claim",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.lifecycle.release_delivery",
        "cursor.lifecycle",
        "typed delivery release",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.reconcile_uncertain_agent",
        "cursor.recovery",
        "session observation to next agent_dispatch_state",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.reconcile_uncertain_fork",
        "cursor.recovery",
        "fork observation to next fork_operation_state",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.reconcile_uncertain_issue_creation",
        "cursor.recovery",
        "GitHub issue-create observation",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.reconcile_uncertain_linear_ticket_creation",
        "cursor.recovery",
        "Linear ticket-create observation",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.reconcile_uncertain_worktree",
        "cursor.recovery",
        "checkout observation to next worktree_provision_state",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.reconcile_uncertain_operations",
        "cursor.recovery",
        "fan-out to operation reconcilers",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.reconcile_prompt_and_pane_operations",
        "cursor.recovery",
        "prompt and pane uncertainty",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.recover_jobs",
        "cursor.recovery",
        "startup and periodic recovery admission",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.cancel_target_and_release",
        "cursor.recovery",
        "cancellation cleanup and target release",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.stage_terminal_intent",
        "cursor.recovery",
        "terminal intent before cleanup",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.acknowledge_worktree_quarantine",
        "cursor.recovery",
        "worktree quarantine acknowledgement",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.recovery.resolve_manual_reconciliation",
        "cursor.recovery",
        "manual reconcile token resolution",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.claim_delivery",
        "cursor.delivery",
        "store-backed delivery claim",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.renew_delivery",
        "cursor.delivery",
        "store-backed delivery renew",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.acknowledge_delivery",
        "cursor.delivery",
        "store-backed spoken ack",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.acknowledge_desktop_delivery",
        "cursor.delivery",
        "store-backed desktop ack",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.acknowledge_deferred_delivery",
        "cursor.delivery",
        "store-backed deferred ack",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.release_delivery",
        "cursor.delivery",
        "store-backed delivery release",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.acknowledge_deliveries",
        "cursor.delivery",
        "batch delivery ack",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.delivery.release_deliveries",
        "cursor.delivery",
        "batch delivery release",
        AuthorityKind.CANONICAL,
    ),
    _entry(
        "local_voice_harness.cursor.provisioning.run_claimed_worker",
        "cursor.provisioning",
        "worker-owned provisioning and prompt dispatch",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.provisioning.complete_from_output",
        "cursor.provisioning",
        "completion from provider output",
        AuthorityKind.CALLER_INTERPRETS,
    ),
    _entry(
        "local_voice_harness.cursor.store.JobStore.create",
        "cursor.store",
        "durable create write boundary",
        AuthorityKind.WRITE_BOUNDARY,
    ),
    _entry(
        "local_voice_harness.cursor.store.JobStore.update",
        "cursor.store",
        "durable update write boundary",
        AuthorityKind.WRITE_BOUNDARY,
    ),
    _entry(
        "local_voice_harness.cursor.store.JobStore.apply",
        "cursor.store",
        "coordinator command write boundary",
        AuthorityKind.WRITE_BOUNDARY,
    ),
    _entry(
        "local_voice_harness.cursor.store.JobStore.create_follow_up",
        "cursor.store",
        "follow-up create write boundary",
        AuthorityKind.WRITE_BOUNDARY,
    ),
    _entry(
        "local_voice_harness.cursor.store.JobStore.reserve_target",
        "cursor.store",
        "target reservation write boundary",
        AuthorityKind.WRITE_BOUNDARY,
    ),
    _entry(
        "local_voice_harness.cursor.store.JobStore.reserve_worktree",
        "cursor.store",
        "worktree reservation write boundary",
        AuthorityKind.WRITE_BOUNDARY,
    ),
    _entry(
        "local_voice_harness.cursor.store.JobStore.update_unless_maintenance",
        "cursor.store",
        "maintenance-fenced update write boundary",
        AuthorityKind.WRITE_BOUNDARY,
    ),
)

TRANSITION_ENTRY_POINT_NAMES: frozenset[str] = frozenset(
    entry.qualname for entry in TRANSITION_ENTRY_POINTS
)

COMPATIBILITY_ADAPTERS: frozenset[str] = frozenset(
    {
        "local_voice_harness.prompt_operations.load_prompt_operation",
        "local_voice_harness.prompt_operations.legacy_prompt_fields",
        "local_voice_harness.cursor.model._typed_prompt_operation",
        "local_voice_harness.cursor.model._optional_typed_prompt_operation",
        "local_voice_harness.cursor.model._consume_typed_prompt_operation",
        "local_voice_harness.cursor.model._typed_checkout_operation",
        "local_voice_harness.cursor.model._optional_typed_checkout_operation",
        "local_voice_harness.cursor.model._consume_typed_checkout_operation",
        "local_voice_harness.cursor.model._typed_agent_session_operation",
        "local_voice_harness.cursor.model._optional_typed_agent_session_operation",
        "local_voice_harness.cursor.model._consume_typed_agent_session_operation",
        "local_voice_harness.cursor.operations.load_checkout_operation",
        "local_voice_harness.cursor.operations.checkout_fields",
        "local_voice_harness.cursor.operations.load_agent_session_operation",
        "local_voice_harness.cursor.operations.agent_session_fields",
        "local_voice_harness.cursor.operations.load_worker_ownership",
        "local_voice_harness.cursor.operations.worker_ownership_fields",
        "local_voice_harness.cursor.lifecycle.load_terminal_state",
        "local_voice_harness.cursor.lifecycle.load_cleanup_state",
        "local_voice_harness.cursor.lifecycle.cleanup_fields",
        "local_voice_harness.cursor.lifecycle.load_delivery_state",
        "local_voice_harness.cursor.lifecycle.delivery_fields",
        "local_voice_harness.cursor.model.migrate_job_record",
        "local_voice_harness.cursor.model.CursorJob.from_dict",
        "local_voice_harness.cursor.model.CursorJob.to_dict",
        "local_voice_harness.questions.Question.from_dict",
        "local_voice_harness.questions.Question.to_dict",
        "local_voice_harness.cursor.questions.current",
        "local_voice_harness.cursor.questions.envelope",
        "local_voice_harness.cursor.questions.shared_prompt_identity",
        "local_voice_harness.questions.question_prompt_identity",
        "local_voice_harness.questions.load_question_prompt",
        "local_voice_harness.questions.bind_question_prompt",
        "local_voice_harness.cursor.sqlite_store.SQLiteJobDatabase.load_job",
        "local_voice_harness.cursor.sqlite_store.SQLiteJobDatabase.save_job",
        "local_voice_harness.cursor.model.AgentJob.checkout_operation",
        "local_voice_harness.cursor.model.AgentJob.agent_session_operation",
        "local_voice_harness.cursor.model.AgentJob.prompt_operation",
    }
)

LIFECYCLE_MODULE_PATHS: tuple[str, ...] = (
    "src/local_voice_harness/cursor/model.py",
    "src/local_voice_harness/cursor/provisioning.py",
    "src/local_voice_harness/cursor/recovery.py",
    "src/local_voice_harness/cursor/store.py",
    "src/local_voice_harness/cursor/sqlite_store.py",
    "src/local_voice_harness/cursor/operations.py",
    "src/local_voice_harness/cursor/lifecycle.py",
    "src/local_voice_harness/cursor/coordinator.py",
    "src/local_voice_harness/cursor/questions.py",
    "src/local_voice_harness/cursor/delivery.py",
    "src/local_voice_harness/cursor/workflow.py",
    "src/local_voice_harness/cursor/service.py",
    "src/local_voice_harness/prompt_operations.py",
    "src/local_voice_harness/questions/__init__.py",
    "src/local_voice_harness/job_lifecycle.py",
)


def resolve_qualname(qualname: str) -> Any:
    parts = qualname.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        try:
            target: Any = __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError:
            continue
        try:
            for part in parts[index:]:
                target = getattr(target, part)
        except AttributeError:
            continue
        return target
    raise AttributeError(qualname)


def module_line_counts(root: str | None = None) -> dict[str, int]:
    from pathlib import Path

    base = (
        Path(root) if root is not None else Path(__file__).resolve().parents[2].parent
    )
    counts: dict[str, int] = {}
    for relative in LIFECYCLE_MODULE_PATHS:
        path = base / relative
        counts[relative] = len(path.read_text().splitlines())
    return counts


def measured_baseline_counts() -> dict[str, int]:
    named = [field for fields in _NAMED_TABLE_FIELDS.values() for field in fields]
    return {
        "persisted_field_names": len(persisted_field_names()),
        "named_table_fields": len(named),
        "import_only_fields": len(_IMPORT_ONLY_FIELDS),
        "cursor_job_public_properties": len(cursor_job_public_properties()),
        "compatibility_adapters": len(COMPATIBILITY_ADAPTERS),
        "transition_entry_points": len(TRANSITION_ENTRY_POINT_NAMES),
        "duplicate_authorities": len(DUPLICATE_AUTHORITIES),
        "lifecycle_module_lines": sum(module_line_counts().values()),
    }


# Measured on the #357 baseline checkout. Tests fail if these drift without an
# intentional inventory update. They are evidence, not an optimization target.
BASELINE_COUNTS: dict[str, int] = {
    "persisted_field_names": 213,
    "named_table_fields": 208,
    "import_only_fields": 4,
    "cursor_job_public_properties": 152,
    "compatibility_adapters": 38,
    "transition_entry_points": 72,
    "duplicate_authorities": 0,
    "lifecycle_module_lines": 25586,
}
