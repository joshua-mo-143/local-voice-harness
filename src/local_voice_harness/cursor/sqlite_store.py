from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .coordinator import DurableEffect

DATABASE_SCHEMA_VERSION = 2
DATABASE_FILENAME = "jobs.sqlite3"
BOOTSTRAP_LOCK_FILENAME = ".sqlite-bootstrap.lock"

# Schema-v18 inventory.  Every canonical value belongs to exactly one named
# row.  The four compatibility values are accepted only by the JSON/EAV import
# adapter and are never lifecycle state in schema v2.
_NAMED_TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "job_identity": (
        "parent_job_id",
        "revision",
        "status",
        "harness_kind",
        "issue_provider",
        "request",
        "utterance",
        "trusted_utterance",
        "created_at",
        "updated_at",
        "queued_at",
        "started_at",
        "completed_at",
        "foreground_until",
        "continuation",
        "continuation_answer",
        "reconcile",
        "context_repository",
        "repository_hint",
        "speakable_label",
        "grouped_repository_targets",
        "grouped_repository_candidates",
        "grouped_repository_launches",
        "grouped_repository_coordinator_id",
    ),
    "job_prompt_question": (
        "clarification_kind",
        "clarifications",
        "interactive_questionnaire_blocked",
        "prompt_baseline_sequence",
        "prompt_context_sessions",
        "prompt_manifest",
        "prompt_operation_agent_session",
        "prompt_operation_phase",
        "prompt_operation_state",
        "prompt_operation_target",
        "prompt_operation_turn",
        "question",
        "turn",
        "turn_token",
        "voice_question",
    ),
    "job_terminal_cleanup": (
        "error",
        "result",
        "terminal_intent_completed_at",
        "terminal_intent_error",
        "terminal_intent_result",
        "terminal_intent_status",
    ),
    "job_delivery_announcement": (
        "announcement_ack",
        "announcement_dismissed",
        "announcement_repeated",
        "delivered",
        "delivered_at",
        "delivery_attempts",
        "delivery_claim_token",
        "delivery_claimed_at",
        "delivery_generation",
        "delivery_retry_at",
    ),
    "job_workflow_review_approval_participant": (
        "active_participant",
        "implementer_target",
        "participant_admission_state",
        "participant_creation_checkout",
        "participant_creation_label",
        "participant_creation_pane_id",
        "participant_creation_participant",
        "participant_creation_state",
        "participant_creation_target",
        "participant_creation_workspace_id",
        "participant_session_owners",
        "plan_approval_agent_session",
        "plan_approval_completion_pending",
        "plan_approval_counted",
        "plan_approval_id",
        "plan_approval_plan_artifact",
        "plan_approval_review_artifact",
        "plan_approval_revision",
        "plan_approval_source",
        "plan_approval_state",
        "plan_approval_state_change_sequence",
        "plan_artifact",
        "planner_target",
        "review_approval_source",
        "review_approved",
        "review_artifact",
        "review_decision",
        "review_round",
        "reviewer_target",
        "workflow_classification_reason",
        "workflow_phase",
        "workflow_tier",
        "workflow_turn_phase",
    ),
    "job_checkout_fork": (
        "fork_absent_observations",
        "fork_automatic_reconcile_stopped_at",
        "fork_committed",
        "fork_committed_at",
        "fork_confirmed",
        "fork_confirmed_absent_at",
        "fork_dispatch_exited",
        "fork_exists",
        "fork_last_reconciled_at",
        "fork_next_reconcile_at",
        "fork_operation_login",
        "fork_operation_source",
        "fork_operation_source_default_branch",
        "fork_operation_source_parent",
        "fork_operation_source_private",
        "fork_operation_source_url",
        "fork_operation_state",
        "fork_operation_target",
        "fork_reconcile_attempts",
        "fork_repository",
        "fork_requested",
        "fork_retained_at",
        "pull_request_branch",
        "pull_request_head_oid",
        "pull_request_head_ref",
        "pull_request_remote_url",
        "pull_request_worktree_error",
        "pull_request_worktree_state",
        "repository",
        "worktree_absent_observations",
        "worktree_automatic_reconcile_stopped_at",
        "worktree_branch",
        "worktree_confirmed_absent_at",
        "worktree_dispatch_exited",
        "worktree_label",
        "worktree_last_reconciled_at",
        "worktree_manual_inspection_required",
        "worktree_next_reconcile_at",
        "worktree_path",
        "worktree_provision_error",
        "worktree_provision_state",
        "worktree_quarantine_acknowledged_at",
        "worktree_reconcile_attempts",
        "worktree_retained_at",
        "worktree_root_pane_id",
        "worktree_workspace_id",
    ),
    "job_provider_ticket": (
        "github_issue",
        "github_issue_context",
        "github_issue_create_body",
        "github_issue_create_confirmed",
        "github_issue_create_marker",
        "github_issue_create_operation_state",
        "github_issue_create_requested",
        "github_issue_create_title",
        "github_issue_created_number",
        "github_issue_created_url",
        "github_issue_url",
        "github_pull_request",
        "github_repository",
        "issue_key",
        "linear_ticket_create_baseline_sequence",
        "linear_ticket_create_confirmed",
        "linear_ticket_create_description",
        "linear_ticket_create_marker",
        "linear_ticket_create_operation_state",
        "linear_ticket_create_prompt_session",
        "linear_ticket_create_prompt_target",
        "linear_ticket_create_prompt_token",
        "linear_ticket_create_requested",
        "linear_ticket_create_team",
        "linear_ticket_create_team_id",
        "linear_ticket_create_title",
        "linear_ticket_created_identifier",
        "linear_ticket_created_url",
    ),
    "job_session_pane": (
        "agent_absent_observations",
        "agent_automatic_reconcile_stopped_at",
        "agent_confirmed_absent_at",
        "agent_dispatch_exited",
        "agent_dispatch_state",
        "agent_hint",
        "agent_last_reconciled_at",
        "agent_name",
        "agent_next_reconcile_at",
        "agent_operation_checkout",
        "agent_operation_pane_id",
        "agent_operation_target",
        "agent_operation_workspace_id",
        "agent_provider",
        "agent_provider_session_id",
        "agent_reconcile_attempts",
        "agent_retained_at",
        "agent_state_sequence",
        "attempt_started_at",
        "cancellation_reconciliation_pending",
        "herdr_pane_id",
        "herdr_target",
        "herdr_workspace_id",
        "manual_reconcile_operation",
        "manual_reconcile_outcome",
        "manual_reconcile_required_at",
        "manual_reconcile_resolved_at",
        "manual_reconcile_token",
        "next_reconcile_at",
        "pane_retained_at",
        "reconciliation_base_error",
        "session_id",
        "target_release_manual_required",
        "target_release_owner_boot_id",
        "target_release_owner_pid",
        "target_release_owner_start",
        "target_release_pending",
        "target_release_token",
        "target_release_unverified_targets",
    ),
    "job_worker": (
        "worker_boot_id",
        "worker_claim_operation",
        "worker_claimed_at",
        "worker_operation",
        "worker_pid",
        "worker_process_start",
        "worker_token",
    ),
}
_IMPORT_ONLY_FIELDS = frozenset(
    {
        "schema_version",
        "migration_source_schema_version",
        "phase_prompt_active",
        "agent_identity_legacy_compatible",
    }
)
_STRUCTURED_FIELDS = frozenset(
    {
        "voice_question",
        "clarifications",
        "prompt_context_sessions",
        "prompt_manifest",
        "grouped_repository_targets",
        "grouped_repository_candidates",
        "grouped_repository_launches",
        "target_release_unverified_targets",
        "participant_session_owners",
    }
)
_BOOL_FIELDS = frozenset(
    {
        "agent_dispatch_exited",
        "announcement_dismissed",
        "announcement_repeated",
        "cancellation_reconciliation_pending",
        "continuation",
        "delivered",
        "fork_committed",
        "fork_confirmed",
        "fork_dispatch_exited",
        "fork_exists",
        "fork_operation_source_private",
        "fork_requested",
        "github_issue_create_confirmed",
        "github_issue_create_requested",
        "interactive_questionnaire_blocked",
        "linear_ticket_create_confirmed",
        "linear_ticket_create_requested",
        "plan_approval_completion_pending",
        "plan_approval_counted",
        "reconcile",
        "review_approved",
        "target_release_manual_required",
        "target_release_pending",
        "worktree_dispatch_exited",
        "worktree_manual_inspection_required",
    }
)
_INTEGER_FIELDS = frozenset(
    {
        "agent_absent_observations",
        "agent_reconcile_attempts",
        "agent_state_sequence",
        "delivery_attempts",
        "delivery_generation",
        "fork_absent_observations",
        "fork_reconcile_attempts",
        "github_issue",
        "github_issue_created_number",
        "github_pull_request",
        "linear_ticket_create_baseline_sequence",
        "plan_approval_revision",
        "plan_approval_state_change_sequence",
        "prompt_baseline_sequence",
        "prompt_operation_turn",
        "review_round",
        "revision",
        "target_release_owner_pid",
        "turn",
        "worker_pid",
        "worktree_absent_observations",
        "worktree_reconcile_attempts",
    }
)
_REAL_FIELDS = frozenset(
    {
        "agent_automatic_reconcile_stopped_at",
        "agent_confirmed_absent_at",
        "agent_last_reconciled_at",
        "agent_next_reconcile_at",
        "agent_retained_at",
        "attempt_started_at",
        "completed_at",
        "created_at",
        "delivered_at",
        "delivery_claimed_at",
        "delivery_retry_at",
        "foreground_until",
        "fork_automatic_reconcile_stopped_at",
        "fork_committed_at",
        "fork_confirmed_absent_at",
        "fork_last_reconciled_at",
        "fork_next_reconcile_at",
        "fork_retained_at",
        "manual_reconcile_required_at",
        "manual_reconcile_resolved_at",
        "next_reconcile_at",
        "pane_retained_at",
        "queued_at",
        "started_at",
        "terminal_intent_completed_at",
        "updated_at",
        "worker_claimed_at",
        "worktree_automatic_reconcile_stopped_at",
        "worktree_confirmed_absent_at",
        "worktree_last_reconciled_at",
        "worktree_next_reconcile_at",
        "worktree_quarantine_acknowledged_at",
        "worktree_retained_at",
    }
)


def _column_definition(field: str) -> str:
    if field in _BOOL_FIELDS:
        return f'"{field}" INTEGER CHECK("{field}" IN (0, 1))'
    if field in _INTEGER_FIELDS:
        if field == "review_round":
            return '"review_round" INTEGER CHECK("review_round" BETWEEN 0 AND 2)'
        if field == "revision":
            return '"revision" INTEGER CHECK("revision" >= 0)'
        return f'"{field}" INTEGER'
    if field in _REAL_FIELDS:
        return f'"{field}" REAL'
    if field in _STRUCTURED_FIELDS:
        return f'"{field}" TEXT CHECK("{field}" IS NULL OR json_valid("{field}"))'
    return f'"{field}" TEXT'


def _named_schema_statements() -> tuple[str, ...]:
    statements = []
    for table, fields in _NAMED_TABLE_FIELDS.items():
        columns = ", ".join(_column_definition(field) for field in fields)
        discriminator = (
            ", lifecycle_kind TEXT NOT NULL CHECK(lifecycle_kind IN "
            "('queued','routing','running','awaiting_user','blocked',"
            "'reconciling','terminal'))"
            if table == "job_identity"
            else ""
        )
        statements.append(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE"
            f"{discriminator}, {columns}) STRICT"
        )
    statements.append(
        "CREATE TABLE IF NOT EXISTS job_field_presence ("
        "job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE, "
        "field_name TEXT NOT NULL, PRIMARY KEY(job_id, field_name)) STRICT"
    )
    return tuple(statements)


def _schema_statements(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    pending = ""
    for line in script.splitlines():
        pending += line + "\n"
        if sqlite3.complete_statement(pending):
            statement = pending.strip()
            pending = ""
            if statement and not statement.startswith("PRAGMA"):
                statements.append(statement)
    if pending.strip():
        raise RuntimeError("incomplete SQLite schema statement")
    return tuple(statements)


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY
        CHECK(length(job_id) = 12 AND job_id NOT GLOB '*[^0-9a-f]*'),
    parent_job_id TEXT,
    revision INTEGER NOT NULL CHECK(revision >= 0),
    status TEXT NOT NULL,
    harness_kind TEXT NOT NULL,
    issue_provider TEXT,
    created_at REAL NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS job_fields (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value_kind TEXT NOT NULL
        CHECK(value_kind IN ('null', 'bool', 'integer', 'real', 'text', 'json')),
    integer_value INTEGER,
    real_value REAL,
    text_value TEXT,
    json_value TEXT,
    PRIMARY KEY(job_id, name),
    CHECK (
        (value_kind = 'null' AND integer_value IS NULL AND real_value IS NULL
            AND text_value IS NULL AND json_value IS NULL)
        OR (value_kind IN ('bool', 'integer') AND integer_value IS NOT NULL
            AND real_value IS NULL AND text_value IS NULL AND json_value IS NULL)
        OR (value_kind = 'real' AND integer_value IS NULL AND real_value IS NOT NULL
            AND text_value IS NULL AND json_value IS NULL)
        OR (value_kind = 'text' AND integer_value IS NULL AND real_value IS NULL
            AND text_value IS NOT NULL AND json_value IS NULL)
        OR (value_kind = 'json' AND integer_value IS NULL AND real_value IS NULL
            AND text_value IS NULL AND json_value IS NOT NULL)
    ),
    CHECK(value_kind != 'bool' OR integer_value IN (0, 1))
) STRICT;

CREATE TABLE IF NOT EXISTS reservations (
    resource_kind TEXT NOT NULL CHECK(resource_kind IN ('ticket', 'target', 'worktree')),
    resource_key TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    PRIMARY KEY(resource_kind, resource_key)
) STRICT;

CREATE TABLE IF NOT EXISTS delivery_claims (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    claim_token TEXT UNIQUE,
    claimed_at REAL,
    generation INTEGER NOT NULL CHECK(generation >= 0),
    attempts INTEGER NOT NULL CHECK(attempts >= 0),
    retry_at REAL NOT NULL,
    delivered INTEGER NOT NULL CHECK(delivered IN (0, 1)),
    delivered_at REAL,
    CHECK((claim_token IS NULL) = (claimed_at IS NULL)),
    CHECK(NOT delivered OR claim_token IS NULL)
) STRICT;

CREATE TABLE IF NOT EXISTS worker_claims (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    token TEXT,
    pid INTEGER,
    boot_id TEXT,
    process_start TEXT,
    operation TEXT,
    CHECK(
        (token IS NULL AND pid IS NULL AND boot_id IS NULL AND process_start IS NULL)
        OR (token IS NOT NULL AND pid > 0 AND boot_id IS NOT NULL
            AND process_start IS NOT NULL)
    )
) STRICT;

CREATE TABLE IF NOT EXISTS maintenance (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    token TEXT NOT NULL UNIQUE,
    operation TEXT NOT NULL CHECK(operation = 'delete_all'),
    started_at REAL NOT NULL,
    owner_pid INTEGER NOT NULL CHECK(owner_pid > 0),
    owner_boot_id TEXT NOT NULL,
    owner_process_start TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS quarantine (
    evidence_id TEXT PRIMARY KEY,
    job_id TEXT,
    metadata_path TEXT NOT NULL UNIQUE,
    payload_path TEXT,
    payload_digest TEXT,
    error TEXT NOT NULL,
    quarantined_at REAL,
    target_key TEXT,
    worktree_key TEXT,
    blocks_all INTEGER NOT NULL CHECK(blocks_all IN (0, 1)),
    reserves_target INTEGER NOT NULL CHECK(reserves_target IN (0, 1)),
    reserves_worktree INTEGER NOT NULL CHECK(reserves_worktree IN (0, 1)),
    resolved_at REAL,
    resolution_reason TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS artifacts (
    reference TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('plan', 'review')),
    round INTEGER NOT NULL CHECK(round BETWEEN 0 AND 2),
    path TEXT NOT NULL,
    digest TEXT NOT NULL,
    source_digest TEXT,
    UNIQUE(job_id, kind, round, digest)
) STRICT;

CREATE TABLE IF NOT EXISTS outbox (
    effect_id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(job_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    next_at REAL,
    lease_token TEXT,
    leased_at REAL,
    completed_at REAL,
    payload_json TEXT,
    outcome_json TEXT,
    last_error TEXT,
    CHECK((lease_token IS NULL) = (leased_at IS NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    command_id TEXT NOT NULL UNIQUE,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    revision INTEGER NOT NULL CHECK(revision >= 0),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE INDEX IF NOT EXISTS reservations_job_idx ON reservations(job_id);
CREATE INDEX IF NOT EXISTS quarantine_job_idx ON quarantine(job_id);
"""


class SQLiteJobDatabase:
    """Low-level normalized SQLite persistence used by ``JobStore``."""

    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.path = jobs_dir / DATABASE_FILENAME

    def connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(
                f"file:{self.path}?mode=ro",
                uri=True,
                timeout=5,
                isolation_level=None,
            )
        else:
            self.jobs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.jobs_dir.chmod(0o700)
            connection = sqlite3.connect(
                self.path,
                timeout=5,
                isolation_level=None,
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        if not readonly:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            try:
                self.path.chmod(0o600)
            except FileNotFoundError:
                pass
        return connection

    @contextmanager
    def _bootstrap_lock(self) -> Iterator[None]:
        self.jobs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.jobs_dir.chmod(0o700)
        path = self.jobs_dir / BOOTSTRAP_LOCK_FILENAME
        with path.open("a+b") as lock:
            path.chmod(0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def initialize(
        self, *, normalize_legacy: Callable[[Any], Any] | None = None
    ) -> None:
        with self._bootstrap_lock(), self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._initialize_transaction(
                    connection, normalize_legacy=normalize_legacy
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _initialize_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        normalize_legacy: Callable[[Any], Any] | None,
    ) -> None:
        has_metadata = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'store_meta'"
        ).fetchone()
        if has_metadata is None:
            existing_tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if existing_tables:
                raise sqlite3.DatabaseError(
                    "job database schema marker is missing from a non-empty database"
                )
            self._create_schema(connection)
            connection.execute(
                "INSERT INTO store_meta(key, value) VALUES('schema_version', ?)",
                (str(DATABASE_SCHEMA_VERSION),),
            )
            return
        row = connection.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            evidence_tables = {
                str(item["name"])
                for item in connection.execute(
                    "SELECT name FROM sqlite_schema "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                    "AND name != 'store_meta'"
                )
            }
            for table in evidence_tables:
                try:
                    populated = connection.execute(
                        f'SELECT 1 FROM "{table}" LIMIT 1'
                    ).fetchone()
                except sqlite3.DatabaseError as exc:
                    raise sqlite3.DatabaseError(
                        "job database schema marker is missing and evidence "
                        f"table {table} is unreadable"
                    ) from exc
                if populated is not None:
                    raise sqlite3.DatabaseError(
                        "job database schema marker is missing while persisted "
                        "evidence exists"
                    )
            self._create_schema(connection)
            connection.execute(
                "INSERT INTO store_meta(key, value) VALUES('schema_version', ?)",
                (str(DATABASE_SCHEMA_VERSION),),
            )
            return
        found = str(row["value"])
        if found == str(DATABASE_SCHEMA_VERSION):
            self._create_schema(connection)
            return
        if found != "1":
            raise sqlite3.DatabaseError(
                f"unsupported job database schema version {found}"
            )
        if normalize_legacy is None:
            raise sqlite3.DatabaseError(
                "schema-v1 migration requires the canonical legacy normalizer"
            )
        self._migrate_v1(
            connection,
            normalize_legacy=normalize_legacy,
        )
        self._create_schema(connection)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        for statement in _schema_statements(_SCHEMA):
            connection.execute(statement)
        for statement in _named_schema_statements():
            connection.execute(statement)

    def _migrate_v1(
        self,
        connection: sqlite3.Connection,
        *,
        normalize_legacy: Callable[[Any], Any],
    ) -> None:
        """Atomically normalize and project every complete v1 EAV record."""

        from .model import CURRENT_SCHEMA_VERSION, CursorJob

        for statement in _named_schema_statements():
            connection.execute(statement)
        job_ids = connection.execute(
            "SELECT job_id FROM jobs ORDER BY job_id"
        ).fetchall()
        for row in job_ids:
            job_id = str(row["job_id"])
            raw = self._load_eav_job(connection, job_id)
            imported = CursorJob.from_dict(raw)
            candidate = normalize_legacy(imported)
            candidate.validate_invariants(require_worker_owner=True)
            values = candidate.to_dict()
            self._save_job_core(connection, values)
            self._save_named_state(connection, values)
            projected = CursorJob.from_dict(self.load_job(connection, job_id))
            if projected.loaded_schema_version != CURRENT_SCHEMA_VERSION:
                raise sqlite3.DatabaseError(
                    f"{job_id}: projected lifecycle is not native schema v18"
                )
            projected.validate_invariants(require_worker_owner=True)
            if projected.to_dict() != candidate.to_dict():
                raise sqlite3.DatabaseError(
                    f"{job_id}: relational lifecycle projection is not lossless"
                )
            core = connection.execute(
                """
                SELECT parent_job_id, revision, status, harness_kind,
                    issue_provider, created_at
                FROM jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            identity = connection.execute(
                """
                SELECT parent_job_id, revision, status, harness_kind,
                    issue_provider, created_at
                FROM job_identity WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if core is None or identity is None or tuple(core) != tuple(identity):
                raise sqlite3.DatabaseError(
                    f"{job_id}: jobs core disagrees with canonical identity"
                )
        connection.execute(
            "UPDATE store_meta SET value = ? WHERE key = 'schema_version'",
            (str(DATABASE_SCHEMA_VERSION),),
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def meta(connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute(
            "SELECT value FROM store_meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None

    @staticmethod
    def set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            "INSERT INTO store_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @staticmethod
    def _encoded(
        value: object,
    ) -> tuple[str, int | None, float | None, str | None, str | None]:
        if value is None:
            return ("null", None, None, None, None)
        if isinstance(value, bool):
            return ("bool", int(value), None, None, None)
        if isinstance(value, int):
            return ("integer", value, None, None, None)
        if isinstance(value, float):
            return ("real", None, value, None, None)
        if isinstance(value, str):
            return ("text", None, None, value, None)
        return (
            "json",
            None,
            None,
            None,
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True),
        )

    @staticmethod
    def _decoded(row: sqlite3.Row) -> object:
        kind = str(row["value_kind"])
        if kind == "null":
            return None
        if kind == "bool":
            return bool(row["integer_value"])
        if kind == "integer":
            return int(row["integer_value"])
        if kind == "real":
            return float(row["real_value"])
        if kind == "text":
            return str(row["text_value"])
        return json.loads(str(row["json_value"]))

    def _load_eav_job(
        self, connection: sqlite3.Connection, job_id: str
    ) -> dict[str, object]:
        if (
            connection.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            is None
        ):
            raise FileNotFoundError(job_id)
        rows = connection.execute(
            "SELECT name, value_kind, integer_value, real_value, text_value, json_value "
            "FROM job_fields WHERE job_id = ? ORDER BY name",
            (job_id,),
        )
        return {str(row["name"]): self._decoded(row) for row in rows}

    @staticmethod
    def _lifecycle_kind(status: object) -> str:
        value = str(status)
        if value in {"completed", "failed", "cancelled"}:
            return "terminal"
        if value == "awaiting_user":
            return "awaiting_user"
        if value in {"queued", "routing", "running", "blocked", "reconciling"}:
            return value
        raise ValueError(f"unsupported top-level lifecycle discriminator {value}")

    @staticmethod
    def _database_value(field: str, value: object) -> object:
        if value is None:
            return None
        if field in _STRUCTURED_FIELDS:
            return json.dumps(
                value, allow_nan=False, separators=(",", ":"), sort_keys=True
            )
        if field in _BOOL_FIELDS:
            return int(bool(value))
        return value

    @staticmethod
    def _model_value(field: str, value: object) -> object:
        if field in _STRUCTURED_FIELDS:
            return json.loads(str(value))
        if field in _BOOL_FIELDS:
            return bool(value)
        return value

    def _save_named_state(
        self, connection: sqlite3.Connection, values: Mapping[str, object]
    ) -> None:
        job_id = str(values["id"])
        known = (
            {field for fields in _NAMED_TABLE_FIELDS.values() for field in fields}
            | _IMPORT_ONLY_FIELDS
            | {"id"}
        )
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                "schema-v18 persistence inventory is incomplete: "
                + ", ".join(sorted(unknown))
            )
        for table, fields in _NAMED_TABLE_FIELDS.items():
            names = ["job_id"]
            encoded: list[object] = [job_id]
            if table == "job_identity":
                names.append("lifecycle_kind")
                encoded.append(self._lifecycle_kind(values["status"]))
            names.extend(fields)
            encoded.extend(
                self._database_value(field, values.get(field)) for field in fields
            )
            columns = ", ".join(f'"{name}"' for name in names)
            placeholders = ", ".join("?" for _ in names)
            updates = ", ".join(f'"{name}" = excluded."{name}"' for name in names[1:])
            connection.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(job_id) DO UPDATE SET {updates}",
                encoded,
            )
        connection.execute("DELETE FROM job_field_presence WHERE job_id = ?", (job_id,))
        connection.executemany(
            "INSERT INTO job_field_presence(job_id, field_name) VALUES(?, ?)",
            (
                (job_id, field)
                for field in sorted(set(values) & (known - _IMPORT_ONLY_FIELDS))
                if field != "id"
            ),
        )

    def load_job(
        self, connection: sqlite3.Connection, job_id: str
    ) -> dict[str, object]:
        identity = connection.execute(
            "SELECT * FROM job_identity WHERE job_id = ?", (job_id,)
        ).fetchone()
        if identity is None:
            if (
                connection.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                is None
            ):
                raise FileNotFoundError(job_id)
            raise sqlite3.DatabaseError(f"{job_id}: missing schema-v2 identity state")
        if str(identity["lifecycle_kind"]) != self._lifecycle_kind(identity["status"]):
            raise sqlite3.DatabaseError(
                f"{job_id}: top-level lifecycle discriminator disagrees with status"
            )
        present = {
            str(row["field_name"])
            for row in connection.execute(
                "SELECT field_name FROM job_field_presence WHERE job_id = ?",
                (job_id,),
            )
        }
        known = {field for fields in _NAMED_TABLE_FIELDS.values() for field in fields}
        if unknown := present - known:
            raise sqlite3.DatabaseError(
                f"{job_id}: unknown schema-v2 field presence "
                + ", ".join(sorted(unknown))
            )
        values: dict[str, object] = {
            "id": job_id,
            "schema_version": 18,
        }
        for table, fields in _NAMED_TABLE_FIELDS.items():
            row = (
                identity
                if table == "job_identity"
                else connection.execute(
                    f"SELECT * FROM {table} WHERE job_id = ?", (job_id,)
                ).fetchone()
            )
            if row is None:
                raise sqlite3.DatabaseError(
                    f"{job_id}: missing schema-v2 {table} state"
                )
            for field in fields:
                if field in present:
                    raw = row[field]
                    values[field] = (
                        None if raw is None else self._model_value(field, raw)
                    )
        return values

    def list_jobs(self, connection: sqlite3.Connection) -> list[dict[str, object]]:
        ids = connection.execute("SELECT job_id FROM jobs ORDER BY job_id").fetchall()
        return [self.load_job(connection, str(row["job_id"])) for row in ids]

    def save_job(
        self,
        connection: sqlite3.Connection,
        values: Mapping[str, object],
        *,
        reservations: Iterable[tuple[str, str, str]],
    ) -> None:
        job_id = str(values["id"])
        self._save_job_core(connection, values)
        self._save_named_state(connection, values)
        connection.execute("DELETE FROM reservations WHERE job_id = ?", (job_id,))
        connection.executemany(
            "INSERT INTO reservations(resource_kind, resource_key, job_id, reason) "
            "VALUES(?, ?, ?, ?)",
            ((kind, key, job_id, reason) for kind, key, reason in reservations),
        )
        connection.execute("DELETE FROM delivery_claims WHERE job_id = ?", (job_id,))
        connection.execute(
            """
            INSERT INTO delivery_claims(
                job_id, claim_token, claimed_at, generation, attempts,
                retry_at, delivered, delivered_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                values.get("delivery_claim_token"),
                values.get("delivery_claimed_at"),
                int(str(values.get("delivery_generation") or 0)),
                int(str(values.get("delivery_attempts") or 0)),
                float(str(values.get("delivery_retry_at") or 0)),
                int(bool(values.get("delivered"))),
                values.get("delivered_at"),
            ),
        )
        connection.execute("DELETE FROM worker_claims WHERE job_id = ?", (job_id,))
        worker_identity = (
            values.get("worker_token"),
            values.get("worker_pid"),
            values.get("worker_boot_id"),
            values.get("worker_process_start"),
        )
        # Pre-current JSON may carry a partial worker identity. Keep those
        # import-only fields losslessly in job_fields, but do not bless them as
        # a relational claim until recovery verifies and rewrites the owner.
        if not any(worker_identity) or all(
            value is not None for value in worker_identity
        ):
            connection.execute(
                """
                INSERT INTO worker_claims(
                    job_id, token, pid, boot_id, process_start, operation
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    *worker_identity,
                    values.get("worker_operation"),
                ),
            )
        connection.execute("DELETE FROM artifacts WHERE job_id = ?", (job_id,))
        for kind in ("plan", "review"):
            reference = values.get(f"{kind}_artifact")
            if not isinstance(reference, str):
                continue
            name = Path(reference).name
            parts = name.removesuffix(".json").split("-")
            if len(parts) < 3:
                continue
            connection.execute(
                """
                INSERT INTO artifacts(
                    reference, job_id, kind, round, path, digest, source_digest
                ) VALUES(?, ?, ?, ?, ?, ?, NULL)
                """,
                (reference, job_id, kind, int(parts[1]), reference, parts[-1]),
            )

    @staticmethod
    def _save_job_core(
        connection: sqlite3.Connection, values: Mapping[str, object]
    ) -> None:
        job_id = str(values["id"])
        connection.execute(
            """
            INSERT INTO jobs(
                job_id, parent_job_id, revision, status, harness_kind,
                issue_provider, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                parent_job_id = excluded.parent_job_id,
                revision = excluded.revision,
                status = excluded.status,
                harness_kind = excluded.harness_kind,
                issue_provider = excluded.issue_provider,
                created_at = excluded.created_at
            """,
            (
                job_id,
                values.get("parent_job_id"),
                values["revision"],
                values["status"],
                values.get("harness_kind", "cursor"),
                values.get("issue_provider"),
                values["created_at"],
            ),
        )

    def load_command_event(
        self, connection: sqlite3.Connection, command_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT event_id, job_id, revision, kind, payload_json "
            "FROM events WHERE command_id = ?",
            (command_id,),
        ).fetchone()

    def insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_id: str,
        command_id: str,
        job_id: str,
        revision: int,
        kind: str,
        payload: Mapping[str, object],
        created_at: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(
                event_id, command_id, job_id, revision, kind, payload_json,
                created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                command_id,
                job_id,
                revision,
                kind,
                json.dumps(
                    dict(payload),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                created_at,
            ),
        )

    def insert_outbox_effects(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        effects: Iterable[DurableEffect],
    ) -> None:
        for effect in effects:
            connection.execute(
                """
                INSERT INTO outbox(
                    effect_id, job_id, kind, idempotency_key, status, attempts,
                    payload_json
                ) VALUES(?, ?, ?, ?, 'pending', 0, ?)
                """,
                (
                    uuid.uuid4().hex,
                    job_id,
                    effect.kind,
                    effect.idempotency_key,
                    json.dumps(
                        dict(effect.payload),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            )

    @staticmethod
    def delete_job(connection: sqlite3.Connection, job_id: str) -> None:
        connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def diagnostics(self) -> dict[str, Any]:
        with self.connect(readonly=True) as connection:
            schema = self.meta(connection, "schema_version")
            migration = self.meta(connection, "migration_status")
            failures = int(
                connection.execute(
                    "SELECT count(*) FROM quarantine WHERE resolved_at IS NULL"
                ).fetchone()[0]
            )
            jobs = int(connection.execute("SELECT count(*) FROM jobs").fetchone()[0])
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        return {
            "path": self.path,
            "schema_version": schema,
            "migration_status": migration,
            "import_failures": failures,
            "jobs": jobs,
            "integrity": integrity,
        }


def fsync_database_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
