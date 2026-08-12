from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DATABASE_SCHEMA_VERSION = 1
DATABASE_FILENAME = "jobs.sqlite3"

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

    def initialize(self) -> None:
        with self.connect() as connection:
            has_metadata = connection.execute(
                "SELECT 1 FROM sqlite_schema "
                "WHERE type = 'table' AND name = 'store_meta'"
            ).fetchone()
            if has_metadata is not None:
                row = connection.execute(
                    "SELECT value FROM store_meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is None or str(row["value"]) != str(DATABASE_SCHEMA_VERSION):
                    found = str(row["value"]) if row is not None else "missing"
                    raise sqlite3.DatabaseError(
                        f"unsupported job database schema version {found}"
                    )
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT INTO store_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO NOTHING",
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

    def load_job(
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
        connection.execute("DELETE FROM job_fields WHERE job_id = ?", (job_id,))
        connection.executemany(
            """
            INSERT INTO job_fields(
                job_id, name, value_kind, integer_value, real_value,
                text_value, json_value
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (job_id, name, *self._encoded(value))
                for name, value in sorted(values.items())
            ),
        )
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
