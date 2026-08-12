from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import pytest

from local_voice_harness.config import JOBS_DB, JOBS_DIR
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobValidationError,
)
from local_voice_harness.cursor.sqlite_store import DATABASE_SCHEMA_VERSION
from local_voice_harness.cursor.store import JobQuarantineWarning, JobStore


def _job(job_id: str, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "id": job_id,
        "revision": 0,
        "request": "persist it",
        "status": "queued",
        "created_at": 1,
        "queued_at": 1,
        "delivered": False,
    }
    values.update(changes)
    return values


def test_configured_database_is_inside_durable_jobs_directory() -> None:
    assert JOBS_DB == JOBS_DIR / "jobs.sqlite3"


def test_fresh_store_uses_normalized_private_wal_database(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    source = _job("123456789abc", voice_question={"version": 1, "id": "q"})
    # Use a model-valid structured value while still proving it is not stored
    # as one whole-record payload.
    source.pop("voice_question")
    created = store.create(CursorJob.from_dict(source))

    assert store.db_path == tmp_path / "jobs" / "jobs.sqlite3"
    assert stat.S_IMODE(store.db_path.stat().st_mode) == 0o600
    assert not store.path(created.id).exists()
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone() == (str(DATABASE_SCHEMA_VERSION),)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        assert {
            "jobs",
            "job_fields",
            "reservations",
            "delivery_claims",
            "worker_claims",
            "maintenance",
            "quarantine",
            "artifacts",
            "outbox",
        } <= tables
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        assert "payload_json" not in columns
        persisted_fields = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM job_fields WHERE job_id = ?", (created.id,)
            )
        }
        assert persisted_fields == set(created.to_dict())


def test_first_open_imports_json_and_archives_original_bytes(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    source = jobs / "123456789abc.json"
    contents = json.dumps(_job("123456789abc"), sort_keys=True).encode()
    source.write_bytes(contents)

    store = JobStore(jobs, tmp_path / "legacy")

    assert [job.id for job in store.list()] == ["123456789abc"]
    assert not source.exists()
    assert source.with_suffix(".json.imported").read_bytes() == contents
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT value FROM store_meta WHERE key = 'migration_status'"
        ).fetchone() == ("complete",)


def test_invalid_import_is_quarantined_while_safe_rows_cut_over(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "aaaaaaaaaaaa.json").write_text(json.dumps(_job("aaaaaaaaaaaa")))
    invalid = json.dumps(
        _job("bbbbbbbbbbbb", status="running", parent_job_id="invalid")
    ).encode()
    (jobs / "bbbbbbbbbbbb.json").write_bytes(invalid)

    store = JobStore(jobs, tmp_path / "legacy")
    with pytest.warns(JobQuarantineWarning, match="legacy import is quarantined"):
        imported = store.list()

    assert [job.id for job in imported] == ["aaaaaaaaaaaa"]
    assert (jobs / "bbbbbbbbbbbb.json.failed").read_bytes() == invalid
    assert list((jobs / ".quarantine").glob("bbbbbbbbbbbb-*.metadata.json"))
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT value FROM store_meta WHERE key = 'migration_status'"
        ).fetchone() == ("complete_with_quarantine",)
        assert connection.execute(
            "SELECT count(*) FROM quarantine WHERE resolved_at IS NULL"
        ).fetchone() == (1,)


def test_existing_maintenance_fence_is_imported_before_cutover(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / ".maintenance").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "delete_all",
                "token": "lease",
                "started_at": 1,
                "owner_pid": 42,
                "owner_boot_id": "boot",
                "owner_process_start": "start",
            }
        )
    )

    store = JobStore(jobs, tmp_path / "legacy")

    assert store.maintenance_active()
    assert (jobs / ".maintenance.imported").exists()


def test_unknown_database_schema_fails_closed_without_consuming_json(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    source = jobs / "123456789abc.json"
    source.write_text(json.dumps(_job("123456789abc")))
    database = jobs / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE store_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO store_meta(key, value) VALUES('schema_version', '999')"
        )

    with pytest.raises(sqlite3.DatabaseError, match="unsupported.*999"):
        JobStore(jobs, tmp_path / "legacy").list()

    assert source.exists()
    assert not (jobs / "123456789abc.json.imported").exists()


def test_database_quarantine_fence_survives_premature_resolution_file(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "aaaaaaaaaaaa.json").write_text(
        json.dumps(
            _job(
                "aaaaaaaaaaaa",
                status="running",
                herdr_target="shared-agent",
                parent_job_id="invalid",
            )
        )
    )
    store = JobStore(jobs, tmp_path / "legacy")
    with pytest.warns(JobQuarantineWarning):
        store.list()
    store.acknowledge_quarantine_reservations(
        "aaaaaaaaaaaa", reason="simulate file published before SQL commit"
    )
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE quarantine SET resolved_at = NULL, resolution_reason = NULL"
        )

    candidate = CursorJob.from_dict(_job("bbbbbbbbbbbb", herdr_target="shared-agent"))
    with pytest.raises(JobValidationError, match="unresolved quarantine evidence"):
        store.create(candidate)
