from __future__ import annotations

import json
import sqlite3
import stat
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from local_voice_harness.config import JOBS_DB, JOBS_DIR
from local_voice_harness.cursor import model as cursor_model
from local_voice_harness.cursor import sqlite_store as cursor_sqlite_store
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobValidationError,
)
from local_voice_harness.cursor.sqlite_store import (
    _IMPORT_ONLY_FIELDS,
    _NAMED_TABLE_FIELDS,
    _QUARANTINE_ADDITIVE_COLUMNS,
    _STRUCTURED_FIELDS,
    DATABASE_SCHEMA_VERSION,
    SQLiteJobDatabase,
)
from local_voice_harness.cursor.store import JobQuarantineWarning, JobStore

FIXTURES = Path(__file__).parent / "fixtures" / "jobs"


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


def _install_v1_database(
    jobs: Path, records: list[dict[str, object]]
) -> SQLiteJobDatabase:
    database = SQLiteJobDatabase(jobs)
    database.initialize()
    with database.connect() as connection:
        connection.execute("DROP TABLE job_field_presence")
        for table in reversed(_NAMED_TABLE_FIELDS):
            connection.execute(f"DROP TABLE {table}")
        for raw in records:
            values = CursorJob.from_dict(raw).to_dict(preserve_loaded_version=True)
            job_id = str(values["id"])
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, parent_job_id, revision, status, harness_kind,
                    issue_provider, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    values.get("parent_job_id"),
                    values["revision"],
                    values["status"],
                    values["harness_kind"],
                    values.get("issue_provider"),
                    values["created_at"],
                ),
            )
            connection.executemany(
                """
                INSERT INTO job_fields(
                    job_id, name, value_kind, integer_value, real_value,
                    text_value, json_value
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (job_id, name, *database._encoded(value))
                    for name, value in sorted(values.items())
                ),
            )
        connection.execute(
            "UPDATE store_meta SET value = '1' WHERE key = 'schema_version'"
        )
    return database


_LEGACY_QUARANTINE_COLUMNS = (
    "evidence_id",
    "job_id",
    "metadata_path",
    "payload_path",
    "payload_digest",
    "error",
    "quarantined_at",
    "resolved_at",
    "resolution_reason",
)
_CURRENT_QUARANTINE_COLUMNS = (
    "evidence_id",
    "job_id",
    "metadata_path",
    "payload_path",
    "payload_digest",
    "error",
    "quarantined_at",
    "target_key",
    "worktree_key",
    "blocks_all",
    "reserves_target",
    "reserves_worktree",
    "resolved_at",
    "resolution_reason",
)


def _legacy_quarantine_rows(jobs: Path) -> list[tuple[object, ...]]:
    return [
        (
            "open-evidence",
            "aaaaaaaaaaaa",
            str(jobs / ".quarantine" / "aaaaaaaaaaaa-open.metadata.json"),
            str(jobs / ".quarantine" / "aaaaaaaaaaaa-open.json"),
            "open-digest",
            "invalid parent",
            1.0,
            None,
            None,
        ),
        (
            "resolved-evidence",
            "bbbbbbbbbbbb",
            str(jobs / ".quarantine" / "bbbbbbbbbbbb-resolved.metadata.json"),
            str(jobs / ".quarantine" / "bbbbbbbbbbbb-resolved.json"),
            "resolved-digest",
            "unreadable payload",
            2.0,
            3.0,
            "operator checked",
        ),
    ]


def _install_legacy_quarantine_database(
    jobs: Path,
    *,
    schema_version: str,
    rows: list[tuple[object, ...]] | None = None,
) -> Path:
    jobs.mkdir(parents=True, exist_ok=True)
    path = jobs / "jobs.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE store_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT"
        )
        connection.execute(
            "INSERT INTO store_meta(key, value) VALUES('schema_version', ?)",
            (schema_version,),
        )
        if schema_version == "1":
            connection.execute(
                """
                CREATE TABLE jobs (
                    job_id TEXT PRIMARY KEY
                        CHECK(length(job_id) = 12 AND job_id NOT GLOB '*[^0-9a-f]*'),
                    parent_job_id TEXT,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    status TEXT NOT NULL,
                    harness_kind TEXT NOT NULL,
                    issue_provider TEXT,
                    created_at REAL NOT NULL
                ) STRICT
                """
            )
        connection.execute(
            """
            CREATE TABLE quarantine (
                evidence_id TEXT PRIMARY KEY,
                job_id TEXT,
                metadata_path TEXT NOT NULL UNIQUE,
                payload_path TEXT,
                payload_digest TEXT,
                error TEXT NOT NULL,
                quarantined_at REAL,
                resolved_at REAL,
                resolution_reason TEXT
            ) STRICT
            """
        )
        connection.executemany(
            """
            INSERT INTO quarantine(
                evidence_id, job_id, metadata_path, payload_path,
                payload_digest, error, quarantined_at, resolved_at,
                resolution_reason
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows if rows is not None else _legacy_quarantine_rows(jobs),
        )
    return path


def _quarantine_column_names(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(quarantine)")
    )


def _quarantine_snapshots(
    connection: sqlite3.Connection,
) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM quarantine ORDER BY evidence_id"
        ).fetchall()
    ]


def _write_quarantine_source(
    jobs: Path,
    job_id: str,
    stem: str,
    *,
    error: str,
    payload: dict[str, object],
) -> tuple[Path, Path]:
    quarantine = jobs / ".quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    payload_path = quarantine / f"{stem}.json"
    payload_path.write_text(json.dumps(payload))
    metadata_path = quarantine / f"{stem}.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "quarantined_name": payload_path.name,
                "quarantined_at": 10,
                "sha256": "imported-digest",
                "error": error,
            }
        )
    )
    return metadata_path, payload_path


def _bootstrap_process(paths: tuple[str, str]) -> tuple[str, ...]:
    jobs, legacy = paths
    return tuple(job.id for job in JobStore(Path(jobs), Path(legacy)).list())


def test_configured_database_is_inside_durable_jobs_directory() -> None:
    assert JOBS_DB == JOBS_DIR / "jobs.sqlite3"


def test_every_v18_persisted_field_has_exactly_one_disposition() -> None:
    model_fields = (
        cursor_model._BOOL_FIELDS
        | cursor_model._INT_FIELDS
        | cursor_model._FLOAT_FIELDS
        | cursor_model._STRING_FIELDS
        | _STRUCTURED_FIELDS
    )
    grouped = [field for fields in _NAMED_TABLE_FIELDS.values() for field in fields]
    named = set(grouped)

    assert len(grouped) == len(set(grouped))
    assert named | _IMPORT_ONLY_FIELDS | {"id"} == model_fields
    assert cursor_sqlite_store._BOOL_FIELDS == cursor_model._BOOL_FIELDS & named
    assert cursor_sqlite_store._INTEGER_FIELDS == cursor_model._INT_FIELDS & named
    assert cursor_sqlite_store._REAL_FIELDS == cursor_model._FLOAT_FIELDS & named
    assert (
        named
        - cursor_sqlite_store._BOOL_FIELDS
        - cursor_sqlite_store._INTEGER_FIELDS
        - cursor_sqlite_store._REAL_FIELDS
        - _STRUCTURED_FIELDS
        == cursor_model._STRING_FIELDS & named
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grouped_repository_coordinator_id", 42),
        ("pane_retained_at", "not-a-number"),
        ("pane_retained_at", True),
    ],
)
def test_dynamic_relational_fields_reject_invalid_model_types(
    field: str, value: object
) -> None:
    with pytest.raises(JobValidationError, match=field):
        CursorJob.from_dict(_job("aaaaaaaaaaaa", **{field: value}))


def test_canonical_save_return_exactly_matches_relational_reload(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    candidate = CursorJob.from_dict(
        _job(
            "aaaaaaaaaaaa",
            grouped_repository_coordinator_id="bbbbbbbbbbbb",
            pane_retained_at=2.5,
        )
    )

    saved = store.create(candidate)
    reloaded = store.get(candidate.id)

    assert saved.to_dict() == reloaded.to_dict()
    assert reloaded.to_dict()["grouped_repository_coordinator_id"] == "bbbbbbbbbbbb"
    assert reloaded.to_dict()["pane_retained_at"] == 2.5


@pytest.mark.parametrize("source_kind", ["eav", "json"])
@pytest.mark.parametrize(
    ("fixture_name", "expected_status", "expected_uncertain"),
    [
        ("cursor-v8.json", "running", True),
        ("agent-v9.json", "running", True),
        ("agent-v12-github.json", "routing", True),
        ("agent-v17-uncertain.json", "routing", True),
    ],
)
def test_legacy_active_and_uncertain_records_project_to_native_v18(
    tmp_path: Path,
    source_kind: str,
    fixture_name: str,
    expected_status: str,
    expected_uncertain: bool,
) -> None:
    raw = json.loads((FIXTURES / fixture_name).read_text())
    assert isinstance(raw, dict)
    job_id = str(raw["id"])
    jobs = tmp_path / "jobs"
    if source_kind == "eav":
        _install_v1_database(jobs, [raw])
    else:
        jobs.mkdir()
        (jobs / f"{job_id}.json").write_text(json.dumps(raw))

    first = JobStore(jobs, tmp_path / "legacy").get(job_id)

    assert first.loaded_schema_version == CURRENT_SCHEMA_VERSION
    assert first.status.value == expected_status
    assert first.has_uncertain_operation() is expected_uncertain
    first.validate_invariants(require_worker_owner=True)
    assert first.worker_token is not None
    if fixture_name in {"cursor-v8.json", "agent-v9.json"}:
        assert first.agent_dispatch_state == "ambiguous"
    if fixture_name == "agent-v12-github.json":
        assert first.worktree_provision_state == "ambiguous"
        assert first.fork_operation_state == "submitted"

    reopened = JobStore(jobs, tmp_path / "legacy").get(job_id)
    assert reopened.to_dict() == first.to_dict()
    assert reopened.loaded_schema_version == CURRENT_SCHEMA_VERSION
    reopened.validate_invariants(require_worker_owner=True)
    if source_kind == "json":
        assert (jobs / f"{job_id}.json.imported").exists()
    else:
        with sqlite3.connect(jobs / "jobs.sqlite3") as connection:
            assert connection.execute(
                "SELECT integer_value FROM job_fields "
                "WHERE job_id = ? AND name = 'schema_version'",
                (job_id,),
            ).fetchone() == (int(raw["schema_version"]),)


def test_concurrent_processes_serialize_fresh_bootstrap(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    legacy = tmp_path / "legacy"
    paths = (str(jobs), str(legacy))

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_bootstrap_process, [paths] * 4))

    assert results == [()] * 4
    with sqlite3.connect(jobs / "jobs.sqlite3") as connection:
        assert connection.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone() == (str(DATABASE_SCHEMA_VERSION),)
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)


def test_missing_marker_bootstrap_is_idempotent_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    repairable = tmp_path / "repairable"
    repairable.mkdir()
    with sqlite3.connect(repairable / "jobs.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE store_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    database = SQLiteJobDatabase(repairable)
    database.initialize()
    database.initialize()
    assert database.diagnostics()["schema_version"] == str(DATABASE_SCHEMA_VERSION)

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    with sqlite3.connect(blocked / "jobs.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE store_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute("CREATE TABLE unknown_evidence(value TEXT NOT NULL)")
        connection.execute("INSERT INTO unknown_evidence VALUES('preserve me')")

    with pytest.raises(sqlite3.DatabaseError, match="marker is missing.*evidence"):
        SQLiteJobDatabase(blocked).initialize()
    with sqlite3.connect(blocked / "jobs.sqlite3") as connection:
        assert connection.execute("SELECT value FROM unknown_evidence").fetchone() == (
            "preserve me",
        )
        assert (
            connection.execute(
                "SELECT value FROM store_meta WHERE key = 'schema_version'"
            ).fetchone()
            is None
        )


def test_interrupted_fresh_bootstrap_rolls_back_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = tmp_path / "jobs"
    database = SQLiteJobDatabase(jobs)
    create_schema = SQLiteJobDatabase._create_schema

    def interrupt(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE interrupted_evidence(value TEXT)")
        raise RuntimeError("simulated bootstrap interruption")

    monkeypatch.setattr(SQLiteJobDatabase, "_create_schema", staticmethod(interrupt))
    with pytest.raises(RuntimeError, match="bootstrap interruption"):
        database.initialize()

    with sqlite3.connect(database.path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM sqlite_schema "
            "WHERE type = 'table' AND name = 'interrupted_evidence'"
        ).fetchone() == (0,)

    monkeypatch.setattr(
        SQLiteJobDatabase, "_create_schema", staticmethod(create_schema)
    )
    database.initialize()
    assert database.diagnostics()["schema_version"] == str(DATABASE_SCHEMA_VERSION)


@pytest.mark.parametrize(
    ("table", "field"),
    [(table, fields[-1]) for table, fields in _NAMED_TABLE_FIELDS.items()],
)
def test_v2_initialization_adds_every_missing_named_table_field(
    tmp_path: Path,
    table: str,
    field: str,
) -> None:
    database = SQLiteJobDatabase(tmp_path / "jobs")
    database.initialize()
    with database.connect() as connection:
        connection.execute(f'ALTER TABLE "{table}" DROP COLUMN "{field}"')

    database.initialize()
    with database.connect(readonly=True) as connection:
        columns = {
            str(row["name"])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        schema_after_upgrade = connection.execute(
            "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
        ).fetchall()
    assert field in columns

    database.initialize()
    with database.connect(readonly=True) as connection:
        assert (
            connection.execute(
                "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
            ).fetchall()
            == schema_after_upgrade
        )


def test_v2_named_schema_upgrade_preserves_rows_and_supports_current_writes(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    legacy = tmp_path / "legacy"
    original = CursorJob.from_dict(
        _job(
            "aaaaaaaaaaaa",
            request="preserve durable evidence",
            grouped_repository_targets=["owner/repository"],
            speakable_label="existing job",
        )
    )
    first_store = JobStore(jobs, legacy)
    first_store.create(original)
    with sqlite3.connect(first_store.db_path) as connection:
        evidence_before = connection.execute(
            "SELECT job_id, revision, status, created_at FROM jobs"
        ).fetchall()
        presence_before = connection.execute(
            "SELECT job_id, field_name FROM job_field_presence ORDER BY field_name"
        ).fetchall()
        connection.execute(
            "ALTER TABLE job_identity DROP COLUMN grouped_repository_coordinator_id"
        )

    reopened = JobStore(jobs, legacy)
    preserved = reopened.get(original.id)
    child = reopened.create(
        CursorJob.from_dict(
            _job(
                "bbbbbbbbbbbb",
                request="write the current field",
                grouped_repository_coordinator_id=original.id,
            )
        )
    )

    assert preserved.to_dict() == original.to_dict()
    assert (
        reopened.get(child.id).to_dict()["grouped_repository_coordinator_id"]
        == original.id
    )
    with sqlite3.connect(reopened.db_path) as connection:
        assert (
            connection.execute(
                "SELECT job_id, revision, status, created_at FROM jobs "
                "WHERE job_id = 'aaaaaaaaaaaa'"
            ).fetchall()
            == evidence_before
        )
        assert (
            connection.execute(
                "SELECT job_id, field_name FROM job_field_presence "
                "WHERE job_id = 'aaaaaaaaaaaa' ORDER BY field_name"
            ).fetchall()
            == presence_before
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(job_identity)")
        }
    assert "grouped_repository_coordinator_id" in columns


def test_v2_ambiguous_named_schema_fails_without_partial_upgrade(
    tmp_path: Path,
) -> None:
    database = SQLiteJobDatabase(tmp_path / "jobs")
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            "ALTER TABLE job_identity DROP COLUMN grouped_repository_coordinator_id"
        )
        connection.execute("DROP TABLE job_prompt_question")
        connection.execute("CREATE TABLE job_prompt_question(legacy_value TEXT) STRICT")

    with pytest.raises(
        sqlite3.DatabaseError,
        match=r"missing foundational columns: job_prompt_question\.job_id",
    ):
        database.initialize()

    with database.connect(readonly=True) as connection:
        identity_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(job_identity)")
        }
        prompt_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(job_prompt_question)")
        }
    assert "grouped_repository_coordinator_id" not in identity_columns
    assert prompt_columns == {"legacy_value"}


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
            "job_identity",
            "job_prompt_question",
            "job_terminal_cleanup",
            "job_delivery_announcement",
            "job_workflow_review_approval_participant",
            "job_checkout_fork",
            "job_provider_ticket",
            "job_session_pane",
            "job_worker",
            "job_field_presence",
            "reservations",
            "delivery_claims",
            "worker_claims",
            "maintenance",
            "quarantine",
            "artifacts",
            "outbox",
            "outbox_concurrency",
            "outbox_consumptions",
            "events",
        } <= tables
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        assert "payload_json" not in columns
        assert connection.execute(
            "SELECT count(*) FROM job_fields WHERE job_id = ?", (created.id,)
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT lifecycle_kind, request FROM job_identity WHERE job_id = ?",
            (created.id,),
        ).fetchone() == ("queued", "persist it")


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
        schema_before = connection.execute(
            "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
        ).fetchall()

    with pytest.raises(sqlite3.DatabaseError, match="unsupported.*999"):
        JobStore(jobs, tmp_path / "legacy").list()

    assert source.exists()
    assert not (jobs / "123456789abc.json.imported").exists()
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
            ).fetchall()
            == schema_before
        )


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


def test_v1_eav_migration_round_trips_named_state_and_preserves_auxiliary_rows(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    record = _job(
        "aaaaaaaaaaaa",
        revision=3,
        worker_token="worker-token",
        worker_pid=42,
        worker_boot_id="boot",
        worker_process_start="start",
        worker_claim_operation="run",
        worker_claimed_at=1,
        target_release_pending=True,
        target_release_manual_required=True,
        target_release_unverified_targets=["pane-1"],
    )
    terminal_record = _job(
        "bbbbbbbbbbbb",
        revision=5,
        status="completed",
        completed_at=8,
        result="done",
        announcement_ack="deferred",
        delivery_generation=2,
        delivery_attempts=1,
        delivery_retry_at=9,
    )
    database = _install_v1_database(jobs, [record, terminal_record])
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO reservations VALUES('target', 'pane-1', "
            "'aaaaaaaaaaaa', 'active target')"
        )
        connection.execute(
            "INSERT INTO delivery_claims VALUES("
            "'aaaaaaaaaaaa', NULL, NULL, 0, 0, 0, 0, NULL)"
        )
        connection.execute(
            "INSERT INTO delivery_claims VALUES("
            "'bbbbbbbbbbbb', NULL, NULL, 2, 1, 9, 0, NULL)"
        )
        connection.execute(
            "INSERT INTO worker_claims VALUES("
            "'aaaaaaaaaaaa', 'worker-token', 42, 'boot', 'start', 'run')"
        )
        connection.execute(
            "INSERT INTO artifacts VALUES("
            "'.artifacts/aaaaaaaaaaaa/plan-0-deadbeef.json', "
            "'aaaaaaaaaaaa', 'plan', 0, 'artifact-path', 'deadbeef', NULL)"
        )
        connection.execute(
            "INSERT INTO quarantine VALUES("
            "'evidence', 'aaaaaaaaaaaa', 'meta', 'payload', 'digest', 'error', "
            "1, 'pane-1', NULL, 0, 1, 0, 2, 'checked')"
        )
        connection.execute(
            "INSERT INTO maintenance VALUES("
            "1, 'lease', 'delete_all', 1, 42, 'boot', 'start')"
        )
        auxiliary_before = {
            table: [
                tuple(row)
                for row in connection.execute(f"SELECT * FROM {table}").fetchall()
            ]
            for table in (
                "reservations",
                "delivery_claims",
                "worker_claims",
                "artifacts",
                "quarantine",
                "maintenance",
            )
        }
        evidence_before = [
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM job_fields ORDER BY name"
            ).fetchall()
        ]

    store = JobStore(jobs, tmp_path / "legacy")
    migrated = store.get("aaaaaaaaaaaa")
    terminal = store.get("bbbbbbbbbbbb")

    assert migrated.to_dict() == CursorJob.from_dict(record).to_dict()
    assert migrated.revision == 3
    assert terminal.to_dict() == CursorJob.from_dict(terminal_record).to_dict()
    assert terminal.completed_at == 8
    assert not terminal.delivered
    assert terminal.announcement_ack == "deferred"
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone() == ("2",)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_schema "
            "WHERE type = 'table' AND name = 'events'"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT lifecycle_kind FROM job_identity ORDER BY job_id"
        ).fetchall() == [("queued",), ("terminal",)]
        assert connection.execute(
            "SELECT target_release_unverified_targets FROM job_session_pane"
        ).fetchone() == ('["pane-1"]',)
        assert (
            connection.execute("SELECT * FROM job_fields ORDER BY name").fetchall()
            == evidence_before
        )
        for table, rows in auxiliary_before.items():
            assert connection.execute(f"SELECT * FROM {table}").fetchall() == rows
        connection.execute("DELETE FROM maintenance")

    updated = store.update(migrated.id, lambda current: current.evolve(reconcile=True))
    assert updated is not None
    assert updated.revision == 4


def test_v1_eav_migration_failure_rolls_back_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    database = _install_v1_database(jobs, [_job("aaaaaaaaaaaa")])
    with database.connect() as connection:
        kind, integer, real, text, encoded = database._encoded("not-an-id")
        connection.execute(
            """
            INSERT INTO job_fields(
                job_id, name, value_kind, integer_value, real_value,
                text_value, json_value
            ) VALUES('aaaaaaaaaaaa', 'parent_job_id', ?, ?, ?, ?, ?)
            """,
            (kind, integer, real, text, encoded),
        )
        evidence = [
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM job_fields ORDER BY name"
            ).fetchall()
        ]

    with pytest.raises(JobValidationError, match="parent_job_id"):
        JobStore(jobs, tmp_path / "legacy").list()

    with sqlite3.connect(database.path) as connection:
        assert connection.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone() == ("1",)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_schema "
            "WHERE type = 'table' AND name = 'job_identity'"
        ).fetchone() == (0,)
        assert (
            connection.execute("SELECT * FROM job_fields ORDER BY name").fetchall()
            == evidence
        )
        connection.execute(
            "DELETE FROM job_fields "
            "WHERE job_id = 'aaaaaaaaaaaa' AND name = 'parent_job_id'"
        )

    assert [job.id for job in JobStore(jobs, tmp_path / "legacy").list()] == [
        "aaaaaaaaaaaa"
    ]


def test_v1_status_normalization_updates_jobs_core_and_identity(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    _install_v1_database(
        jobs,
        [
            {
                "schema_version": 5,
                "id": "aaaaaaaaaaaa",
                "revision": 4,
                "request": "recover unsafe active owner",
                "status": "running",
                "created_at": 1,
                "delivered": False,
            }
        ],
    )

    migrated = JobStore(jobs, tmp_path / "legacy").get("aaaaaaaaaaaa")

    assert migrated.status.value == "queued"
    assert migrated.loaded_schema_version == CURRENT_SCHEMA_VERSION
    assert migrated.worker_token is None
    with sqlite3.connect(jobs / "jobs.sqlite3") as connection:
        core = connection.execute(
            "SELECT revision, status, created_at FROM jobs"
        ).fetchone()
        identity = connection.execute(
            "SELECT revision, status, created_at FROM job_identity"
        ).fetchone()
        assert core == identity == (4, "queued", 1.0)


@pytest.mark.parametrize("schema_version", ["1", str(DATABASE_SCHEMA_VERSION)])
def test_initialize_adds_missing_quarantine_columns_without_rewriting_rows(
    tmp_path: Path,
    schema_version: str,
) -> None:
    jobs = tmp_path / "jobs"
    rows = _legacy_quarantine_rows(jobs)
    path = _install_legacy_quarantine_database(
        jobs, schema_version=schema_version, rows=rows
    )
    with sqlite3.connect(path) as connection:
        assert _quarantine_column_names(connection) == _LEGACY_QUARANTINE_COLUMNS
        evidence_before = _quarantine_snapshots(connection)

    database = SQLiteJobDatabase(jobs)
    database.initialize(normalize_legacy=lambda job: job)
    with database.connect(readonly=True) as connection:
        columns = _quarantine_column_names(connection)
        upgraded = connection.execute(
            """
            SELECT evidence_id, job_id, metadata_path, payload_path,
                payload_digest, error, quarantined_at, target_key,
                worktree_key, blocks_all, reserves_target, reserves_worktree,
                resolved_at, resolution_reason
            FROM quarantine ORDER BY evidence_id
            """
        ).fetchall()
        rows_after_upgrade = _quarantine_snapshots(connection)
        schema_after_upgrade = connection.execute(
            "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
        ).fetchall()

    assert set(columns) == set(_CURRENT_QUARANTINE_COLUMNS)
    assert {column for column, _definition in _QUARANTINE_ADDITIVE_COLUMNS} <= set(
        columns
    )
    assert [tuple(row)[:7] + tuple(row)[12:] for row in upgraded] == evidence_before
    assert [tuple(row)[7:12] for row in upgraded] == [
        (None, None, 0, 0, 0),
        (None, None, 0, 0, 0),
    ]

    database.initialize(normalize_legacy=lambda job: job)
    with database.connect(readonly=True) as connection:
        assert _quarantine_snapshots(connection) == rows_after_upgrade
        assert (
            connection.execute(
                "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
            ).fetchall()
            == schema_after_upgrade
        )


def test_legacy_quarantine_survives_startup_import_and_recovery(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    metadata_path, payload_path = _write_quarantine_source(
        jobs,
        "aaaaaaaaaaaa",
        "aaaaaaaaaaaa-open",
        error="invalid parent",
        payload=_job(
            "aaaaaaaaaaaa",
            status="running",
            herdr_target="held-agent",
            parent_job_id="invalid",
        ),
    )
    rows = [
        (
            "open-evidence",
            "aaaaaaaaaaaa",
            str(metadata_path),
            str(payload_path),
            "open-digest",
            "invalid parent",
            1.0,
            None,
            None,
        ),
        (
            "orphan-evidence",
            "cccccccccccc",
            str(jobs / ".quarantine" / "cccccccccccc-orphan.metadata.json"),
            str(jobs / ".quarantine" / "cccccccccccc-orphan.json"),
            "orphan-digest",
            "missing source",
            4.0,
            None,
            None,
        ),
        (
            "resolved-evidence",
            "bbbbbbbbbbbb",
            str(jobs / ".quarantine" / "bbbbbbbbbbbb-resolved.metadata.json"),
            str(jobs / ".quarantine" / "bbbbbbbbbbbb-resolved.json"),
            "resolved-digest",
            "unreadable payload",
            2.0,
            3.0,
            "operator checked",
        ),
    ]
    _install_v1_database(jobs, [_job("dddddddddddd", request="recover after upgrade")])
    with sqlite3.connect(jobs / "jobs.sqlite3") as connection:
        connection.execute("DROP TABLE quarantine")
        connection.execute(
            """
            CREATE TABLE quarantine (
                evidence_id TEXT PRIMARY KEY,
                job_id TEXT,
                metadata_path TEXT NOT NULL UNIQUE,
                payload_path TEXT,
                payload_digest TEXT,
                error TEXT NOT NULL,
                quarantined_at REAL,
                resolved_at REAL,
                resolution_reason TEXT
            ) STRICT
            """
        )
        connection.executemany(
            """
            INSERT INTO quarantine(
                evidence_id, job_id, metadata_path, payload_path,
                payload_digest, error, quarantined_at, resolved_at,
                resolution_reason
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute(
            "UPDATE store_meta SET value = '1' WHERE key = 'schema_version'"
        )

    store = JobStore(jobs, tmp_path / "legacy")
    recovered = store.get("dddddddddddd")
    store.create(CursorJob.from_dict(_job("eeeeeeeeeeee", herdr_target="other-agent")))

    assert recovered.request == "recover after upgrade"
    with sqlite3.connect(store.db_path) as connection:
        assert set(_quarantine_column_names(connection)) == set(
            _CURRENT_QUARANTINE_COLUMNS
        )
        imported, orphan, resolved = connection.execute(
            """
            SELECT evidence_id, job_id, metadata_path, payload_path,
                payload_digest, error, quarantined_at, target_key,
                worktree_key, blocks_all, reserves_target, reserves_worktree,
                resolved_at, resolution_reason
            FROM quarantine ORDER BY evidence_id
            """
        ).fetchall()
    assert imported[0] == "open-evidence"
    assert imported[2] == str(metadata_path)
    assert imported[3] == str(payload_path)
    assert imported[5] == "invalid parent"
    assert imported[7] == "held-agent"
    assert imported[9:] == (0, 1, 0, None, None)
    assert tuple(orphan) == (
        "orphan-evidence",
        "cccccccccccc",
        rows[1][2],
        rows[1][3],
        "orphan-digest",
        "missing source",
        4.0,
        None,
        None,
        0,
        0,
        0,
        None,
        None,
    )
    assert tuple(resolved) == (
        "resolved-evidence",
        "bbbbbbbbbbbb",
        rows[2][2],
        rows[2][3],
        "resolved-digest",
        "unreadable payload",
        2.0,
        None,
        None,
        0,
        0,
        0,
        3.0,
        "operator checked",
    )


def test_unsupported_schema_does_not_alter_legacy_quarantine(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    rows = _legacy_quarantine_rows(jobs)
    path = _install_legacy_quarantine_database(jobs, schema_version="999", rows=rows)
    with sqlite3.connect(path) as connection:
        schema_before = connection.execute(
            "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
        ).fetchall()
        evidence_before = _quarantine_snapshots(connection)

    with pytest.raises(sqlite3.DatabaseError, match="unsupported.*999"):
        SQLiteJobDatabase(jobs).initialize(normalize_legacy=lambda job: job)

    with sqlite3.connect(path) as connection:
        assert _quarantine_column_names(connection) == _LEGACY_QUARANTINE_COLUMNS
        assert _quarantine_snapshots(connection) == evidence_before
        assert (
            connection.execute(
                "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
            ).fetchall()
            == schema_before
        )
