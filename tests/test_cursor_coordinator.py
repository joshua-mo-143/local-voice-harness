from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from local_voice_harness.cursor.coordinator import (
    CoordinatorCommand,
    CoordinatorDecision,
    CoordinatorError,
    DurableEffect,
)
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
    transition,
)
from local_voice_harness.cursor.store import JobStore


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


def _counts(store: JobStore, job_id: str) -> tuple[int, int, int]:
    with sqlite3.connect(store.db_path) as connection:
        revision = connection.execute(
            "SELECT revision FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
        events = connection.execute(
            "SELECT count(*) FROM events WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
        outbox = connection.execute(
            "SELECT count(*) FROM outbox WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
    return int(revision), int(events), int(outbox)


def test_command_and_effect_require_identity() -> None:
    with pytest.raises(CoordinatorError, match="job_id and command_id"):
        CoordinatorCommand(
            job_id=" ",
            expected_revision=0,
            command_id="cmd",
            kind="session.create",
        )
    with pytest.raises(CoordinatorError, match="kind"):
        CoordinatorCommand(
            job_id="aaaaaaaaaaaa",
            expected_revision=0,
            command_id="cmd",
            kind=" ",
        )
    with pytest.raises(CoordinatorError, match="negative"):
        CoordinatorCommand(
            job_id="aaaaaaaaaaaa",
            expected_revision=-1,
            command_id="cmd",
            kind="session.create",
        )
    with pytest.raises(CoordinatorError, match="kind and idempotency_key"):
        DurableEffect(kind=" ", idempotency_key="session:a")


def test_apply_commits_state_event_and_outbox_atomically(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    created = store.create(CursorJob.from_dict(_job("aaaaaaaaaaaa")))

    def decide(job: CursorJob) -> CoordinatorDecision:
        return CoordinatorDecision(
            job=job.evolve(speakable_label="named"),
            effects=(
                DurableEffect(
                    kind="session.create",
                    idempotency_key="session:aaaaaaaaaaaa:voice",
                    payload={"target": "voice"},
                ),
            ),
            event_kind="session.create",
        )

    updated = store.apply(
        CoordinatorCommand(
            job_id=created.id,
            expected_revision=0,
            command_id="cmd-session-create",
            kind="session.create",
        ),
        decide,
    )

    assert updated is not None
    assert updated.revision == 1
    assert updated.speakable_label == "named"
    assert store.get(created.id).speakable_label == "named"
    with sqlite3.connect(store.db_path) as connection:
        event = connection.execute(
            "SELECT command_id, revision, kind, payload_json FROM events "
            "WHERE job_id = ? AND command_id = ?",
            (created.id, "cmd-session-create"),
        ).fetchone()
        assert event == (
            "cmd-session-create",
            1,
            "session.create",
            json.dumps(
                {"command_kind": "session.create", "effects": ["session.create"]},
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        outbox = connection.execute(
            "SELECT kind, idempotency_key, status, payload_json FROM outbox "
            "WHERE job_id = ?",
            (created.id,),
        ).fetchone()
        assert outbox == (
            "session.create",
            "session:aaaaaaaaaaaa:voice",
            "pending",
            json.dumps({"target": "voice"}, separators=(",", ":"), sort_keys=True),
        )


def test_stale_revision_does_not_write(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    created = store.create(CursorJob.from_dict(_job("aaaaaaaaaaaa")))

    result = store.apply(
        CoordinatorCommand(
            job_id=created.id,
            expected_revision=4,
            command_id="cmd-stale",
            kind="session.create",
        ),
        lambda job: CoordinatorDecision(job=job.evolve(speakable_label="nope")),
    )

    assert result is None
    assert _counts(store, created.id) == (0, 1, 0)
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM events WHERE command_id = ?",
            ("cmd-stale",),
        ).fetchone() == (0,)


def test_duplicate_command_does_not_apply_twice(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    created = store.create(CursorJob.from_dict(_job("aaaaaaaaaaaa")))
    command = CoordinatorCommand(
        job_id=created.id,
        expected_revision=0,
        command_id="cmd-once",
        kind="session.create",
    )
    effect = DurableEffect(
        kind="session.create",
        idempotency_key="session:aaaaaaaaaaaa:once",
        payload={"target": "once"},
    )

    first = store.apply(
        command,
        lambda job: CoordinatorDecision(
            job=job.evolve(speakable_label="first"),
            effects=(effect,),
            event_kind="session.create",
        ),
    )
    second = store.apply(
        CoordinatorCommand(
            job_id=created.id,
            expected_revision=0,
            command_id="cmd-once",
            kind="session.create",
        ),
        lambda job: CoordinatorDecision(
            job=job.evolve(speakable_label="second"),
            effects=(
                DurableEffect(
                    kind="session.create",
                    idempotency_key="session:aaaaaaaaaaaa:twice",
                    payload={"target": "twice"},
                ),
            ),
        ),
    )

    assert first is not None
    assert second is not None
    assert second.revision == first.revision == 1
    assert second.speakable_label == "first"
    assert _counts(store, created.id) == (1, 2, 1)
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT speakable_label FROM job_identity WHERE job_id = ?",
            (created.id,),
        ).fetchone() == ("first",)
        assert connection.execute(
            "SELECT count(*) FROM outbox WHERE idempotency_key = ?",
            ("session:aaaaaaaaaaaa:twice",),
        ).fetchone() == (0,)


def test_reservation_conflict_rolls_back_event_and_outbox(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    store.create(CursorJob.from_dict(_job("aaaaaaaaaaaa")))
    reserved = store.update(
        "aaaaaaaaaaaa",
        lambda job: transition(job, JobStatus.QUEUED, herdr_target="shared-agent"),
    )
    assert reserved is not None
    store.create(CursorJob.from_dict(_job("bbbbbbbbbbbb")))

    with pytest.raises(JobValidationError, match="reserved by both"):
        store.apply(
            CoordinatorCommand(
                job_id="bbbbbbbbbbbb",
                expected_revision=0,
                command_id="cmd-conflict",
                kind="target.reserve",
            ),
            lambda job: CoordinatorDecision(
                job=transition(job, JobStatus.QUEUED, herdr_target="shared-agent"),
                effects=(
                    DurableEffect(
                        kind="session.create",
                        idempotency_key="session:bbbbbbbbbbbb:shared-agent",
                        payload={"target": "shared-agent"},
                    ),
                ),
                event_kind="target.reserve",
            ),
        )

    assert _counts(store, "bbbbbbbbbbbb") == (0, 1, 0)
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM events WHERE command_id = ?",
            ("cmd-conflict",),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM outbox WHERE idempotency_key = ?",
            ("session:bbbbbbbbbbbb:shared-agent",),
        ).fetchone() == (0,)


def test_store_update_records_an_audit_event(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    created = store.create(CursorJob.from_dict(_job("aaaaaaaaaaaa")))
    updated = store.update(
        created.id, lambda job: job.evolve(speakable_label="via-update")
    )

    assert updated is not None
    assert updated.speakable_label == "via-update"
    revision, events, outbox = _counts(store, created.id)
    assert revision == 1
    assert events == 2
    assert outbox == 0
