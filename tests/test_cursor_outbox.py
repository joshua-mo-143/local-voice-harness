from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from local_voice_harness.cursor import outbox as outbox_module
from local_voice_harness.cursor.coordinator import (
    OUTBOX_LEASE_SECONDS,
    OUTBOX_MAX_ATTEMPTS,
    OUTBOX_RETRY_SECONDS,
    CoordinatorCommand,
    CoordinatorDecision,
    DurableEffect,
    EffectObservation,
    OutboxLease,
)
from local_voice_harness.cursor.model import CURRENT_SCHEMA_VERSION, CursorJob
from local_voice_harness.cursor.outbox import drain_outbox, recover_outbox
from local_voice_harness.cursor.recovery import recover_jobs
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


def _admit_effect(
    store: JobStore,
    job_id: str = "aaaaaaaaaaaa",
    *,
    kind: str = "session.create",
    key: str = "session:aaaaaaaaaaaa:voice",
    concurrency_key: str = "session:voice",
) -> CursorJob:
    created = store.create(CursorJob.from_dict(_job(job_id)))
    updated = store.apply(
        CoordinatorCommand(
            job_id=created.id,
            expected_revision=0,
            command_id=f"cmd-{key}",
            kind=kind,
        ),
        lambda job: CoordinatorDecision(
            job=job.evolve(speakable_label="named"),
            effects=(
                DurableEffect(
                    kind=kind,
                    idempotency_key=key,
                    concurrency_key=concurrency_key,
                    payload={"target": "voice"},
                ),
            ),
            event_kind=kind,
        ),
    )
    assert updated is not None
    return updated


def _outbox_row(store: JobStore, job_id: str) -> sqlite3.Row:
    with sqlite3.connect(store.db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM outbox WHERE job_id = ?", (job_id,)
        ).fetchone()
    assert row is not None
    return row


def test_drain_commits_claim_before_handler_runs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        row = _outbox_row(store, job.id)
        assert row["status"] == "running"
        assert row["lease_token"] == lease.lease_token
        return EffectObservation(outcome="Confirmed")

    assert drain_outbox(store, {"session.create": handler}, now=10) == 1
    assert store.get(job.id).revision == job.revision
    row = _outbox_row(store, job.id)
    assert row["status"] == "succeeded"
    assert json.loads(row["outcome_json"])["outcome"] == "Confirmed"


def test_observe_does_not_update_job_rows(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    lease = store.claim_outbox(("session.create",), now=10)
    assert lease is not None
    assert (
        store.observe_outbox(lease, EffectObservation(outcome="Confirmed"), now=11)
        == "applied"
    )
    assert store.get(job.id).revision == job.revision
    assert store.get(job.id).speakable_label == "named"


def test_lease_contention_allows_only_one_claim(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    _admit_effect(store)
    first = store.claim_outbox(("session.create",), now=10, lease_token="lease-a")
    second = store.claim_outbox(("session.create",), now=10, lease_token="lease-b")
    assert first is not None
    assert second is None
    assert _outbox_row(store, first.job_id)["lease_token"] == "lease-a"


def test_concurrency_key_serializes_distinct_effects(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    first_job = _admit_effect(
        store,
        "aaaaaaaaaaaa",
        key="session:aaaaaaaaaaaa:voice",
        concurrency_key="session:voice",
    )
    second_job = _admit_effect(
        store,
        "bbbbbbbbbbbb",
        key="session:bbbbbbbbbbbb:voice",
        concurrency_key="session:voice",
    )

    first = store.claim_outbox(("session.create",), now=10)
    assert first is not None
    assert first.job_id == first_job.id
    assert first.concurrency_key == "session:voice"
    assert store.claim_outbox(("session.create",), now=10) is None

    assert (
        store.observe_outbox(first, EffectObservation(outcome="Confirmed"), now=11)
        == "applied"
    )
    second = store.claim_outbox(("session.create",), now=11)
    assert second is not None
    assert second.job_id == second_job.id


def test_lease_renewal_prevents_expiry(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    _admit_effect(store)
    lease = store.claim_outbox(("session.create",), now=10)
    assert lease is not None

    assert store.renew_outbox_lease(lease, now=35)
    assert store.reap_expired_outbox_leases(now=50) == 0
    assert _outbox_row(store, lease.job_id)["status"] == "running"
    assert store.reap_expired_outbox_leases(now=65) == 1


def test_supervisor_renews_lease_while_handler_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    monkeypatch.setattr(outbox_module, "OUTBOX_RENEW_SECONDS", 0.01)
    renewed: list[float] = []

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        time.sleep(0.04)
        renewed.append(float(_outbox_row(store, job.id)["leased_at"]))
        return EffectObservation(outcome="Confirmed")

    assert drain_outbox(store, {"session.create": handler}, now=10) == 1
    assert renewed[0] > 10


def test_duplicate_observation_is_idempotent(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    lease = store.claim_outbox(("session.create",), now=10)
    assert lease is not None
    assert (
        store.observe_outbox(lease, EffectObservation(outcome="Confirmed"), now=11)
        == "applied"
    )
    assert (
        store.observe_outbox(
            lease,
            EffectObservation(outcome="Failed", detail={"error": "late"}),
            now=12,
        )
        == "duplicate"
    )
    row = _outbox_row(store, job.id)
    assert row["status"] == "succeeded"
    assert json.loads(row["outcome_json"])["outcome"] == "Confirmed"


def test_completed_effect_is_not_delivered_to_handler_twice(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    _admit_effect(store)
    calls = 0

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        nonlocal calls
        calls += 1
        return EffectObservation(outcome="Confirmed")

    handlers = {"session.create": handler}
    assert drain_outbox(store, handlers, now=10) == 1
    assert drain_outbox(store, handlers, now=11) == 0
    assert calls == 1


def test_handler_failure_is_terminal_and_leaves_job_untouched(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        return EffectObservation(outcome="Failed", detail={"error": "rejected"})

    assert drain_outbox(store, {"session.create": handler}, now=10) == 1
    assert store.get(job.id).revision == job.revision
    row = _outbox_row(store, job.id)
    assert row["status"] == "failed"
    assert json.loads(row["outcome_json"])["outcome"] == "Failed"
    assert row["last_error"] == "rejected"


def test_retryable_failure_is_reclaimed_after_backoff(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    _admit_effect(store)
    first = store.claim_outbox(("session.create",), now=10)
    assert first is not None
    assert (
        store.observe_outbox(
            first,
            EffectObservation(outcome="Failed", retryable=True),
            now=10,
        )
        == "applied"
    )
    assert (
        store.claim_outbox(("session.create",), now=10 + OUTBOX_RETRY_SECONDS - 0.1)
        is None
    )
    second = store.claim_outbox(("session.create",), now=10 + OUTBOX_RETRY_SECONDS)
    assert second is not None
    assert second.effect_id == first.effect_id
    assert second.attempts == 2


def test_retryable_observation_after_dispatch_becomes_unknown(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    lease = store.claim_outbox(("session.create",), now=10)
    assert lease is not None
    assert store.mark_outbox_dispatched(lease)

    assert (
        store.observe_outbox(
            lease,
            EffectObservation(outcome="Failed", retryable=True),
            now=11,
        )
        == "applied"
    )
    row = _outbox_row(store, job.id)
    assert row["status"] == "unknown"
    outcome = json.loads(row["outcome_json"])
    assert outcome["outcome"] == "OutcomeUnknown"
    assert outcome["retryable"] is False


def test_crash_before_dispatch_is_replayed_after_lease_expiry(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    first = store.claim_outbox(("session.create",), now=10)
    assert first is not None
    assert store.reap_expired_outbox_leases(now=10 + OUTBOX_LEASE_SECONDS) == 1
    row = _outbox_row(store, job.id)
    assert row["status"] == "failed-retryable"
    assert row["lease_token"] is None
    second = store.claim_outbox(
        ("session.create",),
        now=10 + OUTBOX_LEASE_SECONDS + OUTBOX_RETRY_SECONDS,
    )
    assert second is not None
    assert second.effect_id == first.effect_id
    assert second.lease_token != first.lease_token


def test_crash_after_dispatch_becomes_outcome_unknown(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    lease = store.claim_outbox(("session.create",), now=10)
    assert lease is not None
    assert store.mark_outbox_dispatched(lease)
    assert store.reap_expired_outbox_leases(now=10 + OUTBOX_LEASE_SECONDS) == 1
    row = _outbox_row(store, job.id)
    assert row["status"] == "unknown"
    assert json.loads(row["outcome_json"])["outcome"] == "OutcomeUnknown"
    assert (
        store.claim_outbox(("session.create",), now=20 + OUTBOX_LEASE_SECONDS) is None
    )


def test_handler_crash_after_dispatch_reports_unknown(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        assert callable(mark_dispatched)
        mark_dispatched()
        raise RuntimeError("crashed after submit")

    assert drain_outbox(store, {"session.create": handler}, now=10) == 1
    row = _outbox_row(store, job.id)
    assert row["status"] == "unknown"
    assert "crashed after submit" in str(row["last_error"])
    assert store.get(job.id).revision == job.revision


def test_handler_crash_before_dispatch_is_replayable(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        raise RuntimeError("crashed before submit")

    assert drain_outbox(store, {"session.create": handler}, now=10) == 1
    row = _outbox_row(store, job.id)
    assert row["status"] == "failed-retryable"
    replayed = store.claim_outbox(("session.create",), now=10 + OUTBOX_RETRY_SECONDS)
    assert replayed is not None
    assert replayed.attempts == 2


def test_handler_exceptions_stop_after_max_attempts(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE outbox SET attempts = ? WHERE job_id = ?",
            (OUTBOX_MAX_ATTEMPTS - 1, job.id),
        )

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        raise RuntimeError("still broken")

    assert drain_outbox(store, {"session.create": handler}, now=10) == 1
    row = _outbox_row(store, job.id)
    assert row["status"] == "failed"
    outcome = json.loads(row["outcome_json"])
    assert outcome["outcome"] == "Failed"
    assert outcome["retryable"] is False
    assert store.claim_outbox(("session.create",), now=100) is None


def test_recover_jobs_reaps_expired_leases(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    lease = store.claim_outbox(("session.create",), now=10)
    assert lease is not None
    recover_jobs(
        store, launch_worker=lambda _job_id: None, now=10 + OUTBOX_LEASE_SECONDS
    )
    row = _outbox_row(store, job.id)
    assert row["status"] == "failed-retryable"
    assert row["lease_token"] is None


def test_recover_jobs_executes_registered_handler(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    calls: list[str] = []

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        calls.append(lease.effect_id)
        return EffectObservation(outcome="Confirmed")

    recover_jobs(
        store,
        launch_worker=lambda _job_id: None,
        outbox_handlers={"session.create": handler},
        now=10,
    )
    assert len(calls) == 1
    assert _outbox_row(store, job.id)["status"] == "succeeded"


def test_recover_outbox_drains_registered_handlers_after_restart(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = _admit_effect(store)
    lease = store.claim_outbox(("session.create",), now=10)
    assert lease is not None

    def handler(lease: OutboxLease, mark_dispatched: object) -> EffectObservation:
        return EffectObservation(outcome="ConfirmedAbsent")

    recover_outbox(
        store,
        handlers={"session.create": handler},
        now=10 + OUTBOX_LEASE_SECONDS,
    )
    recover_outbox(
        store,
        handlers={"session.create": handler},
        now=10 + OUTBOX_LEASE_SECONDS + OUTBOX_RETRY_SECONDS,
    )
    row = _outbox_row(store, job.id)
    assert row["status"] == "succeeded"
    assert json.loads(row["outcome_json"])["outcome"] == "ConfirmedAbsent"
