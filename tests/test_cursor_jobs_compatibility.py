from __future__ import annotations

from pathlib import Path

import pytest

from local_voice_harness.cursor import jobs
from local_voice_harness.cursor.store import JobQuarantineWarning


def configure_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "LEGACY_JOBS_DIR", tmp_path / "legacy")


def test_legacy_job_store_and_query_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_store(tmp_path, monkeypatch)
    jobs.write_job(
        {
            "id": "123456789abc",
            "status": "queued",
            "request": "compatibility",
            "created_at": 1,
            "queued_at": 1,
            "delivered": False,
            "herdr_target": "reserved-agent",
        }
    )

    loaded = jobs.read_job("123456789abc")

    assert loaded["request"] == "compatibility"
    assert [job["id"] for job in jobs.active_jobs()] == ["123456789abc"]
    assert jobs.reserved_targets() == {"reserved-agent"}
    assert jobs.job_path("123456789abc").name == "123456789abc.json"


def test_legacy_delivery_claim_preserves_private_token_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_store(tmp_path, monkeypatch)
    jobs.write_job(
        {
            "id": "123456789abc",
            "status": "completed",
            "request": "compatibility",
            "created_at": 1,
            "completed_at": 2,
            "foreground_until": 0,
            "result": "done",
            "delivered": False,
        }
    )

    claimed = jobs.claim_delivery("123456789abc", foreground=True)

    assert claimed is not None
    token = str(claimed["_delivery_token"])
    assert jobs.acknowledge_delivery("123456789abc", token)
    assert jobs.read_job("123456789abc")["delivered"] is True


def test_legacy_helpers_keep_dict_mutation_and_confirmation_contract() -> None:
    job: dict[str, object] = {
        "id": "123456789abc",
        "turn_token": "turn",
        "herdr_target": "agent",
    }

    jobs.complete_from_output(
        job,
        output="VOICE_SUMMARY[turn]: completed",
        agent_status="idle",
    )

    assert job["status"] == "reconciling"
    assert job["terminal_intent_status"] == "completed"
    assert job["terminal_intent_result"] == "completed"
    assert job["target_release_pending"] is True
    assert jobs.decide_fork_confirmation("Yes, please!") is True
    assert jobs.decide_fork_confirmation("no thanks") is False
    assert jobs.decide_fork_confirmation("maybe") is None


def test_read_modify_write_increments_revision_once_and_rejects_stale_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_store(tmp_path, monkeypatch)
    jobs.write_job(
        {
            "id": "123456789abc",
            "status": "queued",
            "request": "original",
            "created_at": 1,
            "queued_at": 1,
            "delivered": False,
        }
    )
    first = jobs.read_job("123456789abc")
    stale = dict(first)
    first["repository_hint"] = "edited"

    jobs.write_job(first)

    persisted = jobs.read_job("123456789abc")
    assert persisted["repository_hint"] == "edited"
    assert persisted["request"] == "original"
    assert persisted["revision"] == 1
    stale["repository_hint"] = "stale edit"
    with pytest.raises(jobs.HarnessError, match="stale Cursor job revision"):
        jobs.write_job(stale)
    assert jobs.read_job("123456789abc")["revision"] == 1


def test_read_modify_write_rejects_original_request_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_store(tmp_path, monkeypatch)
    jobs.write_job(
        {
            "id": "123456789abc",
            "status": "queued",
            "request": "original",
            "created_at": 1,
            "queued_at": 1,
            "delivered": False,
        }
    )
    edited = jobs.read_job("123456789abc")
    edited["request"] = "changed"

    with pytest.raises(jobs.HarnessError, match="cannot change the original request"):
        jobs.write_job(edited)


def test_compatibility_store_errors_use_harness_error_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_store(tmp_path, monkeypatch)
    with pytest.raises(jobs.HarnessError, match="could not read Cursor job"):
        jobs.read_job("123456789abc")
    with pytest.raises(jobs.HarnessError, match="could not write Cursor job"):
        jobs.write_job({"id": "invalid", "status": "queued"})

    jobs.JOBS_DIR.mkdir(exist_ok=True)
    corrupted = jobs.JOBS_DIR / "123456789abc.json"
    corrupted.write_text("{corrupt")
    with (
        pytest.warns(JobQuarantineWarning),
        pytest.raises(jobs.HarnessError, match="job is quarantined"),
    ):
        jobs.read_job("123456789abc")
    assert list((jobs.JOBS_DIR / ".quarantine").glob("123456789abc-*.json"))
