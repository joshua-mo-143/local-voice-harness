from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest

from local_voice_harness.cursor import service
from local_voice_harness.cursor.delivery import claim_delivery
from local_voice_harness.cursor.model import CURRENT_SCHEMA_VERSION, CursorJob
from local_voice_harness.cursor.service import CursorTurnRequest, cursor_turn
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.responses import as_assistant_response
from tests.support import run_concurrently


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobStore:
    jobs_dir = tmp_path / "jobs"
    legacy_dir = tmp_path / "legacy"
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", legacy_dir)
    return JobStore(jobs_dir, legacy_dir)


def _make(
    store: JobStore,
    job_id: str,
    *,
    status: str = "queued",
    label: str | None = None,
    request: str = "do the thing",
    issue_key: str | None = None,
    issue_provider: str | None = None,
    github_repository: str | None = None,
    github_issue: int | None = None,
) -> CursorJob:
    now = time.time()
    values: dict[str, object] = {
        "id": job_id,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "revision": 0,
        "request": request,
        "status": status,
        "created_at": now,
        "delivered": False,
        "foreground_until": 0,
        "speakable_label": label,
        "issue_key": issue_key,
        "issue_provider": issue_provider,
        "github_repository": github_repository,
        "github_issue": github_issue,
    }
    if status == "queued":
        values["queued_at"] = now
    elif status == "awaiting_user":
        values.update(question="which repo?", result="which repo?")
    elif status in {"completed", "cancelled", "blocked"}:
        values.update(result="done", completed_at=now)
    elif status == "failed":
        values.update(result="boom", error="boom", completed_at=now)
    if status in {"routing", "running", "reconciling"}:
        values.update(
            worker_token="claim",
            worker_pid=42,
            worker_boot_id="boot",
            worker_process_start="start",
            worker_claim_operation="test",
            worker_claimed_at=now,
        )
    return store.create(CursorJob.from_dict(values))


def test_list_jobs_summarizes_inbox(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", label="issue 42")
    _make(store, "bbbbbbbbbbbb", status="awaiting_user", label="api refactor")

    result = cursor_turn(CursorTurnRequest("", action="list"))
    text = as_assistant_response(result.text).display_text

    assert "You have 2 Cursor jobs." in text
    assert "api refactor" in text
    assert "issue 42" in text


def test_ambiguous_cancel_clarifies_without_cancelling(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="running", label="issue 42")
    _make(store, "bbbbbbbbbbbb", status="running", label="issue 43")

    result = cursor_turn(CursorTurnRequest("cancel the issue job", action="cancel"))
    response = as_assistant_response(result.text)

    assert "Which one" in response.display_text
    assert "issue 42" in response.spoken_text
    assert "issue 43" in response.spoken_text
    assert "aaaa" not in response.spoken_text
    assert "bbbb" not in response.spoken_text
    assert store.get("aaaaaaaaaaaa").status.value == "running"
    assert store.get("bbbbbbbbbbbb").status.value == "running"


def test_reference_cancel_targets_intended_job(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="running", label="issue 42")
    _make(store, "bbbbbbbbbbbb", status="running", label="venice fix")

    with mock.patch.object(service, "_stop_worker", return_value=True):
        result = cursor_turn(
            CursorTurnRequest("cancel the venice fix", action="cancel")
        )

    response = as_assistant_response(result.text)
    assert response.spoken_text == "Cursor cancelled venice fix."
    assert "bbbbbbbbbbbb" not in response.spoken_text
    assert "bbbbbbbbbbbb" in response.display_text

    assert store.get("bbbbbbbbbbbb").status.value == "cancelled"
    assert store.get("aaaaaaaaaaaa").status.value == "running"


def test_status_reports_speakable_label(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="running", label="issue 42")

    result = cursor_turn(CursorTurnRequest("aaaa", action="status", reference="aaaa"))

    assert result.text == "issue 42 is running."


def test_compatibility_status_uses_labels_instead_of_job_ids(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="running", label="issue 42")
    _make(store, "bbbbbbbbbbbb", status="running", label="readme update")

    single = service.job_status("aaaaaaaaaaaa")
    active = service.job_status()

    assert "issue 42 is running" in single
    assert "aaaaaaaaaaaa" not in single
    assert "issue 42" in active
    assert "readme update" in active
    assert "aaaaaaaaaaaa" not in active
    assert "bbbbbbbbbbbb" not in active


def test_status_resolves_linear_ticket_over_same_github_number(
    store: JobStore,
) -> None:
    _make(
        store,
        "aaaaaaaaaaaa",
        status="running",
        label="APP-43",
        issue_key="APP-43",
        issue_provider="linear",
    )
    _make(
        store,
        "bbbbbbbbbbbb",
        status="completed",
        label="widgets issue 43",
        issue_provider="github",
        github_repository="acme/widgets",
        github_issue=43,
    )

    result = cursor_turn(CursorTurnRequest("how is APP-43 doing?", action="status"))

    assert result.text == "APP-43 is running."


def test_dismiss_suppresses_delivery_and_persists_state(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="completed", label="bug fix")
    assert claim_delivery(store, "aaaaaaaaaaaa", foreground=True) is not None
    # release the claim so dismissal is the only thing suppressing delivery
    service.release_delivery(
        "aaaaaaaaaaaa",
        store.get("aaaaaaaaaaaa").delivery_claim_token or "",
        retry=False,
    )

    result = cursor_turn(CursorTurnRequest("bug fix", action="dismiss"))

    assert "Dismissed" in as_assistant_response(result.text).display_text
    dismissed = store.get("aaaaaaaaaaaa")
    assert dismissed.announcement_dismissed is True
    assert dismissed.delivered is True
    assert claim_delivery(store, "aaaaaaaaaaaa", foreground=True) is None


def test_repeat_rearms_delivery_preserving_at_least_once(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="completed", label="bug fix")
    claim = claim_delivery(store, "aaaaaaaaaaaa", foreground=True)
    assert claim is not None
    service.acknowledge_delivery("aaaaaaaaaaaa", claim.token)
    assert store.get("aaaaaaaaaaaa").delivered is True
    generation = store.get("aaaaaaaaaaaa").delivery_generation

    result = cursor_turn(CursorTurnRequest("bug fix", action="repeat"))

    assert "repeat" in as_assistant_response(result.text).display_text
    repeated = store.get("aaaaaaaaaaaa")
    assert repeated.announcement_repeated is True
    assert repeated.delivered is False
    assert repeated.delivery_generation == generation + 1
    assert claim_delivery(store, "aaaaaaaaaaaa", foreground=True) is not None


def test_reply_without_session_resolves_single_awaiting_job(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="awaiting_user", label="api refactor")

    with (
        mock.patch.object(service, "launch_worker") as launch,
        mock.patch.object(service, "_await_foreground") as foreground,
    ):
        cursor_turn(CursorTurnRequest("use the api repository", action="reply"))

    launch.assert_called_once_with("aaaaaaaaaaaa")
    foreground.assert_called_once_with(
        "aaaaaaaaaaaa",
        None,
        timeout=5.0,
        continuation=True,
    )
    assert store.get("aaaaaaaaaaaa").status.value == "queued"


def test_new_fields_survive_restart(store: JobStore, tmp_path: Path) -> None:
    _make(store, "aaaaaaaaaaaa", status="completed", label="bug fix")
    cursor_turn(CursorTurnRequest("bug fix", action="dismiss"))

    # A fresh store instance models a process restart reading the same directory.
    restarted = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    reloaded = restarted.get("aaaaaaaaaaaa")
    assert reloaded.speakable_label == "bug fix"
    assert reloaded.announcement_dismissed is True


def test_recovery_preserves_announcement_state(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="completed", label="bug fix")
    cursor_turn(CursorTurnRequest("bug fix", action="repeat"))

    with mock.patch.object(service, "launch_worker"):
        service.recover_jobs()

    recovered = store.get("aaaaaaaaaaaa")
    assert recovered.speakable_label == "bug fix"
    assert recovered.announcement_repeated is True


def test_concurrent_dismiss_has_single_effective_write(store: JobStore) -> None:
    _make(store, "aaaaaaaaaaaa", status="completed", label="bug fix")
    base = store.get("aaaaaaaaaaaa").revision

    outcomes = run_concurrently(
        [
            lambda: service.dismiss_announcement("aaaaaaaaaaaa"),
            lambda: service.dismiss_announcement("aaaaaaaaaaaa"),
        ]
    )

    assert all(error is None for error in outcomes.errors)
    final = store.get("aaaaaaaaaaaa")
    assert final.announcement_dismissed is True
    # Exactly one dismissal mutates the record; the other observes it as done.
    assert final.revision == base + 1
