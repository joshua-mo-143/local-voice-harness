from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from unittest import mock

import pytest

from local_voice_harness.cursor import service
from local_voice_harness.cursor.model import CursorJob, NewCursorJob
from local_voice_harness.cursor.service import (
    CursorTurnRequest,
    CursorTurnResult,
    StartJobRequest,
)
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.errors import HarnessError


def test_service_request_and_result_types_are_explicit() -> None:
    start = StartJobRequest("fix the bug", repository="project")
    turn = CursorTurnRequest("continue", session_id="123456789abc", action="reply")
    result = CursorTurnResult("done", None)

    assert start.repository == "project"
    assert turn.session_id == "123456789abc"
    assert tuple(result) == ("done", None)


def test_submit_notifies_only_after_job_starts() -> None:
    events: list[str] = []

    def start(*_args: object, **_kwargs: object) -> str:
        events.append("started")
        return "123456789abc"

    with (
        mock.patch.object(service, "start_job", side_effect=start),
        mock.patch.object(
            service,
            "_await_foreground",
            return_value=CursorTurnResult("working", None),
        ),
    ):
        result = service.cursor_turn(
            CursorTurnRequest(
                "fix it",
                on_job_started=lambda: events.append("notified"),
            )
        )

    assert result == CursorTurnResult("working", None)
    assert events == ["started", "notified"]


def test_submit_failure_does_not_notify() -> None:
    notified = mock.Mock()

    with (
        mock.patch.object(service, "start_job", side_effect=HarnessError("failed")),
        pytest.raises(HarnessError, match="failed"),
    ):
        service.cursor_turn(CursorTurnRequest("fix it", on_job_started=notified))

    notified.assert_not_called()


def test_production_modules_do_not_import_jobs_facade() -> None:
    source_root = Path(__file__).parents[1] / "src" / "local_voice_harness"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        if path.name == "jobs.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module == "local_voice_harness.cursor.jobs"
                or (node.module == "jobs" and path.parent.name == "cursor")
                or node.module == "cursor.jobs"
            ):
                offenders.append(str(path.relative_to(source_root)))
    assert offenders == []


def test_service_and_store_reads_return_cursor_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    legacy_dir = tmp_path / "legacy"
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", legacy_dir)
    now = time.time()
    job = CursorJob.new(
        NewCursorJob(
            id="123456789abc",
            request="typed boundary",
            created_at=now,
            foreground_until=now,
        )
    )
    store = JobStore(jobs_dir, legacy_dir)
    created = store.create(job)

    assert isinstance(created, CursorJob)
    assert isinstance(store.get(job.id), CursorJob)
    assert isinstance(service.read_job(job.id), CursorJob)


def test_cancellation_refuses_unsafe_durable_legacy_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    legacy_dir = tmp_path / "legacy"
    jobs_dir.mkdir()
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", legacy_dir)
    path = jobs_dir / "123456789abc.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "id": "123456789abc",
                "revision": 0,
                "request": "legacy",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "worker_token": "legacy-claim",
                "worker_pid": 42,
                "worker_process_start": "start",
            }
        )
    )
    before = path.read_bytes()

    with (
        mock.patch.object(
            service.worker_lifecycle,
            "inspect_and_stop_legacy_worker",
            return_value="unsafe",
        ),
        pytest.raises(HarnessError, match="could not safely stop legacy"),
    ):
        service.cancel_job("123456789abc")

    assert path.read_bytes() == before


def test_cancellation_clears_safely_absent_durable_legacy_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    legacy_dir = tmp_path / "legacy"
    jobs_dir.mkdir()
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", legacy_dir)
    path = jobs_dir / "123456789abc.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "id": "123456789abc",
                "revision": 0,
                "request": "legacy",
                "status": "running",
                "created_at": 1,
                "delivered": False,
                "worker_token": "legacy-claim",
                "worker_pid": 42,
                "worker_process_start": "start",
            }
        )
    )

    with mock.patch.object(
        service.worker_lifecycle,
        "inspect_and_stop_legacy_worker",
        return_value="absent",
    ):
        service.cancel_job("123456789abc")

    cancelled = service.read_job("123456789abc")
    assert cancelled.status.value == "cancelled"
    assert cancelled.worker_token is None
    assert cancelled.worker_pid is None


def _write_queued_job(jobs_dir: Path, job_id: str, **fields: object) -> None:
    value: dict[str, object] = {
        "schema_version": 5,
        "id": job_id,
        "revision": 0,
        "request": "do it",
        "status": "queued",
        "created_at": 1,
        "queued_at": 1,
        "delivered": False,
    }
    value.update(fields)
    (jobs_dir / f"{job_id}.json").write_text(json.dumps(value))


def test_count_jobs_reports_durable_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", tmp_path / "legacy")

    assert service.count_jobs() == 0
    _write_queued_job(jobs_dir, "aaaaaaaaaaaa")
    _write_queued_job(jobs_dir, "bbbbbbbbbbbb")
    assert service.count_jobs() == 2


def test_nuke_jobs_deletes_all_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", tmp_path / "legacy")
    _write_queued_job(jobs_dir, "aaaaaaaaaaaa")
    _write_queued_job(jobs_dir, "bbbbbbbbbbbb")

    message = service.nuke_jobs()

    assert message == "Deleted all 2 Cursor jobs."
    assert list(jobs_dir.glob("*.json")) == []
    assert service.count_jobs() == 0


def test_nuke_jobs_stops_running_worker_before_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", tmp_path / "legacy")
    _write_queued_job(
        jobs_dir,
        "aaaaaaaaaaaa",
        status="running",
        worker_token="claim",
        worker_pid=42,
        worker_boot_id="boot",
        worker_process_start="start",
    )

    with mock.patch.object(service, "_stop_worker", return_value=True) as stop_worker:
        message = service.nuke_jobs()

    stop_worker.assert_called_once()
    assert message == "Deleted all 1 Cursor job."
    assert list(jobs_dir.glob("*.json")) == []


def test_nuke_jobs_with_no_jobs_reports_nothing_to_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", tmp_path / "legacy")

    assert service.nuke_jobs() == "There were no Cursor jobs to delete."


def test_behavior_modules_reject_low_level_store_and_job_dict_mutation() -> None:
    cursor_root = Path(__file__).parents[1] / "src" / "local_voice_harness" / "cursor"
    behavior_modules = (
        "service.py",
        "provisioning.py",
        "recovery.py",
        "worker_lifecycle.py",
    )
    forbidden_store_names = {
        "locked",
        "read_unlocked",
        "read_all_unlocked",
        "write_unlocked",
    }
    offenders: list[str] = []
    for name in behavior_modules:
        path = cursor_root / name
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "store"
                and node.level == 1
            ):
                imported = {alias.name for alias in node.names}
                if imported & forbidden_store_names:
                    offenders.append(f"{name}: low-level store import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"job", "current"}
                    and node.func.attr in {"get", "update", "pop", "to_dict"}
                ):
                    offenders.append(
                        f"{name}:{node.lineno}: {node.func.value.id}.{node.func.attr}"
                    )
            if isinstance(node, ast.AnnAssign):
                annotation = ast.unparse(node.annotation)
                if annotation == "dict[str, object]" and name != "recovery.py":
                    offenders.append(f"{name}:{node.lineno}: raw job dictionary")
    assert offenders == []
