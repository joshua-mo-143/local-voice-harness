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
