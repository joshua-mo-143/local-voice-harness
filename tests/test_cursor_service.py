from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from local_voice_harness.cursor import service
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    NewCursorJob,
)
from local_voice_harness.cursor.service import (
    CursorTurnRequest,
    CursorTurnResult,
    StartJobRequest,
    TicketJobRequest,
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


def test_fanout_preflights_every_target_before_bounded_background_starts() -> None:
    events: list[str] = []
    client = mock.Mock()

    def details(issue: service.GitHubIssue) -> dict[str, object]:
        number = issue.number
        events.append(f"preflight-{number}")
        if number == 2:
            raise service.GitHubError("issue was not found")
        return {
            "number": number,
            "title": f"Issue {number}",
            "state": "OPEN",
            "url": f"https://github.com/example/project/issues/{number}",
        }

    def start(request: StartJobRequest) -> str:
        assert events[:3] == ["preflight-1", "preflight-2", "preflight-3"]
        assert not request.foreground
        assert request.github_issue in {1, 3}
        assert f"example/project#{request.github_issue}" in request.text
        events.append(f"start-{request.github_issue}")
        if request.github_issue == 3:
            raise HarnessError("job deletion maintenance is active")
        return "job-one"

    client.issue_details.side_effect = details
    with (
        mock.patch.object(service, "GitHubClient", return_value=client),
        mock.patch.object(service, "integration_enabled", return_value=True),
        mock.patch.object(service, "start_job", side_effect=start),
        mock.patch.object(service, "read_job", side_effect=KeyError),
        mock.patch.object(service, "_await_foreground") as foreground,
    ):
        result = service.cursor_turn(
            CursorTurnRequest(
                "Work on issues 1, 2, and 3",
                utterance="Work on issues 1, 2, and 3",
                issue_scope="example/project",
                issue_scope_source="github",
            )
        )

    assert result.session_id is None
    assert result.text.index("example/project#1: accepted") < result.text.index(
        "example/project#2: rejected"
    )
    assert result.text.index("example/project#2: rejected") < result.text.index(
        "example/project#3: start-failed"
    )
    assert "job-one" in result.text
    assert "job deletion maintenance is active" in result.text
    foreground.assert_not_called()


def test_start_jobs_enforces_bound_and_preserves_outcome_order() -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0

    def start(request: StartJobRequest) -> str:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return f"job-{request.text}"

    requests = tuple(
        TicketJobRequest(str(index), StartJobRequest(str(index), foreground=False))
        for index in range(6)
    )
    with (
        mock.patch.object(service, "start_job", side_effect=start),
        mock.patch.object(service, "read_job", side_effect=KeyError),
    ):
        outcomes = service.start_jobs(requests, concurrency=2)

    assert 1 < maximum <= 2
    assert [outcome.target for outcome in outcomes] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert all(outcome.status == "accepted" for outcome in outcomes)


def test_single_scoped_ticket_keeps_foreground_behavior() -> None:
    client = mock.Mock()
    client.issue_details.return_value = {
        "number": 7,
        "title": "Do it",
        "state": "OPEN",
        "url": "https://github.com/example/project/issues/7",
    }
    started: list[StartJobRequest] = []

    def start(request: StartJobRequest) -> str:
        started.append(request)
        return "123456789abc"

    with (
        mock.patch.object(service, "GitHubClient", return_value=client),
        mock.patch.object(service, "integration_enabled", return_value=True),
        mock.patch.object(service, "start_job", side_effect=start),
        mock.patch.object(service, "read_job", side_effect=KeyError),
        mock.patch.object(
            service,
            "_await_foreground",
            return_value=CursorTurnResult("working", None),
        ) as foreground,
    ):
        result = service.cursor_turn(
            CursorTurnRequest(
                "Work on issue 7",
                utterance="Work on issue 7",
                issue_scope="example/project",
                issue_scope_source="github",
            )
        )

    assert result == CursorTurnResult("working", None)
    assert len(started) == 1
    assert started[0].foreground
    foreground.assert_called_once_with("123456789abc", None)


def test_fanout_linear_capability_preflight_happens_before_any_start() -> None:
    events: list[str] = []

    def require(reference: str) -> None:
        events.append(f"capability-{reference}")

    def resolve(reference: str | None) -> str | None:
        events.append(f"resolve-{reference}")
        return reference

    def start(request: StartJobRequest) -> str:
        assert events[:3] == [
            "capability-ENG-1",
            "resolve-ENG-1",
            "resolve-ENG-2",
        ]
        events.append(f"start-{request.issue_key}")
        return f"job-{request.issue_key}"

    with (
        mock.patch.object(service, "require_issue_capabilities", side_effect=require),
        mock.patch.object(service, "resolve_issue_reference", side_effect=resolve),
        mock.patch.object(service, "start_job", side_effect=start),
        mock.patch.object(service, "read_job", side_effect=KeyError),
    ):
        result = service.cursor_turn(
            CursorTurnRequest(
                "Work on ENG-1 and ENG-2",
                utterance="Work on ENG-1 and ENG-2",
            )
        )

    assert result.text.startswith("Ticket starts: ENG-1: accepted")
    assert "ENG-2: accepted" in result.text


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
        "schema_version": CURRENT_SCHEMA_VERSION,
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


def test_nuke_jobs_preserves_record_reservations_and_artifacts_on_stop_failure(
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
        worker_operation="agent_start",
        herdr_target="agent-target",
        agent_dispatch_state="dispatching",
    )
    artifacts = jobs_dir / ".artifacts" / "aaaaaaaaaaaa"
    artifacts.mkdir(parents=True)
    evidence = artifacts / "evidence.txt"
    evidence.write_text("keep me")

    with (
        mock.patch.object(service, "_stop_worker", return_value=False),
        mock.patch.object(
            service.worker_lifecycle, "process_owner_alive", return_value=True
        ),
        pytest.raises(HarnessError, match="preserved.*still running"),
    ):
        service.nuke_jobs()

    retained = service.read_job("aaaaaaaaaaaa")
    assert retained.worker_token == "claim"
    assert retained.target_release_pending
    assert retained.cancellation_reconciliation_pending
    assert retained.herdr_target == "agent-target"
    assert retained.worker_operation == "agent_start"
    assert evidence.read_text() == "keep me"
    assert not (jobs_dir / ".maintenance").exists()


def test_nuke_jobs_preserves_uncertain_external_operation(
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
        worker_operation="agent_start",
        herdr_target="agent-target",
        agent_dispatch_state="dispatching",
    )

    with (
        mock.patch.object(service, "_stop_worker", return_value=True),
        mock.patch.object(service, "_cancel_target_and_release"),
        pytest.raises(HarnessError, match="recovery or reservation fence remains"),
    ):
        service.nuke_jobs()

    retained = service.read_job("aaaaaaaaaaaa")
    assert retained.agent_dispatch_state == "dispatching"
    assert retained.target_release_pending
    assert not (jobs_dir / ".maintenance").exists()


def test_nuke_jobs_refuses_unverifiable_process_exit(
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

    with (
        mock.patch.object(service, "_stop_worker", return_value=True),
        mock.patch.object(
            service.worker_lifecycle, "process_owner_alive", return_value=None
        ),
        pytest.raises(HarnessError, match="exit could not be verified"),
    ):
        service.nuke_jobs()

    assert service.read_job("aaaaaaaaaaaa").worker_token == "claim"
    assert not (jobs_dir / ".maintenance").exists()


def test_nuke_and_concurrent_cancellation_share_one_terminal_fence(
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
    stopping = threading.Event()
    allow_stop = threading.Event()
    outcomes: list[str] = []
    failures: list[BaseException] = []

    def stop(_job: CursorJob) -> bool:
        stopping.set()
        assert allow_stop.wait(2)
        return True

    def nuke() -> None:
        try:
            outcomes.append(service.nuke_jobs())
        except BaseException as exc:
            failures.append(exc)

    with mock.patch.object(service, "_stop_worker", side_effect=stop):
        thread = threading.Thread(target=nuke)
        thread.start()
        assert stopping.wait(2)
        assert service.cancel_job("aaaaaaaaaaaa") == (
            "Cursor job aaaaaaaaaaaa was cancelled."
        )
        allow_stop.set()
        thread.join(2)

    assert not thread.is_alive()
    assert failures == []
    assert outcomes == ["Deleted all 1 Cursor job."]
    assert list(jobs_dir.glob("*.json")) == []


def test_nuke_jobs_checks_legacy_claim_before_modern_token(
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
        worker_boot_id="legacy-unknown",
        worker_process_start="start",
        schema_version=5,
    )

    with (
        mock.patch.object(
            service.worker_lifecycle,
            "inspect_and_stop_legacy_worker",
            return_value="unsafe",
        ) as inspect,
        mock.patch.object(service, "_stop_worker") as stop,
        pytest.raises(HarnessError, match="legacy worker identity"),
    ):
        service.nuke_jobs()

    inspect.assert_called_once()
    stop.assert_not_called()
    assert (jobs_dir / "aaaaaaaaaaaa.json").exists()


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
