from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

import pytest

from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
    transition,
)
from local_voice_harness.cursor.store import JobStore
from tests.support import run_fresh_interpreter

CHILD = Path(__file__).with_name("cursor_crash_child.py")
FIXTURES = Path(__file__).parent / "fixtures" / "jobs"
JOB_ID = "123456789abc"


def _crash(root: Path, command: str, *arguments: object) -> None:
    result = run_fresh_interpreter(CHILD, command, root, *arguments)
    assert result.returncode == -signal.SIGKILL, (
        f"child did not reach killpoint; exit={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _recover(root: Path) -> None:
    result = run_fresh_interpreter(CHILD, "recover", root, timeout=10)
    assert result.returncode == 0, (
        f"recovery failed with exit={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _effects(root: Path) -> dict[str, Any]:
    path = root / "effects.json"
    if not path.exists():
        return {
            "agents": {},
            "calls": {},
            "forks": {},
            "launches": [],
            "panes": {},
            "playbacks": [],
            "worktrees": {},
        }
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("fixture", ["cursor-v8.json", "agent-v9.json"])
@pytest.mark.parametrize(
    "killpoint",
    [
        "file-fsync",
        "pre-rename",
        "post-rename",
        "directory-fsync",
        "pre-source-unlink",
        "post-source-unlink",
    ],
)
def test_active_schema_import_survives_process_death(
    tmp_path: Path, fixture: str, killpoint: str
) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    source = legacy / f"{JOB_ID}.json"
    source.write_bytes((FIXTURES / fixture).read_bytes())

    _crash(tmp_path, "migrate", killpoint)
    _recover(tmp_path)

    durable_jobs = list((tmp_path / "jobs").glob("*.json"))
    assert [path.name for path in durable_jobs] == [f"{JOB_ID}.json"]
    assert not source.exists()
    persisted = json.loads(durable_jobs[0].read_text())
    assert persisted["schema_version"] == CURRENT_SCHEMA_VERSION
    assert persisted["id"] == JOB_ID
    assert _effects(tmp_path)["launches"] == [JOB_ID]


@pytest.mark.parametrize(
    "killpoint",
    ["quarantine-metadata", "quarantine-move", "quarantine-fsync"],
)
@pytest.mark.parametrize("payload", ["malformed", "torn"])
def test_quarantine_crash_retains_reservation_evidence(
    tmp_path: Path, killpoint: str, payload: str
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    malformed = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "id": JOB_ID,
        "revision": 0,
        "request": "malformed active job",
        "status": "running",
        "created_at": 1,
        "delivered": False,
        "herdr_target": "reserved-agent",
        "worktree_path": "/worktrees/reserved",
        "parent_job_id": "invalid",
    }
    contents = (
        json.dumps(malformed)
        if payload == "malformed"
        else (
            f'{{"schema_version":{CURRENT_SCHEMA_VERSION},"id":"{JOB_ID}",'
            '"status":"running","herdr_target":"reserved-agent"'
        )
    )
    (jobs / f"{JOB_ID}.json").write_text(contents)

    _crash(tmp_path, "quarantine", killpoint)
    _recover(tmp_path)

    store = JobStore(jobs, tmp_path / "legacy")
    assert store.list() == []
    assert list((jobs / ".quarantine").glob(f"{JOB_ID}-*.metadata.json"))
    candidate = store.create(
        CursorJob.from_dict(
            {
                "id": "bbbbbbbbbbbb",
                "status": "queued",
                "request": "conflicting job",
                "created_at": 2,
                "queued_at": 2,
                "delivered": False,
            }
        )
    )
    with pytest.raises(JobValidationError, match="unresolved quarantine evidence"):
        store.reserve_target(
            candidate.id,
            lambda job: transition(
                job, JobStatus.QUEUED, herdr_target="reserved-agent"
            ),
        )
    with pytest.raises(JobValidationError, match="unresolved quarantine evidence"):
        store.reserve_worktree(
            candidate.id,
            lambda job: transition(
                job, JobStatus.QUEUED, worktree_path="/worktrees/reserved"
            ),
        )


def _seed_effect_job(root: Path, effect: str) -> None:
    values: dict[str, object] = {
        "id": JOB_ID,
        "status": "queued",
        "request": "exercise durable side effect",
        "created_at": 1,
        "queued_at": 1,
        "delivered": False,
    }
    if effect == "prompt":
        values.update(
            workflow_phase="classifying",
            turn=1,
            turn_token=f"{JOB_ID}-1",
            workflow_turn_phase="classifying",
            herdr_target="planner",
            planner_target="planner",
            active_participant="planner",
            agent_dispatch_state="ready",
        )
    JobStore(root / "jobs", root / "legacy").create(CursorJob.from_dict(values))


_EFFECT_CALL = {
    "worktree": "worktrees:voice/issue-100",
    "agent": "agents:voice-issue-100",
    "fork": "forks:voice-user/project",
    "pane": "panes:pane-reviewer",
    "prompt": f"prompts:{JOB_ID}-1",
}


@pytest.mark.parametrize("effect", ["worktree", "agent", "fork", "pane", "prompt"])
@pytest.mark.parametrize("killpoint", ["pre-submit", "post-submit", "post-settle"])
def test_external_effect_recovery_never_blindly_replays(
    tmp_path: Path, effect: str, killpoint: str
) -> None:
    _seed_effect_job(tmp_path, effect)

    _crash(tmp_path, "effect", effect, killpoint)
    _recover(tmp_path)

    state = _effects(tmp_path)
    calls = state["calls"]
    expected_calls = 0 if killpoint == "pre-submit" else 1
    assert calls.get(_EFFECT_CALL[effect], 0) == expected_calls
    job = JobStore(tmp_path / "jobs", tmp_path / "legacy").get(JOB_ID)

    if killpoint == "pre-submit":
        if effect == "prompt":
            assert job.prompt_operation_state == "ambiguous"
            assert job.manual_reconcile_operation == "prompt"
        else:
            assert job.manual_reconcile_operation == effect
        assert job.manual_reconcile_token
        return

    if effect == "worktree":
        expected = "retained" if killpoint == "post-submit" else "ready"
        assert job.worktree_provision_state == expected
    elif effect == "agent":
        assert job.agent_dispatch_state == "ready"
    elif effect == "fork":
        assert job.fork_operation_state == "exists"
        assert job.fork_exists
    elif effect == "pane":
        expected = "manual_required" if killpoint == "post-submit" else "created"
        assert job.participant_creation_state == expected
        if killpoint == "post-submit":
            assert job.manual_reconcile_operation == "pane"
            assert job.manual_reconcile_token
    else:
        assert job.prompt_operation_state == "submitted"


def _seed_delivery(root: Path) -> None:
    JobStore(root / "jobs", root / "legacy").create(
        CursorJob.from_dict(
            {
                "id": JOB_ID,
                "status": "completed",
                "request": "deliver result",
                "result": "done",
                "created_at": 1,
                "completed_at": 10,
                "delivered": False,
            }
        )
    )


@pytest.mark.parametrize(
    ("killpoint", "expected_playbacks"),
    [
        ("after-claim", 1),
        ("after-playback", 2),
        ("after-ack", 1),
    ],
)
def test_delivery_crash_is_at_least_once_without_false_acknowledgement(
    tmp_path: Path, killpoint: str, expected_playbacks: int
) -> None:
    _seed_delivery(tmp_path)

    _crash(tmp_path, "delivery", killpoint, 100)
    _recover(tmp_path)
    result = run_fresh_interpreter(CHILD, "delivery", tmp_path, "none", 401)
    assert result.returncode == 0, result.stderr

    state = _effects(tmp_path)
    assert state["playbacks"] == [JOB_ID] * expected_playbacks
    job = JobStore(tmp_path / "jobs", tmp_path / "legacy").get(JOB_ID)
    assert job.delivered
