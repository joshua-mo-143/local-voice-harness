from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_voice_harness.agents.model import (
    CURRENT_SCHEMA_VERSION,
    AgentJob,
    AgentJobValidationError,
    HarnessKind,
    NewAgentJob,
)
from local_voice_harness.agents.store import migrate_legacy_jobs

FIXTURES = Path(__file__).parent / "fixtures" / "jobs"


def fixture(name: str) -> dict[str, object]:
    value = json.loads((FIXTURES / name).read_text())
    assert isinstance(value, dict)
    return value


def test_v8_cursor_fixture_migrates_to_structured_agent_schema() -> None:
    job = AgentJob.from_dict(fixture("cursor-v8.json"))

    assert job.loaded_schema_version == 8
    assert job.schema_version == CURRENT_SCHEMA_VERSION
    assert job.harness_kind == HarnessKind.CURSOR
    assert job.session_id == "cursor-agent-61"
    assert job.herdr_target == job.session_id
    assert job.to_record() == fixture("agent-v9.json")


def test_v9_agent_fixture_loads_with_legacy_recovery_accessors() -> None:
    job = AgentJob.from_dict(fixture("agent-v9.json"))

    assert job.loaded_schema_version == CURRENT_SCHEMA_VERSION
    assert job.herdr_pane_id == "pane-1"
    assert job.agent_dispatch_state == "ready"
    assert job.github_repository == "owner/project"
    assert job.github_issue == 61
    assert job.repository == "/repo"
    assert job.worktree_path == "/repo-worktree"
    assert job.issue_key == "VOICE-61"


def test_v9_structured_record_requires_harness_kind() -> None:
    record = fixture("agent-v9.json")
    record.pop("harness_kind")

    with pytest.raises(AgentJobValidationError, match="harness_kind"):
        AgentJob.from_dict(record)


def test_new_job_record_has_no_cursor_or_herdr_core_fields() -> None:
    job = AgentJob.new(
        NewAgentJob(
            id="abcdef123456",
            request="build support for another harness",
            created_at=10,
            foreground_until=20,
            harness_kind=HarnessKind.CURSOR,
            session_id="session-1",
        )
    )

    record = job.to_record()
    assert record["harness_kind"] == "cursor"
    assert record["session_id"] == "session-1"
    assert "herdr_target" not in record
    assert "herdr_pane_id" not in record
    assert "github_repository" not in record
    assert isinstance(record["harness_state"], dict)
    assert isinstance(record["provider_state"], dict)


def test_legacy_directory_import_rewrites_v8_record(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    durable = tmp_path / "durable"
    legacy.mkdir()
    source = legacy / "123456789abc.json"
    source.write_text(json.dumps(fixture("cursor-v8.json")))

    assert migrate_legacy_jobs(legacy, durable) == set()

    persisted = json.loads((durable / source.name).read_text())
    assert persisted == fixture("agent-v9.json")
    assert not source.exists()


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4, 5, 6, 7, 8])
def test_every_legacy_schema_version_has_an_explicit_upgrade(
    version: int | None,
) -> None:
    record: dict[str, object] = {
        "id": "abcdef123456",
        "status": "queued",
        "request": "migrate",
        "created_at": 1,
        "queued_at": 1,
        "delivered": False,
    }
    if version is not None:
        record["schema_version"] = version

    job = AgentJob.from_dict(record)

    assert job.loaded_schema_version == (version or 0)
    assert job.harness_kind == HarnessKind.CURSOR
    assert job.to_record()["schema_version"] == CURRENT_SCHEMA_VERSION
