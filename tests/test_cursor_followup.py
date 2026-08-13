from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import cast
from unittest import mock

import pytest

from local_voice_harness.cursor import provisioning, recovery, service, worker_lifecycle
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    JobStatus,
    JobValidationError,
)
from local_voice_harness.cursor.service import CursorTurnRequest, cursor_turn
from local_voice_harness.cursor.store import (
    FollowUpCheckoutBusy,
    FollowUpUnavailable,
    JobStore,
)
from local_voice_harness.errors import HarnessError
from local_voice_harness.integrations.herdr import (
    AgentSelection,
    HerdrClient,
    HerdrError,
)
from local_voice_harness.responses import as_assistant_response


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobStore:
    jobs_dir = tmp_path / "jobs"
    legacy_dir = tmp_path / "legacy"
    monkeypatch.setattr(service, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(service, "LEGACY_JOBS_DIR", legacy_dir)
    return JobStore(jobs_dir, legacy_dir)


def _completed_parent(
    store: JobStore,
    tmp_path: Path,
    job_id: str = "aaaaaaaaaaaa",
    *,
    worktree_provision_state: str = "ready",
    completed_at: float | None = None,
    issue_key: str | None = None,
) -> CursorJob:
    now = completed_at if completed_at is not None else time.time()
    repository = tmp_path / "repo"
    checkout = tmp_path / "worktrees" / "wt"
    values: dict[str, object] = {
        "id": job_id,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "revision": 0,
        "request": "add a feature",
        "status": JobStatus.COMPLETED.value,
        "created_at": now,
        "completed_at": now,
        "delivered": True,
        "result": "done",
        "repository": str(repository),
        "worktree_branch": "voice/feature",
        "worktree_path": str(checkout),
        "worktree_provision_state": worktree_provision_state,
        "speakable_label": "the feature",
        "issue_key": issue_key,
    }
    return store.create(CursorJob.from_dict(values))


def _active_worktree_holder(store: JobStore, checkout: str) -> None:
    now = time.time()
    store.create(
        CursorJob.from_dict(
            {
                "id": "cccccccccccc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "request": "busy",
                "status": JobStatus.QUEUED.value,
                "created_at": now,
                "queued_at": now,
                "delivered": False,
                "repository": str(Path(checkout).parent / "repo"),
                "worktree_branch": "voice/feature",
                "worktree_path": checkout,
                "worktree_provision_state": "ready",
            }
        )
    )


def _build_child(parent: CursorJob, child_id: str = "bbbbbbbbbbbb") -> CursorJob:
    now = time.time()
    from local_voice_harness.cursor.model import NewCursorJob

    return CursorJob.new(
        NewCursorJob(
            id=child_id,
            parent_job_id=parent.id,
            request="review the changes",
            created_at=now,
            foreground_until=0,
            repository=parent.repository,
            worktree_branch=parent.worktree_branch,
            worktree_path=parent.worktree_path,
            worktree_provision_state="ready",
            harness_kind=parent.harness_kind,
            issue_provider=parent.issue_provider,
        )
    )


def test_create_follow_up_links_parent_and_copies_checkout(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path)

    child = store.create_follow_up(
        parent.id, _build_child, expected_completed_at=parent.completed_at
    )

    assert child.parent_job_id == parent.id
    assert child.status == JobStatus.QUEUED
    assert child.repository == parent.repository
    assert child.worktree_branch == parent.worktree_branch
    assert child.worktree_path == parent.worktree_path
    # The parent record is never rewritten by follow-up creation.
    assert store.get(parent.id).revision == parent.revision


def test_create_follow_up_accepts_retained_worktree(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path, worktree_provision_state="retained")
    child = store.create_follow_up(parent.id, _build_child)
    assert child.parent_job_id == parent.id


def test_create_follow_up_rejects_non_completed_parent(
    store: JobStore, tmp_path: Path
) -> None:
    now = time.time()
    store.create(
        CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "request": "x",
                "status": JobStatus.AWAITING_USER.value,
                "created_at": now,
                "question": "which repo?",
                "result": "which repo?",
                "delivered": False,
            }
        )
    )
    with pytest.raises(FollowUpUnavailable):
        store.create_follow_up("aaaaaaaaaaaa", _build_child)


def test_create_follow_up_rejects_completion_identity_mismatch(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path)
    with pytest.raises(FollowUpUnavailable):
        store.create_follow_up(
            parent.id,
            _build_child,
            expected_completed_at=(parent.completed_at or 0) + 5,
        )


def test_create_follow_up_rejects_shared_clone(store: JobStore, tmp_path: Path) -> None:
    now = time.time()
    shared = tmp_path / "repo"
    store.create(
        CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "request": "x",
                "status": JobStatus.COMPLETED.value,
                "created_at": now,
                "completed_at": now,
                "delivered": True,
                "result": "done",
                "repository": str(shared),
                "worktree_branch": "voice/feature",
                "worktree_path": str(shared),
                "worktree_provision_state": "ready",
            }
        )
    )
    with pytest.raises(FollowUpUnavailable):
        store.create_follow_up("aaaaaaaaaaaa", _build_child)


def test_create_follow_up_checkout_busy_when_reserved(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path)
    assert parent.worktree_path is not None
    _active_worktree_holder(store, parent.worktree_path)
    with pytest.raises(FollowUpCheckoutBusy):
        store.create_follow_up(parent.id, _build_child)


def test_create_follow_up_respects_unresolved_quarantine_reservation(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path)
    assert parent.worktree_path is not None
    quarantined = store.path("cccccccccccc")
    quarantined.write_text(
        json.dumps(
            {
                "id": "cccccccccccc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "request": "uncertain owner",
                "status": JobStatus.RUNNING.value,
                "created_at": 1,
                "delivered": False,
                "repository": parent.repository,
                "worktree_branch": parent.worktree_branch,
                "worktree_path": parent.worktree_path,
                "worktree_provision_state": "ready",
            }
        )
    )
    with pytest.warns(UserWarning):
        store.list()

    with pytest.raises(FollowUpCheckoutBusy):
        store.create_follow_up(parent.id, _build_child)


def test_create_follow_up_rechecks_newly_quarantined_peer(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path)
    assert parent.worktree_path is not None
    store.path("cccccccccccc").write_text(
        json.dumps(
            {
                "id": "cccccccccccc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "request": "unscanned malformed owner",
                "status": JobStatus.RUNNING.value,
                "created_at": 1,
                "delivered": False,
                "repository": parent.repository,
                "worktree_branch": parent.worktree_branch,
                "worktree_path": parent.worktree_path,
                "worktree_provision_state": "ready",
            }
        )
    )

    with (
        pytest.warns(UserWarning),
        pytest.raises(FollowUpCheckoutBusy),
    ):
        store.create_follow_up(parent.id, _build_child)
    assert not store.path("bbbbbbbbbbbb").exists()


def test_terminal_quarantine_release_fence_blocks_follow_up(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path)
    store.path("cccccccccccc").write_text(
        json.dumps(
            {
                "id": "cccccccccccc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "parent_job_id": "invalid",
                "request": "malformed fenced terminal",
                "status": JobStatus.FAILED.value,
                "created_at": 1,
                "completed_at": 2,
                "delivered": True,
                "error": "failed",
                "result": "failed",
                "repository": parent.repository,
                "worktree_branch": parent.worktree_branch,
                "worktree_path": parent.worktree_path,
                "worktree_provision_state": "ready",
                "herdr_target": "uncertain-agent",
                "target_release_pending": True,
            }
        )
    )
    with pytest.warns(UserWarning):
        store.list()

    with pytest.raises(FollowUpCheckoutBusy):
        store.create_follow_up(parent.id, _build_child)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "/different/repository"),
        ("worktree_branch", "voice/different"),
        ("worktree_path", "/different/worktree"),
    ],
)
def test_create_follow_up_rejects_mismatched_inherited_checkout(
    store: JobStore,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    parent = _completed_parent(store, tmp_path)

    def mismatched(source: CursorJob) -> CursorJob:
        values = _build_child(source).to_dict()
        values[field] = value
        return CursorJob.from_dict(values)

    with pytest.raises(JobValidationError, match=f"inherit parent {field} exactly"):
        store.create_follow_up(parent.id, mismatched)


@pytest.mark.parametrize("parent_job_id", ["bad", "AAAAAAAAAAAA", "bbbbbbbbbbbb"])
def test_store_rejects_malformed_or_self_referential_lineage(
    parent_job_id: str,
) -> None:
    with pytest.raises(JobValidationError, match="parent_job_id"):
        CursorJob.from_dict(
            {
                "id": "bbbbbbbbbbbb",
                "parent_job_id": parent_job_id,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "request": "follow up",
                "status": JobStatus.QUEUED.value,
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
            }
        )


def test_start_follow_up_creates_and_launches_child(
    store: JobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _completed_parent(store, tmp_path)
    launched: list[str] = []
    events: list[str] = []

    def launch(job_id: str) -> None:
        events.append("launched")
        launched.append(job_id)

    monkeypatch.setattr(service, "launch_worker", launch)

    child_id = service.start_follow_up(
        parent.id,
        "review the changes",
        expected_completed_at=parent.completed_at,
        on_created=lambda: events.append("created"),
    )

    assert events == ["created", "launched"]
    assert launched == [child_id]
    child = store.get(child_id)
    assert child.parent_job_id == parent.id
    assert child.worktree_path == parent.worktree_path


def test_start_follow_up_checks_capability_before_creating_child(
    store: JobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _completed_parent(store, tmp_path, issue_key="ENG-123")
    launch = mock.Mock()
    monkeypatch.setattr(service, "launch_worker", launch)
    monkeypatch.setattr(
        service,
        "resolve_issue_reference",
        lambda _reference, *_args, **_kwargs: "ENG-123",
    )
    monkeypatch.setattr(service, "require_issue_provider", lambda *_args: None)

    def reject(_reference: str, *_args: object, **_kwargs: object) -> None:
        raise HarnessError("Linear MCP requires authentication")

    monkeypatch.setattr(service, "require_issue_capabilities", reject)

    with pytest.raises(HarnessError, match="requires authentication"):
        service.start_follow_up(parent.id, "review the changes")

    assert [job.id for job in store.list()] == [parent.id]
    launch.assert_not_called()


def test_cursor_turn_follow_up_reports_busy(store: JobStore, tmp_path: Path) -> None:
    parent = _completed_parent(store, tmp_path)
    assert parent.worktree_path is not None
    _active_worktree_holder(store, parent.worktree_path)

    result = cursor_turn(
        CursorTurnRequest(
            "review the changes",
            action="follow_up",
            job_id=parent.id,
            expected_completed_at=parent.completed_at,
        )
    )

    assert result.session_id is None
    assert "busy" in as_assistant_response(result.text).display_text.lower()


def test_cursor_turn_follow_up_reports_unavailable_for_stale_source(
    store: JobStore, tmp_path: Path
) -> None:
    parent = _completed_parent(store, tmp_path)

    result = cursor_turn(
        CursorTurnRequest(
            "review the changes",
            action="follow_up",
            job_id=parent.id,
            expected_completed_at=(parent.completed_at or 0) + 99,
        )
    )

    assert result.session_id is None
    assert "follow up" in as_assistant_response(result.text).display_text.lower()


def test_cursor_turn_follow_up_reports_unavailable_for_missing_parent(
    store: JobStore,
) -> None:
    result = cursor_turn(
        CursorTurnRequest(
            "review the changes",
            action="follow_up",
            job_id="aaaaaaaaaaaa",
            expected_completed_at=1,
        )
    )

    assert result.session_id is None
    assert (
        "no longer follow up" in as_assistant_response(result.text).display_text.lower()
    )


def test_cursor_turn_follow_up_reports_unavailable_for_quarantined_parent(
    store: JobStore,
) -> None:
    store.durable_dir.mkdir(parents=True)
    store.path("aaaaaaaaaaaa").write_text("{invalid")

    with pytest.warns(UserWarning):
        result = cursor_turn(
            CursorTurnRequest(
                "review the changes",
                action="follow_up",
                job_id="aaaaaaaaaaaa",
            )
        )

    assert result.session_id is None
    assert (
        "no longer follow up" in as_assistant_response(result.text).display_text.lower()
    )


def test_cursor_turn_follow_up_without_source_is_graceful(store: JobStore) -> None:
    result = cursor_turn(CursorTurnRequest("review the changes", action="follow_up"))
    assert result.session_id is None
    assert "recent completed" in as_assistant_response(result.text).display_text.lower()


class _FakeHerdr:
    def __init__(
        self,
        worktrees: list[dict[str, object]],
        *,
        agent: AgentSelection | None = None,
    ) -> None:
        self._worktrees = worktrees
        self._agent = agent
        self.started: list[AgentSelection] = []
        self.started_modes: list[str | None] = []
        self.new_pane_calls = 0

    def allowed_repository(self, path: Path) -> bool:
        return True

    def run_json(self, *args: str, **kwargs: object) -> dict[str, object]:
        return {"worktrees": self._worktrees}

    def find_agent(
        self, *, checkout: Path | None = None, reserved: set[str] | None = None
    ) -> AgentSelection | None:
        return self._agent

    def workspace_for(self, checkout: Path) -> dict[str, object] | None:
        return None

    def new_pane(
        self,
        checkout: Path,
        label: str,
        workspace_id: str | None,
        *,
        checkpoint: object | None = None,
    ) -> tuple[str, str]:
        self.new_pane_calls += 1
        return "pane-1", "ws-1"

    def start_agent(
        self,
        checkout: Path,
        label: str,
        pane: str,
        workspace: str,
        *,
        name: str | None = None,
        mode: str | None = None,
        checkpoint: object | None = None,
    ) -> AgentSelection:
        target = name or "voice-agent"
        selection = AgentSelection(
            target=target,
            pane_id=pane,
            workspace_id=workspace,
            cwd=str(checkout),
            name=target,
            worktree_path=str(checkout),
        )
        self.started.append(selection)
        self.started_modes.append(mode)
        return selection


def test_find_agent_rejects_multiple_exact_checkout_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "worktree"
    client = HerdrClient("herdr")
    monkeypatch.setattr(
        client,
        "live_agents",
        lambda: [
            {
                "agent": "cursor",
                "agent_status": "idle",
                "name": "z-agent",
                "pane_id": "p2",
                "workspace_id": "w",
                "cwd": str(checkout),
            },
            {
                "agent": "cursor",
                "agent_status": "idle",
                "name": "a-agent",
                "pane_id": "p1",
                "workspace_id": "w",
                "cwd": str(checkout),
            },
        ],
    )

    with pytest.raises(HerdrError, match="a-agent, z-agent") as raised:
        client.find_agent(checkout=checkout)

    assert raised.value.code == "agent_ambiguous"


def _isolated_checkout(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    checkout = tmp_path / "worktrees" / "wt"
    (checkout / ".git").mkdir(parents=True)
    repository.mkdir(parents=True, exist_ok=True)
    return repository, checkout


def _routing_child(
    store: JobStore,
    repository: Path,
    checkout: Path,
    *,
    retained_pane: bool = True,
) -> CursorJob:
    now = time.time()
    values: dict[str, object] = {
        "id": "bbbbbbbbbbbb",
        "parent_job_id": "aaaaaaaaaaaa",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "revision": 0,
        "request": "review the changes",
        "status": JobStatus.ROUTING.value,
        "created_at": now,
        "delivered": False,
        "repository": str(repository),
        "worktree_branch": "voice/feature",
        "worktree_path": str(checkout),
        "worktree_provision_state": "ready",
        "worker_token": "tok",
        "worker_pid": 42,
        "worker_boot_id": "boot",
        "worker_process_start": "start",
    }
    if retained_pane:
        values.update(
            worktree_workspace_id="ws-1",
            worktree_root_pane_id="pane-1",
        )
    return store.create(CursorJob.from_dict(values))


def test_validate_followup_checkout_rejects_missing_worktree(tmp_path: Path) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    client = _FakeHerdr(worktrees=[])
    job = CursorJob.from_dict(
        {
            "id": "bbbbbbbbbbbb",
            "parent_job_id": "aaaaaaaaaaaa",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "revision": 0,
            "request": "x",
            "status": JobStatus.QUEUED.value,
            "created_at": 1,
            "queued_at": 1,
            "delivered": False,
            "repository": str(repository),
            "worktree_branch": "voice/feature",
            "worktree_path": str(checkout),
        }
    )
    with pytest.raises(HarnessError):
        provisioning._validate_followup_checkout(cast(HerdrClient, client), job)


def test_provision_followup_agent_starts_fresh_plan_agent(
    store: JobStore, tmp_path: Path
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    existing = AgentSelection(
        target="reused-agent",
        pane_id="p",
        workspace_id="w",
        cwd=str(checkout),
        name="reused-agent",
        worktree_path=str(checkout),
    )
    client = _FakeHerdr(
        worktrees=[{"branch": "voice/feature", "path": str(checkout)}],
        agent=existing,
    )
    job = _routing_child(store, repository, checkout)

    updated, target = provisioning._provision_followup_agent(
        store,
        job.id,
        "tok",
        job,
        cast(HerdrClient, client),
        set(),
        lambda: None,
    )

    assert target.startswith("voice-")
    assert updated.herdr_target == target
    assert len(client.started) == 1
    assert client.started_modes == ["plan"]
    assert client.new_pane_calls == 1


def test_provision_followup_agent_starts_agent_when_absent(
    store: JobStore, tmp_path: Path
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    client = _FakeHerdr(
        worktrees=[{"branch": "voice/feature", "path": str(checkout)}],
        agent=None,
    )
    job = _routing_child(store, repository, checkout)

    updated, target = provisioning._provision_followup_agent(
        store,
        job.id,
        "tok",
        job,
        cast(HerdrClient, client),
        set(),
        lambda: None,
    )

    assert target.startswith("voice-")
    assert updated.herdr_target == target
    assert len(client.started) == 1
    assert client.started_modes == ["plan"]
    assert client.new_pane_calls == 1


def test_provision_followup_agent_fails_closed_without_retained_pane(
    store: JobStore, tmp_path: Path
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    client = _FakeHerdr(
        worktrees=[{"branch": "voice/feature", "path": str(checkout)}],
        agent=None,
    )
    job = _routing_child(store, repository, checkout, retained_pane=False)

    with pytest.raises(HarnessError, match="no retained Herdr pane"):
        provisioning._provision_followup_agent(
            store,
            job.id,
            "tok",
            job,
            cast(HerdrClient, client),
            set(),
            lambda: None,
        )

    assert client.new_pane_calls == 0
    assert client.started == []
    assert store.get(job.id).herdr_target is None


def test_followup_agent_start_is_fenced_before_crash_and_recovered(
    store: JobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    client = _FakeHerdr(
        worktrees=[{"branch": "voice/feature", "path": str(checkout)}],
        agent=None,
    )
    job = _routing_child(store, repository, checkout)
    original_start = client.start_agent

    def start_with_fence(*args: object, **kwargs: object) -> AgentSelection:
        fenced = store.get(job.id)
        assert fenced.agent_dispatch_state == "dispatching"
        assert fenced.herdr_target == kwargs["name"]
        assert fenced.herdr_pane_id == "pane-1"
        return original_start(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(client, "start_agent", start_with_fence)
    original_settle = provisioning._settle_worker_agent
    monkeypatch.setattr(
        provisioning,
        "_settle_worker_agent",
        mock.Mock(side_effect=RuntimeError("worker crashed after agent start")),
    )

    with pytest.raises(RuntimeError, match="worker crashed"):
        provisioning._provision_followup_agent(
            store,
            job.id,
            "tok",
            job,
            cast(HerdrClient, client),
            set(),
            lambda: None,
        )

    dispatching = store.get(job.id)
    assert dispatching.agent_dispatch_state == "dispatching"
    assert dispatching.herdr_target
    started = client.started[0]
    monkeypatch.setattr(provisioning, "_settle_worker_agent", original_settle)
    recovery_client = mock.Mock()
    recovery_client.ensure_server.return_value = None
    recovery_client.get_agent.return_value = {
        "name": started.target,
        "pane_id": started.pane_id,
        "workspace_id": started.workspace_id,
        "cwd": str(checkout),
    }

    recovery.reconcile_uncertain_agent(
        store,
        dispatching,
        now=time.time() + 1,
        herdr_factory=lambda: cast(HerdrClient, recovery_client),
    )

    recovered = store.get(job.id)
    assert recovered.agent_dispatch_state == "ready"
    assert recovered.herdr_target == started.target


def test_followup_agent_timeout_keeps_reserved_identity_for_recovery(
    store: JobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    client = _FakeHerdr(
        worktrees=[{"branch": "voice/feature", "path": str(checkout)}],
        agent=None,
    )
    job = _routing_child(store, repository, checkout)
    materialized: dict[str, str] = {}

    def timeout(*_args: object, **kwargs: object) -> AgentSelection:
        target = str(kwargs["name"])
        fenced = store.get(job.id)
        assert fenced.herdr_target == target
        assert fenced.agent_dispatch_state == "dispatching"
        materialized["target"] = target
        raise HerdrError("agent start timed out", code="operation_timeout")

    monkeypatch.setattr(client, "start_agent", timeout)
    with pytest.raises(HerdrError, match="timed out"):
        provisioning._provision_followup_agent(
            store,
            job.id,
            "tok",
            job,
            cast(HerdrClient, client),
            set(),
            lambda: None,
        )

    ambiguous = store.get(job.id)
    assert ambiguous.agent_dispatch_state == "ambiguous"
    assert ambiguous.herdr_target == materialized["target"]
    recovery_client = mock.Mock()
    recovery_client.ensure_server.return_value = None
    recovery_client.get_agent.return_value = {
        "name": materialized["target"],
        "pane_id": "pane-1",
        "workspace_id": "ws-1",
        "cwd": str(checkout),
    }
    recovery.reconcile_uncertain_agent(
        store,
        ambiguous,
        now=time.time() + 1,
        herdr_factory=lambda: cast(HerdrClient, recovery_client),
    )

    assert store.get(job.id).agent_dispatch_state == "ready"


def test_uncertain_followup_agent_wrong_cwd_is_failed_and_cancelled(
    store: JobStore, tmp_path: Path
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    job = _routing_child(store, repository, checkout)
    dispatching = store.update(
        job.id,
        lambda current: current.evolve(
            herdr_target="wrong-agent",
            herdr_pane_id="pane-1",
            herdr_workspace_id="ws-1",
            agent_name="wrong-agent",
            agent_dispatch_state="dispatching",
            worker_operation="agent_start",
        ),
    )
    assert dispatching is not None
    client = mock.Mock()
    client.ensure_server.return_value = None
    client.get_agent.return_value = {
        "name": "wrong-agent",
        "pane_id": "pane-1",
        "workspace_id": "ws-1",
        "cwd": str(tmp_path / "different-checkout"),
    }

    recovery.reconcile_uncertain_agent(
        store,
        dispatching,
        now=time.time() + 1,
        herdr_factory=lambda: cast(HerdrClient, client),
    )

    staged = store.get(job.id)
    assert staged.status == JobStatus.RECONCILING
    assert staged.terminal_intent_status == JobStatus.FAILED
    assert staged.agent_dispatch_state == "ready"
    assert staged.target_release_pending
    assert staged.cancellation_reconciliation_pending
    assert staged.worktree_path == str(checkout)

    recovery.recover_jobs(
        store,
        launch_worker=lambda _job_id: None,
        herdr_factory=lambda: cast(HerdrClient, client),
        is_worker_alive=lambda _job: False,
        now=time.time() + 2,
    )

    client.close_owned_pane.assert_not_called()
    released = store.get(job.id)
    assert released.target_release_pending
    assert released.cancellation_reconciliation_pending


def _routing_child_with_ready_target(
    store: JobStore, repository: Path, checkout: Path
) -> CursorJob:
    job = _routing_child(store, repository, checkout)
    updated = store.update(
        job.id,
        lambda current: current.evolve(
            herdr_target="retained-agent",
            herdr_pane_id="pane",
            herdr_workspace_id="workspace",
            agent_name="retained-agent",
            agent_dispatch_state="ready",
            reconcile=True,
        ),
    )
    assert updated is not None
    return updated


def test_recovered_followup_revalidates_stale_checkout_before_dispatch(
    store: JobStore, tmp_path: Path
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    job = _routing_child_with_ready_target(store, repository, checkout)
    client = mock.Mock()
    client.ensure_server.return_value = None
    client.allowed_repository.return_value = True
    client.run_json.return_value = {"worktrees": []}
    context = worker_lifecycle.WorkerContext(store, job, "tok", threading.Event())

    provisioning.run_claimed_worker(
        context,
        provisioning.ClientFactories(
            herdr=lambda: cast(HerdrClient, client),
            github=mock.Mock(),
        ),
    )

    failed = store.get(job.id)
    assert failed.status == JobStatus.FAILED
    assert "no longer matches Herdr" in str(failed.error)
    client.get_agent.assert_not_called()
    client.prompt_and_wait.assert_not_called()


def test_recovered_followup_rejects_agent_with_mismatched_cwd(
    store: JobStore, tmp_path: Path
) -> None:
    repository, checkout = _isolated_checkout(tmp_path)
    job = _routing_child_with_ready_target(store, repository, checkout)
    client = mock.Mock()
    client.ensure_server.return_value = None
    client.allowed_repository.return_value = True
    client.run_json.return_value = {
        "worktrees": [{"branch": "voice/feature", "path": str(checkout)}]
    }
    client.get_agent.return_value = {
        "name": "retained-agent",
        "pane_id": "pane",
        "workspace_id": "workspace",
        "cwd": str(tmp_path / "different-worktree"),
    }
    context = worker_lifecycle.WorkerContext(store, job, "tok", threading.Event())

    provisioning.run_claimed_worker(
        context,
        provisioning.ClientFactories(
            herdr=lambda: cast(HerdrClient, client),
            github=mock.Mock(),
        ),
    )

    failed = store.get(job.id)
    assert failed.status == JobStatus.FAILED
    assert "different checkout" in str(failed.error)
    client.prompt_and_wait.assert_not_called()
