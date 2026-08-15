from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from local_voice_harness.agents import (
    HarnessCapability,
    HarnessSession,
    ReconciliationState,
    SessionReconciliation,
    SessionRequest,
    UnsupportedCapabilityError,
)
from local_voice_harness.cursor.model import (
    CURRENT_SCHEMA_VERSION,
    CursorJob,
    HarnessKind,
)
from local_voice_harness.cursor.recovery import reconcile_uncertain_agent
from local_voice_harness.cursor.service import StartJobRequest, start_job
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.integrations import herdr
from local_voice_harness.integrations.herdr import OpenCodeSession
from local_voice_harness.integrations.linear import CapabilityStatus, LinearIntegration
from local_voice_harness.integrations.registry import (
    build_integration_registry,
    require_harness_capabilities,
)
from local_voice_harness.user_config import IntegrationSettings, default_user_config


def test_start_job_stamps_opencode_without_changing_cursor_default(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    registry = build_integration_registry(default_user_config(tmp_path))
    with (
        mock.patch("local_voice_harness.cursor.service._job_store", return_value=store),
        mock.patch("local_voice_harness.cursor.service.launch_worker"),
        mock.patch.object(OpenCodeSession, "require_ready"),
    ):
        default_id = start_job(
            "fix a local bug", foreground=False, integrations=registry
        )
        opencode_id = start_job(
            "fix a local bug with OpenCode",
            harness_kind=HarnessKind.OPENCODE,
            foreground=False,
            integrations=registry,
        )
        request_id = start_job(
            StartJobRequest("another OpenCode task", harness_kind=HarnessKind.OPENCODE),
            foreground=False,
            integrations=registry,
        )

    default_job = store.get(default_id)
    opencode_job = store.get(opencode_id)
    request_job = store.get(request_id)
    assert default_job.harness_kind == HarnessKind.CURSOR
    assert opencode_job.harness_kind == HarnessKind.OPENCODE
    assert request_job.harness_kind == HarnessKind.OPENCODE
    assert default_job.to_dict()["harness_kind"] == "cursor"
    assert opencode_job.to_dict()["harness_kind"] == "opencode"


def test_opencode_linear_request_fails_before_persistence(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    config = replace(
        default_user_config(tmp_path),
        integrations=IntegrationSettings(linear_enabled=True),
    )
    registry = build_integration_registry(config)
    with (
        mock.patch.object(
            LinearIntegration,
            "capability_status",
            return_value=CapabilityStatus(True, "ready"),
        ),
        mock.patch("local_voice_harness.cursor.service._job_store", return_value=store),
        mock.patch("local_voice_harness.cursor.service.launch_worker") as launch,
        pytest.raises(UnsupportedCapabilityError, match="mcp_connectors") as raised,
    ):
        start_job(
            "work on API-42",
            harness_kind=HarnessKind.OPENCODE,
            foreground=False,
            integrations=registry,
        )

    assert raised.value.provider == "opencode/herdr"
    assert raised.value.capability == HarnessCapability.MCP_CONNECTORS
    assert store.list() == []
    launch.assert_not_called()


def test_opencode_linear_ticket_create_fails_before_persistence(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    config = replace(
        default_user_config(tmp_path),
        integrations=IntegrationSettings(linear_enabled=True),
    )
    registry = build_integration_registry(config)
    with (
        mock.patch.object(
            LinearIntegration,
            "capability_status",
            return_value=CapabilityStatus(True, "ready"),
        ),
        mock.patch("local_voice_harness.cursor.service._job_store", return_value=store),
        mock.patch("local_voice_harness.cursor.service.launch_worker") as launch,
        pytest.raises(UnsupportedCapabilityError, match="choose a compatible harness"),
    ):
        start_job(
            "create a Linear ticket",
            linear_team="API",
            linear_ticket_create_requested=True,
            harness_kind=HarnessKind.OPENCODE,
            foreground=False,
            integrations=registry,
        )

    assert store.list() == []
    launch.assert_not_called()


def test_require_harness_capabilities_is_a_no_op_for_cursor() -> None:
    require_harness_capabilities(
        HarnessKind.CURSOR,
        provider="linear",
        linear_ticket_create_requested=True,
    )


def test_start_agent_starts_opencode_without_cursor_flags(tmp_path: Path) -> None:
    client = herdr.HerdrClient("herdr")
    client.bind_harness_kind("opencode")
    agent = {
        "agent": "opencode",
        "name": "planner",
        "pane_id": "pane",
        "workspace_id": "workspace",
        "cwd": str(tmp_path),
        "agent_session": "opencode-session",
        "interactive_ready": True,
        "state_change_seq": 1,
    }
    with mock.patch.object(client, "run_json", return_value={"agent": agent}) as run:
        selection = client.start_agent(
            tmp_path,
            "planning",
            "pane",
            "workspace",
            name="planner",
            mode="plan",
        )

    assert selection.provider == "opencode/herdr"
    assert selection.provider_session_id == "opencode-session"
    args = run.call_args.args
    assert args[:2] == ("agent", "start")
    assert args[args.index("--kind") + 1] == "opencode"
    assert "--trust" not in args
    assert "--approve-mcps" not in args
    assert "--mode" not in args
    assert args[args.index("--") + 1 :] == ("--agent", "plan")


def test_start_agent_maps_ask_and_build_onto_opencode_agents(tmp_path: Path) -> None:
    client = herdr.HerdrClient("herdr")
    client.bind_harness_kind("opencode")
    agent = {
        "agent": "opencode",
        "name": "reviewer",
        "pane_id": "pane",
        "workspace_id": "workspace",
        "cwd": str(tmp_path),
        "agent_session": "session",
        "interactive_ready": True,
    }
    with mock.patch.object(client, "run_json", return_value={"agent": agent}) as run:
        client.start_agent(
            tmp_path,
            "review",
            "pane",
            "workspace",
            name="reviewer",
            mode="ask",
        )
        ask_args = run.call_args.args
        client.start_agent(
            tmp_path,
            "implement",
            "pane",
            "workspace",
            name="implementer",
        )
        build_args = run.call_args.args

    assert ask_args[ask_args.index("--") + 1 :] == ("--agent", "plan")
    assert build_args[build_args.index("--") + 1 :] == ("--agent", "build")


def test_opencode_start_does_not_link_cursor_mcp(tmp_path: Path) -> None:
    operations = mock.Mock()
    operations.create_session.return_value = HarnessSession(
        provider="opencode/herdr",
        session_id="session",
        target="participant",
        state_sequence=0,
        metadata={
            "name": "participant",
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": str(tmp_path),
        },
    )
    workspace = herdr.HerdrWorkspace(
        operations,
        cursor_mcp_auth_source=Path("/authenticated"),
    )
    workspace.bind_kind("opencode")
    assert workspace._cursor_mcp_auth is not None
    with mock.patch.object(workspace._cursor_mcp_auth, "link") as link:
        workspace.start_agent(
            tmp_path,
            "participant",
            "pane",
            "workspace",
            name="participant",
            mode="plan",
        )

    link.assert_not_called()
    request = operations.create_session.call_args.args[0]
    assert request.provider == "opencode/herdr"
    assert request.required_capabilities == frozenset()


def test_opencode_live_agents_do_not_reuse_cursor_panes() -> None:
    operations = mock.Mock()
    operations.list_agents.return_value = [
        {
            "agent": "cursor",
            "name": "cursor-agent",
            "interactive_ready": True,
            "agent_status": "idle",
        },
        {
            "agent": "opencode",
            "name": "opencode-agent",
            "interactive_ready": True,
            "agent_status": "idle",
        },
    ]
    workspace = herdr.HerdrWorkspace(operations)
    workspace.bind_kind("opencode")
    live = workspace.live_agents()
    assert [agent["name"] for agent in live] == ["opencode-agent"]


def test_opencode_session_fails_closed_without_durable_identity() -> None:
    client = herdr.HerdrClient("herdr")
    harness = OpenCodeSession(client)
    starting = {
        "agent": "opencode",
        "name": "planner",
        "pane_id": "pane",
        "workspace_id": "workspace",
        "interactive_ready": True,
    }
    with (
        mock.patch.object(client, "run_json", return_value={"agent": starting}),
        mock.patch.object(client, "get_agent", return_value=starting),
        mock.patch(
            "local_voice_harness.integrations.herdr.session.time.monotonic",
            side_effect=[0, 0, 16],
        ),
        mock.patch("local_voice_harness.integrations.herdr.session.time.sleep"),
        pytest.raises(herdr.HerdrError, match="ready OpenCode session") as raised,
    ):
        harness.create_session(
            SessionRequest(
                name="planner",
                provider="opencode/herdr",
                launch_context={"pane_id": "pane", "workspace_id": "workspace"},
            )
        )

    assert raised.value.code == "operation_ambiguous"


def test_opencode_readiness_fails_actionably_when_binary_is_missing() -> None:
    client = herdr.HerdrClient("herdr")
    client.bind_harness_kind("opencode")

    with (
        mock.patch(
            "local_voice_harness.integrations.herdr.session.shutil.which",
            return_value=None,
        ),
        pytest.raises(herdr.HerdrError, match="on PATH") as raised,
    ):
        client.require_harness_ready()

    assert raised.value.code == "operation_spawn_failed"


def test_opencode_readiness_requires_an_authenticated_provider() -> None:
    client = herdr.HerdrClient("herdr")
    client.bind_harness_kind("opencode")
    process = mock.Mock(returncode=0, stdout="0 credentials", stderr="")

    with (
        mock.patch(
            "local_voice_harness.integrations.herdr.session.shutil.which",
            return_value="/usr/bin/opencode",
        ),
        mock.patch(
            "local_voice_harness.integrations.herdr.session.subprocess.run",
            return_value=process,
        ),
        pytest.raises(herdr.HerdrError, match="opencode auth login") as raised,
    ):
        client.require_harness_ready()

    assert raised.value.code == "authentication_unavailable"


def test_missing_opencode_binary_prevents_job_persistence(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    registry = build_integration_registry(default_user_config(tmp_path))

    with (
        mock.patch("local_voice_harness.cursor.service._job_store", return_value=store),
        mock.patch("local_voice_harness.cursor.service.launch_worker") as launch,
        mock.patch(
            "local_voice_harness.integrations.herdr.session.shutil.which",
            return_value=None,
        ),
        pytest.raises(herdr.HerdrError, match="OpenCode is selected"),
    ):
        start_job(
            "fix a local bug",
            harness_kind=HarnessKind.OPENCODE,
            foreground=False,
            integrations=registry,
        )

    assert store.list() == []
    launch.assert_not_called()


def test_opencode_reconcile_rejects_cursor_agent_kind() -> None:
    client = herdr.HerdrClient("herdr")
    client.bind_harness_kind("opencode")
    observed = {
        "agent": "cursor",
        "name": "planner",
        "agent_session": "cursor-session",
        "agent_status": "working",
    }

    with (
        mock.patch.object(client, "run_json", return_value={"agent": observed}),
        pytest.raises(herdr.HerdrError, match="expected 'opencode'") as raised,
    ):
        client.reconcile_session("planner", expected_session_id="cursor-session")

    assert raised.value.code == "agent_kind_mismatch"


def test_opencode_cancellation_rejects_cursor_agent_kind() -> None:
    client = herdr.HerdrClient("herdr")
    client.bind_harness_kind("opencode")
    observed = {
        "agent": "cursor",
        "name": "planner",
        "agent_session": "cursor-session",
        "agent_status": "working",
    }

    with (
        mock.patch.object(client, "run_json", return_value={"agent": observed}) as run,
        pytest.raises(herdr.HerdrError, match="expected 'opencode'"),
    ):
        client.harness.cancel(
            HarnessSession("opencode/herdr", "cursor-session", "planner", 1)
        )

    assert run.call_count == 1


def test_restart_recovery_binds_persisted_kind_before_observation(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    job = store.create(
        CursorJob.from_dict(
            {
                "id": "123456789abc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "worktree_path": "/worktree",
                "herdr_target": "agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_dispatch_state": "ambiguous",
                "agent_operation_target": "agent",
                "agent_operation_checkout": "/worktree",
                "agent_operation_workspace_id": "workspace",
                "agent_operation_pane_id": "pane",
                "agent_provider": "opencode/herdr",
                "agent_provider_session_id": "opencode-session",
                "agent_state_sequence": 1,
                "harness_kind": "opencode",
            }
        )
    )
    client = mock.Mock()
    client.reconcile_session.return_value = SessionReconciliation(
        ReconciliationState.ACTIVE,
        HarnessSession(
            "opencode/herdr",
            "opencode-session",
            "agent",
            2,
            {
                "name": "agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
                "cwd": "/worktree",
            },
        ),
        "working",
        True,
    )

    reconcile_uncertain_agent(store, job, now=100, herdr_factory=lambda: client)

    assert client.method_calls[0] == mock.call.bind_harness_kind("opencode")
    client.reconcile_session.assert_called_once_with(
        "agent",
        expected_session_id="opencode-session",
    )
    assert store.get(job.id).agent_dispatch_state == "ready"


def test_restart_replays_outbox_through_persisted_opencode_harness(
    tmp_path: Path,
) -> None:
    from local_voice_harness.cursor.agent_outbox import SESSION_CREATE
    from local_voice_harness.cursor.coordinator import (
        CoordinatorCommand,
        CoordinatorDecision,
        DurableEffect,
    )
    from local_voice_harness.cursor.recovery import recover_jobs

    store = JobStore(tmp_path / "jobs", tmp_path / "legacy")
    created = store.create(
        CursorJob.from_dict(
            {
                "id": "123456789abc",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "revision": 0,
                "status": "queued",
                "request": "test",
                "created_at": 1,
                "queued_at": 1,
                "delivered": False,
                "harness_kind": "opencode",
            }
        )
    )
    key = f"{SESSION_CREATE}:{created.id}:1"
    admitted = store.apply(
        CoordinatorCommand(
            job_id=created.id,
            expected_revision=created.revision,
            command_id=f"admit:{key}",
            kind=f"{SESSION_CREATE}.admit",
        ),
        lambda job: CoordinatorDecision(
            job=job.evolve(),
            effects=(
                DurableEffect(
                    kind=SESSION_CREATE,
                    idempotency_key=key,
                    concurrency_key="opencode/herdr:voice",
                    payload={
                        "name": "voice",
                        "provider": "opencode/herdr",
                        "launch_context": {
                            "pane_id": "pane",
                            "workspace_id": "workspace",
                        },
                        "required_capabilities": [],
                    },
                ),
            ),
        ),
    )
    assert admitted is not None
    cursor_harness = mock.Mock()
    opencode_harness = mock.Mock()

    def create_session(request, *, checkpoint=None, before_submit=None):
        assert request.provider == "opencode/herdr"
        assert before_submit is not None
        before_submit()
        return HarnessSession(
            provider="opencode/herdr",
            session_id="opencode-session",
            target="voice",
            state_sequence=1,
        )

    opencode_harness.create_session.side_effect = create_session
    client = mock.Mock()
    client.harness = cursor_harness

    def bind(kind: str) -> None:
        client.harness = opencode_harness if kind == "opencode" else cursor_harness

    client.bind_harness_kind.side_effect = bind

    recover_jobs(
        store,
        launch_worker=mock.Mock(),
        herdr_factory=lambda: client,
        now=100,
    )

    client.bind_harness_kind.assert_called_with("opencode")
    opencode_harness.create_session.assert_called_once()
    cursor_harness.create_session.assert_not_called()
