from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from local_voice_harness.agents import (
    HarnessCapability,
    HarnessSession,
    SessionRequest,
    UnsupportedCapabilityError,
)
from local_voice_harness.cursor.model import HarnessKind
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
