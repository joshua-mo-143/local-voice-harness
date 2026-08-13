from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.agents import HarnessCapability, HarnessSession
from local_voice_harness.integrations.herdr import (
    AgentSelection,
    HerdrClient,
    HerdrError,
    HerdrRepository,
    HerdrTransport,
    HerdrWorkspace,
)


class HerdrComponentBoundaryTests(unittest.TestCase):
    def test_client_exposes_split_components(self) -> None:
        client = HerdrClient("herdr")
        self.assertIsInstance(client.transport, HerdrTransport)
        self.assertIsInstance(client.workspace, HerdrWorkspace)
        self.assertIsInstance(client.repository, HerdrRepository)

    def test_transport_handles_server_and_agent_session_ops(self) -> None:
        client = HerdrClient("herdr")
        with mock.patch.object(
            client,
            "run_json",
            side_effect=[{}, {"agent": {"agent_status": "idle"}}],
        ) as run_json:
            client.cancel_agent("agent")

        self.assertEqual(
            run_json.call_args_list,
            [
                mock.call("agent", "send-keys", "agent", "ctrl+c"),
                mock.call("agent", "wait", "agent", "--timeout", "5000", timeout=10),
            ],
        )

    def test_owned_pane_close_verifies_binding_and_post_close_absence(self) -> None:
        client = HerdrClient("herdr")
        agent = {
            "name": "planner",
            "pane_id": "workspace:participant",
            "workspace_id": "workspace",
        }
        pane = {
            "pane_id": "workspace:participant",
            "workspace_id": "workspace",
        }
        with (
            mock.patch.object(
                client,
                "get_agent",
                side_effect=[
                    agent,
                    HerdrError("absent", code="agent_not_found"),
                ],
            ),
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"pane": pane},
                    {},
                    HerdrError("absent", code="pane_not_found"),
                ],
            ) as run_json,
            mock.patch.object(client.session, "cancel_agent") as cancel,
        ):
            client.close_owned_pane(
                "planner",
                "workspace:participant",
                "workspace",
            )

        cancel.assert_called_once_with("planner")
        self.assertEqual(
            run_json.call_args_list,
            [
                mock.call("pane", "get", "workspace:participant"),
                mock.call("pane", "close", "workspace:participant"),
                mock.call("pane", "get", "workspace:participant"),
            ],
        )

    def test_owned_pane_close_is_idempotent_when_agent_and_pane_are_absent(
        self,
    ) -> None:
        client = HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                side_effect=HerdrError("absent", code="agent_not_found"),
            ),
            mock.patch.object(
                client,
                "run_json",
                side_effect=HerdrError("absent", code="pane_not_found"),
            ) as run_json,
            mock.patch.object(client.session, "cancel_agent") as cancel,
        ):
            client.close_owned_pane("planner", "workspace:pane", "workspace")

        run_json.assert_called_once_with("pane", "get", "workspace:pane")
        cancel.assert_not_called()

    def test_owned_pane_close_fails_closed_on_binding_mismatch(self) -> None:
        client = HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "get_agent",
                return_value={
                    "name": "planner",
                    "pane_id": "workspace:other",
                    "workspace_id": "workspace",
                },
            ),
            mock.patch.object(client, "run_json") as run_json,
            self.assertRaises(HerdrError) as raised,
        ):
            client.close_owned_pane("planner", "workspace:pane", "workspace")

        self.assertEqual(raised.exception.code, "ownership_mismatch")
        run_json.assert_not_called()

    def test_owned_pane_close_retains_failure_when_close_needs_confirmation(
        self,
    ) -> None:
        client = HerdrClient("herdr")
        pane = {"pane_id": "workspace:pane", "workspace_id": "workspace"}
        with (
            mock.patch.object(
                client,
                "get_agent",
                side_effect=HerdrError("absent", code="agent_not_found"),
            ),
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"pane": pane},
                    HerdrError("confirmation required", code="confirmation_required"),
                ],
            ),
            self.assertRaises(HerdrError) as raised,
        ):
            client.close_owned_pane("planner", "workspace:pane", "workspace")

        self.assertEqual(raised.exception.code, "confirmation_required")

    def test_repository_resolution_is_independent_of_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = root / "alpha-app"
            alpha.mkdir()
            operations = mock.Mock()
            repository = HerdrRepository(operations)
            resolved, matches = repository.resolve_repository(
                "alpha-app", "task", [alpha]
            )
        self.assertEqual(resolved, alpha)
        self.assertEqual(matches, [alpha])
        operations.list_workspaces.assert_not_called()

    def test_workspace_reuses_existing_agent_without_transport_pane_creation(
        self,
    ) -> None:
        operations = mock.Mock()
        agent = {
            "agent": "cursor",
            "agent_status": "idle",
            "cwd": "/checkout",
            "pane_id": "w1:p1",
            "workspace_id": "w1",
        }
        operations.list_agents.return_value = [agent]
        operations.live_agents.return_value = [agent]
        operations.find_agent.return_value = HerdrWorkspace(operations).selection(
            agent, "/checkout"
        )
        operations.run_json.return_value = {
            "worktrees": [
                {
                    "branch": "voice/api-98",
                    "path": "/checkout",
                    "open_workspace_id": "w1",
                }
            ]
        }
        workspace = HerdrWorkspace(operations)
        selection = workspace.ensure_agent(
            Path("/repo"),
            issue_key="API-98",
            agent_hint=None,
            reserved=set(),
        )
        self.assertEqual(selection.target, "w1:p1")
        self.assertFalse(
            any(
                call.args[:2] in {("tab", "create"), ("workspace", "create")}
                for call in operations.run_json.call_args_list
            )
        )

    def test_facade_delegates_repository_clone_to_repository_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = HerdrClient("herdr", repository_root=root)

            def clone(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                destination = Path(command[-1])
                destination.mkdir()
                (destination / ".git").mkdir()
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(
                    client.repository,
                    "choose_or_clone_repository",
                    wraps=client.repository.choose_or_clone_repository,
                ) as choose,
                mock.patch(
                    "local_voice_harness.integrations.herdr.repository.choose_repository",
                    return_value="https://github.com/example/project.git",
                ),
                mock.patch(
                    "local_voice_harness.integrations.herdr.repository.confirm_clone",
                    return_value=True,
                ),
                mock.patch(
                    "local_voice_harness.local_git.run_command",
                    side_effect=clone,
                ),
            ):
                repository, reason = client.choose_or_clone_repository([])

            choose.assert_called_once()
            self.assertEqual(repository, root / "project")
            self.assertEqual(reason, "")

    def test_workspace_reservation_callbacks_fire_before_agent_start(self) -> None:
        operations = mock.Mock()
        checkout = Path("/tmp/worktree")
        operations.run_json.side_effect = [
            {"worktrees": []},
            {
                "worktree": {"path": str(checkout)},
                "workspace": {"workspace_id": "workspace"},
                "root_pane": {"pane_id": "pane"},
            },
        ]
        operations.list_agents.return_value = []
        operations.live_agents.return_value = []
        operations.find_agent.return_value = None
        operations.workspace_for.return_value = None
        operations.planned_worktree_path.return_value = checkout
        operations.new_pane.return_value = ("participant-pane", "workspace")
        workspace = HerdrWorkspace(operations)
        events: list[str] = []
        selection = AgentSelection(
            "planned-agent",
            "participant-pane",
            "workspace",
            str(checkout),
            "planned-agent",
            str(checkout),
        )

        def reserve_worktree(
            _repository: Path, _branch: str, _checkout: Path, state: str
        ) -> None:
            events.append(f"worktree:{state}")

        def settle_worktree(
            _checkout: Path, _workspace: str | None, _pane: str | None
        ) -> None:
            events.append("worktree:settled")

        def reserve_agent(_selection: AgentSelection, dispatching: bool) -> None:
            events.append(f"agent:reserved:{dispatching}")

        with mock.patch.object(
            operations,
            "start_agent",
            side_effect=lambda *_args, **_kwargs: (
                events.append("agent:started"),
                selection,
            )[1],
        ) as start_agent:
            workspace.ensure_agent(
                Path("/tmp/repository"),
                issue_key=None,
                agent_hint=None,
                reserved=set(),
                worktree_branch="voice/task",
                reserve=reserve_agent,
                settle=lambda _selection: events.append("agent:settled"),
                reserve_worktree=reserve_worktree,
                settle_worktree=settle_worktree,
            )

        self.assertEqual(
            events,
            [
                "worktree:planned",
                "worktree:dispatching",
                "worktree:settled",
                "agent:reserved:True",
                "agent:started",
                "agent:settled",
            ],
        )
        start_agent.assert_called_once_with(
            checkout,
            "repository",
            "participant-pane",
            "workspace",
            name=mock.ANY,
            mode=None,
            checkpoint=None,
        )

    def test_router_workspace_is_not_a_transport_concern(self) -> None:
        root = Path("/repositories")
        selection = AgentSelection(
            "voice-router",
            "pane",
            "workspace",
            str(root),
            "voice-router",
            None,
        )
        operations = mock.Mock()
        operations.find_agent.return_value = None
        operations.list_workspaces.return_value = []
        operations.new_pane.return_value = ("pane", "workspace")
        operations.start_agent.return_value = selection
        workspace = HerdrWorkspace(operations, repository_root=root)
        self.assertIs(
            workspace.ensure_router(set()),
            selection,
        )

        operations.start_agent.assert_called_once_with(
            root,
            "router",
            "pane",
            "workspace",
            name="voice-router",
            checkpoint=None,
        )
        self.assertFalse(
            hasattr(operations, "prompt_and_wait") and operations.prompt_and_wait.called
        )

    def test_every_cursor_launch_mode_links_mcp_auth_at_shared_boundary(self) -> None:
        checkout = Path("/tmp/worktree")
        for mode in (None, "ask", "plan"):
            with self.subTest(mode=mode):
                operations = mock.Mock()
                operations.create_session.return_value = HarnessSession(
                    provider="cursor/herdr",
                    session_id="session",
                    target="participant",
                    state_sequence=0,
                    metadata={
                        "name": "participant",
                        "pane_id": "pane",
                        "workspace_id": "workspace",
                        "cwd": str(checkout),
                    },
                )
                workspace = HerdrWorkspace(
                    operations,
                    cursor_mcp_auth_source=Path("/authenticated"),
                )
                assert workspace._cursor_mcp_auth is not None
                with mock.patch.object(workspace._cursor_mcp_auth, "link") as link:
                    workspace.start_agent(
                        checkout,
                        "participant",
                        "pane",
                        "workspace",
                        name="participant",
                        mode=mode,
                    )

                link.assert_called_once_with(checkout)
                request = operations.create_session.call_args.args[0]
                self.assertEqual(request.mode, mode)
                self.assertEqual(
                    request.required_capabilities,
                    frozenset({HarnessCapability.MCP_CONNECTORS}),
                )


if __name__ == "__main__":
    unittest.main()
