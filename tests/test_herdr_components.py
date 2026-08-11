from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.integrations.herdr import (
    HOME_ROOT,
    AgentSelection,
    HerdrClient,
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
            client = HerdrClient("herdr")

            def clone(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                destination = Path(command[-1])
                destination.mkdir()
                (destination / ".git").mkdir()
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch(
                    "local_voice_harness.integrations.herdr.repository.HOME_ROOT",
                    root,
                ),
                mock.patch(
                    "local_voice_harness.integrations.herdr.types.HOME_ROOT",
                    root,
                ),
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
                    "local_voice_harness.integrations.herdr.repository.subprocess.run",
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
        operations.new_pane.return_value = ("pane", "workspace")
        workspace = HerdrWorkspace(operations)
        events: list[str] = []
        selection = AgentSelection(
            "planned-agent",
            "pane",
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
        ):
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

    def test_router_workspace_is_not_a_transport_concern(self) -> None:
        selection = AgentSelection(
            "voice-router",
            "pane",
            "workspace",
            str(HOME_ROOT),
            "voice-router",
            None,
        )
        operations = mock.Mock()
        operations.find_agent.return_value = None
        operations.list_workspaces.return_value = []
        operations.new_pane.return_value = ("pane", "workspace")
        operations.start_agent.return_value = selection
        workspace = HerdrWorkspace(operations)
        self.assertIs(
            workspace.ensure_router(set()),
            selection,
        )

        operations.start_agent.assert_called_once_with(
            HOME_ROOT,
            "router",
            "pane",
            "workspace",
            name="voice-router",
            checkpoint=None,
        )
        self.assertFalse(
            hasattr(operations, "prompt_and_wait") and operations.prompt_and_wait.called
        )


if __name__ == "__main__":
    unittest.main()
