from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.integrations import herdr


class HerdrIntegrationTests(unittest.TestCase):
    def test_normalizes_spoken_linear_issue_key(self) -> None:
        self.assertEqual(herdr.extract_linear_issue("work on API 77"), "API-77")
        self.assertEqual(herdr.extract_linear_issue("work on api - 78"), "API-78")
        self.assertEqual(herdr.extract_linear_issue("work on API-79"), "API-79")

    def test_extracts_latest_multiline_marker(self) -> None:
        output = """
VOICE_SUMMARY[token]: old
more old text
VOICE_QUESTION[token]: Which repository
should I use?
"""
        self.assertEqual(
            herdr.extract_marker(output, "VOICE_QUESTION", "token"),
            "Which repository should I use?",
        )

    def test_repository_resolution_requires_unique_valid_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = root / "alpha-app"
            alphabet = root / "alphabet"
            alpha.mkdir()
            alphabet.mkdir()
            client = herdr.HerdrClient("herdr")
            repository, matches = client.resolve_repository(
                "alpha", "task", [alpha, alphabet]
            )
            self.assertIsNone(repository)
            self.assertEqual(matches, [alpha, alphabet])
            repository, matches = client.resolve_repository(
                "alpha-app", "task", [alpha, alphabet]
            )
            self.assertEqual(repository, alpha)
            self.assertEqual(matches, [alpha])

    def test_live_agents_accepts_detected_idle_cursor_without_readiness(self) -> None:
        client = herdr.HerdrClient("herdr")
        agents = [
            {
                "agent": "cursor",
                "agent_status": "idle",
                "cwd": "/repo",
                "pane_id": "w1:p1",
            },
            {
                "agent": "cursor",
                "agent_status": "idle",
                "cwd": "/repo",
                "interactive_ready": False,
                "pane_id": "w1:p2",
            },
        ]
        with mock.patch.object(client, "list_agents", return_value=agents):
            live = client.live_agents()
        self.assertEqual([agent["pane_id"] for agent in live], ["w1:p1"])

    def test_existing_detected_agent_avoids_new_tab(self) -> None:
        client = herdr.HerdrClient("herdr")
        repository = Path("/repo")
        checkout = Path("/checkout")
        agent = {
            "agent": "cursor",
            "agent_status": "idle",
            "cwd": str(checkout),
            "pane_id": "w1:p1",
            "workspace_id": "w1",
        }
        listing = {
            "worktrees": [
                {
                    "branch": "voice/api-98",
                    "path": str(checkout),
                    "open_workspace_id": "w1",
                }
            ]
        }
        with (
            mock.patch.object(client, "run_json", return_value=listing),
            mock.patch.object(client, "list_agents", return_value=[agent]),
            mock.patch.object(client, "new_pane") as new_pane,
            mock.patch.object(client, "start_agent") as start_agent,
        ):
            selection = client.ensure_agent(
                repository,
                issue_key="API-98",
                agent_hint=None,
                reserved=set(),
            )
        self.assertEqual(selection.target, "w1:p1")
        new_pane.assert_not_called()
        start_agent.assert_not_called()

    def test_extracts_safe_repository_name_from_git_url(self) -> None:
        self.assertEqual(
            herdr.repository_name_from_url(
                "https://github.com/example/example-project.git"
            ),
            "example-project",
        )
        self.assertEqual(
            herdr.repository_name_from_url("git@github.com:example/project.git"),
            "project",
        )
        self.assertIsNone(herdr.repository_name_from_url("file:///tmp/project"))
        self.assertIsNone(
            herdr.repository_name_from_url("https://github.com/example/.hidden.git")
        )

    def test_confirmed_rofi_url_is_cloned_under_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = herdr.HerdrClient("herdr")

            def clone(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                destination = Path(command[-1])
                destination.mkdir()
                (destination / ".git").mkdir()
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(herdr, "HOME_ROOT", root),
                mock.patch.object(
                    herdr,
                    "choose_repository",
                    return_value="https://github.com/example/project.git",
                ),
                mock.patch.object(herdr, "confirm_clone", return_value=True),
                mock.patch("subprocess.run", side_effect=clone) as run,
            ):
                repository, reason = client.choose_or_clone_repository([])

            self.assertEqual(repository, root / "project")
            self.assertEqual(reason, "")
            command = run.call_args.args[0]
            self.assertEqual(
                command[:4],
                ["git", "clone", "--", "https://github.com/example/project.git"],
            )
            self.assertEqual(Path(command[-1]).name, "project")

    def test_cancelled_rofi_clone_returns_spoken_fallback_reason(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                herdr,
                "choose_repository",
                return_value="https://github.com/example/project.git",
            ),
            mock.patch.object(herdr, "confirm_clone", return_value=False),
        ):
            repository, reason = client.choose_or_clone_repository([])

        self.assertIsNone(repository)
        self.assertEqual(reason, "Repository cloning was cancelled.")

    def test_server_start_falls_back_to_transient_unit(self) -> None:
        client = herdr.HerdrClient("herdr")
        states = iter([False, False, True])
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            mock.patch.object(client, "is_running", side_effect=lambda: next(states)),
            mock.patch("subprocess.run", return_value=completed) as run,
            mock.patch("time.sleep"),
        ):
            client.ensure_server(timeout=1)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(any(command[0] == "systemd-run" for command in commands))

    def test_agent_name_stays_within_herdr_limit_for_long_label(self) -> None:
        client = herdr.HerdrClient("herdr")
        agent = {
            "name": "voice-issue-22-aaaaaaaaaa",
            "pane_id": "pane",
            "workspace_id": "workspace",
            "cwd": "/tmp/worktree",
        }
        with (
            mock.patch.object(client, "run_json", return_value={"agent": agent}) as run,
            mock.patch.object(
                herdr.uuid,
                "uuid4",
                return_value=mock.Mock(hex="a" * 32),
            ),
        ):
            client.start_agent(
                Path("/tmp/worktree"),
                "github-joshua-mo-143-local-voice-harness-22",
                "pane",
                "workspace",
            )

        name = run.call_args.args[2]
        self.assertLessEqual(len(name), 32)
        self.assertRegex(name, r"^[a-z][a-z0-9_-]{0,31}$")
        self.assertTrue(name.endswith("-aaaaaaaaaa"))

    def test_non_linear_task_creates_requested_worktree(self) -> None:
        client = herdr.HerdrClient("herdr")
        selection = herdr.AgentSelection(
            "agent", "pane", "workspace", "/tmp/worktree", "agent", "/tmp/worktree"
        )
        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"worktrees": []},
                    {
                        "worktree": {"path": "/tmp/worktree"},
                        "workspace": {"workspace_id": "workspace"},
                        "root_pane": {"pane_id": "pane"},
                    },
                ],
            ) as run_json,
            mock.patch.object(client, "find_agent", return_value=None),
            mock.patch.object(client, "workspace_for", return_value=None),
            mock.patch.object(
                client,
                "planned_worktree_path",
                return_value=Path("/tmp/worktree"),
            ),
            mock.patch.object(
                client, "start_agent", return_value=selection
            ) as start_agent,
        ):
            result = client.ensure_agent(
                Path("/tmp/repository"),
                issue_key=None,
                agent_hint=None,
                reserved=set(),
                worktree_branch="voice/github-123456",
                worktree_label="github-123456",
            )

        self.assertEqual(result, selection)
        self.assertEqual(
            run_json.call_args_list[1].args[:9],
            (
                "worktree",
                "create",
                "--cwd",
                "/tmp/repository",
                "--branch",
                "voice/github-123456",
                "--path",
                "/tmp/worktree",
                "--label",
            ),
        )
        start_agent.assert_called_once_with(
            Path("/tmp/worktree"),
            "github-123456",
            "pane",
            "workspace",
            name=mock.ANY,
            checkpoint=None,
        )

    def test_worktree_and_agent_are_reserved_before_dispatch(self) -> None:
        client = herdr.HerdrClient("herdr")
        checkout = Path("/tmp/worktree")
        selection = herdr.AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(checkout),
            "planned-agent",
            str(checkout),
        )
        events: list[str] = []

        def reserve_worktree(
            _repository: Path, _branch: str, _checkout: Path, state: str
        ) -> None:
            events.append(f"worktree:{state}")

        def settle_worktree(
            _checkout: Path, _workspace: str | None, _pane: str | None
        ) -> None:
            events.append("worktree:settled")

        def reserve_agent(_selection: herdr.AgentSelection, dispatching: bool) -> None:
            events.append(f"agent:reserved:{dispatching}")

        def start_agent(*_args: object, **_kwargs: object) -> herdr.AgentSelection:
            events.append("agent:started")
            return selection

        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"worktrees": []},
                    {
                        "worktree": {"path": str(checkout)},
                        "workspace": {"workspace_id": "workspace"},
                        "root_pane": {"pane_id": "pane"},
                    },
                ],
            ),
            mock.patch.object(client, "find_agent", return_value=None),
            mock.patch.object(client, "workspace_for", return_value=None),
            mock.patch.object(client, "planned_worktree_path", return_value=checkout),
            mock.patch.object(client, "start_agent", side_effect=start_agent),
        ):
            client.ensure_agent(
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

    def test_worktree_post_checkpoint_prevents_pane_and_agent_creation(self) -> None:
        client = herdr.HerdrClient("herdr")
        checkpoints = 0

        def checkpoint() -> None:
            nonlocal checkpoints
            checkpoints += 1
            if checkpoints == 4:
                raise RuntimeError("cancelled after worktree creation")

        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[
                    {"worktrees": []},
                    {
                        "worktree": {"path": "/tmp/worktree"},
                        "workspace": {"workspace_id": "workspace"},
                        "root_pane": {"pane_id": "pane"},
                    },
                ],
            ),
            mock.patch.object(client, "find_agent") as find_agent,
            mock.patch.object(client, "new_pane") as new_pane,
            mock.patch.object(client, "start_agent") as start_agent,
            mock.patch.object(
                client,
                "planned_worktree_path",
                return_value=Path("/tmp/worktree"),
            ),
            self.assertRaisesRegex(RuntimeError, "after worktree creation"),
        ):
            client.ensure_agent(
                Path("/tmp/repository"),
                issue_key=None,
                agent_hint=None,
                reserved=set(),
                worktree_branch="voice/github-123456",
                checkpoint=checkpoint,
            )

        find_agent.assert_not_called()
        new_pane.assert_not_called()
        start_agent.assert_not_called()

    def test_cancel_agent_waits_until_agent_is_settled(self) -> None:
        client = herdr.HerdrClient("herdr")
        with mock.patch.object(
            client,
            "run_json",
            side_effect=[{}, {"agent": {"agent_status": "idle"}}],
        ) as run_json:
            client.cancel_agent("agent")

        self.assertEqual(
            run_json.call_args_list,
            [
                mock.call("agent", "send-keys", "agent", "ctrl-c"),
                mock.call(
                    "agent",
                    "wait",
                    "agent",
                    "--timeout",
                    "5000",
                    timeout=10,
                ),
            ],
        )

    def test_cancel_agent_rejects_still_working_agent(self) -> None:
        client = herdr.HerdrClient("herdr")
        with (
            mock.patch.object(
                client,
                "run_json",
                side_effect=[{}, {"agent": {"agent_status": "working"}}],
            ),
            self.assertRaisesRegex(herdr.HerdrError, "did not stop"),
        ):
            client.cancel_agent("agent")


if __name__ == "__main__":
    unittest.main()
