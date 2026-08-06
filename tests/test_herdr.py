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
            run_json.call_args_list[1].args[:7],
            (
                "worktree",
                "create",
                "--cwd",
                "/tmp/repository",
                "--branch",
                "voice/github-123456",
                "--label",
            ),
        )
        start_agent.assert_called_once_with(
            Path("/tmp/worktree"), "github-123456", "pane", "workspace"
        )


if __name__ == "__main__":
    unittest.main()
