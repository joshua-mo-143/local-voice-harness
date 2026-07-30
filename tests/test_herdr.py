from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.integrations import herdr


class HerdrIntegrationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
