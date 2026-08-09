from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from local_voice_harness.integrations import rofi


class RofiIntegrationTests(unittest.TestCase):
    def test_missing_rofi_falls_back_without_prompting(self) -> None:
        with (
            mock.patch.object(rofi.shutil, "which", return_value=None),
            mock.patch.object(rofi.subprocess, "run") as run,
        ):
            self.assertIsNone(rofi.choose_repository(["example"]))
        run.assert_not_called()

    def test_repository_prompt_accepts_custom_input(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 0, "https://github.com/example/project.git\n", ""
        )
        with (
            mock.patch.object(rofi.shutil, "which", return_value="/usr/bin/rofi"),
            mock.patch.object(rofi.subprocess, "run", return_value=completed) as run,
        ):
            selected = rofi.choose_repository(["local-project"])

        self.assertEqual(selected, "https://github.com/example/project.git")
        self.assertEqual(run.call_args.kwargs["input"], "local-project")
        self.assertIn("-no-click-to-exit", run.call_args.args[0])
        self.assertNotIn("-no-custom", run.call_args.args[0])

    def test_empty_repository_list_hides_results_view(self) -> None:
        completed = subprocess.CompletedProcess([], 1, "", "")
        with (
            mock.patch.object(rofi.shutil, "which", return_value="/usr/bin/rofi"),
            mock.patch.object(rofi.subprocess, "run", return_value=completed) as run,
        ):
            self.assertIsNone(rofi.choose_repository([]))

        command = run.call_args.args[0]
        self.assertIn("listview { enabled: false; }", command)

    def test_clone_confirmation_disallows_custom_values(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "Clone\n", "")
        with (
            mock.patch.object(rofi.shutil, "which", return_value="/usr/bin/rofi"),
            mock.patch.object(rofi.subprocess, "run", return_value=completed) as run,
        ):
            self.assertTrue(
                rofi.confirm_clone("https://github.com/example/project.git")
            )

        self.assertIn("-no-custom", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
