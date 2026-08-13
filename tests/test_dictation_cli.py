from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import dictation_cli


class DictationCliTests(unittest.TestCase):
    def test_all_existing_commands_use_lightweight_dispatch(self) -> None:
        for command in dictation_cli.COMMANDS:
            with (
                self.subTest(command=command),
                mock.patch.object(dictation_cli, "_run") as run,
            ):
                dictation_cli.main([command])
                run.assert_called_once_with(command)

    def test_invalid_command_exits_before_loading_dictation(self) -> None:
        with (
            mock.patch.object(dictation_cli, "_run") as run,
            self.assertRaisesRegex(SystemExit, "2"),
        ):
            dictation_cli.main(["unknown"])

        run.assert_not_called()

    def test_entrypoint_import_does_not_load_general_cli_or_dictation(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import json, sys; "
            "import local_voice_harness.dictation_cli; "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name.startswith('local_voice_harness.'))))"
        )

        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )

        loaded = json.loads(process.stdout)
        self.assertEqual(loaded, ["local_voice_harness.dictation_cli"])


if __name__ == "__main__":
    unittest.main()
