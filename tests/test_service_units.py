from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import service_units


class ServiceUnitValidationTests(unittest.TestCase):
    def test_repository_units_have_parity_and_consistent_defaults(self) -> None:
        self.assertEqual(service_units.parity_errors(), [])
        self.assertEqual(service_units.consistency_errors(), [])

    def test_parity_checks_the_complete_inventory_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            source = project_root / service_units.SOURCE_RELATIVE
            shutil.copytree(
                service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE, source
            )
            (source / "voice-harness-llm.service").write_text("[Unit]\n")
            (source / "unexpected.service").write_text("[Unit]\n")
            (source / "dictation.service").unlink()

            errors = service_units.parity_errors(project_root)

        self.assertIn("source units missing: dictation.service", errors)
        self.assertIn("source units unexpected: unexpected.service", errors)
        self.assertIn(
            "source and packaged units differ: voice-harness-llm.service", errors
        )

    def test_systemd_analyze_is_optional_when_not_installed(self) -> None:
        with mock.patch.object(service_units.shutil, "which", return_value=None):
            self.assertIsNone(service_units.systemd_analyze([]))

    def test_stages_normal_absolute_executable_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()

            staged = service_units.stage_executable_stub(root, "/usr/local/bin/tool")

            self.assertEqual(staged, root.resolve() / "usr/local/bin/tool")
            self.assertTrue(staged.is_file())
            self.assertTrue(staged.stat().st_mode & 0o111)
            self.assertTrue(staged.resolve().is_relative_to(root.resolve()))

    def test_rejects_lexical_traversal_and_non_normalized_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            outside = base / "outside"

            for executable in (
                "/../outside/tool",
                "/../../outside/tool",
                "/usr/../outside/tool",
                "/usr/./bin/tool",
                "/usr//bin/tool",
            ):
                with (
                    self.subTest(executable=executable),
                    self.assertRaisesRegex(ValueError, "unsafe staged executable"),
                ):
                    service_units.stage_executable_stub(root, executable)

            self.assertFalse(outside.exists())

    def test_rejects_parent_and_destination_symlink_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            outside = base / "outside"
            outside.mkdir()

            parent_root = base / "parent-root"
            parent_root.mkdir()
            (parent_root / "usr").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                service_units.stage_executable_stub(parent_root, "/usr/bin/tool")
            self.assertFalse((outside / "bin/tool").exists())

            destination_root = base / "destination-root"
            (destination_root / "usr/bin").mkdir(parents=True)
            outside_target = outside / "tool"
            (destination_root / "usr/bin/tool").symlink_to(outside_target)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                service_units.stage_executable_stub(destination_root, "/usr/bin/tool")
            self.assertFalse(outside_target.exists())

    def test_systemd_analyze_stages_missing_commands_and_known_dependencies(
        self,
    ) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        paths = [
            service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE / name
            for name in service_units.SERVICE_FILES
        ]

        def inspect_staging(
            arguments: list[str], **options: object
        ) -> subprocess.CompletedProcess[str]:
            root = Path(arguments[1].removeprefix("--root="))
            self.assertEqual(arguments[2:5], ["--man=no", "--generators=no", "verify"])
            self.assertEqual(arguments[5:], list(service_units.SERVICE_FILES))
            self.assertEqual(
                options,
                {"capture_output": True, "text": True, "check": False},
            )
            expected_commands = (
                Path("/usr/sbin/llama-server"),
                Path.home()
                / "local-voice-harness/.venv-dictation/bin/voice-harness-dictation",
                Path.home() / "local-voice-harness/.venv/bin/voice-harness-wake",
                Path.home() / "chatterbox-audition/.venv/bin/voice-harness-tts",
            )
            for command in expected_commands:
                staged = root / command.relative_to("/")
                self.assertTrue(staged.is_file(), command)
                self.assertTrue(staged.stat().st_mode & 0o111, command)
            for name in service_units.EXTERNAL_UNIT_STUBS:
                self.assertTrue((root / "etc/systemd/system" / name).is_file())
            return completed

        with mock.patch.object(
            service_units.subprocess, "run", side_effect=inspect_staging
        ):
            result = service_units.systemd_analyze(paths, executable="/tool")

        self.assertIs(result, completed)

    @unittest.skipUnless(shutil.which("systemd-analyze"), "systemd-analyze unavailable")
    def test_systemd_analyze_rejects_genuine_unit_syntax_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.service"
            invalid.write_text(
                "[Unit]\n"
                "Description=Invalid fixture\n"
                "[Service]\n"
                "Type=oneshot\n"
                "ExecStart=relative/path/is/not/valid\n"
            )

            result = service_units.systemd_analyze([invalid])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Neither a valid executable name nor an absolute path", result.stderr
        )

    @unittest.skipUnless(shutil.which("systemd-analyze"), "systemd-analyze unavailable")
    def test_systemd_analyze_rejects_unknown_required_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid-dependency.service"
            invalid.write_text(
                "[Unit]\n"
                "Description=Invalid dependency fixture\n"
                "Requires=misspelled-runtime.service\n"
                "[Service]\n"
                "Type=oneshot\n"
                "ExecStart=/missing/on/clean/runner\n"
            )

            result = service_units.systemd_analyze([invalid])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("misspelled-runtime.service", result.stderr)


if __name__ == "__main__":
    unittest.main()
