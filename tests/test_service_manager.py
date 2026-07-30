from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import service_manager
from local_voice_harness.config import START_SERVICES


class ServiceManagementTests(unittest.TestCase):
    def test_start_uses_only_always_on_services(self) -> None:
        with mock.patch.object(service_manager, "systemctl") as systemctl:
            service_manager.start_services()
        systemctl.assert_called_once_with("start", *START_SERVICES)

    def test_install_preserves_external_dictation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            systemd = Path(temporary)
            dictation = systemd / "dictation.service"
            dictation.write_text("external dictation\n")
            with (
                mock.patch.object(service_manager, "SYSTEMD_USER_DIR", systemd),
                mock.patch.object(service_manager, "systemctl"),
            ):
                service_manager.install_services(force=True)
            self.assertEqual(dictation.read_text(), "external dictation\n")
            self.assertIn(
                "voice-harness-wake",
                (systemd / "voice-harness-wake.service").read_text(),
            )

    def test_packaged_systemd_resources_are_available(self) -> None:
        with mock.patch.object(
            service_manager, "PROJECT_ROOT", Path("/definitely/not/a/source/tree")
        ):
            text = service_manager.unit_text("voice-harness-wake.service")
        self.assertIn("voice-harness-wake", text)

    def test_source_units_match_packaged_resources(self) -> None:
        source_root = service_manager.PROJECT_ROOT / "systemd" / "user"
        for source in source_root.glob("*.service"):
            with (
                self.subTest(unit=source.name),
                mock.patch.object(
                    service_manager,
                    "PROJECT_ROOT",
                    Path("/definitely/not/a/source/tree"),
                ),
            ):
                self.assertEqual(source.read_text(), service_manager.unit_text(source.name))

    def test_all_console_entry_points_load(self) -> None:
        expected = {
            "voice-harness",
            "voice-harness-wake",
            "voice-harness-dictation",
            "voice-harness-tts",
            "voice-harness-cursor-worker",
        }
        scripts = {
            entry.name: entry
            for entry in importlib.metadata.distribution(
                "local-voice-harness"
            ).entry_points
            if entry.group == "console_scripts"
        }
        self.assertEqual(set(scripts), expected)
        for entry in scripts.values():
            self.assertTrue(callable(entry.load()))

    def test_console_entry_points_smoke(self) -> None:
        scripts = Path(sys.executable).parent
        commands = {
            "voice-harness": ["--help"],
            "voice-harness-wake": ["--check"],
            "voice-harness-dictation": ["--check"],
            "voice-harness-tts": ["--check"],
            "voice-harness-cursor-worker": ["--help"],
        }
        for name, arguments in commands.items():
            with self.subTest(name=name):
                process = subprocess.run(
                    [str(scripts / name), *arguments],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
