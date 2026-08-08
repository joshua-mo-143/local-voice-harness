from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import cli, config, service_manager
from local_voice_harness.config import START_SERVICES
from local_voice_harness.stt import server as stt_server


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
                mock.patch("builtins.print") as output,
            ):
                service_manager.install_services(force=True)
            self.assertEqual(dictation.read_text(), "external dictation\n")
            self.assertIn(
                "voice-harness-wake",
                (systemd / "voice-harness-wake.service").read_text(),
            )
            self.assertTrue(
                any(
                    "--force --replace-dictation" in str(call)
                    for call in output.call_args_list
                )
            )

    def test_install_replaces_dictation_only_with_explicit_migration_flag(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            systemd = Path(temporary)
            dictation = systemd / "dictation.service"
            dictation.write_text("external dictation\n")
            with (
                mock.patch.object(service_manager, "SYSTEMD_USER_DIR", systemd),
                mock.patch.object(service_manager, "systemctl"),
            ):
                service_manager.install_services(force=True, replace_dictation=True)

            self.assertEqual(
                dictation.read_text(),
                service_manager.unit_text("dictation.service"),
            )

    def test_services_audit_is_read_only_and_exposed_by_cli(self) -> None:
        parsed = cli.parser().parse_args(["services", "audit"])
        install = cli.parser().parse_args(
            ["services", "install", "--force", "--replace-dictation"]
        )
        self.assertEqual(parsed.service_command, "audit")
        self.assertTrue(install.force)
        self.assertTrue(install.replace_dictation)

        with mock.patch.object(service_manager, "audit_installed", return_value=0):
            self.assertEqual(service_manager.audit_services(), 0)
        with (
            mock.patch.object(cli, "audit_services", return_value=0) as audit,
            self.assertRaises(SystemExit) as exited,
        ):
            cli.dispatch(parsed)
        audit.assert_called_once_with()
        self.assertEqual(exited.exception.code, 0)

    def test_hardening_docs_require_dictation_replacement_and_audit(self) -> None:
        root = service_manager.PROJECT_ROOT
        hardening = (root / "docs/service-hardening.md").read_text()
        readme = (root / "README.md").read_text()

        for text in (hardening, readme):
            self.assertIn(
                "voice-harness services install --force --replace-dictation",
                text,
            )
            self.assertIn("voice-harness services audit", text)
        self.assertIn("intentionally replaces the standalone unit", readme)

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
                self.assertEqual(
                    source.read_text(), service_manager.unit_text(source.name)
                )

    def test_llm_model_matches_service_and_documentation(self) -> None:
        project_root = service_manager.PROJECT_ROOT
        unit = (project_root / "systemd/user/voice-harness-llm.service").read_text()
        readme = (project_root / "README.md").read_text()
        match = re.search(r"--model \S*/(?P<filename>\S+\.gguf)\b", unit)

        self.assertIsNotNone(match)
        assert match is not None
        filename = match.group("filename")
        self.assertIn(f"--alias {config.LLM_MODEL}", unit)
        self.assertEqual(
            set(re.findall(r"Qwen3\.5-[A-Za-z0-9_.-]+\.gguf", readme)),
            {filename},
        )

    def test_dictation_defaults_have_documented_installable_backends(self) -> None:
        project_root = service_manager.PROJECT_ROOT
        unit = (project_root / "systemd/user/dictation.service").read_text()
        readme = (project_root / "README.md").read_text()
        metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
        extras = metadata["project"]["optional-dependencies"]

        self.assertIn("Environment=DICTATION_BACKEND=parakeet", unit)
        self.assertIn(
            f"Environment=DICTATION_MODEL={stt_server.PARAKEET_DEFAULT_MODEL}",
            unit,
        )
        self.assertIn(stt_server.PARAKEET_DEFAULT_MODEL, readme)
        self.assertTrue(
            any(dependency.startswith("onnx-asr") for dependency in extras["dictation"])
        )
        self.assertTrue(
            any(
                dependency.startswith("faster-whisper")
                for dependency in extras["dictation-whisper"]
            )
        )

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
