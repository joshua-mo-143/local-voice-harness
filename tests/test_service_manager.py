from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import cli, llm_launcher, service_manager
from local_voice_harness.config import START_SERVICES
from local_voice_harness.integrations.registry import build_integration_registry
from local_voice_harness.stt import server as stt_server
from local_voice_harness.user_config import default_user_config


def _snapshot():
    config = default_user_config()
    return service_manager.ServiceManagementSnapshot(
        config,
        build_integration_registry(config),
    )


class ServiceManagementTests(unittest.TestCase):
    def test_restart_reuses_one_service_management_snapshot(self) -> None:
        snapshot = _snapshot()
        with (
            mock.patch.object(
                service_manager.ServiceManagementSnapshot,
                "load",
                return_value=snapshot,
            ) as load,
            mock.patch.object(service_manager, "systemctl"),
        ):
            service_manager.restart_services(include_herdr=False)

        load.assert_called_once_with()

    def test_stop_herdr_uses_configured_snapshot_client(self) -> None:
        snapshot = _snapshot()
        client = mock.Mock(executable="/opt/herdr", timeout=17)
        client.is_running.return_value = True
        client.run.return_value = subprocess.CompletedProcess([], 0, "", "")
        configured = replace(
            snapshot,
            registry=replace(snapshot.registry, herdr_client=lambda: client),
        )
        service_manager.stop_herdr(configured)

        client.run.assert_called_once_with("server", "stop", check=False)

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

        with mock.patch.object(
            service_manager, "audit_installed", return_value=0
        ) as audit_installed:
            self.assertEqual(service_manager.audit_services(), 0)
        audit_installed.assert_called_once_with(config=mock.ANY)
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
        installation = (root / "docs/installation.md").read_text()

        for text in (hardening, installation):
            self.assertIn(
                "voice-harness services install --force --replace-dictation",
                text,
            )
            self.assertIn("voice-harness services audit", text)
        self.assertIn("intentionally replaces the standalone unit", installation)

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

    def test_llm_unit_defers_model_and_device_to_typed_launcher(self) -> None:
        project_root = service_manager.PROJECT_ROOT
        unit = (project_root / "systemd/user/voice-harness-llm.service").read_text()
        installation = (project_root / "docs/installation.md").read_text()
        snapshot = default_user_config()
        configured = replace(
            snapshot,
            providers=replace(
                snapshot.providers,
                llm_provider="local",
                llm_model="configured-model",
            ),
            compute=replace(snapshot.compute, cuda_device="CUDA7"),
        )
        with mock.patch.object(
            llm_launcher, "load_user_config", return_value=configured
        ):
            command = llm_launcher.command()

        self.assertNotIn("--model", unit)
        self.assertNotIn("--device", unit)
        self.assertIn("voice-harness-llm", unit)
        self.assertEqual(command[command.index("--alias") + 1], "configured-model")
        self.assertEqual(command[command.index("--device") + 1], "CUDA7")
        self.assertEqual(
            set(re.findall(r"Qwen3\.5-[A-Za-z0-9_.-]+\.gguf", installation)),
            {llm_launcher.MODEL_FILE.name},
        )

    def test_dictation_defaults_have_documented_installable_backends(self) -> None:
        project_root = service_manager.PROJECT_ROOT
        unit = (project_root / "systemd/user/dictation.service").read_text()
        configuration = (project_root / "docs/configuration.md").read_text()
        installer = (project_root / "scripts/install.sh").read_text()
        metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
        extras = metadata["project"]["optional-dependencies"]

        self.assertNotIn("Environment=DICTATION_BACKEND", unit)
        self.assertNotIn("Environment=DICTATION_MODEL", unit)
        defaults = default_user_config()
        self.assertEqual(defaults.compute.dictation_backend, "parakeet")
        self.assertEqual(
            defaults.compute.dictation_model, stt_server.PARAKEET_DEFAULT_MODEL
        )
        self.assertIn(stt_server.PARAKEET_DEFAULT_MODEL, configuration)
        self.assertTrue(
            any(dependency.startswith("onnx-asr") for dependency in extras["dictation"])
        )
        self.assertTrue(
            any(
                dependency.startswith("onnx-asr")
                for dependency in extras["dictation-cuda"]
            )
        )
        self.assertIn("--extra dictation-cuda", installer)
        self.assertTrue(
            any(
                dependency.startswith("onnxruntime")
                for dependency in extras["dictation"]
            )
        )
        self.assertFalse(
            any(
                "gpu" in dependency.casefold() or "nvidia" in dependency.casefold()
                for dependency in extras["dictation"]
            )
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
            "voice-harness-llm",
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
            "voice-harness-llm": ["--check"],
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
