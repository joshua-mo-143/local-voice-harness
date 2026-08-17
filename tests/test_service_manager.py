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
from local_voice_harness.config import START_SERVICES, STOP_SERVICES
from local_voice_harness.install_profile import resolve_installation_plan
from local_voice_harness.integrations.registry import build_integration_registry
from local_voice_harness.stt import server as stt_server
from local_voice_harness.user_config import default_user_config


def _snapshot():
    config = default_user_config()
    return service_manager.ServiceManagementSnapshot(
        config,
        build_integration_registry(config),
    )


def _fake_clock(start: float = 0.0):
    clock = [start]

    def monotonic() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    return monotonic, sleep


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

    def test_wake_mode_stop_uses_normal_stop_set_without_disabling_units(self) -> None:
        stopped = subprocess.CompletedProcess([], 0, "", "")
        inactive = subprocess.CompletedProcess([], 3, "inactive\n", "")
        with (
            mock.patch.object(
                service_manager,
                "systemctl",
                side_effect=[stopped, *(inactive for _service in STOP_SERVICES)],
            ) as systemctl,
            mock.patch.object(service_manager, "notify") as notify,
            mock.patch.object(service_manager.time, "sleep") as sleep,
        ):
            service_manager.execute_wake_mode_stop()

        sleep.assert_not_called()

        self.assertEqual(
            systemctl.call_args_list,
            [
                mock.call("stop", *STOP_SERVICES, check=False),
                *(
                    mock.call("is-active", service, check=False)
                    for service in STOP_SERVICES
                ),
            ],
        )
        notify.assert_not_called()

    def test_wake_mode_stop_accepts_deactivating_units(self) -> None:
        stopped = subprocess.CompletedProcess([], 0, "", "")
        states = [
            subprocess.CompletedProcess(
                [],
                3,
                (
                    "deactivating\n"
                    if service == "voice-harness-wake.service"
                    else "inactive\n"
                ),
                "",
            )
            for service in STOP_SERVICES
        ]
        with (
            mock.patch.object(
                service_manager,
                "systemctl",
                side_effect=[stopped, *states],
            ),
            mock.patch.object(service_manager, "notify") as notify,
            mock.patch.object(service_manager.time, "sleep") as sleep,
        ):
            service_manager.execute_wake_mode_stop()

        notify.assert_not_called()
        sleep.assert_not_called()

    def test_wake_mode_stop_retries_active_state_until_inactive(self) -> None:
        stopped = subprocess.CompletedProcess([], 0, "", "")
        active = subprocess.CompletedProcess([], 0, "active\n", "")
        inactive = subprocess.CompletedProcess([], 3, "inactive\n", "")
        effects: list[subprocess.CompletedProcess[str]] = [stopped]
        for service in STOP_SERVICES:
            if service == "voice-harness-wake.service":
                effects.extend((active, inactive))
            else:
                effects.append(inactive)
        with (
            mock.patch.object(service_manager, "systemctl", side_effect=effects),
            mock.patch.object(service_manager, "notify") as notify,
            mock.patch.object(service_manager.time, "monotonic", return_value=0.0),
            mock.patch.object(service_manager.time, "sleep") as sleep,
        ):
            service_manager.execute_wake_mode_stop()

        notify.assert_not_called()
        sleep.assert_called_once_with(service_manager._WAKE_STOP_POLL_SECONDS)

    def test_wake_mode_stop_reports_unverified_active_service(self) -> None:
        stopped = subprocess.CompletedProcess([], 1, "", "stop failed")
        active = subprocess.CompletedProcess([], 0, "active\n", "")
        inactive = subprocess.CompletedProcess([], 3, "inactive\n", "")

        def systemctl(command: str, *arguments: str, check: bool = True):
            if command == "stop":
                return stopped
            service = arguments[0]
            if service == "voice-harness-wake.service":
                return active
            return inactive

        monotonic, sleep = _fake_clock()
        with (
            mock.patch.object(service_manager, "systemctl", side_effect=systemctl),
            mock.patch.object(service_manager, "notify") as notify,
            mock.patch.object(service_manager.time, "monotonic", side_effect=monotonic),
            mock.patch.object(service_manager.time, "sleep", side_effect=sleep),
            self.assertRaises(service_manager.HarnessError) as raised,
        ):
            service_manager.execute_wake_mode_stop()

        self.assertIn("voice-harness-wake.service: active", str(raised.exception))
        notify.assert_called_once_with(str(raised.exception), error=True)

    def test_wake_mode_stop_worker_runs_outside_wake_service_cgroup(self) -> None:
        process = mock.Mock()
        with (
            mock.patch.object(
                service_manager.uuid,
                "uuid4",
                return_value=mock.Mock(hex="a" * 32),
            ),
            mock.patch.object(
                service_manager.subprocess,
                "Popen",
                return_value=process,
            ) as popen,
        ):
            returned = service_manager.launch_wake_mode_stop_worker()

        self.assertIs(returned, process)
        command = popen.call_args.args[0]
        self.assertEqual(
            command[:5], ["systemd-run", "--user", "--scope", "--collect", "--quiet"]
        )
        self.assertIn("--unit=voice-harness-wake-stop-aaaaaaaaaaaaaaaa", command)
        self.assertEqual(
            command[-3:],
            [
                "-m",
                "local_voice_harness.service_manager",
                "--stop-for-wake-request",
            ],
        )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

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
            command = llm_launcher.command(cuda_available=True)

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
        self.assertIn('--extra "$INSTALL_DICTATION_EXTRA"', installer)
        self.assertEqual(
            resolve_installation_plan(profile="local-cuda").dictation_extra,
            "dictation-cuda",
        )
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
            "voice-harness-dictate",
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
            "voice-harness-dictate": ["--help"],
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
