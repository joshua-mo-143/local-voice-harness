from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import service_units


class ServiceUnitValidationTests(unittest.TestCase):
    def test_ci_explicitly_runs_staged_verifier_with_pinned_actions(self) -> None:
        workflow = (
            service_units.PROJECT_ROOT / ".github/workflows/quality.yml"
        ).read_text()

        self.assertIn("Verify staged systemd unit templates", workflow)
        self.assertIn(
            "python -m local_voice_harness.service_units \\\n"
            "            --require-systemd-analyze",
            workflow,
        )
        self.assertNotIn("--audit-installed", workflow)
        action_lines = re.findall(
            r"^\s*(?:-\s*)?uses:\s*(\S+)\s+#\s*(\S+)$", workflow, re.M
        )
        self.assertEqual(len(action_lines), workflow.count("uses:"))
        for reference, version in action_lines:
            with self.subTest(action=reference):
                self.assertRegex(reference, r"^[^@]+@[0-9a-f]{40}$")
                self.assertRegex(version, r"^v\d")

    def test_repository_units_have_parity_and_consistent_defaults(self) -> None:
        self.assertEqual(service_units.parity_errors(), [])
        self.assertEqual(service_units.consistency_errors(), [])
        self.assertEqual(service_units.security_errors(), [])

    def test_state_directory_is_service_owned_not_user_configurable(self) -> None:
        wake = "voice-harness-wake.service"
        self.assertIn("STATE_DIRECTORY", service_units.SERVICE_OWNED_ENVIRONMENT[wake])
        self.assertNotIn(
            "STATE_DIRECTORY", service_units.OPTIONAL_ENVIRONMENT_POLICY[wake]
        )

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

    def test_security_policy_rejects_weakened_or_incompatible_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            source = project_root / service_units.SOURCE_RELATIVE
            shutil.copytree(
                service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE, source
            )
            llm_path = source / "voice-harness-llm.service"
            llm_path.write_text(
                llm_path.read_text()
                .replace("--host 127.0.0.1", "--host 0.0.0.0")
                .replace("MemoryMax=8G", "MemoryMax=infinity")
                .replace(
                    "ProtectSystem=strict",
                    "ProtectSystem=strict\n"
                    "PrivateDevices=true\n"
                    "SystemCallFilter=@system-service",
                )
            )

            errors = service_units.security_errors(project_root)

        self.assertIn(
            "voice-harness-llm.service MemoryMax must be '8G', got 'infinity'",
            errors,
        )
        self.assertIn(
            "voice-harness-llm.service must not restrict CUDA devices with "
            "PrivateDevices",
            errors,
        )
        self.assertIn(
            "voice-harness-llm.service must not set compatibility-sensitive "
            "SystemCallFilter",
            errors,
        )
        self.assertIn(
            "voice-harness-llm.service must bind to IPv4 loopback",
            errors,
        )

    def test_security_policy_rejects_duplicates_resets_and_last_overrides(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            source = project_root / service_units.SOURCE_RELATIVE
            shutil.copytree(
                service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE, source
            )
            llm_path = source / "voice-harness-llm.service"
            llm_path.write_text(
                llm_path.read_text()
                .replace("MemoryMax=8G", "MemoryMax=8G\nMemoryMax=8G")
                .replace(
                    "ProtectHome=read-only",
                    "ProtectHome=read-only\nProtectHome=\nProtectHome=false",
                )
                .replace(
                    "NoNewPrivileges=true",
                    "NoNewPrivileges=true\nNoNewPrivileges=false",
                )
                .replace(
                    "Environment=CUDA_CACHE_PATH=%t/voice-harness-llm/cuda-cache",
                    "Environment=CUDA_CACHE_PATH="
                    "%t/voice-harness-llm/cuda-cache\n"
                    "Environment=\n"
                    "Environment=CUDA_CACHE_PATH=/tmp/cuda",
                )
            )

            errors = service_units.security_errors(project_root)

        self.assertIn(
            "voice-harness-llm.service MemoryMax must have exactly one "
            "assignment, got ['8G', '8G']",
            errors,
        )
        self.assertIn(
            "voice-harness-llm.service ProtectHome must have exactly one "
            "assignment, got ['read-only', '', 'false']",
            errors,
        )
        self.assertIn(
            "voice-harness-llm.service NoNewPrivileges must have exactly one "
            "assignment, got ['true', 'false']",
            errors,
        )
        self.assertIn(
            "voice-harness-llm.service Environment=CUDA_CACHE_PATH must be "
            "assigned exactly once to "
            "'%t/voice-harness-llm/cuda-cache', got "
            "['%t/voice-harness-llm/cuda-cache', '<reset>', '/tmp/cuda']",
            errors,
        )

    def test_security_policy_captures_each_services_resource_boundaries(self) -> None:
        policy = service_units.SERVICE_SECURITY_POLICY

        self.assertEqual(set(policy), set(service_units.SERVICE_FILES))
        self.assertEqual(
            set(service_units.OPTIONAL_ENVIRONMENT_POLICY),
            set(service_units.SERVICE_FILES),
        )
        self.assertTrue(
            all(
                not variables
                for name, variables in service_units.OPTIONAL_ENVIRONMENT_POLICY.items()
                if name != "voice-harness-wake.service"
            )
        )
        configuration = (
            service_units.PROJECT_ROOT / "docs" / "configuration.md"
        ).read_text()
        documented_wake_variables = set(
            re.findall(
                r"^\| `([A-Z0-9_]+)` .*\| Wake drop-in \|$",
                configuration,
                re.M,
            )
        )
        self.assertEqual(
            documented_wake_variables,
            set(
                service_units.OPTIONAL_ENVIRONMENT_POLICY["voice-harness-wake.service"]
            ),
        )
        self.assertEqual(
            policy["voice-harness-tts.service"]["RestrictAddressFamilies"],
            "AF_UNIX AF_NETLINK",
        )
        self.assertEqual(
            policy["dictation.service"]["ReadWritePaths"],
            "%t",
        )
        self.assertEqual(
            policy["voice-harness-wake.service"]["ProtectHome"],
            "false",
        )
        self.assertEqual(
            policy["voice-harness-wake.service"]["StateDirectory"],
            "voice-harness",
        )
        self.assertEqual(
            policy["voice-harness-wake.service"]["StateDirectoryMode"],
            "0700",
        )
        for name in service_units.GPU_SERVICES:
            self.assertNotIn("PrivateDevices", policy[name])
            runtime = service_units.CUDA_RUNTIME_DIRECTORIES[name]
            cache = service_units.REQUIRED_ENVIRONMENT[name]["CUDA_CACHE_PATH"]
            self.assertTrue(cache.startswith(f"%t/{runtime}/"))
        self.assertEqual(
            service_units.STT_SOCKET,
            service_units.RUNTIME / "dictation.sock",
        )
        self.assertEqual(
            service_units.TTS_SOCKET,
            service_units.RUNTIME / "voice-harness-tts.sock",
        )
        self.assertEqual(
            service_units.REQUIRED_ENVIRONMENT["dictation.service"]["DICTATION_SOCKET"],
            "%t/dictation.sock",
        )
        self.assertNotIn(
            "VOICE_HARNESS_TTS_SOCKET",
            service_units.REQUIRED_ENVIRONMENT["voice-harness-tts.service"],
        )

    def test_tts_socket_override_is_rejected_to_preserve_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            source = project_root / service_units.SOURCE_RELATIVE
            shutil.copytree(
                service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE, source
            )
            path = source / "voice-harness-tts.service"
            path.write_text(
                path.read_text().replace(
                    "Environment=HF_HUB_OFFLINE=1",
                    "Environment=HF_HUB_OFFLINE=1\n"
                    "Environment=VOICE_HARNESS_TTS_SOCKET=%t/other.sock",
                )
            )

            errors = service_units.security_errors(project_root)

        self.assertIn(
            "voice-harness-tts.service must use the fixed compatible socket",
            errors,
        )

    def test_all_templates_reject_unrestricted_environment_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            source = project_root / service_units.SOURCE_RELATIVE
            shutil.copytree(
                service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE, source
            )
            for name in service_units.SERVICE_FILES:
                path = source / name
                path.write_text(
                    path.read_text().replace(
                        "[Service]",
                        "[Service]\nEnvironmentFile=-%h/.config/unsafe.env",
                    )
                )

            errors = service_units.security_errors(project_root)

        for name in service_units.SERVICE_FILES:
            with self.subTest(service=name):
                self.assertIn(
                    f"{name} must not import unrestricted EnvironmentFile values",
                    errors,
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
            self.assertEqual(arguments[1], "--user")
            root = Path(arguments[2].removeprefix("--root="))
            self.assertEqual(arguments[3:6], ["--man=no", "--generators=no", "verify"])
            self.assertEqual(arguments[6:], list(service_units.SERVICE_FILES))
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
                self.assertTrue((root / "etc/systemd/user" / name).is_file())
            return completed

        with mock.patch.object(
            service_units.subprocess, "run", side_effect=inspect_staging
        ):
            result = service_units.systemd_analyze(paths, executable="/tool")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIs(result.process, completed)
        self.assertEqual(result.context, "staged-user")
        self.assertEqual(result.rooted_user_context, "supported")

    def test_systemd_analyze_falls_back_when_rooted_user_mode_is_unsupported(
        self,
    ) -> None:
        path = (
            service_units.PROJECT_ROOT
            / service_units.SOURCE_RELATIVE
            / "voice-harness-llm.service"
        )
        unsupported = subprocess.CompletedProcess(
            [],
            1,
            "",
            "Failed to initialize unit search paths for root directory: "
            "Invalid argument",
        )
        completed = subprocess.CompletedProcess([], 0, "", "")

        with mock.patch.object(
            service_units.subprocess,
            "run",
            side_effect=[unsupported, completed],
        ) as run:
            result = service_units.systemd_analyze([path], executable="/tool")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIs(result.process, completed)
        self.assertEqual(result.context, "staged-system")
        self.assertEqual(result.rooted_user_context, "unsupported")
        self.assertIn("Invalid argument", result.reason or "")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][1], "--user")
        self.assertTrue(run.call_args_list[1].args[0][1].startswith("--root="))

    def test_require_user_context_fails_on_honest_unsupported_result(self) -> None:
        verification = service_units.StagedVerification(
            subprocess.CompletedProcess([], 0, "", ""),
            context="staged-system",
            rooted_user_context="unsupported",
            reason="Invalid argument",
        )
        with (
            mock.patch.object(service_units, "parity_errors", return_value=[]),
            mock.patch.object(service_units, "consistency_errors", return_value=[]),
            mock.patch.object(service_units, "security_errors", return_value=[]),
            mock.patch.object(
                service_units, "systemd_analyze", return_value=verification
            ),
            mock.patch("builtins.print") as output,
        ):
            result = service_units.main(
                ["--require-systemd-analyze", "--require-user-context"]
            )

        self.assertEqual(result, 1)
        messages = [str(call) for call in output.call_args_list]
        self.assertTrue(any("staged-system" in message for message in messages))
        self.assertTrue(any("unsupported" in message for message in messages))

    def test_systemd_analyze_does_not_stub_an_invented_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "voice-harness-llm.service"
            source.write_text(
                (
                    service_units.PROJECT_ROOT
                    / service_units.SOURCE_RELATIVE
                    / source.name
                )
                .read_text()
                .replace("/usr/sbin/llama-server", "/usr/sbin/llama-sevrer")
            )

            def inspect_staging(
                arguments: list[str], **_options: object
            ) -> subprocess.CompletedProcess[str]:
                root = Path(arguments[2].removeprefix("--root="))
                invented = root / "usr/sbin/llama-sevrer"
                expected = root / "usr/sbin/llama-server"
                self.assertFalse(invented.exists())
                self.assertFalse(expected.exists())
                return subprocess.CompletedProcess([], 1, "", "missing executable")

            with mock.patch.object(
                service_units.subprocess, "run", side_effect=inspect_staging
            ):
                result = service_units.systemd_analyze([source], executable="/tool")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(result.returncode, 0)

    def test_installed_audit_reads_effective_unit_and_runtime_state(self) -> None:
        name = "voice-harness-llm.service"
        observational = {
            "ActiveState",
            "SubState",
            "Result",
            "NRestarts",
            "ExecMainStatus",
            "MemoryCurrent",
            "TasksCurrent",
            "Environment",
            "ExecStart",
        }
        self.assertEqual(
            set(service_units._audit_expected_properties(name)),
            set(service_units.AUDIT_SHOW_PROPERTIES) - observational,
        )
        unit = (
            service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE / name
        ).read_text()
        command = service_units._expanded_unit_value(
            service_units.EXPECTED_EXECSTART[name]
        )
        properties = {
            **service_units._audit_expected_properties(name),
            "ActiveState": "inactive",
            "SubState": "dead",
            "Result": "success",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "MemoryCurrent": "[not set]",
            "TasksCurrent": "[not set]",
            "Environment": ("CUDA_CACHE_PATH=%t/voice-harness-llm/cuda-cache"),
            "ExecStart": (
                f"{{ path=/usr/sbin/llama-server ; argv[]={command} ; "
                "ignore_errors=no ; }}"
            ),
        }
        shown = "\n".join(f"{key}={value}" for key, value in properties.items())
        completed = [
            subprocess.CompletedProcess([], 0, unit, ""),
            subprocess.CompletedProcess([], 0, shown, ""),
        ]

        with (
            mock.patch.object(
                service_units.subprocess, "run", side_effect=completed
            ) as run,
            mock.patch("builtins.print"),
        ):
            result = service_units.audit_installed([name])

        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["systemctl", "--user", "cat", name],
        )
        self.assertEqual(
            run.call_args_list[1].args[0][:4],
            ["systemctl", "--user", "show", name],
        )
        self.assertIn("--property=EnvironmentFiles", run.call_args_list[1].args[0])

    def audit_fixture(
        self,
        name: str,
        environment: dict[str, str],
        *,
        unit_suffix: str = "",
        property_overrides: dict[str, str] | None = None,
    ) -> list[subprocess.CompletedProcess[str]]:
        unit = (
            service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE / name
        ).read_text()
        command = service_units._expanded_unit_value(
            service_units.EXPECTED_EXECSTART[name]
        )
        properties = {
            **service_units._audit_expected_properties(name),
            "ActiveState": "inactive",
            "SubState": "dead",
            "Result": "success",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "MemoryCurrent": "[not set]",
            "TasksCurrent": "[not set]",
            "Environment": " ".join(
                f"{key}={value}" for key, value in environment.items()
            ),
            "ExecStart": (
                f"{{ path={command.split()[0]} ; argv[]={command} ; "
                "ignore_errors=no ; }}"
            ),
            **(property_overrides or {}),
        }
        shown = "\n".join(f"{key}={value}" for key, value in properties.items())
        return [
            subprocess.CompletedProcess([], 0, unit + unit_suffix, ""),
            subprocess.CompletedProcess([], 0, shown, ""),
        ]

    def test_installed_audit_rejects_environment_files_for_every_service(
        self,
    ) -> None:
        for name in service_units.SERVICE_FILES:
            with self.subTest(service=name):
                completed = self.audit_fixture(
                    name,
                    service_units.REQUIRED_ENVIRONMENT[name],
                    unit_suffix=(
                        "\n# drop-in\n[Service]\n"
                        "EnvironmentFile=-%h/.config/unsafe.env\n"
                    ),
                    property_overrides={
                        "EnvironmentFiles": "/home/test/.config/unsafe.env"
                    },
                )
                with (
                    mock.patch.object(
                        service_units.subprocess, "run", side_effect=completed
                    ),
                    mock.patch("builtins.print") as output,
                ):
                    result = service_units.audit_installed([name])

                self.assertEqual(result, 1)
                messages = [str(call) for call in output.call_args_list]
                self.assertTrue(
                    any("EnvironmentFile must be empty" in item for item in messages)
                )
                self.assertTrue(any("EnvironmentFiles" in item for item in messages))

    def test_installed_audit_accepts_documented_wake_overrides(self) -> None:
        name = "voice-harness-wake.service"
        environment = {
            **service_units.REQUIRED_ENVIRONMENT[name],
            "VOICE_HARNESS_SOURCE": "voice_harness_aec",
            "VOICE_HARNESS_VOICE": "/home/test/reference.wav",
            "VOICE_HARNESS_WAKE_THRESHOLD": "0.6",
            "VOICE_HARNESS_MIN_SPEECH_RMS": "1000",
            "VOICE_HARNESS_BARGE_IN_MODE": "vad",
            "VOICE_HARNESS_BARGE_IN_SPEECH_FRAMES": "3",
            "VOICE_HARNESS_PLAYBACK_QUIET_FRAMES": "2",
            "VOICE_HARNESS_PLAYBACK_QUIET_TIMEOUT_SECONDS": "1.5",
            "VOICE_HARNESS_PLAYBACK_LATENCY": "100ms",
            "VOICE_HARNESS_CURSOR_FOREGROUND_SECONDS": "5",
            "VOICE_HARNESS_HERDR_BIN": "/home/test/.local/bin/herdr",
            "VOICE_HARNESS_PROJECT_ROOT": "/home/test/projects",
            "VOICE_HARNESS_GITHUB_ROOT": "/home/test/projects/forks",
            "DICTATION_INJECT": "paste",
            "DICTATION_REPLACEMENTS": "cursor=Cursor",
        }
        completed = self.audit_fixture(name, environment)

        with (
            mock.patch.object(service_units.subprocess, "run", side_effect=completed),
            mock.patch("builtins.print"),
        ):
            result = service_units.audit_installed([name])

        self.assertEqual(result, 0)

    def test_installed_audit_rejects_unknown_and_protected_wake_overrides(
        self,
    ) -> None:
        name = "voice-harness-wake.service"
        environment = {
            **service_units.REQUIRED_ENVIRONMENT[name],
            "LD_PRELOAD": "/tmp/inject.so",
            "DICTATION_SOCKET": "/tmp/dictation.sock",
            "CUDA_CACHE_PATH": "/tmp/cuda",
            "XDG_RUNTIME_DIR": "/tmp/runtime",
            "VOICE_HARNESS_LLM_HOST": "0.0.0.0",
        }
        completed = self.audit_fixture(name, environment)

        with (
            mock.patch.object(service_units.subprocess, "run", side_effect=completed),
            mock.patch("builtins.print") as output,
        ):
            result = service_units.audit_installed([name])

        self.assertEqual(result, 1)
        messages = " ".join(str(call) for call in output.call_args_list)
        for variable in environment.keys() - service_units.REQUIRED_ENVIRONMENT[name]:
            self.assertIn(variable, messages)

    def test_installed_audit_rejects_changed_required_environment(self) -> None:
        cases = {
            "dictation.service": ("DICTATION_SOCKET", "/tmp/dictation.sock"),
            "voice-harness-llm.service": ("CUDA_CACHE_PATH", "/tmp/llm-cache"),
            "voice-harness-tts.service": ("CUDA_CACHE_PATH", "/tmp/tts-cache"),
            "voice-harness-wake.service": ("PYTHONUNBUFFERED", "0"),
        }
        for name, (variable, value) in cases.items():
            with self.subTest(service=name, variable=variable):
                environment = {
                    **service_units.REQUIRED_ENVIRONMENT[name],
                    variable: value,
                }
                completed = self.audit_fixture(name, environment)
                with (
                    mock.patch.object(
                        service_units.subprocess, "run", side_effect=completed
                    ),
                    mock.patch("builtins.print") as output,
                ):
                    result = service_units.audit_installed([name])

                self.assertEqual(result, 1)
                self.assertIn(
                    variable, " ".join(str(call) for call in output.call_args_list)
                )

    def test_installed_audit_validates_security_relevant_wake_values(self) -> None:
        name = "voice-harness-wake.service"
        environment = {
            **service_units.REQUIRED_ENVIRONMENT[name],
            "VOICE_HARNESS_VOICE": "relative.wav",
            "VOICE_HARNESS_WAKE_THRESHOLD": "2",
            "VOICE_HARNESS_BARGE_IN_MODE": "always",
            "VOICE_HARNESS_BARGE_IN_SPEECH_FRAMES": "0",
            "VOICE_HARNESS_PROJECT_ROOT": "/home/test/projects",
            "VOICE_HARNESS_GITHUB_ROOT": "/home/test/projects/../outside",
        }
        completed = self.audit_fixture(name, environment)

        with (
            mock.patch.object(service_units.subprocess, "run", side_effect=completed),
            mock.patch("builtins.print") as output,
        ):
            result = service_units.audit_installed([name])

        self.assertEqual(result, 1)
        messages = " ".join(str(call) for call in output.call_args_list)
        for variable in environment.keys() - service_units.REQUIRED_ENVIRONMENT[name]:
            self.assertIn(variable, messages)

    def test_installed_audit_rejects_effective_weakening_override(self) -> None:
        name = "voice-harness-llm.service"
        source_unit = (
            service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE / name
        ).read_text()
        weakened_command = service_units._expanded_unit_value(
            service_units.EXPECTED_EXECSTART[name]
        ).replace("--host 127.0.0.1", "--host 0.0.0.0")
        unit = (
            source_unit + "\n# weakening drop-in\n[Service]\n"
            "NoNewPrivileges=false\n"
            "MemoryMax=infinity\n"
            "ExecStart=\n"
            f"ExecStart={weakened_command}\n"
        )
        properties = {
            **service_units._audit_expected_properties(name),
            "ActiveState": "active",
            "SubState": "running",
            "Result": "success",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "MemoryCurrent": "1",
            "TasksCurrent": "1",
            "Environment": ("CUDA_CACHE_PATH=%t/voice-harness-llm/cuda-cache"),
            "MemoryMax": "infinity",
            "NoNewPrivileges": "no",
            "ExecStart": (
                f"{{ path=/usr/sbin/llama-server ; argv[]={weakened_command} ; "
                "ignore_errors=no ; }}"
            ),
        }
        shown = "\n".join(f"{key}={value}" for key, value in properties.items())
        completed = [
            subprocess.CompletedProcess([], 0, unit, ""),
            subprocess.CompletedProcess([], 0, shown, ""),
        ]

        with (
            mock.patch.object(service_units.subprocess, "run", side_effect=completed),
            mock.patch("builtins.print") as output,
        ):
            result = service_units.audit_installed([name])

        self.assertEqual(result, 1)
        messages = [str(call) for call in output.call_args_list]
        self.assertTrue(any("MemoryMax" in message for message in messages))
        self.assertTrue(any("NoNewPrivileges" in message for message in messages))
        self.assertTrue(any("ExecStart" in message for message in messages))

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
