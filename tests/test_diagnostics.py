from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import cli
from local_voice_harness.diagnostics import checks, runner
from local_voice_harness.diagnostics.model import (
    PROBLEM_SEVERITIES,
    SEVERITY_RANK,
    CheckResult,
    Repair,
    Severity,
)
from local_voice_harness.integrations.linear import CapabilityStatus
from local_voice_harness.user_config import (
    UserConfigurationError,
    default_user_config,
)


def _result(severity: Severity, name: str = "x", **kwargs: object) -> CheckResult:
    return CheckResult(
        name=name,
        category="test",
        severity=severity,
        detail="detail",
        **kwargs,  # type: ignore[arg-type]
    )


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["stub"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _snapshot():
    config = default_user_config()
    return checks.DiagnosticSnapshot(
        config=config,
        registry=checks.build_integration_registry(config),
    )


class ModelTests(unittest.TestCase):
    def test_severity_rank_orders_fatal_first(self) -> None:
        ordered = sorted(Severity, key=lambda s: SEVERITY_RANK[s])
        self.assertEqual(ordered[0], Severity.FATAL)

    def test_check_result_to_dict_serializes_repair_summary(self) -> None:
        repair = Repair(summary="restart x", action=lambda: "done")
        result = _result(Severity.WARNING, suggestion="do it", repair=repair)
        payload = result.to_dict()
        self.assertEqual(payload["severity"], "warning")
        self.assertEqual(payload["repair"], "restart x")
        self.assertEqual(payload["suggestion"], "do it")

    def test_problem_severities_exclude_ok(self) -> None:
        self.assertNotIn(Severity.OK, PROBLEM_SEVERITIES)
        self.assertIn(Severity.FATAL, PROBLEM_SEVERITIES)


class RunnerTests(unittest.TestCase):
    def test_run_diagnostics_converts_crash_into_fatal(self) -> None:
        def boom() -> list[CheckResult]:
            raise RuntimeError("kaboom")

        results = runner.run_diagnostics([boom])
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertIn("kaboom", results[0].detail)

    def test_run_diagnostics_redacts_crash_details(self) -> None:
        def boom() -> list[CheckResult]:
            raise RuntimeError("token=supersecretvalue")

        results = runner.run_diagnostics([boom])

        self.assertNotIn("supersecretvalue", results[0].detail)
        self.assertIn("[REDACTED]", results[0].detail)

    def test_run_diagnostics_aggregates_all_checks(self) -> None:
        results = runner.run_diagnostics(
            [
                lambda: [_result(Severity.OK, "a")],
                lambda: [_result(Severity.WARNING, "b")],
            ]
        )
        self.assertEqual([r.name for r in results], ["a", "b"])

    def test_run_diagnostics_resolves_one_snapshot_for_every_check(self) -> None:
        snapshot = _snapshot()
        observed: list[checks.DiagnosticSnapshot] = []

        def observe(value: checks.DiagnosticSnapshot) -> list[CheckResult]:
            observed.append(value)
            return []

        with mock.patch.object(
            checks.DiagnosticSnapshot, "load", return_value=snapshot
        ) as load:
            runner.run_diagnostics([observe, observe])

        load.assert_called_once_with()
        self.assertEqual(observed, [snapshot, snapshot])

    def test_malformed_config_is_one_direct_fatal_beside_independent_checks(
        self,
    ) -> None:
        snapshot = checks.DiagnosticSnapshot(
            config=None,
            registry=None,
            error=UserConfigurationError("bad config.toml"),
        )
        results = runner.run_diagnostics(
            [
                checks.check_backend_configuration,
                lambda _snapshot: [_result(Severity.OK, "independent")],
                checks.check_model_file,
                checks.check_optional_integrations,
            ],
            snapshot=snapshot,
        )

        self.assertEqual(
            [result.name for result in results],
            ["configuration:user", "independent"],
        )
        self.assertIn("bad config.toml", results[0].detail)

    def test_malformed_config_diagnostic_is_redacted(self) -> None:
        snapshot = checks.DiagnosticSnapshot(
            config=None,
            registry=None,
            error=UserConfigurationError("bad config.toml: token=supersecretvalue"),
        )

        results = runner.run_diagnostics(
            [checks.check_backend_configuration],
            snapshot=snapshot,
        )

        self.assertNotIn("supersecretvalue", results[0].detail)
        self.assertIn("[REDACTED]", results[0].detail)

    def test_render_json_reports_summary_and_health(self) -> None:
        results = [_result(Severity.FATAL, "a"), _result(Severity.OK, "b")]
        payload = json.loads(runner.render_json(results))
        self.assertFalse(payload["healthy"])
        self.assertEqual(payload["summary"]["fatal"], 1)
        self.assertEqual(payload["summary"]["ok"], 1)
        self.assertEqual(len(payload["checks"]), 2)

    def test_render_json_healthy_when_no_fatal(self) -> None:
        payload = json.loads(runner.render_json([_result(Severity.WARNING, "a")]))
        self.assertTrue(payload["healthy"])

    def test_render_human_groups_and_headline(self) -> None:
        repair = Repair(summary="restart svc", action=lambda: "ok")
        results = [
            _result(Severity.FATAL, "fatal-one"),
            _result(Severity.WARNING, "warn-one", repair=repair),
            _result(Severity.UNAVAILABLE, "opt-one"),
            _result(Severity.OK, "ok-one"),
        ]
        text = runner.render_human(results)
        self.assertIn("FATAL (1):", text)
        self.assertIn("WARNING (1):", text)
        self.assertIn("UNAVAILABLE (1):", text)
        self.assertIn("unhealthy", text)
        self.assertIn("Summary: 1 fatal, 1 warning, 1 unavailable, 1 ok", text)
        self.assertIn("doctor --fix", text)

    def test_render_human_healthy_message(self) -> None:
        text = runner.render_human([_result(Severity.OK, "a")])
        self.assertIn("looks healthy", text)

    def test_render_human_degraded_message(self) -> None:
        text = runner.render_human([_result(Severity.WARNING, "a")])
        self.assertIn("usable but some features are degraded", text)


class ApplyRepairTests(unittest.TestCase):
    def test_confirmed_repair_runs_and_reports(self) -> None:
        calls: list[str] = []
        repair = Repair(summary="fix it", action=lambda: calls.append("ran") or "done")
        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=repair)],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertEqual(calls, ["ran"])
        self.assertIn("repaired svc: done", out.getvalue())

    def test_declined_repair_is_skipped(self) -> None:
        ran: list[str] = []
        repair = Repair(summary="fix it", action=lambda: ran.append("x") or "done")
        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=repair)],
            confirm=lambda _prompt: False,
            out=out,
        )
        self.assertEqual(ran, [])
        self.assertIn("skipped svc", out.getvalue())

    def test_failed_repair_does_not_raise(self) -> None:
        def action() -> str:
            raise RuntimeError("nope")

        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=Repair("fix", action))],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertIn("repair failed for svc: nope", out.getvalue())

    def test_failed_repair_redacts_diagnostic(self) -> None:
        def action() -> str:
            raise RuntimeError("password=supersecretvalue")

        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=Repair("fix", action))],
            confirm=lambda _prompt: True,
            out=out,
        )

        self.assertNotIn("supersecretvalue", out.getvalue())
        self.assertIn("[REDACTED]", out.getvalue())

    def test_ok_results_with_repair_are_ignored(self) -> None:
        ran: list[str] = []
        repair = Repair(summary="fix", action=lambda: ran.append("x") or "done")
        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.OK, "svc", repair=repair)],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertEqual(ran, [])


class DoctorEntryTests(unittest.TestCase):
    def test_doctor_json_mode_returns_zero_without_fatal(self) -> None:
        out = io.StringIO()
        code = runner.doctor(
            json_output=True,
            checks=[lambda: [_result(Severity.WARNING, "a")]],
            out=out,
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["healthy"])

    def test_doctor_returns_one_on_fatal(self) -> None:
        out = io.StringIO()
        code = runner.doctor(
            checks=[lambda: [_result(Severity.FATAL, "a")]],
            out=out,
        )
        self.assertEqual(code, 1)

    def test_doctor_fix_applies_confirmed_repairs(self) -> None:
        ran: list[str] = []
        repair = Repair(summary="restart", action=lambda: ran.append("x") or "ok")
        out = io.StringIO()
        code = runner.doctor(
            fix=True,
            checks=[lambda: [_result(Severity.FATAL, "svc", repair=repair)]],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertEqual(code, 1)
        self.assertEqual(ran, ["x"])
        self.assertIn("Guided recovery", out.getvalue())

    def test_doctor_fix_reports_when_no_repairs(self) -> None:
        out = io.StringIO()
        runner.doctor(
            fix=True,
            checks=[lambda: [_result(Severity.WARNING, "a")]],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertIn("No confirmation-gated repairs", out.getvalue())

    def test_doctor_json_mode_never_prompts_for_repairs(self) -> None:
        repair = Repair(summary="restart", action=lambda: "ok")
        confirm = mock.Mock(return_value=True)
        out = io.StringIO()
        runner.doctor(
            json_output=True,
            fix=True,
            checks=[lambda: [_result(Severity.FATAL, "svc", repair=repair)]],
            confirm=confirm,
            out=out,
        )
        confirm.assert_not_called()


class ExecutableCheckTests(unittest.TestCase):
    def test_missing_required_executable_is_fatal(self) -> None:
        with mock.patch.object(checks, "_which", return_value=None):
            results = checks.check_required_executables()
        self.assertTrue(all(r.severity is Severity.FATAL for r in results))
        self.assertTrue(all(r.suggestion for r in results))

    def test_present_required_executable_is_ok(self) -> None:
        with mock.patch.object(checks, "_which", return_value="/usr/bin/tool"):
            results = checks.check_required_executables()
        self.assertTrue(all(r.severity is Severity.OK for r in results))

    def test_executable_checks_use_configured_client_paths(self) -> None:
        config = default_user_config()
        configured = replace(
            config,
            platform=replace(
                config.platform,
                herdr_bin=Path("/opt/herdr"),
                gh_bin=Path("/opt/gh"),
            ),
        )
        snapshot = checks.DiagnosticSnapshot(
            configured,
            checks.build_integration_registry(configured),
        )

        with mock.patch.object(
            checks, "_which", return_value="/configured/tool"
        ) as which:
            checks.check_required_executables(snapshot)
            checks.check_optional_executables(snapshot)

        self.assertIn(mock.call("/opt/herdr"), which.call_args_list)
        self.assertIn(mock.call("/opt/gh"), which.call_args_list)

    def test_missing_optional_executable_is_warning(self) -> None:
        with mock.patch.object(checks, "_which", return_value=None):
            results = checks.check_optional_executables()
        self.assertTrue(all(r.severity is Severity.WARNING for r in results))

    def test_focus_automation_ok_when_x11_stack_present(self) -> None:
        def which(name: str) -> str | None:
            return "/usr/bin/tool" if name in {"xdotool", "xclip"} else None

        with mock.patch.object(checks, "_which", side_effect=which):
            results = checks.check_focus_automation()
        self.assertIs(results[0].severity, Severity.OK)
        self.assertIn("X11", results[0].detail)

    def test_focus_automation_ok_when_wayland_stack_present(self) -> None:
        def which(name: str) -> str | None:
            return "/usr/bin/tool" if name in {"wtype", "wl-copy", "wl-paste"} else None

        with mock.patch.object(checks, "_which", side_effect=which):
            results = checks.check_focus_automation()
        self.assertIs(results[0].severity, Severity.OK)

    def test_focus_automation_warns_without_any_stack(self) -> None:
        with mock.patch.object(checks, "_which", return_value=None):
            results = checks.check_focus_automation()
        self.assertIs(results[0].severity, Severity.WARNING)


def _local_tts_snapshot():
    config = default_user_config()
    config = replace(config, providers=replace(config.providers, tts_provider="local"))
    return checks.DiagnosticSnapshot(
        config=config,
        registry=checks.build_integration_registry(config),
    )


class PythonEnvironmentTests(unittest.TestCase):
    def test_present_environment_is_ok(self) -> None:
        with mock.patch.object(Path, "exists", return_value=True):
            results = checks.check_python_environments(_local_tts_snapshot())
        self.assertTrue(all(r.severity is Severity.OK for r in results))
        self.assertTrue(any("chatterbox" in result.name for result in results))

    def test_missing_environment_is_fatal(self) -> None:
        with mock.patch.object(Path, "exists", return_value=False):
            results = checks.check_python_environments(_local_tts_snapshot())
        self.assertTrue(all(r.severity is Severity.FATAL for r in results))
        self.assertTrue(all(r.suggestion for r in results))

    def test_incomplete_environment_is_warning(self) -> None:
        def exists(self: Path) -> bool:
            return "bin" not in self.parts

        with mock.patch.object(Path, "exists", exists):
            results = checks.check_python_environments(_local_tts_snapshot())
        self.assertTrue(all(r.severity is Severity.WARNING for r in results))

    def test_hosted_tts_skips_local_chatterbox_environment(self) -> None:
        with mock.patch.object(Path, "exists", return_value=False):
            results = checks.check_python_environments(_snapshot())
        self.assertFalse(any("chatterbox" in result.name for result in results))
        self.assertTrue(results)


class ModelAndCudaTests(unittest.TestCase):
    def test_model_present(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            model.write_bytes(b"0" * 1024)
            with mock.patch.object(checks, "MODEL_FILE", model):
                results = checks.check_model_file(_snapshot())
        self.assertIs(results[0].severity, Severity.OK)

    def test_model_missing_is_fatal(self) -> None:
        snapshot = _snapshot()
        assert snapshot.config is not None
        config = snapshot.config
        snapshot = replace(
            snapshot,
            config=replace(
                config,
                providers=replace(config.providers, llm_provider="local"),
            ),
        )
        with mock.patch.object(checks, "MODEL_FILE", Path("/nonexistent/model.gguf")):
            results = checks.check_model_file(snapshot)
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertIn("hf download", results[0].suggestion or "")

    def test_model_cache_present(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(checks, "HUGGINGFACE_CACHE", Path(temporary)):
                results = checks.check_model_caches()
        self.assertIs(results[0].severity, Severity.OK)

    def test_model_cache_absent_is_warning(self) -> None:
        with mock.patch.object(checks, "HUGGINGFACE_CACHE", Path("/nonexistent/cache")):
            results = checks.check_model_caches()
        self.assertIs(results[0].severity, Severity.WARNING)

    def test_cuda_missing_nvidia_smi_is_warning(self) -> None:
        with mock.patch.object(checks, "_which", return_value=None):
            results = checks.check_cuda()
        self.assertIs(results[0].severity, Severity.WARNING)

    def test_cuda_ok_when_nvidia_smi_succeeds(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(checks, "_run", return_value=_completed(0, "GPU")),
        ):
            results = checks.check_cuda()
        self.assertIs(results[0].severity, Severity.OK)

    def test_cuda_warns_when_nvidia_smi_fails(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(
                checks, "_run", return_value=_completed(1, "", "driver error")
            ),
        ):
            results = checks.check_cuda()
        self.assertIs(results[0].severity, Severity.WARNING)


_WPCTL_STATUS = """\
Audio
 ├─ Devices:
 │      40. alsa_card.pci-0000_00_1f.3
 ├─ Sinks:
 │  *   62. alsa_output.pci-0000_00_1f.3.analog-stereo [vol: 0.50]
 ├─ Sources:
 │  *   63. alsa_input.pci-0000_00_1f.3.analog-stereo [vol: 1.00]
 ├─ Filters:
"""

_WPCTL_FILTER_STATUS = _WPCTL_STATUS.replace(
    " ├─ Filters:\n",
    " ├─ Filters:\n │      72. voice_harness_aec_sink\n │  *   73. voice_harness_aec\n",
)


class PipewireTests(unittest.TestCase):
    def test_terminal_section_header_is_parsed(self) -> None:
        status = """\
Audio
 ├─ Sinks:
 │  *   62. output.node
 └─ Sources:
    *   63. input.node
"""
        self.assertEqual(
            checks.pipewire_section_devices(status, "Sources"),
            ("input.node",),
        )

    def test_missing_wpctl_is_fatal(self) -> None:
        with mock.patch.object(checks, "_which", return_value=None):
            results = checks.check_pipewire_devices(_snapshot())
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertIn("system default source", results[0].detail)

    def test_wpctl_ok_uses_system_default_when_unconfigured(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(
                checks, "_run", return_value=_completed(0, _WPCTL_STATUS)
            ) as run,
        ):
            results = checks.check_pipewire_devices(_snapshot())
        self.assertIs(results[0].severity, Severity.OK)
        self.assertIn("system default source", results[0].detail)
        self.assertIn("system default sink", results[0].detail)
        run.assert_called_once_with(["wpctl", "status", "--name"], timeout=5)

    def test_configured_node_names_match_capture_targets(self) -> None:
        snapshot = _snapshot()
        assert snapshot.config is not None
        snapshot = checks.DiagnosticSnapshot(
            config=replace(
                snapshot.config,
                audio=replace(
                    snapshot.config.audio,
                    source="alsa_input.pci-0000_00_1f.3.analog-stereo",
                    sink="alsa_output.pci-0000_00_1f.3.analog-stereo",
                ),
            ),
            registry=snapshot.registry,
        )
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(
                checks, "_run", return_value=_completed(0, _WPCTL_STATUS)
            ),
        ):
            results = checks.check_pipewire_devices(snapshot)
        self.assertIs(results[0].severity, Severity.OK)

    def test_configured_filter_node_matches_virtual_capture_target(self) -> None:
        snapshot = _snapshot()
        assert snapshot.config is not None
        snapshot = checks.DiagnosticSnapshot(
            config=replace(
                snapshot.config,
                audio=replace(snapshot.config.audio, source="voice_harness_aec"),
            ),
            registry=snapshot.registry,
        )
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(
                checks, "_run", return_value=_completed(0, _WPCTL_FILTER_STATUS)
            ),
        ):
            results = checks.check_pipewire_devices(snapshot)
        self.assertIs(results[0].severity, Severity.OK)

    def test_missing_configured_dictation_source_is_fatal(self) -> None:
        snapshot = _snapshot()
        assert snapshot.config is not None
        snapshot = checks.DiagnosticSnapshot(
            config=replace(
                snapshot.config,
                dictation=replace(
                    snapshot.config.dictation,
                    source="missing-dictation-mic",
                ),
            ),
            registry=snapshot.registry,
        )
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(
                checks, "_run", return_value=_completed(0, _WPCTL_STATUS)
            ),
        ):
            results = checks.check_pipewire_devices(snapshot)
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertIn("missing-dictation-mic", results[0].detail)
        self.assertIn("dictation.source", results[0].suggestion or "")

    def test_wpctl_failure_is_fatal(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(checks, "_run", return_value=_completed(1)),
        ):
            results = checks.check_pipewire_devices(_snapshot())
        self.assertIs(results[0].severity, Severity.FATAL)

    def test_missing_configured_source_is_fatal(self) -> None:
        snapshot = _snapshot()
        assert snapshot.config is not None
        snapshot = checks.DiagnosticSnapshot(
            config=replace(
                snapshot.config,
                audio=replace(snapshot.config.audio, source="missing-mic"),
            ),
            registry=snapshot.registry,
        )
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(
                checks, "_run", return_value=_completed(0, _WPCTL_STATUS)
            ),
        ):
            results = checks.check_pipewire_devices(snapshot)
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertIn("missing-mic", results[0].detail)


class SystemdUnitTests(unittest.TestCase):
    def _run_with(
        self, tmp: Path, installed: set[str], props: dict[str, dict[str, str]]
    ) -> list[CheckResult]:
        for name in installed:
            (tmp / name).write_text("[Unit]\n")

        def show(name: str, _properties: object) -> dict[str, str]:
            return props.get(name, {})

        with (
            mock.patch.object(checks, "SYSTEMD_USER_DIR", tmp),
            mock.patch.object(checks, "_systemctl_show", side_effect=show),
        ):
            return checks.check_systemd_units()

    def test_not_installed_is_fatal(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            results = self._run_with(Path(temporary), set(), {})
        self.assertTrue(all(r.severity is Severity.FATAL for r in results))
        self.assertTrue(all("not installed" in r.detail for r in results))

    def test_restart_loop_is_fatal_with_repair(self) -> None:
        import tempfile

        props = {
            name: {"ActiveState": "activating", "Result": "success", "NRestarts": "9"}
            for name in checks.SERVICE_FILES
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = self._run_with(Path(temporary), set(checks.SERVICE_FILES), props)
        self.assertTrue(all(r.severity is Severity.FATAL for r in results))
        self.assertTrue(all(r.repair is not None for r in results))

    def test_failed_unit_is_fatal(self) -> None:
        import tempfile

        props = {
            name: {
                "ActiveState": "failed",
                "SubState": "failed",
                "Result": "exit-code",
                "NRestarts": "0",
            }
            for name in checks.SERVICE_FILES
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = self._run_with(Path(temporary), set(checks.SERVICE_FILES), props)
        self.assertTrue(all(r.severity is Severity.FATAL for r in results))

    def test_always_on_inactive_is_warning(self) -> None:
        import tempfile

        props = {
            name: {
                "ActiveState": "inactive",
                "SubState": "dead",
                "Result": "success",
                "NRestarts": "0",
                "UnitFileState": "enabled",
            }
            for name in checks.SERVICE_FILES
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = self._run_with(Path(temporary), set(checks.SERVICE_FILES), props)
        by_name = {r.name: r for r in results}
        self.assertIs(
            by_name["unit:voice-harness-wake.service"].severity, Severity.WARNING
        )
        # on-demand services inactive is healthy
        self.assertIs(by_name["unit:voice-harness-llm.service"].severity, Severity.OK)

    def test_disabled_always_on_unit_is_warning(self) -> None:
        import tempfile

        props = {
            name: {
                "ActiveState": "active",
                "SubState": "running",
                "Result": "success",
                "NRestarts": "0",
                "UnitFileState": "disabled",
            }
            for name in checks.SERVICE_FILES
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = self._run_with(Path(temporary), set(checks.SERVICE_FILES), props)
        by_name = {r.name: r for r in results}
        self.assertIs(by_name["unit:dictation.service"].severity, Severity.WARNING)

    def test_healthy_units_are_ok(self) -> None:
        import tempfile

        props = {
            name: {
                "ActiveState": "active",
                "SubState": "running",
                "Result": "success",
                "NRestarts": "0",
                "UnitFileState": "enabled",
            }
            for name in checks.SERVICE_FILES
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = self._run_with(Path(temporary), set(checks.SERVICE_FILES), props)
        self.assertTrue(all(r.severity is Severity.OK for r in results))


class SocketCheckTests(unittest.TestCase):
    def test_socket_ready_is_ok(self) -> None:
        with mock.patch.object(checks, "socket_ready", return_value=True):
            result = checks._socket_result(
                name="socket:stt",
                label="STT",
                socket_path=Path("/run/x.sock"),
                unit="dictation.service",
                always_on=True,
            )
        self.assertIs(result.severity, Severity.OK)

    def test_socket_down_while_service_active_offers_restart(self) -> None:
        with (
            mock.patch.object(checks, "socket_ready", return_value=False),
            mock.patch.object(checks, "_service_active_state", return_value="active"),
        ):
            result = checks._socket_result(
                name="socket:stt",
                label="STT",
                socket_path=Path("/run/x.sock"),
                unit="dictation.service",
                always_on=True,
            )
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIsNotNone(result.repair)
        self.assertIn("restart", (result.repair.summary if result.repair else ""))

    def test_stale_socket_offers_removal(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "x.sock"
            socket_path.write_text("stale")
            with (
                mock.patch.object(checks, "socket_ready", return_value=False),
                mock.patch.object(
                    checks, "_service_active_state", return_value="inactive"
                ),
            ):
                result = checks._socket_result(
                    name="socket:tts",
                    label="TTS",
                    socket_path=socket_path,
                    unit="voice-harness-tts.service",
                    always_on=False,
                )
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("stale", result.detail)
        self.assertIsNotNone(result.repair)

    def test_absent_on_demand_socket_is_ok(self) -> None:
        with (
            mock.patch.object(checks, "socket_ready", return_value=False),
            mock.patch.object(checks, "_service_active_state", return_value="inactive"),
        ):
            result = checks._socket_result(
                name="socket:tts",
                label="TTS",
                socket_path=Path("/nonexistent/x.sock"),
                unit="voice-harness-tts.service",
                always_on=False,
            )
        self.assertIs(result.severity, Severity.OK)

    def test_absent_always_on_socket_is_warning(self) -> None:
        with (
            mock.patch.object(checks, "socket_ready", return_value=False),
            mock.patch.object(checks, "_service_active_state", return_value="inactive"),
        ):
            result = checks._socket_result(
                name="socket:stt",
                label="STT",
                socket_path=Path("/nonexistent/x.sock"),
                unit="dictation.service",
                always_on=True,
            )
        self.assertIs(result.severity, Severity.WARNING)

    def test_llm_ready_reports_ok(self) -> None:
        with (
            mock.patch.object(checks, "socket_ready", return_value=True),
            mock.patch.object(checks, "llm_ready", return_value=True),
        ):
            results = checks.check_runtime_sockets()
        by_name = {r.name: r for r in results}
        self.assertIs(by_name["socket:llm"].severity, Severity.OK)

    def test_llm_down_while_active_offers_restart(self) -> None:
        with (
            mock.patch.object(checks, "socket_ready", return_value=True),
            mock.patch.object(checks, "llm_ready", return_value=False),
            mock.patch.object(checks, "_service_active_state", return_value="active"),
        ):
            results = checks.check_runtime_sockets()
        by_name = {r.name: r for r in results}
        self.assertIs(by_name["socket:llm"].severity, Severity.WARNING)
        self.assertIsNotNone(by_name["socket:llm"].repair)


class RepairActionTests(unittest.TestCase):
    def test_restart_service_repair_success(self) -> None:
        with mock.patch.object(checks, "_run", return_value=_completed(0)):
            repair = checks._restart_service_repair("dictation.service")
            message = repair.action()
        self.assertIn("restarted dictation.service", message)

    def test_restart_service_repair_failure_raises(self) -> None:
        with mock.patch.object(checks, "_run", return_value=_completed(1, "", "boom")):
            repair = checks._restart_service_repair("dictation.service")
            with self.assertRaises(checks.HarnessError) as caught:
                repair.action()
        self.assertIn("boom", str(caught.exception))

    def test_restart_service_repair_spawn_failure_raises(self) -> None:
        with mock.patch.object(checks, "_run", return_value=None):
            repair = checks._restart_service_repair("dictation.service")
            with self.assertRaises(checks.HarnessError):
                repair.action()

    def test_remove_stale_socket_is_idempotent(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "gone.sock"
            repair = checks._remove_stale_socket_repair(missing)
            message = repair.action()
        self.assertIn("nothing to do", message)


class RuntimeDirectoryTests(unittest.TestCase):
    def test_writable_runtime_is_ok(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(checks, "RUNTIME", Path(temporary)),
                mock.patch.object(checks, "STATE_DIR", Path(temporary) / "state"),
                mock.patch.object(checks, "JOBS_DIR", Path(temporary) / "jobs"),
            ):
                results = checks.check_runtime_directories()
        by_name = {r.name: r for r in results}
        self.assertIs(by_name["runtime:xdg"].severity, Severity.OK)

    def test_missing_runtime_is_fatal(self) -> None:
        with (
            mock.patch.object(checks, "RUNTIME", Path("/nonexistent/runtime")),
            mock.patch.object(checks, "STATE_DIR", Path("/nonexistent/state")),
            mock.patch.object(checks, "JOBS_DIR", Path("/nonexistent/jobs")),
        ):
            results = checks.check_runtime_directories()
        by_name = {r.name: r for r in results}
        self.assertIs(by_name["runtime:xdg"].severity, Severity.FATAL)

    def test_directory_owned_by_other_uid_is_warning(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            state.mkdir(mode=0o700)
            fake_stat = mock.Mock(st_uid=999999, st_mode=0o040700)
            with mock.patch.object(Path, "stat", return_value=fake_stat):
                result = checks._directory_result(
                    "runtime:state", state, required=False
                )
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("owned by uid", result.detail)


class CursorJobTests(unittest.TestCase):
    def _write(self, jobs_dir: Path, name: str, payload: dict[str, object]) -> None:
        (jobs_dir / name).write_text(json.dumps(payload))

    def test_no_store_is_ok(self) -> None:
        with mock.patch.object(checks, "JOBS_DIR", Path("/nonexistent/jobs")):
            results = checks.check_cursor_jobs()
        self.assertIs(results[0].severity, Severity.OK)

    def test_counts_and_flags_are_reported(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary)
            self._write(jobs, "aaaaaaaaaaaa.json", {"status": "completed"})
            self._write(jobs, "bbbbbbbbbbbb.json", {"status": "awaiting_user"})
            self._write(
                jobs,
                "cccccccccccc.json",
                {"status": "running", "updated_at": time.time() - 10_000},
            )
            (jobs / "dddddddddddd.json").write_text("{ not json")
            quarantine = jobs / ".quarantine"
            quarantine.mkdir()
            (quarantine / "eeee-abc.metadata.json").write_text("{}")
            with mock.patch.object(checks, "JOBS_DIR", jobs):
                results = checks.check_cursor_jobs()
        names = {r.name for r in results}
        self.assertIn("jobs:store", names)
        self.assertIn("jobs:attention", names)
        self.assertIn("jobs:stuck", names)
        self.assertIn("jobs:unreadable", names)
        self.assertIn("jobs:quarantine", names)
        quarantine_result = next(
            result for result in results if result.name == "jobs:quarantine"
        )
        self.assertEqual(
            quarantine_result.suggestion,
            "voice-harness jobs quarantine list",
        )

    def test_reading_jobs_does_not_mutate_files(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary)
            (jobs / "dddddddddddd.json").write_text("{ not valid json")
            before = sorted(p.name for p in jobs.iterdir())
            with mock.patch.object(checks, "JOBS_DIR", jobs):
                checks.check_cursor_jobs()
            after = sorted(p.name for p in jobs.iterdir())
        self.assertEqual(before, after)

    def test_sqlite_store_reports_path_schema_migration_and_integrity(self) -> None:
        import tempfile

        from local_voice_harness.cursor.model import CursorJob
        from local_voice_harness.cursor.store import JobStore

        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            store = JobStore(jobs, Path(temporary) / "legacy")
            store.create(
                CursorJob.from_dict(
                    {
                        "id": "123456789abc",
                        "revision": 0,
                        "request": "diagnose",
                        "status": "queued",
                        "created_at": 1,
                        "queued_at": 1,
                        "delivered": False,
                    }
                )
            )
            before = store.db_path.stat().st_mtime_ns
            with mock.patch.object(checks, "JOBS_DIR", jobs):
                results = checks.check_cursor_jobs()
            after = store.db_path.stat().st_mtime_ns

        database = next(result for result in results if result.name == "jobs:database")
        self.assertIs(database.severity, Severity.OK)
        self.assertIn(str(store.db_path), database.detail)
        self.assertIn("schema=2", database.detail)
        self.assertIn("migration=complete", database.detail)
        self.assertIn("integrity=ok", database.detail)
        self.assertEqual(after, before)

    def test_sqlite_store_reports_missing_required_named_column_as_fatal(self) -> None:
        import tempfile

        from local_voice_harness.cursor.model import CursorJob
        from local_voice_harness.cursor.store import JobStore

        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            store = JobStore(jobs, Path(temporary) / "legacy")
            store.create(
                CursorJob.from_dict(
                    {
                        "id": "123456789abc",
                        "revision": 0,
                        "request": "diagnose missing schema",
                        "status": "queued",
                        "created_at": 1,
                        "queued_at": 1,
                        "delivered": False,
                    }
                )
            )
            with sqlite3.connect(store.db_path) as connection:
                connection.execute(
                    "ALTER TABLE job_identity "
                    "DROP COLUMN grouped_repository_coordinator_id"
                )

            with mock.patch.object(checks, "JOBS_DIR", jobs):
                results = checks.check_cursor_jobs()

        self.assertEqual(len(results), 1)
        database = results[0]
        self.assertEqual(database.name, "jobs:database")
        self.assertIs(database.severity, Severity.FATAL)
        self.assertIn("named schema is incomplete", database.detail)
        self.assertIn("job_identity.grouped_repository_coordinator_id", database.detail)


class IntegrationCheckTests(unittest.TestCase):
    def test_herdr_uses_configured_snapshot_client_and_path(self) -> None:
        config = default_user_config()
        configured = replace(
            config,
            platform=replace(config.platform, herdr_bin=Path("/opt/herdr")),
        )
        client = mock.Mock()
        client.is_running.return_value = True
        registry = replace(
            checks.build_integration_registry(configured),
            herdr_client=lambda: client,
        )
        snapshot = checks.DiagnosticSnapshot(configured, registry)

        with mock.patch.object(checks, "_which", return_value="/opt/herdr") as which:
            results = checks.check_herdr(snapshot)

        which.assert_called_once_with("/opt/herdr")
        client.is_running.assert_called_once_with()
        self.assertIs(results[0].severity, Severity.OK)

    def test_github_auth_uses_configured_executable_and_timeout(self) -> None:
        config = default_user_config()
        configured = replace(
            config,
            platform=replace(
                config.platform,
                gh_bin=Path("/opt/gh"),
                github_timeout_seconds=17,
            ),
        )
        snapshot = checks.DiagnosticSnapshot(
            configured,
            checks.build_integration_registry(configured),
        )
        with (
            mock.patch.object(checks, "_which", return_value="/opt/gh"),
            mock.patch.object(
                checks, "_run", return_value=_completed(0, "logged in")
            ) as run,
        ):
            results = checks.check_github_auth(snapshot)

        run.assert_called_once_with(["/opt/gh", "auth", "status"], timeout=17)
        self.assertIs(results[0].severity, Severity.OK)

    def test_herdr_absent_binary_is_skipped(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value=None),
            mock.patch.object(Path, "exists", return_value=False),
        ):
            self.assertEqual(checks.check_herdr(_snapshot()), [])

    def test_herdr_running_is_ok(self) -> None:
        snapshot = _snapshot()
        client = mock.Mock()
        client.is_running.return_value = True
        assert snapshot.registry is not None
        snapshot = replace(
            snapshot,
            registry=replace(snapshot.registry, herdr_client=lambda: client),
        )
        with mock.patch.object(checks, "_which", return_value="/usr/bin/herdr"):
            results = checks.check_herdr(snapshot)
        self.assertIs(results[0].severity, Severity.OK)

    def test_herdr_stopped_is_warning(self) -> None:
        snapshot = _snapshot()
        client = mock.Mock()
        client.is_running.return_value = False
        assert snapshot.registry is not None
        snapshot = replace(
            snapshot,
            registry=replace(snapshot.registry, herdr_client=lambda: client),
        )
        with mock.patch.object(checks, "_which", return_value="/usr/bin/herdr"):
            results = checks.check_herdr(snapshot)
        self.assertIs(results[0].severity, Severity.WARNING)

    def test_herdr_error_is_warning(self) -> None:
        snapshot = _snapshot()
        client = mock.Mock()
        client.is_running.side_effect = checks.HerdrError("bad")
        assert snapshot.registry is not None
        snapshot = replace(
            snapshot,
            registry=replace(snapshot.registry, herdr_client=lambda: client),
        )
        with mock.patch.object(checks, "_which", return_value="/usr/bin/herdr"):
            results = checks.check_herdr(snapshot)
        self.assertIs(results[0].severity, Severity.WARNING)

    def test_cursor_cli_absent_is_skipped(self) -> None:
        with mock.patch.object(checks, "_which", return_value=None):
            self.assertEqual(checks.check_cursor_cli(), [])

    def test_cursor_cli_ok(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/agent"),
            mock.patch.object(checks, "_run", return_value=_completed(0, "1.2.3")),
        ):
            results = checks.check_cursor_cli()
        self.assertIs(results[0].severity, Severity.OK)

    def test_cursor_cli_failure_is_warning(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/agent"),
            mock.patch.object(checks, "_run", return_value=_completed(1)),
        ):
            results = checks.check_cursor_cli()
        self.assertIs(results[0].severity, Severity.WARNING)

    def test_github_absent_is_skipped(self) -> None:
        with mock.patch.object(checks, "_which", return_value=None):
            self.assertEqual(checks.check_github_auth(), [])

    def test_github_authenticated_is_ok(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/gh"),
            mock.patch.object(checks, "_run", return_value=_completed(0, "logged in")),
        ):
            results = checks.check_github_auth()
        self.assertIs(results[0].severity, Severity.OK)

    def test_github_unauthenticated_is_warning(self) -> None:
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/gh"),
            mock.patch.object(checks, "_run", return_value=_completed(1)),
        ):
            results = checks.check_github_auth()
        self.assertIs(results[0].severity, Severity.WARNING)
        self.assertIn("gh auth login", results[0].suggestion or "")

    def test_disabled_linear_has_no_diagnostic(self) -> None:
        with mock.patch.object(checks, "capability_statuses", return_value=()):
            self.assertEqual(checks.check_mcp_linear(), [])

    def test_linear_mcp_configured_is_ok(self) -> None:
        with mock.patch.object(
            checks,
            "capability_statuses",
            return_value=(("linear", CapabilityStatus(True, "available")),),
        ):
            results = checks.check_mcp_linear()
        self.assertIs(results[0].severity, Severity.OK)

    def test_enabled_linear_missing_mcp_is_fatal_and_actionable(self) -> None:
        with mock.patch.object(
            checks,
            "capability_statuses",
            return_value=(
                (
                    "linear",
                    CapabilityStatus(False, "cursor-mcp unavailable", "enable Linear"),
                ),
            ),
        ):
            results = checks.check_mcp_linear()
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertEqual(results[0].suggestion, "enable Linear")


class CliWiringTests(unittest.TestCase):
    def test_doctor_flags_parse(self) -> None:
        parsed = cli.parser().parse_args(["doctor", "--json", "--fix"])
        self.assertEqual(parsed.command, "doctor")
        self.assertTrue(parsed.json_output)
        self.assertTrue(parsed.fix)

    def test_doctor_defaults(self) -> None:
        parsed = cli.parser().parse_args(["doctor"])
        self.assertFalse(parsed.json_output)
        self.assertFalse(parsed.fix)

    def test_dispatch_invokes_doctor_and_exits_with_code(self) -> None:
        parsed = cli.parser().parse_args(["doctor", "--json"])
        with mock.patch.object(cli, "doctor", return_value=3) as doctor:
            with self.assertRaises(SystemExit) as exited:
                cli.dispatch(parsed)
        doctor.assert_called_once_with(json_output=True, fix=False)
        self.assertEqual(exited.exception.code, 3)

    def test_all_checks_run_without_crashing(self) -> None:
        results = runner.run_diagnostics()
        self.assertTrue(results)
        self.assertFalse(
            any(r.category == "internal" for r in results),
            "a real check crashed",
        )


if __name__ == "__main__":
    unittest.main()
