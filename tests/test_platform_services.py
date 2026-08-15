from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import (
    components,
    config_activation,
    config_management,
    desktop,
    service_manager,
)
from local_voice_harness.credentials import SecretServiceStore
from local_voice_harness.diagnostics import checks
from local_voice_harness.diagnostics.model import Severity
from local_voice_harness.integrations.herdr import transport as herdr_transport
from local_voice_harness.notifications import NotifySendService
from local_voice_harness.platform_services import (
    SystemdUserSupervisor,
    linux_platform,
)

ORCHESTRATION_MODULES = (
    components,
    config_activation,
    config_management,
    desktop,
    service_manager,
    checks,
    herdr_transport,
)


class PlatformServiceBoundaryTests(unittest.TestCase):
    def test_linux_platform_exposes_service_secret_and_notification_impls(self) -> None:
        platform = linux_platform()
        self.assertIsInstance(platform.services, SystemdUserSupervisor)
        self.assertIsInstance(platform.credentials, SecretServiceStore)
        self.assertIsInstance(platform.notifications, NotifySendService)

    def test_systemd_supervisor_owns_systemctl_invocation(self) -> None:
        from local_voice_harness import platform_services

        supervisor = SystemdUserSupervisor()
        completed = subprocess.CompletedProcess([], 0, "active\n", "")
        with mock.patch.object(
            platform_services.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(
                supervisor.is_active("voice-harness-wake.service"), "active"
            )
            supervisor.start("voice-harness-tts.service")
            supervisor.start_transient("voice-harness-herdr", ("herdr", "server"))

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands[0][:3],
            ["systemctl", "--user", "is-active"],
        )
        self.assertEqual(
            commands[1],
            ["systemctl", "--user", "start", "voice-harness-tts.service"],
        )
        self.assertEqual(
            commands[2][:4],
            ["systemd-run", "--user", "--unit=voice-harness-herdr", "--collect"],
        )

    def test_systemd_availability_probes_user_bus_with_timeout(self) -> None:
        from local_voice_harness import platform_services

        supervisor = SystemdUserSupervisor()
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            mock.patch.object(
                platform_services.shutil, "which", return_value="/usr/bin/systemctl"
            ),
            mock.patch.object(
                platform_services.subprocess, "run", return_value=completed
            ) as run,
        ):
            self.assertTrue(supervisor.available())

        self.assertEqual(
            run.call_args.args[0],
            ["systemctl", "--user", "show-environment"],
        )
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            supervisor.CAPABILITY_PROBE_TIMEOUT_SECONDS,
        )

    def test_systemd_availability_rejects_unusable_or_stalled_user_bus(self) -> None:
        from local_voice_harness import platform_services

        supervisor = SystemdUserSupervisor()
        with (
            mock.patch.object(
                platform_services.shutil, "which", return_value="/usr/bin/systemctl"
            ),
            mock.patch.object(
                platform_services.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("systemctl", 2),
            ),
        ):
            self.assertFalse(supervisor.available())

    def test_orchestration_does_not_invoke_platform_tools_directly(self) -> None:
        forbidden = ('["systemctl"', "['systemctl'", '"secret-tool"', '"notify-send"')
        for module in ORCHESTRATION_MODULES:
            filename = module.__file__
            self.assertIsNotNone(filename)
            assert filename is not None
            path = Path(filename)
            source = path.read_text()
            for token in forbidden:
                self.assertNotIn(
                    token,
                    source,
                    f"{path.name} still constructs {token}",
                )


class PlatformCapabilityDiagnosticTests(unittest.TestCase):
    def test_missing_features_are_reported_not_required(self) -> None:
        supervisor = mock.Mock()
        supervisor.available.return_value = False
        supervisor.binary_available.return_value = True
        backend = desktop.DegradedWaylandDesktop(
            "gnome",
            "GNOME Wayland does not expose a supported focused-window or "
            "keyboard-injection API; dictation insertion degrades to stdout",
        )
        with (
            mock.patch.object(checks, "user_services", return_value=supervisor),
            mock.patch.object(checks, "secret_service_available", return_value=False),
            mock.patch.object(
                checks, "secret_service_binary_available", return_value=True
            ),
            mock.patch.object(
                checks, "notification_service_available", return_value=False
            ),
            mock.patch.object(
                checks, "notification_service_binary_available", return_value=True
            ),
            mock.patch.object(checks, "get_desktop", return_value=backend),
            mock.patch.object(desktop.shutil, "which", return_value=None),
        ):
            results = checks.check_platform_capabilities()
            units = checks.check_systemd_units()
            focus = checks.check_focus_automation()

        names = {result.name: result for result in results}
        self.assertEqual(names["platform:services"].severity, Severity.WARNING)
        self.assertEqual(names["platform:credentials"].severity, Severity.WARNING)
        self.assertEqual(names["platform:notifications"].severity, Severity.WARNING)
        self.assertEqual(names["platform:desktop"].severity, Severity.WARNING)
        self.assertIn("installed", names["platform:services"].detail)
        self.assertIn("installed", names["platform:credentials"].detail)
        self.assertIn("installed", names["platform:notifications"].detail)
        self.assertIn("GNOME", names["platform:desktop"].detail)
        self.assertEqual(units[0].severity, Severity.WARNING)
        self.assertNotIn("not installed", units[0].detail)
        self.assertEqual(focus[0].severity, Severity.WARNING)
        self.assertIn("GNOME", focus[0].detail)

    def test_supported_desktops_remain_ok(self) -> None:
        supervisor = mock.Mock()
        supervisor.available.return_value = True
        with (
            mock.patch.object(checks, "user_services", return_value=supervisor),
            mock.patch.object(checks, "secret_service_available", return_value=True),
            mock.patch.object(
                checks, "notification_service_available", return_value=True
            ),
            mock.patch.object(checks, "get_desktop", return_value=desktop.X11Desktop()),
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"),
        ):
            results = checks.check_platform_capabilities()
            focus = checks.check_focus_automation()

        names = {result.name: result for result in results}
        self.assertTrue(all(result.severity is Severity.OK for result in results))
        self.assertEqual(focus[0].severity, Severity.OK)
        self.assertIn("X11", names["platform:desktop"].detail)


class DesktopCapabilityTests(unittest.TestCase):
    def test_x11_hyprland_and_sway_report_full_capabilities(self) -> None:
        with mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"):
            x11 = desktop.X11Desktop().capabilities()
            hyprland = desktop.HyprlandDesktop().capabilities()
            sway = desktop.SwayDesktop().capabilities()

        self.assertTrue(x11.active_window and x11.type_text and x11.overlay)
        self.assertTrue(
            hyprland.active_window and hyprland.send_key and hyprland.overlay
        )
        self.assertTrue(sway.active_window and sway.clipboard and sway.overlay)
        self.assertEqual(x11.name, "x11")
        self.assertEqual(hyprland.name, "hyprland")
        self.assertEqual(sway.name, "sway")


if __name__ == "__main__":
    unittest.main()
