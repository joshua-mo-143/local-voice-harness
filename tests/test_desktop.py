from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from local_voice_harness import desktop


def completed(
    stdout: str = "",
    *,
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


class DesktopSelectionTests(unittest.TestCase):
    def test_defaults_to_x11_without_wayland_environment(self) -> None:
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(desktop, "_run", return_value=None),
        ):
            self.assertIsInstance(desktop.get_desktop(), desktop.X11Desktop)

    def test_recovers_x11_environment_imported_after_service_start(self) -> None:
        supervisor = mock.Mock()
        supervisor.user_environment.return_value = {
            "DISPLAY": ":0",
            "XAUTHORITY": "/run/user/1000/xauth",
            "UNRELATED_SECRET": "do-not-import",
        }
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(desktop, "user_services", return_value=supervisor),
        ):
            self.assertIsInstance(desktop.get_desktop(), desktop.X11Desktop)
            self.assertEqual(desktop.os.environ["DISPLAY"], ":0")
            self.assertEqual(desktop.os.environ["XAUTHORITY"], "/run/user/1000/xauth")
            self.assertNotIn("UNRELATED_SECRET", desktop.os.environ)

        supervisor.user_environment.assert_called_once_with()

    def test_existing_graphical_environment_is_not_replaced(self) -> None:
        with (
            mock.patch.dict(
                "os.environ",
                {"DISPLAY": ":1", "XAUTHORITY": "/existing"},
                clear=True,
            ),
            mock.patch.object(desktop, "_run") as run,
        ):
            self.assertIsInstance(desktop.get_desktop(), desktop.X11Desktop)
            self.assertEqual(desktop.os.environ["DISPLAY"], ":1")

        run.assert_not_called()

    def test_selects_hyprland(self) -> None:
        environment = {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-1",
            "HYPRLAND_INSTANCE_SIGNATURE": "instance",
        }
        with mock.patch.dict("os.environ", environment, clear=True):
            self.assertIsInstance(desktop.get_desktop(), desktop.HyprlandDesktop)

    def test_selects_sway(self) -> None:
        environment = {"WAYLAND_DISPLAY": "wayland-1", "SWAYSOCK": "/run/sway.sock"}
        with mock.patch.dict("os.environ", environment, clear=True):
            self.assertIsInstance(desktop.get_desktop(), desktop.SwayDesktop)

    def test_gnome_wayland_degrades_without_injection(self) -> None:
        environment = {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-1",
            "XDG_CURRENT_DESKTOP": "GNOME",
        }
        with mock.patch.dict("os.environ", environment, clear=True):
            backend = desktop.get_desktop()
        self.assertIsInstance(backend, desktop.DegradedWaylandDesktop)
        assert backend is not None
        capabilities = backend.capabilities()
        self.assertEqual(capabilities.name, "gnome")
        self.assertFalse(capabilities.active_window)
        self.assertFalse(capabilities.type_text)
        self.assertFalse(capabilities.overlay)
        with self.assertRaisesRegex(desktop.DesktopError, "GNOME Wayland"):
            backend.type_text("hello")

    def test_kde_wayland_degrades_without_injection(self) -> None:
        environment = {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-1",
            "XDG_CURRENT_DESKTOP": "KDE",
        }
        with mock.patch.dict("os.environ", environment, clear=True):
            backend = desktop.get_desktop()
        self.assertIsInstance(backend, desktop.DegradedWaylandDesktop)
        assert backend is not None
        capabilities = backend.capabilities()
        self.assertEqual(capabilities.name, "kde")
        self.assertFalse(capabilities.send_key)
        with self.assertRaisesRegex(desktop.DesktopError, "KDE Plasma"):
            backend.send_key("ctrl+v")


class WindowMetadataTests(unittest.TestCase):
    def test_hyprland_active_window(self) -> None:
        payload = {
            "address": "0x1234",
            "class": "org.mozilla.firefox",
            "pid": 42,
        }
        backend = desktop.HyprlandDesktop()
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/hyprctl"),
            mock.patch.object(
                desktop, "_run", return_value=completed(json.dumps(payload))
            ) as run,
        ):
            window = backend.active_window()

        self.assertEqual(window, desktop.Window("0x1234", "org.mozilla.firefox", 42))
        run.assert_called_once_with(["hyprctl", "-j", "activewindow"])

    def test_sway_native_window(self) -> None:
        payload = {
            "nodes": [
                {
                    "id": 7,
                    "focused": True,
                    "app_id": "foot",
                    "pid": 99,
                    "nodes": [],
                    "floating_nodes": [],
                }
            ]
        }
        backend = desktop.SwayDesktop()
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/swaymsg"),
            mock.patch.object(
                desktop, "_run", return_value=completed(json.dumps(payload))
            ),
        ):
            window = backend.active_window()

        self.assertEqual(window, desktop.Window("7", "foot", 99))

    def test_sway_xwayland_window_uses_x11_class(self) -> None:
        payload = {
            "floating_nodes": [
                {
                    "id": 8,
                    "focused": True,
                    "app_id": None,
                    "window_properties": {"class": "Firefox"},
                    "pid": "100",
                }
            ]
        }
        backend = desktop.SwayDesktop()
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/swaymsg"),
            mock.patch.object(
                desktop, "_run", return_value=completed(json.dumps(payload))
            ),
        ):
            window = backend.active_window()

        self.assertEqual(window, desktop.Window("8", "firefox", 100))

    def test_invalid_compositor_output_returns_no_window(self) -> None:
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"),
            mock.patch.object(desktop, "_run", return_value=completed("not json")),
        ):
            self.assertIsNone(desktop.HyprlandDesktop().active_window())
            self.assertIsNone(desktop.SwayDesktop().active_window())


class DesktopCommandTests(unittest.TestCase):
    def test_wayland_clipboard_commands(self) -> None:
        backend = desktop.HyprlandDesktop()
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"),
            mock.patch.object(
                desktop, "_run", return_value=completed("clipboard")
            ) as run,
            mock.patch.object(desktop, "_write_clipboard", return_value=True) as write,
        ):
            self.assertEqual(backend.read_clipboard(), (True, "clipboard"))
            self.assertTrue(backend.write_clipboard("new"))

        run.assert_called_once_with(["wl-paste", "--no-newline"])
        write.assert_called_once_with(["wl-copy"], "new")

    def test_wayland_key_chord_uses_wtype(self) -> None:
        backend = desktop.HyprlandDesktop()
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/wtype"),
            mock.patch.object(desktop, "_run", return_value=completed()) as run,
        ):
            self.assertTrue(backend.send_key("ctrl+shift+v"))

        run.assert_called_once_with(
            [
                "wtype",
                "-M",
                "ctrl",
                "-M",
                "shift",
                "-k",
                "v",
                "-m",
                "shift",
                "-m",
                "ctrl",
            ]
        )

    def test_wayland_key_rejects_changed_focus(self) -> None:
        backend = desktop.HyprlandDesktop()
        expected = desktop.Window("one", "firefox", 1)
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/wtype"),
            mock.patch.object(
                backend,
                "active_window",
                return_value=desktop.Window("two", "foot", 2),
            ),
            mock.patch.object(desktop, "_run") as run,
        ):
            self.assertFalse(backend.send_key("ctrl+l", window=expected))
        run.assert_not_called()

    def test_wayland_type_reads_text_from_stdin(self) -> None:
        backend = desktop.SwayDesktop()
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/wtype"),
            mock.patch.object(desktop, "_run", return_value=completed()) as run,
        ):
            backend.type_text("- unicode ∇")
        run.assert_called_once_with(["wtype", "-"], input_text="- unicode ∇")

    def test_missing_wayland_input_tool_is_reported(self) -> None:
        with (
            mock.patch.object(desktop.shutil, "which", return_value=None),
            self.assertRaisesRegex(desktop.DesktopError, "wtype"),
        ):
            desktop.SwayDesktop().type_text("hello")

    def test_x11_key_keeps_targeted_window_behavior(self) -> None:
        backend = desktop.X11Desktop()
        window = desktop.Window("42", "firefox", 10)
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/xdotool"),
            mock.patch.object(desktop, "_run", return_value=completed()) as run,
        ):
            self.assertTrue(backend.send_key("ctrl+l", window=window))
        run.assert_called_once_with(
            [
                "xdotool",
                "key",
                "--window",
                "42",
                "--clearmodifiers",
                "ctrl+l",
            ]
        )

    def test_x11_type_targets_validated_window(self) -> None:
        backend = desktop.X11Desktop()
        window = desktop.Window("42", "firefox", 10)
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/xdotool"),
            mock.patch.object(desktop, "_run", return_value=completed()) as run,
        ):
            backend.type_text("hello", window=window)
        run.assert_called_once_with(
            [
                "xdotool",
                "type",
                "--window",
                "42",
                "--clearmodifiers",
                "--",
                "hello",
            ]
        )

    def test_wayland_type_rejects_changed_focus(self) -> None:
        backend = desktop.HyprlandDesktop()
        expected = desktop.Window("one", "firefox", 1)
        with (
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/wtype"),
            mock.patch.object(
                backend,
                "active_window",
                return_value=desktop.Window("two", "foot", 2),
            ),
            mock.patch.object(desktop, "_run") as run,
            self.assertRaisesRegex(desktop.DesktopError, "active window"),
        ):
            backend.type_text("hello", window=expected)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
