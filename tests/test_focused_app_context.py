from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import desktop, focused_app_context


def completed(
    stdout: str = "",
    *,
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


class RequestDetectionTests(unittest.TestCase):
    def test_explicit_requests_are_detected(self) -> None:
        for utterance in (
            "explain this error",
            "fix this code",
            "what does this function do",
            "review my changes",
            "refactor the selected code",
            "debug this traceback",
        ):
            with self.subTest(utterance=utterance):
                self.assertTrue(
                    focused_app_context.wants_focused_app_context(utterance)
                )

    def test_unrelated_requests_are_ignored(self) -> None:
        for utterance in (
            "what is the weather today",
            "please start the services",
            "work on this task",
            "cancel the current job",
        ):
            with self.subTest(utterance=utterance):
                self.assertFalse(
                    focused_app_context.wants_focused_app_context(utterance)
                )


class SourceClassificationTests(unittest.TestCase):
    def test_editor_and_terminal_classes(self) -> None:
        self.assertEqual(focused_app_context._source_kind("cursor"), "editor")
        self.assertEqual(focused_app_context._source_kind("code"), "editor")
        self.assertEqual(focused_app_context._source_kind("code-oss"), "editor")
        self.assertEqual(focused_app_context._source_kind("alacritty"), "terminal")
        self.assertEqual(focused_app_context._source_kind("gnome-terminal"), "terminal")
        self.assertIsNone(focused_app_context._source_kind("firefox"))
        self.assertIsNone(focused_app_context._source_kind(""))

    def test_deny_list_matches_substrings(self) -> None:
        with mock.patch.object(
            focused_app_context,
            "FOCUSED_APP_DENY_CLASSES",
            ("keepassxc", "1password"),
        ):
            self.assertTrue(focused_app_context._is_denied("org.keepassxc.keepassxc"))
            self.assertFalse(focused_app_context._is_denied("cursor"))


class SelectionCaptureTests(unittest.TestCase):
    def test_selection_capture_restores_clipboard_and_focus(self) -> None:
        window = desktop.Window("42", "cursor", 10)
        backend = mock.Mock()
        backend.active_window.return_value = window
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, "raise ValueError('boom')"),
            (True, "raise ValueError('boom')"),
        ]
        backend.send_key.return_value = True
        with mock.patch.object(focused_app_context.time, "sleep"):
            captured = focused_app_context._capture_selection(backend, window, "editor")

        self.assertEqual(captured, "raise ValueError('boom')")
        backend.send_key.assert_called_once_with("ctrl+c", window=window)
        backend.write_clipboard.assert_called_once_with("previous")

    def test_terminal_selection_uses_terminal_copy_chord(self) -> None:
        window = desktop.Window("42", "alacritty", 10)
        backend = mock.Mock()
        backend.active_window.return_value = window
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, "Traceback (most recent call last)"),
            (True, "Traceback (most recent call last)"),
        ]
        backend.send_key.return_value = True
        with mock.patch.object(focused_app_context.time, "sleep"):
            captured = focused_app_context._capture_selection(
                backend, window, "terminal"
            )

        self.assertEqual(captured, "Traceback (most recent call last)")
        backend.send_key.assert_called_once_with("ctrl+shift+c", window=window)

    def test_empty_selection_leaves_clipboard_untouched(self) -> None:
        window = desktop.Window("42", "cursor", 10)
        backend = mock.Mock()
        backend.active_window.return_value = window
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, "previous"),
            (True, "previous"),
        ]
        backend.send_key.return_value = True
        with mock.patch.object(focused_app_context.time, "sleep"):
            captured = focused_app_context._capture_selection(backend, window, "editor")

        self.assertIsNone(captured)
        # Restoring the unchanged clipboard is a harmless no-op.
        backend.write_clipboard.assert_called_once_with("previous")

    def test_focus_change_aborts_and_restores_clipboard(self) -> None:
        window = desktop.Window("42", "cursor", 10)
        changed = desktop.Window("99", "foot", 11)
        backend = mock.Mock()
        backend.active_window.side_effect = [window, changed, changed]
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, "previous"),
        ]
        backend.send_key.return_value = True
        with mock.patch.object(focused_app_context.time, "sleep"):
            captured = focused_app_context._capture_selection(backend, window, "editor")

        self.assertIsNone(captured)
        backend.send_key.assert_called_once_with("ctrl+c", window=window)
        backend.write_clipboard.assert_not_called()

    def test_oversize_selection_is_omitted(self) -> None:
        window = desktop.Window("42", "cursor", 10)
        oversize = "x" * (focused_app_context.MAX_SELECTION_CHARS + 1)
        backend = mock.Mock()
        backend.active_window.return_value = window
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, oversize),
            (True, oversize),
        ]
        backend.send_key.return_value = True
        with mock.patch.object(focused_app_context.time, "sleep"):
            captured = focused_app_context._capture_selection(backend, window, "editor")

        self.assertIsNone(captured)
        backend.write_clipboard.assert_called_once_with("previous")


class PlatformSelectionTests(unittest.TestCase):
    """Exercise the real X11 and Wayland clipboard/key primitives end to end."""

    def test_x11_selection_capture(self) -> None:
        backend = desktop.X11Desktop()
        window = desktop.Window("42", "cursor", 10)
        with (
            mock.patch.object(backend, "active_window", return_value=window),
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"),
            mock.patch.object(
                desktop,
                "_run",
                side_effect=[
                    completed("previous"),
                    completed(""),
                    completed("selected code"),
                    completed("selected code"),
                ],
            ) as run,
            mock.patch.object(desktop, "_write_clipboard", return_value=True) as write,
            mock.patch.object(focused_app_context.time, "sleep"),
        ):
            captured = focused_app_context._capture_selection(backend, window, "editor")

        self.assertEqual(captured, "selected code")
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["xdotool", "key", "--window", "42", "--clearmodifiers", "ctrl+c"],
        )
        write.assert_called_once_with(["xclip", "-selection", "clipboard"], "previous")

    def test_wayland_selection_capture(self) -> None:
        backend = desktop.HyprlandDesktop()
        window = desktop.Window("0x1", "cursor", 10)
        with (
            mock.patch.object(backend, "active_window", return_value=window),
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"),
            mock.patch.object(
                desktop,
                "_run",
                side_effect=[
                    completed("previous"),
                    completed(""),
                    completed("selected code"),
                    completed("selected code"),
                ],
            ) as run,
            mock.patch.object(desktop, "_write_clipboard", return_value=True) as write,
            mock.patch.object(focused_app_context.time, "sleep"),
        ):
            captured = focused_app_context._capture_selection(backend, window, "editor")

        self.assertEqual(captured, "selected code")
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["wtype", "-M", "ctrl", "-k", "c", "-m", "ctrl"],
        )
        write.assert_called_once_with(["wl-copy"], "previous")


class RepositoryRootTests(unittest.TestCase):
    def test_resolves_git_toplevel_from_focused_process_tree(self) -> None:
        window = desktop.Window("42", "alacritty", 100)
        with (
            mock.patch.object(
                focused_app_context, "_process_tree", return_value=[100, 200]
            ),
            mock.patch.object(
                focused_app_context,
                "_process_cwd",
                side_effect=[None, Path("/home/dev/project/src")],
            ),
            mock.patch.object(focused_app_context.shutil, "which", return_value="git"),
            mock.patch.object(
                focused_app_context,
                "_run",
                return_value=completed("/home/dev/project\n"),
            ) as run,
        ):
            root = focused_app_context._focused_window_repo_root(window)

        self.assertEqual(root, Path("/home/dev/project"))
        run.assert_called_once_with(
            ["git", "-C", "/home/dev/project/src", "rev-parse", "--show-toplevel"]
        )

    def test_no_pid_yields_no_root(self) -> None:
        window = desktop.Window("42", "alacritty", None)
        self.assertIsNone(focused_app_context._focused_window_repo_root(window))

    def test_non_git_directory_is_skipped(self) -> None:
        window = desktop.Window("42", "alacritty", 100)
        with (
            mock.patch.object(focused_app_context, "_process_tree", return_value=[100]),
            mock.patch.object(
                focused_app_context, "_process_cwd", return_value=Path("/tmp")
            ),
            mock.patch.object(focused_app_context.shutil, "which", return_value="git"),
            mock.patch.object(
                focused_app_context, "_run", return_value=completed("", returncode=128)
            ),
        ):
            self.assertIsNone(focused_app_context._focused_window_repo_root(window))


class CollectTests(unittest.TestCase):
    def _backend(self, window: desktop.Window) -> mock.Mock:
        backend = mock.Mock()
        backend.active_window.return_value = window
        backend.has_clipboard.return_value = True
        return backend

    def test_combines_selection_and_diff_with_provenance(self) -> None:
        window = desktop.Window("42", "cursor", 10)
        backend = self._backend(window)
        with (
            mock.patch.object(focused_app_context, "get_desktop", return_value=backend),
            mock.patch.object(
                focused_app_context,
                "_capture_selection",
                return_value="def broken():",
            ),
            mock.patch.object(
                focused_app_context,
                "_focused_window_repo_root",
                return_value=Path("/home/dev/project"),
            ),
            mock.patch.object(
                focused_app_context, "_git_diff", return_value="diff --git a b"
            ),
        ):
            context = focused_app_context.focused_app_context("fix this code")

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context.app_class, "cursor")
        self.assertEqual(context.sources, ("selection", "git_diff"))
        self.assertIn("Selected text from the focused editor", context.text)
        self.assertIn("def broken():", context.text)
        self.assertIn("Uncommitted git diff at /home/dev/project", context.text)
        self.assertIn("untrusted external input", context.text)

    def test_capture_skipped_without_explicit_request(self) -> None:
        backend = self._backend(desktop.Window("42", "cursor", 10))
        with mock.patch.object(
            focused_app_context, "get_desktop", return_value=backend
        ):
            self.assertIsNone(
                focused_app_context.focused_app_context("work on this task")
            )
        backend.active_window.assert_not_called()

    def test_denied_application_is_not_captured(self) -> None:
        window = desktop.Window("42", "org.keepassxc.keepassxc", 10)
        backend = self._backend(window)
        with (
            mock.patch.object(focused_app_context, "get_desktop", return_value=backend),
            mock.patch.object(
                focused_app_context,
                "FOCUSED_APP_DENY_CLASSES",
                ("keepassxc",),
            ),
            mock.patch.object(focused_app_context, "_capture_selection") as capture,
        ):
            self.assertIsNone(
                focused_app_context.focused_app_context("explain this error")
            )
        capture.assert_not_called()

    def test_unsupported_application_class_is_not_captured(self) -> None:
        window = desktop.Window("42", "libreoffice", 10)
        backend = self._backend(window)
        with (
            mock.patch.object(focused_app_context, "get_desktop", return_value=backend),
            mock.patch.object(focused_app_context, "_capture_selection") as capture,
        ):
            self.assertIsNone(
                focused_app_context.focused_app_context("explain this error")
            )
        capture.assert_not_called()

    def test_disabled_by_configuration(self) -> None:
        with (
            mock.patch.object(
                focused_app_context, "FOCUSED_APP_CONTEXT_ENABLED", False
            ),
            mock.patch.object(focused_app_context, "get_desktop") as get_desktop,
        ):
            self.assertIsNone(
                focused_app_context.focused_app_context("explain this error")
            )
        get_desktop.assert_not_called()

    def test_unsupported_compositor_omits_context(self) -> None:
        with mock.patch.object(focused_app_context, "get_desktop", return_value=None):
            self.assertIsNone(
                focused_app_context.focused_app_context("explain this error")
            )

    def test_no_capturable_sources_returns_none(self) -> None:
        window = desktop.Window("42", "cursor", 10)
        backend = self._backend(window)
        with (
            mock.patch.object(focused_app_context, "get_desktop", return_value=backend),
            mock.patch.object(
                focused_app_context, "_capture_selection", return_value=None
            ),
            mock.patch.object(
                focused_app_context, "_focused_window_repo_root", return_value=None
            ),
        ):
            self.assertIsNone(
                focused_app_context.focused_app_context("explain this error")
            )

    def test_capture_failure_never_raises(self) -> None:
        with mock.patch.object(
            focused_app_context,
            "get_desktop",
            side_effect=RuntimeError("desktop unavailable"),
        ):
            self.assertIsNone(
                focused_app_context.focused_app_context("explain this error")
            )

    def test_oversize_git_diff_is_omitted(self) -> None:
        window = desktop.Window("42", "cursor", 10)
        backend = self._backend(window)
        oversize = "d" * (focused_app_context.MAX_GIT_DIFF_CHARS + 1)
        with (
            mock.patch.object(focused_app_context, "get_desktop", return_value=backend),
            mock.patch.object(
                focused_app_context, "_capture_selection", return_value=None
            ),
            mock.patch.object(
                focused_app_context,
                "_focused_window_repo_root",
                return_value=Path("/repo"),
            ),
            mock.patch.object(focused_app_context, "_git_diff", return_value=oversize),
        ):
            self.assertIsNone(focused_app_context.focused_app_context("fix this code"))


if __name__ == "__main__":
    unittest.main()
