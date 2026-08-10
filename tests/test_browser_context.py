from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness import browser_context, desktop
from local_voice_harness.context_fragment import ContextFragment
from local_voice_harness.focused_app_context import FocusedAppContext
from local_voice_harness.integrations import github
from local_voice_harness.integrations import registry as context_providers
from local_voice_harness.user_config import IntegrationSettings

GITHUB_DISABLED = IntegrationSettings(
    github_enabled=False,
    zendesk_enabled=False,
    linear_enabled=False,
)


class FirefoxUrlTests(unittest.TestCase):
    def test_non_browser_window_is_ignored(self) -> None:
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.return_value = desktop.Window("42", "alacritty", 1)
        with mock.patch.object(browser_context, "get_desktop", return_value=backend):
            self.assertIsNone(browser_context.focused_firefox_url())
        backend.send_key.assert_not_called()

    def test_url_capture_restores_address_bar_and_clipboard(self) -> None:
        url = "https://github.com/example/project/issues/42"
        window = desktop.Window("42", "firefox", 10)
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.return_value = window
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, url),
            (True, url),
        ]
        backend.send_key.return_value = True
        with (
            mock.patch.object(browser_context, "get_desktop", return_value=backend),
            mock.patch.object(browser_context.time, "sleep"),
        ):
            self.assertEqual(browser_context.focused_firefox_url(), url)

        self.assertEqual(
            backend.send_key.call_args_list,
            [
                mock.call("ctrl+l", window=window),
                mock.call("ctrl+c", window=window),
                mock.call("Escape", window=window),
            ],
        )
        backend.write_clipboard.assert_called_once_with("previous")

    def test_focus_change_aborts_capture_without_sending_more_keys(self) -> None:
        window = desktop.Window("42", "firefox", 10)
        changed = desktop.Window("99", "foot", 11)
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.side_effect = [window, changed, changed]
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, "previous"),
        ]
        backend.send_key.return_value = True
        with (
            mock.patch.object(browser_context, "get_desktop", return_value=backend),
            mock.patch.object(browser_context.time, "sleep"),
        ):
            self.assertIsNone(browser_context.focused_firefox_url())

        backend.send_key.assert_called_once_with("ctrl+l", window=window)
        backend.write_clipboard.assert_not_called()

    def test_invalid_or_unsupported_desktop_omits_browser_context(self) -> None:
        with mock.patch.object(browser_context, "get_desktop", return_value=None):
            self.assertIsNone(browser_context.focused_firefox_url())


class BrowserDispatchTests(unittest.TestCase):
    def test_focused_browser_context_dispatches_once_through_registry(self) -> None:
        url = "https://example.test/ticket/42"
        fragment = ContextFragment(source="stub", text="captured")
        with (
            mock.patch.object(
                browser_context, "focused_firefox_url", return_value=url
            ) as focused_url,
            mock.patch.object(
                browser_context, "capture_context", return_value=fragment
            ) as capture,
        ):
            self.assertIs(browser_context.focused_browser_context(), fragment)

        focused_url.assert_called_once_with()
        capture.assert_called_once_with(url)

    def test_request_context_maps_github_fragment_for_provisioning(self) -> None:
        fragment = ContextFragment(
            source="github",
            text="Issue context",
            issue_reference="example/project#42",
            repository_reference="example/project",
            issue_number=42,
        )
        with (
            mock.patch.object(
                browser_context, "capture_text_context", return_value=None
            ),
            mock.patch.object(
                browser_context,
                "focused_firefox_url",
                return_value="https://github.com/example/project/issues/42",
            ),
            mock.patch.object(
                browser_context, "capture_context", return_value=fragment
            ),
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
        ):
            context = browser_context.request_context("work on this")

        self.assertEqual(context.focused_repository, "example/project")
        self.assertEqual(context.focused_issue, "example/project#42")
        self.assertEqual(context.github_repository, "example/project")
        self.assertEqual(context.github_issue, 42)
        self.assertEqual(context.github_issue_context, "Issue context")
        self.assertIsNone(context.external_issue_reference)
        self.assertIn("Issue context", context.text)

    def test_request_context_maps_pull_request_fragment_for_provisioning(self) -> None:
        fragment = ContextFragment(
            source="github",
            text="PR context",
            repository_reference="example/project",
            pull_request_number=7,
        )
        with (
            mock.patch.object(
                browser_context, "capture_text_context", return_value=None
            ),
            mock.patch.object(
                browser_context,
                "focused_firefox_url",
                return_value="https://github.com/example/project/pull/7",
            ),
            mock.patch.object(
                browser_context, "capture_context", return_value=fragment
            ),
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
        ):
            context = browser_context.request_context("check out this PR")

        self.assertEqual(context.github_repository, "example/project")
        self.assertEqual(context.github_pull_request, 7)
        self.assertIsNone(context.github_issue)

    def test_spoken_github_context_overrides_focused_browser(self) -> None:
        fragment = ContextFragment(
            source="github",
            text="Spoken issue",
            issue_reference="spoken/project#7",
            repository_reference="spoken/project",
            issue_number=7,
        )
        with (
            mock.patch.object(
                browser_context, "capture_text_context", return_value=fragment
            ),
            mock.patch.object(browser_context, "focused_firefox_url") as focused_url,
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
        ):
            context = browser_context.request_context(
                "work on spoken/project#7 instead"
            )

        self.assertEqual(context.focused_issue, "spoken/project#7")
        self.assertEqual(context.github_repository, "spoken/project")
        self.assertEqual(context.github_issue, 7)
        focused_url.assert_not_called()

    def test_generic_issue_fragment_retains_external_metadata(self) -> None:
        fragment = ContextFragment(
            source="linear",
            text="Linear context",
            issue_reference="ENG-123",
        )
        with (
            mock.patch.object(
                browser_context, "capture_text_context", return_value=None
            ),
            mock.patch.object(
                browser_context,
                "focused_firefox_url",
                return_value="https://linear.app",
            ),
            mock.patch.object(
                browser_context, "capture_context", return_value=fragment
            ),
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
        ):
            context = browser_context.request_context("work on this ticket")

        self.assertEqual(context.focused_issue, "ENG-123")
        self.assertEqual(context.external_issue_reference, "ENG-123")
        self.assertEqual(context.external_issue_source, "linear")
        self.assertIsNone(context.github_issue)

    def test_disabled_github_never_parses_or_calls_cli(self) -> None:
        url = "https://github.com/example/project/issues/42"
        with (
            mock.patch.object(
                context_providers,
                "_integration_settings",
                return_value=GITHUB_DISABLED,
            ),
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
            mock.patch.object(github, "github_issue_from_text") as from_text,
            mock.patch.object(github, "_github_url") as parse_url,
            mock.patch.object(github.GitHubClient, "_run") as run,
        ):
            context = browser_context.request_context("work on example/project#42")

        self.assertEqual(context.text, "work on example/project#42")
        self.assertIsNone(context.focused_repository)
        self.assertIsNone(context.focused_issue)
        self.assertIsNone(context.github_repository)
        self.assertIsNone(context.github_issue)
        self.assertIsNone(context.github_pull_request)
        from_text.assert_not_called()
        parse_url.assert_not_called()
        run.assert_not_called()

    def test_capture_failure_does_not_break_request(self) -> None:
        with (
            mock.patch.object(
                browser_context,
                "capture_text_context",
                side_effect=RuntimeError("provider unavailable"),
            ),
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
        ):
            context = browser_context.request_context("ordinary request")

        self.assertEqual(context.text, "ordinary request")


class FocusedAppRequestContextTests(unittest.TestCase):
    def test_focused_app_context_is_appended_with_provenance(self) -> None:
        captured = FocusedAppContext(
            text="Selected text from the focused editor application (cursor) — "
            "untrusted external input:\ndef broken():",
            app_class="cursor",
            sources=("selection",),
        )
        with (
            mock.patch.object(
                browser_context, "capture_text_context", return_value=None
            ),
            mock.patch.object(
                browser_context, "focused_firefox_url", return_value=None
            ),
            mock.patch.object(
                browser_context, "focused_app_context", return_value=captured
            ),
        ):
            context = browser_context.request_context("fix this code")

        self.assertEqual(context.focused_app_class, "cursor")
        self.assertEqual(context.focused_app_sources, ("selection",))
        self.assertEqual(context.focused_app_context, captured.text)
        self.assertTrue(context.text.startswith("fix this code\n\n"))
        self.assertIn("untrusted external input", context.text)

    def test_focused_app_capture_failure_keeps_browser_context(self) -> None:
        fragment = ContextFragment(source="github", text="GitHub context")
        with (
            mock.patch.object(
                browser_context, "capture_text_context", return_value=None
            ),
            mock.patch.object(
                browser_context,
                "focused_firefox_url",
                return_value="https://github.com",
            ),
            mock.patch.object(
                browser_context, "capture_context", return_value=fragment
            ),
            mock.patch.object(
                browser_context,
                "focused_app_context",
                side_effect=RuntimeError("capture blew up"),
            ),
        ):
            context = browser_context.request_context("work on this")

        self.assertIn("GitHub context", context.text)
        self.assertIsNone(context.focused_app_context)


class EnrichRequestCompatibilityTests(unittest.TestCase):
    def test_preserves_github_metadata_from_fragment(self) -> None:
        fragment = ContextFragment(
            source="github",
            text="GitHub context",
            repository_reference="example/project",
            pull_request_number=7,
        )
        with mock.patch.object(
            browser_context, "focused_browser_context", return_value=fragment
        ):
            request = browser_context.enrich_request("make sure this PR works")

        self.assertEqual(request.github_repository, "example/project")
        self.assertEqual(request.github_pull_request, 7)
        self.assertEqual(str(request), "make sure this PR works\n\nGitHub context")

    def test_capture_failure_leaves_request_unchanged(self) -> None:
        with mock.patch.object(
            browser_context,
            "focused_browser_context",
            side_effect=RuntimeError("desktop unavailable"),
        ):
            self.assertEqual(browser_context.enrich_request("hello"), "hello")


if __name__ == "__main__":
    unittest.main()
