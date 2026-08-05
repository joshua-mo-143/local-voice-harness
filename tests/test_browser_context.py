from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from local_voice_harness import browser_context, desktop


class FirefoxUrlTests(unittest.TestCase):
    def test_non_browser_window_is_ignored(self) -> None:
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.return_value = desktop.Window("42", "alacritty", 1)
        with (
            mock.patch.object(browser_context, "get_desktop", return_value=backend),
        ):
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

    def test_unsupported_wayland_session_omits_browser_context(self) -> None:
        with mock.patch.object(browser_context, "get_desktop", return_value=None):
            self.assertIsNone(browser_context.focused_firefox_url())


class GitHubContextTests(unittest.TestCase):
    def test_extracts_canonical_issue_url(self) -> None:
        issue = browser_context.github_issue_from_url(
            "https://github.com/example/project/issues/42?notification_referrer=x"
        )
        self.assertEqual(issue, browser_context.GitHubIssue("example", "project", 42))
        self.assertIsNone(
            browser_context.github_issue_from_url(
                "https://github.example/example/project/issues/42"
            )
        )
        self.assertIsNone(
            browser_context.github_issue_from_url(
                "https://github.com/example/project/pull/42"
            )
        )

    def test_malformed_port_is_not_github_context(self) -> None:
        url = "https://github.com:invalid/example/project/issues/42"
        self.assertIsNone(browser_context.github_issue_from_url(url))
        with mock.patch.object(
            browser_context, "focused_firefox_url", return_value=url
        ):
            self.assertIsNone(browser_context.focused_github_context())

    def test_fetches_and_formats_private_issue_through_gh(self) -> None:
        url = "https://github.com/example/private/issues/42"
        details = {
            "number": 42,
            "title": "Fix the reader",
            "state": "OPEN",
            "author": {"login": "octocat"},
            "labels": [{"name": "bug"}],
            "body": "Expected behavior",
            "comments": [{"author": {"login": "dev"}, "body": "I can reproduce"}],
            "url": url,
        }
        completed = subprocess.CompletedProcess([], 0, json.dumps(details), "")
        with (
            mock.patch.object(
                browser_context.shutil, "which", return_value="/usr/bin/gh"
            ),
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(browser_context, "_run", return_value=completed) as run,
        ):
            context = browser_context.focused_github_context()

        self.assertIsNotNone(context)
        self.assertIn("Repository: example/private", str(context))
        self.assertIn("Title: Fix the reader", str(context))
        self.assertIn("- dev: I can reproduce", str(context))
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertEqual(
            run.call_args.args[0][:6],
            ["gh", "issue", "view", "42", "--repo", "example/private"],
        )

    def test_gh_failure_keeps_validated_issue_identity(self) -> None:
        url = "https://github.com/example/project/issues/42"
        completed = subprocess.CompletedProcess([], 1, "", "not found")
        with (
            mock.patch.object(
                browser_context.shutil, "which", return_value="/usr/bin/gh"
            ),
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(browser_context, "_run", return_value=completed),
        ):
            context = browser_context.focused_github_context()

        self.assertIn("Issue: #42", str(context))
        self.assertIn("could not be fetched", str(context))

    def test_issue_content_is_bounded(self) -> None:
        issue = browser_context.GitHubIssue("example", "project", 42)
        details: dict[str, object] = {
            "title": "Title",
            "state": "OPEN",
            "body": "b" * 20_000,
            "comments": [
                {"author": {"login": "dev"}, "body": "c" * 2_000} for _ in range(20)
            ],
        }
        context = browser_context._format_issue(
            "https://github.com/example/project/issues/42", issue, details
        )
        self.assertLess(len(context), 10_000)
        self.assertEqual(context.count("- dev:"), browser_context.MAX_COMMENTS)

    def test_plain_github_page_adds_url_only(self) -> None:
        url = "https://github.com/example/project"
        with mock.patch.object(
            browser_context, "focused_firefox_url", return_value=url
        ):
            context = browser_context.focused_github_context()
        self.assertIn("focused GitHub page", str(context))
        self.assertIn(url, str(context))

    def test_enrich_request_is_fail_open(self) -> None:
        with mock.patch.object(
            browser_context,
            "focused_github_context",
            side_effect=RuntimeError("desktop unavailable"),
        ):
            self.assertEqual(browser_context.enrich_request("hello"), "hello")


if __name__ == "__main__":
    unittest.main()
