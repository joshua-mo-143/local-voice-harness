from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from local_voice_harness import browser_context, desktop
from local_voice_harness.focused_app_context import FocusedAppContext


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

    def test_extracts_spoken_issue_references(self) -> None:
        issue = browser_context.GitHubIssue("example", "project", 42)
        self.assertEqual(
            browser_context.github_issue_from_text("work on example/project#42"),
            issue,
        )
        self.assertEqual(
            browser_context.github_issue_from_text(
                "please handle issue 42 in example/project"
            ),
            issue,
        )
        self.assertEqual(
            browser_context.github_issue_from_text(
                "work on https://github.com/example/project/issues/42"
            ),
            issue,
        )
        self.assertIsNone(browser_context.github_issue_from_text("work on issue 42"))

    def test_malformed_port_is_not_github_context(self) -> None:
        url = "https://github.com:invalid/example/project/issues/42"
        self.assertIsNone(browser_context.github_issue_from_url(url))
        with mock.patch.object(
            browser_context, "focused_firefox_url", return_value=url
        ):
            self.assertIsNone(browser_context.focused_github_context())

    def test_extracts_repository_from_supported_subpages(self) -> None:
        for suffix in (
            "",
            "/issues/42",
            "/pull/7",
            "/tree/main/src",
            "/blob/main/README.md",
        ):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    browser_context.github_repository_from_url(
                        f"https://github.com/example/project{suffix}"
                    ),
                    "example/project",
                )
        self.assertIsNone(
            browser_context.github_repository_from_url(
                "https://evil.example/example/project"
            )
        )
        self.assertIsNone(
            browser_context.github_repository_from_url("https://github.com/example/..")
        )

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
        with (
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(browser_context, "_issue_details", return_value=details),
        ):
            context = browser_context.focused_github_context()

        self.assertIsNotNone(context)
        self.assertIn("Repository: example/private", str(context))
        self.assertIn("Title: Fix the reader", str(context))
        self.assertIn("- dev: I can reproduce", str(context))
        self.assertEqual(getattr(context, "github_issue", None), 42)

    def test_gh_failure_keeps_validated_issue_identity(self) -> None:
        url = "https://github.com/example/project/issues/42"
        with (
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(browser_context, "_issue_details", return_value=None),
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

    def test_plain_github_page_adds_structured_repository(self) -> None:
        url = "https://github.com/example/project"
        details = {
            "nameWithOwner": "Example/Project",
            "description": "Useful project",
            "isPrivate": False,
            "defaultBranchRef": {"name": "main"},
            "url": url,
        }
        with (
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(
                browser_context, "_repository_details", return_value=details
            ),
        ):
            context = browser_context.focused_github_context()
        self.assertIn("focused GitHub repository", str(context))
        self.assertIn(url, str(context))
        self.assertIn("Default branch: main", str(context))
        self.assertEqual(getattr(context, "github_repository", None), "Example/Project")

    def test_extracts_canonical_pull_request_url(self) -> None:
        pull_request = browser_context.github_pull_request_from_url(
            "https://github.com/example/project/pull/42/files?diff=split"
        )
        self.assertEqual(
            pull_request,
            browser_context.GitHubPullRequest("example", "project", 42),
        )
        self.assertIsNone(
            browser_context.github_pull_request_from_url(
                "https://github.com/example/project/issues/42"
            )
        )
        self.assertIsNone(
            browser_context.github_pull_request_from_url(
                "https://evil.example/example/project/pull/42"
            )
        )

    def test_fetches_and_formats_pull_request_through_gh(self) -> None:
        url = "https://github.com/example/project/pull/7"
        details = {
            "number": 7,
            "title": "Add caching",
            "state": "OPEN",
            "author": {"login": "octocat"},
            "labels": [{"name": "enhancement"}],
            "body": "Speeds things up",
            "comments": [{"author": {"login": "dev"}, "body": "Looks good"}],
            "url": url,
            "isDraft": True,
            "baseRefName": "main",
            "headRefName": "feature/cache",
            "additions": 120,
            "deletions": 4,
            "changedFiles": 6,
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
        self.assertIn("focused GitHub pull request", str(context))
        self.assertIn("Pull request: #7", str(context))
        self.assertIn("Branch: feature/cache into main", str(context))
        self.assertIn("Draft: yes", str(context))
        self.assertIn("Changed files: 6 (+120/-4)", str(context))
        self.assertIn("- dev: Looks good", str(context))
        self.assertEqual(getattr(context, "github_repository", None), "example/project")
        self.assertEqual(getattr(context, "github_pull_request", None), 7)
        self.assertEqual(
            run.call_args.args[0][:6],
            ["gh", "pr", "view", "7", "--repo", "example/project"],
        )

    def test_pull_request_gh_failure_keeps_identity(self) -> None:
        url = "https://github.com/example/project/pull/7"
        completed = subprocess.CompletedProcess([], 1, "", "not found")
        with (
            mock.patch.object(
                browser_context.shutil, "which", return_value="/usr/bin/gh"
            ),
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(browser_context, "_run", return_value=completed),
        ):
            context = browser_context.focused_github_context()

        self.assertIn("Pull request: #7", str(context))
        self.assertIn("could not be fetched", str(context))
        self.assertEqual(getattr(context, "github_pull_request", None), 7)

    def test_enriched_request_preserves_pull_request_identity(self) -> None:
        context = browser_context.GitHubContext(
            "GitHub context",
            github_repository="example/project",
            github_pull_request=7,
        )
        with mock.patch.object(
            browser_context, "focused_browser_context", return_value=context
        ):
            request = browser_context.enrich_request("make sure this PR works")
        self.assertEqual(request.github_repository, "example/project")
        self.assertEqual(request.github_pull_request, 7)

    def test_enriched_request_preserves_validated_repository_identity(self) -> None:
        context = browser_context.GitHubContext(
            "GitHub context", github_repository="example/project"
        )
        with mock.patch.object(
            browser_context, "focused_browser_context", return_value=context
        ):
            request = browser_context.enrich_request("fork this repo")
        self.assertEqual(
            request.github_repository,
            "example/project",
        )
        self.assertEqual(str(request), "fork this repo\n\nGitHub context")

    def test_enrich_request_is_fail_open(self) -> None:
        with mock.patch.object(
            browser_context,
            "focused_browser_context",
            side_effect=RuntimeError("desktop unavailable"),
        ):
            self.assertEqual(browser_context.enrich_request("hello"), "hello")

    def test_request_context_exposes_validated_issue_identity(self) -> None:
        url = "https://github.com/example/project/issues/42"
        with (
            mock.patch.object(
                browser_context, "focused_firefox_url", return_value=url
            ) as focused_url,
            mock.patch.object(browser_context, "_issue_details", return_value=None),
        ):
            context = browser_context.request_context("work on this")

        self.assertEqual(context.focused_repository, "example/project")
        self.assertEqual(context.focused_issue, "example/project#42")
        self.assertEqual(context.github_issue, 42)
        self.assertIn("Issue: #42", str(context.github_issue_context))
        self.assertIn("Repository: example/project", context.text)
        focused_url.assert_called_once_with()

    def test_spoken_issue_overrides_focused_browser_issue(self) -> None:
        spoken = browser_context.GitHubIssue("spoken", "project", 7)
        with (
            mock.patch.object(browser_context, "_issue_details", return_value=None),
            mock.patch.object(browser_context, "focused_firefox_url") as focused_url,
        ):
            context = browser_context.request_context(
                "work on spoken/project#7 instead"
            )

        self.assertEqual(context.focused_issue, spoken.reference)
        self.assertEqual(context.github_repository, spoken.name_with_owner)
        self.assertEqual(context.github_issue, 7)
        focused_url.assert_not_called()


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
        self.assertIn("def broken():", context.text)

    def test_focused_app_context_absent_leaves_request_unchanged(self) -> None:
        with (
            mock.patch.object(
                browser_context, "focused_firefox_url", return_value=None
            ),
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
        ):
            context = browser_context.request_context("fix this code")

        self.assertEqual(context.text, "fix this code")
        self.assertIsNone(context.focused_app_context)
        self.assertEqual(context.focused_app_sources, ())

    def test_focused_app_capture_failure_does_not_lose_browser_context(self) -> None:
        url = "https://github.com/example/project/issues/42"
        with (
            mock.patch.object(browser_context, "focused_firefox_url", return_value=url),
            mock.patch.object(browser_context, "_issue_details", return_value=None),
            mock.patch.object(
                browser_context,
                "focused_app_context",
                side_effect=RuntimeError("capture blew up"),
            ),
        ):
            context = browser_context.request_context("work on this")

        self.assertEqual(context.github_issue, 42)
        self.assertIsNone(context.focused_app_context)


class LinearContextTests(unittest.TestCase):
    def test_extracts_canonical_issue_url(self) -> None:
        issue = browser_context.linear_issue_from_url(
            "https://linear.app/acme/issue/eng-123/fix-routing?utm_source=inbox"
        )
        self.assertEqual(issue, browser_context.LinearIssue("ENG-123"))

        invalid_urls = [
            "http://linear.app/acme/issue/ENG-123/fix-routing",
            "https://linear.app.evil.test/acme/issue/ENG-123/fix-routing",
            "https://user@linear.app/acme/issue/ENG-123/fix-routing",
            "https://linear.app:444/acme/issue/ENG-123/fix-routing",
            "https://linear.app/acme/project/ENG-123",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertIsNone(browser_context.linear_issue_from_url(url))

    def test_request_context_preserves_focused_linear_identifier(self) -> None:
        url = "https://linear.app/acme/issue/ENG-123/fix-routing"
        with (
            mock.patch.object(
                browser_context, "focused_firefox_url", return_value=url
            ) as focused_url,
            mock.patch.object(
                browser_context, "focused_app_context", return_value=None
            ),
        ):
            context = browser_context.request_context("work on this ticket")

        self.assertEqual(context.focused_issue, "ENG-123")
        self.assertEqual(context.linear_issue, "ENG-123")
        self.assertIn("Current focused Linear issue", context.text)
        self.assertIn("Identifier: ENG-123", context.text)
        focused_url.assert_called_once_with()

    def test_browser_context_dispatches_linear_url(self) -> None:
        url = "https://linear.app/acme/issue/ENG-123/fix-routing"
        with mock.patch.object(
            browser_context, "focused_firefox_url", return_value=url
        ):
            context = browser_context.focused_browser_context()

        self.assertIn("Identifier: ENG-123", str(context))


class ZendeskContextTests(unittest.TestCase):
    def test_extracts_canonical_ticket_url(self) -> None:
        ticket = browser_context.zendesk_ticket_from_url(
            "https://Example-Help.zendesk.com/agent/tickets/42?foo=bar"
        )
        self.assertEqual(ticket, browser_context.ZendeskTicket("example-help", 42))

        invalid_urls = [
            "http://example.zendesk.com/agent/tickets/42",
            "https://example.zendesk.com.evil.test/agent/tickets/42",
            "https://user@example.zendesk.com/agent/tickets/42",
            "https://example.zendesk.com:444/agent/tickets/42",
            "https://example.zendesk.com/hc/en-us/requests/42",
            "https://example.zendesk.com/agent/tickets/not-a-number",
            "https://-example.zendesk.com/agent/tickets/42",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertIsNone(browser_context.zendesk_ticket_from_url(url))

    def test_copies_rendered_page_text_and_restores_clipboard(self) -> None:
        url = "https://example.zendesk.com/agent/tickets/42"
        page_text = " Ticket 42 \n\n Customer cannot sign in "
        window = desktop.Window("42", "firefox", 10)
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.return_value = window
        backend.read_clipboard.side_effect = [
            (True, "previous"),
            (True, url),
            (True, page_text),
            (True, page_text),
        ]
        backend.send_key.return_value = True
        with (
            mock.patch.object(browser_context, "get_desktop", return_value=backend),
            mock.patch.object(browser_context.time, "sleep"),
        ):
            captured = browser_context._focused_firefox_page_text(url)

        self.assertEqual(captured, "Ticket 42\nCustomer cannot sign in")
        self.assertEqual(
            backend.send_key.call_args_list,
            [
                mock.call("ctrl+l", window=window),
                mock.call("ctrl+c", window=window),
                mock.call("Escape", window=window),
                mock.call("ctrl+a", window=window),
                mock.call("ctrl+c", window=window),
                mock.call("Escape", window=window),
            ],
        )
        backend.write_clipboard.assert_called_once_with("previous")

    def test_focus_change_aborts_page_capture(self) -> None:
        url = "https://example.zendesk.com/agent/tickets/42"
        window = desktop.Window("42", "firefox", 10)
        changed = desktop.Window("99", "foot", 11)
        backend = mock.Mock()
        backend.has_clipboard.return_value = True
        backend.active_window.side_effect = [window, window, window, changed, changed]
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
            self.assertIsNone(browser_context._focused_firefox_page_text(url))

        self.assertEqual(
            backend.send_key.call_args_list,
            [
                mock.call("ctrl+l", window=window),
                mock.call("ctrl+c", window=window),
                mock.call("Escape", window=window),
            ],
        )
        backend.write_clipboard.assert_called_once_with("previous")

    def test_formats_and_bounds_rendered_ticket_text(self) -> None:
        url = "https://example.zendesk.com/agent/tickets/42"
        with mock.patch.object(
            browser_context,
            "_focused_firefox_page_text",
            return_value="x" * 20_000,
        ):
            context = browser_context._zendesk_context_from_url(url)

        self.assertIsNotNone(context)
        self.assertIn("Tenant: example", str(context))
        self.assertIn("Ticket: #42", str(context))
        self.assertIn("Rendered ticket text:", str(context))
        self.assertLess(len(str(context)), 10_500)
        self.assertTrue(str(context).endswith("…"))

    def test_page_copy_failure_keeps_validated_ticket_identity(self) -> None:
        url = "https://example.zendesk.com/agent/tickets/42"
        with mock.patch.object(
            browser_context, "_focused_firefox_page_text", return_value=None
        ):
            context = browser_context._zendesk_context_from_url(url)

        self.assertIn("Tenant: example", str(context))
        self.assertIn("Ticket: #42", str(context))
        self.assertIn("could not be read", str(context))

    def test_browser_context_dispatches_zendesk_url_once(self) -> None:
        url = "https://example.zendesk.com/agent/tickets/42"
        with (
            mock.patch.object(
                browser_context, "focused_firefox_url", return_value=url
            ) as focused_url,
            mock.patch.object(
                browser_context,
                "_focused_firefox_page_text",
                return_value="Ticket details",
            ),
        ):
            context = browser_context.focused_browser_context()

        focused_url.assert_called_once_with()
        self.assertIn("Current focused Zendesk ticket", str(context))
        self.assertIn("Ticket details", str(context))


if __name__ == "__main__":
    unittest.main()
