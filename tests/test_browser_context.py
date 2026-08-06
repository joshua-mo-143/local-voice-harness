from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from local_voice_harness import browser_context


class FirefoxUrlTests(unittest.TestCase):
    def test_non_browser_window_is_ignored(self) -> None:
        with (
            mock.patch.object(
                browser_context.shutil, "which", return_value="/usr/bin/tool"
            ),
            mock.patch.object(browser_context, "_active_window_id", return_value="42"),
            mock.patch.object(
                browser_context, "_window_class", return_value="Alacritty"
            ),
            mock.patch.object(browser_context, "_send_key") as send_key,
        ):
            self.assertIsNone(browser_context.focused_firefox_url())
        send_key.assert_not_called()

    def test_url_capture_restores_address_bar_and_clipboard(self) -> None:
        url = "https://github.com/example/project/issues/42"
        with (
            mock.patch.object(
                browser_context.shutil, "which", return_value="/usr/bin/tool"
            ),
            mock.patch.object(browser_context, "_active_window_id", return_value="42"),
            mock.patch.object(browser_context, "_window_class", return_value="firefox"),
            mock.patch.object(
                browser_context,
                "_read_clipboard",
                side_effect=[(True, "previous"), (True, url), (True, url)],
            ),
            mock.patch.object(
                browser_context, "_send_key", return_value=True
            ) as send_key,
            mock.patch.object(browser_context, "_write_clipboard") as write,
            mock.patch.object(browser_context.time, "sleep"),
        ):
            self.assertEqual(browser_context.focused_firefox_url(), url)

        self.assertEqual(
            send_key.call_args_list,
            [
                mock.call("42", "ctrl+l"),
                mock.call("42", "ctrl+c"),
                mock.call("42", "Escape"),
            ],
        )
        write.assert_called_once_with("previous")

    def test_focus_change_aborts_capture_without_sending_more_keys(self) -> None:
        with (
            mock.patch.object(
                browser_context.shutil, "which", return_value="/usr/bin/tool"
            ),
            mock.patch.object(
                browser_context, "_active_window_id", side_effect=["42", "99", "99"]
            ),
            mock.patch.object(browser_context, "_window_class", return_value="firefox"),
            mock.patch.object(
                browser_context,
                "_read_clipboard",
                side_effect=[(True, "previous"), (True, "previous")],
            ),
            mock.patch.object(
                browser_context, "_send_key", return_value=True
            ) as send_key,
            mock.patch.object(browser_context, "_write_clipboard") as write,
            mock.patch.object(browser_context.time, "sleep"),
        ):
            self.assertIsNone(browser_context.focused_firefox_url())

        send_key.assert_called_once_with("42", "ctrl+l")
        write.assert_not_called()


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
            browser_context, "focused_github_context", return_value=context
        ):
            request = browser_context.enrich_request("make sure this PR works")
        self.assertEqual(request.github_repository, "example/project")
        self.assertEqual(request.github_pull_request, 7)

    def test_enriched_request_preserves_validated_repository_identity(self) -> None:
        context = browser_context.GitHubContext(
            "GitHub context", github_repository="example/project"
        )
        with mock.patch.object(
            browser_context, "focused_github_context", return_value=context
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
            "focused_github_context",
            side_effect=RuntimeError("desktop unavailable"),
        ):
            self.assertEqual(browser_context.enrich_request("hello"), "hello")


if __name__ == "__main__":
    unittest.main()
