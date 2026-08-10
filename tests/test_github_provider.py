from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness.context_fragment import ContextProvider
from local_voice_harness.integrations import github
from local_voice_harness.integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubIssue,
    GitHubProvider,
    GitHubPullRequest,
)


class GitHubUrlTests(unittest.TestCase):
    def test_extracts_canonical_issue_url(self) -> None:
        issue = github.github_issue_from_url(
            "https://github.com/example/project/issues/42?notification_referrer=x"
        )
        self.assertEqual(issue, GitHubIssue("example", "project", 42))
        self.assertIsNone(
            github.github_issue_from_url(
                "https://github.example/example/project/issues/42"
            )
        )
        self.assertIsNone(
            github.github_issue_from_url("https://github.com/example/project/pull/42")
        )

    def test_extracts_spoken_issue_references(self) -> None:
        issue = GitHubIssue("example", "project", 42)
        for text in (
            "work on example/project#42",
            "please handle issue 42 in example/project",
            "work on https://github.com/example/project/issues/42",
        ):
            with self.subTest(text=text):
                self.assertEqual(github.github_issue_from_text(text), issue)
        self.assertIsNone(github.github_issue_from_text("work on issue 42"))

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
                    github.github_repository_from_url(
                        f"https://github.com/example/project{suffix}"
                    ),
                    "example/project",
                )
        self.assertIsNone(
            github.github_repository_from_url("https://evil.example/example/project")
        )
        self.assertIsNone(
            github.github_repository_from_url("https://github.com/example/..")
        )

    def test_extracts_scope_only_from_repository_issue_list(self) -> None:
        self.assertEqual(
            github.github_issues_repository_from_url(
                "https://github.com/example/project/issues?q=is%3Aopen"
            ),
            "example/project",
        )
        for url in (
            "https://github.com/example/project",
            "https://github.com/example/project/issues/42",
            "https://github.com/example/project/pulls",
        ):
            with self.subTest(url=url):
                self.assertIsNone(github.github_issues_repository_from_url(url))

    def test_extracts_canonical_pull_request_url(self) -> None:
        pull_request = github.github_pull_request_from_url(
            "https://github.com/example/project/pull/42/files?diff=split"
        )
        self.assertEqual(
            pull_request,
            GitHubPullRequest("example", "project", 42),
        )
        self.assertIsNone(
            github.github_pull_request_from_url(
                "https://github.com/example/project/issues/42"
            )
        )

    def test_rejects_malformed_or_untrusted_urls(self) -> None:
        invalid_urls = (
            "http://github.com/example/project/issues/42",
            "https://user@github.com/example/project/issues/42",
            "https://github.com:444/example/project/issues/42",
            "https://github.com:invalid/example/project/issues/42",
        )
        provider = GitHubProvider(mock.create_autospec(GitHubClient, instance=True))
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(provider.matches(url))
                self.assertIsNone(provider.capture(url))

        malformed_path = provider.capture("https://github.com/../project/issues/42")
        self.assertIsNotNone(malformed_path)
        assert malformed_path is not None
        self.assertIsNone(malformed_path.repository_reference)


class GitHubProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = mock.create_autospec(GitHubClient, instance=True)
        self.provider = GitHubProvider(self.client)

    def test_satisfies_context_provider_contract(self) -> None:
        self.assertIsInstance(self.provider, ContextProvider)

    def test_issue_capture_returns_bounded_context_and_structured_metadata(
        self,
    ) -> None:
        url = "https://github.com/example/private/issues/42"
        self.client.issue_details.return_value = {
            "number": 42,
            "title": "Fix the reader",
            "state": "OPEN",
            "author": {"login": "octocat"},
            "labels": [{"name": "bug"}],
            "body": "b" * 20_000,
            "comments": [
                {"author": {"login": "dev"}, "body": "c" * 2_000} for _ in range(20)
            ],
            "url": url,
        }

        fragment = self.provider.capture(url)

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.source, "github")
        self.assertEqual(fragment.repository_reference, "example/private")
        self.assertEqual(fragment.issue_reference, "example/private#42")
        self.assertEqual(fragment.issue_number, 42)
        self.assertIsNone(fragment.pull_request_number)
        self.assertIn("Title: Fix the reader", fragment.text)
        self.assertIn("untrusted external context", fragment.text)
        self.assertLess(len(fragment.text), 10_000)
        self.assertEqual(fragment.text.count("- dev:"), github.MAX_COMMENTS)
        self.client.issue_details.assert_called_once_with(
            GitHubIssue("example", "private", 42)
        )

    def test_issue_cli_failure_keeps_validated_identity(self) -> None:
        self.client.issue_details.side_effect = GitHubError("gh unavailable")

        fragment = self.provider.capture("https://github.com/example/project/issues/42")

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.repository_reference, "example/project")
        self.assertEqual(fragment.issue_reference, "example/project#42")
        self.assertEqual(fragment.issue_number, 42)
        self.assertIn("Issue details could not be fetched", fragment.text)

    def test_spoken_issue_capture_uses_same_structured_fragment(self) -> None:
        self.client.issue_details.side_effect = GitHubError("offline")

        fragment = self.provider.capture_text("work on example/project#7")

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.repository_reference, "example/project")
        self.assertEqual(fragment.issue_reference, "example/project#7")
        self.assertEqual(fragment.issue_number, 7)
        self.assertIn("https://github.com/example/project/issues/7", fragment.text)

    def test_non_issue_text_does_not_call_cli(self) -> None:
        self.assertIsNone(self.provider.capture_text("work on issue 42"))
        self.client.issue_details.assert_not_called()

    def test_repository_capture_returns_canonical_structured_identity(self) -> None:
        url = "https://github.com/example/project"
        self.client.repository_context_details.return_value = {
            "nameWithOwner": "Example/Project",
            "description": "Useful project",
            "isPrivate": False,
            "defaultBranchRef": {"name": "main"},
            "url": url,
        }

        fragment = self.provider.capture(url)

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.source, "github")
        self.assertEqual(fragment.repository_reference, "Example/Project")
        self.assertIsNone(fragment.issue_reference)
        self.assertIsNone(fragment.issue_number)
        self.assertIn("focused GitHub repository", fragment.text)
        self.assertIn("Default branch: main", fragment.text)
        self.client.repository_context_details.assert_called_once_with(
            "example/project"
        )

    def test_repository_response_cannot_replace_focused_identity(self) -> None:
        self.client.repository_context_details.return_value = {
            "nameWithOwner": "attacker/other",
            "isPrivate": False,
        }

        fragment = self.provider.capture("https://github.com/example/project")

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.repository_reference, "example/project")

    def test_repository_cli_failure_keeps_validated_identity(self) -> None:
        self.client.repository_context_details.side_effect = GitHubError("offline")

        fragment = self.provider.capture("https://github.com/example/project")

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.repository_reference, "example/project")
        self.assertIn("Repository details could not be fetched", fragment.text)

    def test_issue_list_capture_adds_repository_scope_without_scraping(self) -> None:
        self.client.repository_context_details.return_value = {
            "nameWithOwner": "Example/Project",
            "isPrivate": False,
        }

        fragment = self.provider.capture(
            "https://github.com/example/project/issues?q=is%3Aopen"
        )

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.repository_reference, "Example/Project")
        self.assertEqual(fragment.issue_scope, "Example/Project")
        self.assertIsNone(fragment.issue_number)
        self.assertNotIn("Issue: #", fragment.text)

    def test_non_issue_repository_subpage_has_no_issue_scope(self) -> None:
        self.client.repository_context_details.return_value = {
            "nameWithOwner": "example/project",
            "isPrivate": False,
        }

        fragment = self.provider.capture("https://github.com/example/project/pulls")

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertIsNone(fragment.issue_scope)

    def test_pull_request_capture_returns_structured_metadata(self) -> None:
        url = "https://github.com/example/project/pull/7"
        self.client.pull_request_details.return_value = {
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

        fragment = self.provider.capture(url)

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.repository_reference, "example/project")
        self.assertEqual(fragment.pull_request_number, 7)
        self.assertIsNone(fragment.issue_number)
        self.assertIn("Pull request: #7", fragment.text)
        self.assertIn("Branch: feature/cache into main", fragment.text)
        self.assertIn("Draft: yes", fragment.text)
        self.assertIn("Changed files: 6 (+120/-4)", fragment.text)
        self.client.pull_request_details.assert_called_once_with(
            GitHubPullRequest("example", "project", 7)
        )

    def test_pull_request_cli_failure_keeps_validated_identity(self) -> None:
        self.client.pull_request_details.side_effect = GitHubError("not found")

        fragment = self.provider.capture("https://github.com/example/project/pull/7")

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.repository_reference, "example/project")
        self.assertEqual(fragment.pull_request_number, 7)
        self.assertIn("Pull request details could not be fetched", fragment.text)

    def test_github_non_repository_page_returns_provenance_only(self) -> None:
        fragment = self.provider.capture("https://github.com/notifications")

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.source, "github")
        self.assertIsNone(fragment.repository_reference)
        self.assertIn("focused GitHub page", fragment.text)
        self.client.repository_context_details.assert_not_called()

    def test_non_github_url_is_not_captured_or_inspected(self) -> None:
        self.assertIsNone(
            self.provider.capture("https://evil.example/example/project/issues/42")
        )
        self.client.issue_details.assert_not_called()
        self.client.pull_request_details.assert_not_called()
        self.client.repository_context_details.assert_not_called()


if __name__ == "__main__":
    unittest.main()
