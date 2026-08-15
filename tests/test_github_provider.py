from __future__ import annotations

import unittest
from unittest import mock

from local_voice_harness.context_fragment import ContextProvider
from local_voice_harness.integrations import github
from local_voice_harness.integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubForkPlan,
    GitHubIssue,
    GitHubIssueCreationPlan,
    GitHubIssueCreationResult,
    GitHubProvider,
    GitHubPullRequest,
    GitHubPullRequestCheckoutInputs,
    GitHubPullRequestPlan,
    GitHubRepoCreationPlan,
    GitHubRepoCreationResult,
    GitHubRepository,
    dump_github_provider_state,
    load_github_provider_state,
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
            "work on example/project issue 42",
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

    def test_owns_and_canonicalizes_repository_and_issue_identities(self) -> None:
        self.assertTrue(self.provider.owns_repository_reference("Owner/Repo.git"))
        self.assertFalse(self.provider.owns_repository_reference("owner/repo#35"))
        self.assertFalse(self.provider.owns_repository_reference("API-79"))
        self.assertEqual(
            self.provider.canonicalize_repository_reference("Owner/Repo.git"),
            "Owner/Repo",
        )
        self.assertTrue(self.provider.owns_issue_reference("owner/repo#35"))
        self.assertFalse(self.provider.owns_issue_reference("owner/repo"))
        self.assertFalse(self.provider.owns_issue_reference("example#42"))
        self.assertFalse(self.provider.owns_issue_reference("API-79"))
        self.assertEqual(
            self.provider.canonicalize_issue_reference("owner/repo#35"),
            "owner/repo#35",
        )

    def test_provider_plans_observes_and_submits_fork(self) -> None:
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        fork = GitHubRepository(
            "me/project",
            "https://github.com/me/project",
            False,
            "main",
            "source/project",
        )
        self.client.prepare_public_fork.return_value = (source, "me", "me/project")
        self.client.inspect_public_repository.return_value = source
        self.client.reconcile_fork.return_value = None
        self.client.ensure_fork.return_value = fork

        plan = self.provider.plan_fork("source/project")

        self.assertEqual(plan, GitHubForkPlan(source, "me", "me/project"))
        self.assertIsNone(self.provider.observe_fork(plan))
        self.assertEqual(self.provider.submit_fork(plan, confirmed=True), fork)
        self.client.reconcile_fork.assert_called_once_with(source, "me/project")
        self.client.ensure_fork.assert_called_once_with(
            source,
            "me",
            checkpoint=None,
            before_submit=None,
        )

    def test_provider_requires_confirmation_before_fork_submission(self) -> None:
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        plan = GitHubForkPlan(source, "me", "me/project")

        with self.assertRaisesRegex(GitHubError, "confirmation"):
            self.provider.submit_fork(plan, confirmed=False)

        self.client.ensure_fork.assert_not_called()

    def test_provider_plans_observes_and_submits_issue_creation(self) -> None:
        result = GitHubIssueCreationResult(
            GitHubIssue("example", "project", 42),
            "https://github.com/example/project/issues/42",
            "a" * 32,
        )
        self.client.observe_issue.return_value = None
        self.client.submit_issue.return_value = result

        plan = self.provider.plan_issue_creation(
            " example/project.git ",
            " Fix the reader ",
            " Detailed body ",
            correlation_marker="a" * 32,
        )

        self.assertEqual(
            plan,
            GitHubIssueCreationPlan(
                "example/project",
                "Fix the reader",
                "Detailed body",
                "a" * 32,
            ),
        )
        self.assertIsNone(self.provider.observe_issue_creation(plan))
        self.assertEqual(
            self.provider.submit_issue_creation(plan, confirmed=True),
            result,
        )
        self.client.observe_issue.assert_called_once_with(plan)
        self.client.submit_issue.assert_called_once_with(plan, confirmed=True)

    def test_provider_generates_unique_lowercase_hex_issue_markers(self) -> None:
        first = self.provider.plan_issue_creation("example/project", "Title", "Body")
        second = self.provider.plan_issue_creation("example/project", "Title", "Body")

        self.assertRegex(first.correlation_marker, r"^[0-9a-f]{32}$")
        self.assertRegex(second.correlation_marker, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first.correlation_marker, second.correlation_marker)

    def test_provider_plans_observes_and_submits_repo_creation(self) -> None:
        created = GitHubRepoCreationResult(
            GitHubRepository(
                "alice/payments",
                "https://github.com/alice/payments",
                True,
                "main",
            ),
            "https://github.com/alice/payments",
            "a" * 32,
        )
        self.client.observe_repository_creation.return_value = None
        self.client.submit_repository_creation.return_value = created

        plan = self.provider.plan_repository_creation(
            " alice ",
            " payments ",
            "private",
            correlation_marker="a" * 32,
        )

        self.assertEqual(
            plan,
            GitHubRepoCreationPlan("alice", "payments", "private", "a" * 32),
        )
        self.assertIsNone(self.provider.observe_repository_creation(plan))
        self.assertEqual(
            self.provider.submit_repository_creation(plan, confirmed=True),
            created,
        )
        self.client.observe_repository_creation.assert_called_once_with(plan)
        self.client.submit_repository_creation.assert_called_once_with(
            plan, confirmed=True
        )

    def test_provider_delegates_organization_membership_identity(self) -> None:
        self.client.list_organizations.return_value = ("acme", "widgets")
        self.client.require_organization_membership.return_value = "acme"

        self.assertEqual(self.provider.list_organizations(), ("acme", "widgets"))
        self.assertEqual(
            self.provider.require_organization_membership("acme"),
            "acme",
        )
        self.client.list_organizations.assert_called_once_with()
        self.client.require_organization_membership.assert_called_once_with("acme")

    def test_provider_requires_confirmation_before_repo_submission(self) -> None:
        plan = GitHubRepoCreationPlan("alice", "payments", "private", "a" * 32)
        with self.assertRaisesRegex(GitHubError, "confirmation"):
            self.provider.submit_repository_creation(plan, confirmed=False)
        self.client.submit_repository_creation.assert_not_called()

    def test_provider_requires_confirmation_before_issue_submission(self) -> None:
        plan = GitHubIssueCreationPlan("example/project", "Title", "Body", "a" * 32)
        with self.assertRaisesRegex(GitHubError, "confirmation"):
            self.provider.submit_issue(plan, confirmed=False)
        self.client.submit_issue.assert_not_called()

    def test_provider_rejects_inconsistent_persisted_fork_plan(self) -> None:
        source = GitHubRepository(
            "source/project",
            "https://evil.example/source/project",
            False,
            "main",
        )
        with self.assertRaisesRegex(GitHubError, "source URL"):
            self.provider.validate_fork_plan(GitHubForkPlan(source, "me", "me/project"))

        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        with self.assertRaisesRegex(GitHubError, "invalid target"):
            self.provider.validate_fork_plan(
                GitHubForkPlan(source, "me", "other/project")
            )
        with self.assertRaisesRegex(GitHubError, "does not match its target"):
            self.provider.validate_fork_plan(
                GitHubForkPlan(source, "me", "me/project"),
                materialized_repository="other/project",
            )

    def test_provider_plans_pull_request_checkout_inputs_without_mutation(
        self,
    ) -> None:
        source = GitHubRepository(
            "Example/Project",
            "https://github.com/Example/Project",
            False,
            "main",
        )
        self.client.inspect_repository.return_value = source
        self.client.pull_request_details.return_value = {
            "number": 7,
            "headRefOid": "a" * 40,
        }

        plan = self.provider.plan_pull_request("example/project", 7)

        self.assertEqual(
            plan,
            GitHubPullRequestPlan(
                source,
                GitHubPullRequest("Example", "Project", 7),
                GitHubPullRequestCheckoutInputs(
                    "https://github.com/Example/Project",
                    "refs/pull/7/head",
                    "a" * 40,
                ),
            ),
        )
        self.client.inspect_repository.assert_called_once_with("example/project")
        self.client.pull_request_details.assert_called_once_with(
            GitHubPullRequest("Example", "Project", 7)
        )
        self.client.checkout_pull_request.assert_not_called()

    def test_provider_rejects_untrusted_pr_checkout_metadata(self) -> None:
        source = GitHubRepository(
            "example/project",
            "https://evil.example/example/project",
            False,
            "main",
        )
        self.client.inspect_repository.return_value = source
        self.client.pull_request_details.return_value = {"headRefOid": "a" * 40}
        with self.assertRaisesRegex(GitHubError, "does not match"):
            self.provider.plan_pull_request("example/project", 7)

        source = GitHubRepository(
            "example/project",
            "https://github.com/example/project",
            False,
            "main",
        )
        self.client.inspect_repository.return_value = source
        self.client.pull_request_details.return_value = {"headRefOid": "not-an-oid"}
        with self.assertRaisesRegex(GitHubError, "head OID"):
            self.provider.plan_pull_request("example/project", 7)

        with self.assertRaisesRegex(GitHubError, "does not match its number"):
            self.provider.validate_pull_request_checkout_inputs(
                "example/project",
                7,
                GitHubPullRequestCheckoutInputs(
                    "https://github.com/example/project",
                    "refs/pull/8/head",
                    "a" * 40,
                ),
            )

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
        self.assertEqual(fragment.issue_scope, "example/private")
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
        self.assertEqual(fragment.issue_scope, "example/project")
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


class GitHubProviderStateTests(unittest.TestCase):
    def test_legacy_state_round_trips_through_nested_v1_state(self) -> None:
        flat = {
            "github_repository": "source/project",
            "github_issue": 42,
            "github_issue_url": "https://github.com/source/project/issues/42",
            "github_issue_context": "issue context",
            "github_pull_request": 7,
            "pull_request_worktree_state": "ready",
            "pull_request_branch": "voice/pr",
            "pull_request_worktree_error": None,
            "pull_request_remote_url": "https://github.com/source/project",
            "pull_request_head_ref": "refs/pull/7/head",
            "pull_request_head_oid": "a" * 40,
            "fork_requested": True,
            "fork_confirmed": True,
            "fork_committed": True,
            "fork_exists": False,
            "fork_dispatch_exited": True,
            "fork_committed_at": 1.5,
            "fork_operation_state": "ambiguous",
            "fork_operation_source": "source/project",
            "fork_operation_source_url": "https://github.com/source/project",
            "fork_operation_source_parent": None,
            "fork_operation_source_default_branch": "main",
            "fork_operation_source_private": False,
            "fork_operation_login": "me",
            "fork_operation_target": "me/project",
            "fork_repository": None,
            "fork_reconcile_attempts": 2,
            "fork_absent_observations": 1,
            "fork_next_reconcile_at": 3.0,
            "fork_last_reconciled_at": 2.0,
            "fork_confirmed_absent_at": None,
            "fork_automatic_reconcile_stopped_at": None,
            "fork_retained_at": None,
        }

        nested = dump_github_provider_state(flat)
        expected = dict(flat)
        expected.pop("pull_request_branch")

        self.assertEqual(nested["version"], 1)
        self.assertEqual(
            nested["repository"],
            {"name": "source/project"},
        )
        self.assertEqual(
            nested["pull_request"],
            {
                "number": 7,
                "worktree_state": "ready",
                "worktree_error": None,
                "remote_url": "https://github.com/source/project",
                "head_ref": "refs/pull/7/head",
                "head_oid": "a" * 40,
            },
        )
        self.assertEqual(load_github_provider_state(nested), expected)
        self.assertEqual(
            GitHubProvider.load_state(GitHubProvider.dump_state(flat)), expected
        )

    def test_legacy_pull_request_branch_still_imports(self) -> None:
        nested = {
            "version": 1,
            "pull_request": {
                "number": 7,
                "branch": "voice/pr",
                "worktree_state": "ready",
            },
        }

        loaded = load_github_provider_state(nested)
        self.assertEqual(loaded["pull_request_branch"], "voice/pr")
        self.assertEqual(loaded["github_pull_request"], 7)

    def test_legacy_created_identity_loads_and_native_dump_omits_mirrors(
        self,
    ) -> None:
        nested = {
            "version": 1,
            "repository": {"name": "source/project"},
            "issue": {
                "number": 42,
                "url": "https://github.com/source/project/issues/42",
            },
            "issue_creation": {
                "requested": True,
                "created_number": 42,
                "created_url": "https://github.com/source/project/issues/42",
            },
        }

        loaded = load_github_provider_state(nested)
        self.assertEqual(loaded["github_issue"], 42)
        self.assertEqual(loaded["github_issue_created_number"], 42)
        self.assertEqual(
            loaded["github_issue_created_url"],
            "https://github.com/source/project/issues/42",
        )

        dumped = dump_github_provider_state(loaded)
        self.assertEqual(
            dumped["issue"],
            {
                "number": 42,
                "url": "https://github.com/source/project/issues/42",
            },
        )
        creation = dumped.get("issue_creation")
        self.assertIsInstance(creation, dict)
        assert isinstance(creation, dict)
        self.assertEqual(creation.get("requested"), True)
        self.assertNotIn("created_number", creation)
        self.assertNotIn("created_url", creation)

    def test_malformed_provider_state_is_rejected(self) -> None:
        malformed = (
            {"unknown": True},
            {"version": 2},
            {"version": 1, "fork": "not-an-object"},
            {"version": 1, "pull_request": {"unknown": True}},
        )
        for state in malformed:
            with self.subTest(state=state), self.assertRaises(GitHubError):
                load_github_provider_state(state)


if __name__ == "__main__":
    unittest.main()
