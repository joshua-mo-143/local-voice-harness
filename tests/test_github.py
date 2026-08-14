from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.integrations.github import (
    GitHubClient,
    GitHubCommandStartError,
    GitHubError,
    GitHubIssue,
    GitHubIssueCreationPlan,
    GitHubIssueCreationResult,
    GitHubIssueLookupError,
    GitHubIssueLookupReason,
    GitHubOperationAmbiguous,
    GitHubPullRequest,
    GitHubRepository,
)
from local_voice_harness.local_git import LocalGitOperationAmbiguous


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _repository(
    name: str,
    *,
    private: bool = False,
    parent: str | None = None,
) -> GitHubRepository:
    return GitHubRepository(
        name_with_owner=name,
        url=f"https://github.com/{name}",
        is_private=private,
        default_branch="main",
        parent=parent,
    )


class GitHubClientTests(unittest.TestCase):
    def test_command_runner_translates_process_failures(self) -> None:
        client = GitHubClient()
        with (
            mock.patch(
                "local_voice_harness.integrations.github.run_command",
                side_effect=OSError("missing"),
            ),
            self.assertRaisesRegex(GitHubError, "missing"),
        ):
            client._run(["gh", "status"])

        with (
            mock.patch(
                "local_voice_harness.integrations.github.run_command",
                side_effect=subprocess.TimeoutExpired(["gh"], 1),
            ),
            self.assertRaisesRegex(
                GitHubOperationAmbiguous,
                "outcome is ambiguous.*external side effect",
            ) as raised,
        ):
            client._run(["gh", "status"])
        self.assertIsInstance(raised.exception.__cause__, subprocess.TimeoutExpired)

        failed = _completed(stdout="fallback error", returncode=2)
        with (
            mock.patch(
                "local_voice_harness.integrations.github.run_command",
                return_value=failed,
            ),
            self.assertRaisesRegex(GitHubError, "fallback error"),
        ):
            client._run(["gh", "status"])
        with mock.patch(
            "local_voice_harness.integrations.github.run_command",
            return_value=failed,
        ):
            self.assertIs(client._run(["gh", "status"], check=False), failed)

    def test_configured_timeout_is_the_generic_command_default(self) -> None:
        client = GitHubClient(timeout=42)
        with mock.patch(
            "local_voice_harness.integrations.github.run_command",
            return_value=_completed(),
        ) as run:
            client._run(["gh", "status"])

        self.assertEqual(run.call_args.kwargs["timeout"], 42)

    def test_repository_identity_validation_rejects_paths(self) -> None:
        self.assertEqual(
            GitHubClient.validate_repository("Example/project.git"),
            "Example/project",
        )
        for invalid in (
            "project",
            "../project",
            "owner-/repo",
            "owner/../repo",
            f"owner/{'r' * 101}",
            "owner/repo/extra",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(GitHubError):
                GitHubClient.validate_repository(invalid)

    def test_optional_repository_lookup_only_counts_not_found_as_absent(
        self,
    ) -> None:
        client = GitHubClient()
        with mock.patch.object(
            client,
            "_run",
            return_value=_completed(
                stderr="GraphQL: Could not resolve to a Repository",
                returncode=1,
            ),
        ):
            self.assertIsNone(client._repo_view("me/project", required=False))

        with (
            mock.patch.object(
                client,
                "_run",
                return_value=_completed(
                    stderr="HTTP 503: service unavailable", returncode=1
                ),
            ),
            self.assertRaisesRegex(GitHubError, "service unavailable"),
        ):
            client._repo_view("me/project", required=False)

    def test_inspection_canonicalizes_and_rejects_private_repository(self) -> None:
        client = GitHubClient()
        public = {
            "nameWithOwner": "Example/Project",
            "url": "https://github.com/Example/Project",
            "isPrivate": False,
            "defaultBranchRef": {"name": "main"},
            "parent": None,
        }
        with mock.patch.object(
            client, "_run", return_value=_completed(json.dumps(public))
        ) as run:
            repository = client.inspect_public_repository("example/project")
        self.assertEqual(repository.name_with_owner, "Example/Project")
        self.assertEqual(
            run.call_args.args[0][:4],
            ["gh", "repo", "view", "example/project"],
        )

        private = dict(public, isPrivate=True)
        with (
            mock.patch.object(
                client, "_run", return_value=_completed(json.dumps(private))
            ),
            self.assertRaisesRegex(GitHubError, "private"),
        ):
            client.inspect_public_repository("example/project")

    def test_repository_and_issue_metadata_validation(self) -> None:
        client = GitHubClient()
        for payload in ("not-json", "[]"):
            with (
                self.subTest(repository_payload=payload),
                mock.patch.object(client, "_run", return_value=_completed(payload)),
                self.assertRaisesRegex(GitHubError, "malformed repository"),
            ):
                client.inspect_repository("example/project")

        with (
            mock.patch.object(
                client,
                "_run",
                return_value=_completed(stderr="permission denied", returncode=1),
            ),
            self.assertRaisesRegex(GitHubError, "permission denied"),
        ):
            client.inspect_repository("example/project")

        with self.assertRaisesRegex(GitHubError, "positive"):
            client.issue_details(GitHubIssue("example", "project", 0))
        for payload in ("not-json", "[]"):
            with (
                self.subTest(issue_payload=payload),
                mock.patch.object(client, "_run", return_value=_completed(payload)),
                self.assertRaisesRegex(GitHubError, "malformed issue"),
            ):
                client.issue_details(GitHubIssue("example", "project", 1))

    def test_login_preparation_and_fork_reconciliation(self) -> None:
        client = GitHubClient()
        with mock.patch.object(client, "_run", return_value=_completed("valid-user\n")):
            self.assertEqual(client.authenticated_login(), "valid-user")
        with (
            mock.patch.object(client, "_run", return_value=_completed("invalid/user")),
            self.assertRaisesRegex(GitHubError, "authenticated user"),
        ):
            client.authenticated_login()

        source = _repository("Source/Project")
        with (
            mock.patch.object(client, "inspect_public_repository", return_value=source),
            mock.patch.object(client, "authenticated_login", return_value="me"),
        ):
            self.assertEqual(
                client.prepare_public_fork("source/project"),
                (source, "me", "me/Project"),
            )

        with mock.patch.object(client, "_repo_view", return_value=None):
            self.assertIsNone(client.reconcile_fork(source, "me/Project"))
        with mock.patch.object(client, "_repo_view", return_value=source):
            self.assertIs(client.reconcile_fork(source, "Source/Project"), source)
        unrelated = _repository("me/Project", parent="other/project")
        with (
            mock.patch.object(client, "_repo_view", return_value=unrelated),
            self.assertRaisesRegex(GitHubError, "not a fork"),
        ):
            client.reconcile_fork(source, "me/Project")

    def test_issue_details_are_read_through_gh(self) -> None:
        client = GitHubClient()
        issue = GitHubIssue("example", "project", 42)
        details = {"number": 42, "title": "Fix it"}
        with mock.patch.object(
            client, "_run", return_value=_completed(json.dumps(details))
        ) as run:
            self.assertEqual(client.issue_details(issue), details)

        self.assertEqual(
            run.call_args.args[0][:6],
            ["gh", "issue", "view", "42", "--repo", "example/project"],
        )
        self.assertEqual(issue.reference, "example/project#42")
        self.assertEqual(issue.url, "https://github.com/example/project/issues/42")

    def test_issue_submission_uses_stdin_and_validates_canonical_result(self) -> None:
        client = GitHubClient()
        plan = GitHubIssueCreationPlan(
            "example/project",
            "Fix the reader",
            "Detailed private body",
            "a" * 32,
        )
        with mock.patch.object(
            client,
            "_run",
            return_value=_completed("https://github.com/example/project/issues/42\n"),
        ) as run:
            result = client.submit_issue(plan, confirmed=True)

        self.assertEqual(
            result,
            GitHubIssueCreationResult(
                GitHubIssue("example", "project", 42),
                "https://github.com/example/project/issues/42",
                "a" * 32,
            ),
        )
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            [
                "gh",
                "issue",
                "create",
                "--repo",
                "example/project",
                "--title",
                "Fix the reader",
                "--body-file",
                "-",
            ],
        )
        self.assertNotIn(plan.body, command)
        self.assertEqual(
            run.call_args.kwargs["stdin"],
            "Detailed private body\n\n"
            f"<!-- local-voice-harness-correlation:{'a' * 32} -->\n",
        )

    def test_issue_submission_requires_confirmation_before_running(self) -> None:
        client = GitHubClient()
        plan = GitHubIssueCreationPlan("example/project", "Title", "Body", "a" * 32)
        with (
            mock.patch.object(client, "_run") as run,
            self.assertRaisesRegex(GitHubError, "confirmation"),
        ):
            client.submit_issue(plan, confirmed=False)
        run.assert_not_called()

    def test_issue_submission_rejects_invalid_plan_and_result(self) -> None:
        client = GitHubClient()
        invalid_plans = (
            GitHubIssueCreationPlan("not-a-repository", "Title", "Body", "a" * 32),
            GitHubIssueCreationPlan("example/project", "", "Body", "a" * 32),
            GitHubIssueCreationPlan("example/project", "Title", "", "a" * 32),
            GitHubIssueCreationPlan("example/project", "Title", "Body", "A" * 32),
            GitHubIssueCreationPlan(
                "example/project", "Title", f"Body {'a' * 32}", "a" * 32
            ),
        )
        for plan in invalid_plans:
            with (
                self.subTest(plan=plan),
                mock.patch.object(client, "_run") as run,
                self.assertRaises(GitHubError),
            ):
                client.submit_issue(plan, confirmed=True)
            run.assert_not_called()

        plan = GitHubIssueCreationPlan("example/project", "Title", "Body", "a" * 32)
        for url in (
            "https://github.com/example/project/issues/42?query=x",
            "https://github.com/other/project/issues/42",
            "not a URL",
        ):
            with (
                self.subTest(url=url),
                mock.patch.object(client, "_run", return_value=_completed(url)),
                self.assertRaises(GitHubOperationAmbiguous),
            ):
                client.submit_issue(plan, confirmed=True)

    def test_issue_submission_timeout_is_ambiguous(self) -> None:
        client = GitHubClient()
        plan = GitHubIssueCreationPlan("example/project", "Title", "Body", "a" * 32)
        with (
            mock.patch(
                "local_voice_harness.integrations.github.run_command",
                side_effect=subprocess.TimeoutExpired(["gh", "issue", "create"], 30),
            ),
            self.assertRaises(GitHubOperationAmbiguous),
        ):
            client.submit_issue(plan, confirmed=True)

    def test_issue_submission_nonzero_exit_is_ambiguous(self) -> None:
        client = GitHubClient()
        plan = GitHubIssueCreationPlan("example/project", "Title", "Body", "a" * 32)
        with (
            mock.patch(
                "local_voice_harness.integrations.github.run_command",
                return_value=_completed(
                    stderr="connection closed after response", returncode=1
                ),
            ),
            self.assertRaisesRegex(
                GitHubOperationAmbiguous,
                "write exited without proving.*side effect may already",
            ),
        ):
            client.submit_issue(plan, confirmed=True)

    def test_issue_submission_start_failure_is_definitive(self) -> None:
        client = GitHubClient()
        plan = GitHubIssueCreationPlan("example/project", "Title", "Body", "a" * 32)
        with (
            mock.patch(
                "local_voice_harness.integrations.github.run_command",
                side_effect=OSError("executable missing"),
            ),
            self.assertRaisesRegex(GitHubCommandStartError, "failed to start"),
        ):
            client.submit_issue(plan, confirmed=True)

    def test_issue_observation_lists_recent_issues_and_matches_marker(self) -> None:
        client = GitHubClient()
        plan = GitHubIssueCreationPlan("example/project", "Title", "Body", "a" * 32)
        payload = [
            {
                "number": 44,
                "html_url": "https://github.com/example/project/issues/44",
                "body": "Other issue",
            },
            {
                "number": 42,
                "html_url": "https://github.com/example/project/issues/42",
                "body": (
                    f"Body\n\n<!-- local-voice-harness-correlation:{'a' * 32} -->"
                ),
            },
        ]
        with mock.patch.object(
            client, "_run", return_value=_completed(json.dumps(payload))
        ) as run:
            result = client.observe_issue(plan)

        self.assertEqual(result, client._creation_result(plan, payload[1]["html_url"]))
        command = run.call_args.args[0]
        self.assertEqual(
            command[:5],
            ["gh", "api", "--method", "GET", "repos/example/project/issues"],
        )
        self.assertNotIn("search", command)

        with mock.patch.object(
            client, "_run", return_value=_completed(json.dumps(payload[:1]))
        ):
            self.assertIsNone(client.observe_issue(plan))

    def test_issue_lookup_failure_is_classified_and_voice_safe(self) -> None:
        client = GitHubClient()
        issue = GitHubIssue(
            "very-long-owner-name",
            "private-repository-with-a-long-name",
            42,
        )
        raw = (
            "GraphQL: Could not resolve to an Issue with the number 42. "
            "Authorization: Bearer ghp_secret-value"
        )
        with mock.patch.object(
            client,
            "_run",
            return_value=_completed(stderr=raw, returncode=1),
        ):
            with self.assertRaises(GitHubIssueLookupError) as raised:
                client.issue_details(issue)

        error = raised.exception
        self.assertEqual(
            error.reason,
            GitHubIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE,
        )
        self.assertEqual(
            str(error),
            "I couldn't find or access that GitHub issue.",
        )
        self.assertNotIn("very-long-owner", str(error))
        self.assertNotIn("ghp_secret-value", error.diagnostic)
        self.assertIn("Could not resolve to an Issue", error.diagnostic)

    def test_issue_lookup_unknown_failure_has_safe_fallback(self) -> None:
        client = GitHubClient()
        with mock.patch.object(
            client,
            "_run",
            return_value=_completed(stderr="new GraphQL failure", returncode=1),
        ):
            with self.assertRaises(GitHubIssueLookupError) as raised:
                client.issue_details(GitHubIssue("example", "project", 42))

        self.assertEqual(raised.exception.reason, GitHubIssueLookupReason.UNKNOWN)
        self.assertEqual(str(raised.exception), "I couldn't verify that GitHub issue.")

    def test_context_metadata_is_read_through_gh(self) -> None:
        client = GitHubClient()
        repository_details = {"nameWithOwner": "example/project"}
        pull_request_details = {"number": 7, "title": "Fix it"}
        with mock.patch.object(
            client,
            "_run",
            side_effect=[
                _completed(json.dumps(repository_details)),
                _completed(json.dumps(pull_request_details)),
            ],
        ) as run:
            self.assertEqual(
                client.repository_context_details("example/project"),
                repository_details,
            )
            self.assertEqual(
                client.pull_request_details(GitHubPullRequest("example", "project", 7)),
                pull_request_details,
            )

        self.assertEqual(
            run.call_args_list[0].args[0][:4],
            ["gh", "repo", "view", "example/project"],
        )
        self.assertEqual(
            run.call_args_list[1].args[0][:6],
            ["gh", "pr", "view", "7", "--repo", "example/project"],
        )
        self.assertIn("headRefOid", run.call_args_list[1].args[0][-1])

    def test_context_metadata_rejects_invalid_responses(self) -> None:
        client = GitHubClient()
        with self.assertRaisesRegex(GitHubError, "positive"):
            client.pull_request_details(GitHubPullRequest("example", "project", 0))
        for method, malformed in (
            (
                lambda: client.repository_context_details("example/project"),
                "repository",
            ),
            (
                lambda: client.pull_request_details(
                    GitHubPullRequest("example", "project", 7)
                ),
                "pull request",
            ),
        ):
            for payload in ("not-json", "[]"):
                with (
                    self.subTest(kind=malformed, payload=payload),
                    mock.patch.object(client, "_run", return_value=_completed(payload)),
                    self.assertRaisesRegex(GitHubError, f"malformed {malformed}"),
                ):
                    method()

    def test_provision_issue_reuses_matching_checkout(self) -> None:
        client = GitHubClient()
        source = _repository("Example/Project", private=True)
        checkout = Path("/repos/project")
        with (
            mock.patch.object(
                client, "inspect_repository", return_value=source
            ) as inspect,
            mock.patch.object(
                client, "find_repository_checkout", return_value=checkout
            ) as find_checkout,
            mock.patch.object(client, "ensure_repository_clone") as clone,
        ):
            provisioned = client.provision_issue(
                GitHubIssue("example", "project", 42),
                candidates=[checkout],
            )

        inspect.assert_called_once_with("example/project")
        find_checkout.assert_called_once_with(source, [checkout])
        clone.assert_not_called()
        self.assertEqual(provisioned.checkout, checkout)
        self.assertEqual(provisioned.issue, GitHubIssue("Example", "Project", 42))

    def test_existing_network_fork_is_reused(self) -> None:
        client = GitHubClient()
        source = _repository("source/project", parent="upstream/project")
        existing = _repository(
            "me/project",
            parent="upstream/project",
        )
        with (
            mock.patch.object(client, "_repo_view", return_value=existing),
            mock.patch.object(client, "_run") as run,
        ):
            self.assertIs(client.ensure_fork(source, "me"), existing)
        run.assert_not_called()

    def test_creates_missing_fork_without_cloning(self) -> None:
        client = GitHubClient(timeout=7)
        source = _repository("source/project")
        fork = _repository("me/project", parent="source/project")
        events: list[str] = []

        def submit(
            command: list[str], *, timeout: float = 30, write: bool = False
        ) -> subprocess.CompletedProcess[str]:
            self.assertTrue(write)
            events.append("submit")
            return _completed()

        with (
            mock.patch.object(client, "_repo_view", side_effect=[None, fork]) as view,
            mock.patch.object(client, "_run", side_effect=submit) as run,
        ):
            self.assertEqual(
                client.ensure_fork(
                    source,
                    "me",
                    checkpoint=lambda: events.append("checkpoint"),
                    before_submit=lambda: events.append("dispatching"),
                ),
                fork,
            )
        self.assertEqual(view.call_count, 2)
        self.assertEqual(events, ["checkpoint", "dispatching", "submit"])
        run.assert_called_once_with(
            ["gh", "repo", "fork", "source/project", "--clone=false"],
            timeout=120,
            write=True,
        )

    def test_public_fork_provisioning_requires_confirmation(self) -> None:
        client = GitHubClient()
        with (
            mock.patch.object(client, "inspect_public_repository") as inspect,
            self.assertRaisesRegex(GitHubError, "confirmation"),
        ):
            client.provision_public_fork("source/project", confirmed=False)
        inspect.assert_not_called()

    def test_clone_delegates_owner_qualified_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary)
            root = allowed / "src"
            client = GitHubClient(clone_root=root, allowed_root=allowed)
            source = _repository("source/project")
            fork = _repository("me/project", parent="source/project")
            checkout = (root / "source" / "project").resolve()

            def materialize(_relative: Path, **kwargs: object) -> Path:
                finalize = kwargs["finalize"]
                assert callable(finalize)
                finalize(checkout)
                return checkout

            with (
                mock.patch.object(
                    client.local_git,
                    "materialize",
                    side_effect=materialize,
                ) as clone,
                mock.patch.object(
                    client.local_git,
                    "ensure_remote",
                ) as upstream,
            ):
                checkout = client.ensure_clone(source, fork)

            self.assertEqual(checkout, (root / "source" / "project").resolve())
            self.assertEqual(clone.call_args.args[0], Path("source/project"))
            self.assertEqual(clone.call_args.kwargs["clone_url"], fork.url)
            self.assertEqual(
                clone.call_args.kwargs["clone_command"],
                ("gh", "repo", "clone", fork.name_with_owner),
            )
            self.assertEqual(
                clone.call_args.kwargs["expected"].identity,
                "github.com/me/project",
            )
            upstream.assert_called_once_with(
                checkout,
                "upstream",
                source.url,
            )

    def test_clone_root_must_remain_inside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            outside = base / "outside"
            client = GitHubClient(
                clone_root=outside,
                allowed_root=base / "allowed",
            )
            with self.assertRaisesRegex(GitHubError, "must be inside"):
                client.ensure_clone(
                    _repository("source/project"),
                    _repository("me/project", parent="source/project"),
                )
            self.assertFalse(outside.exists())

    def test_clone_rejects_remote_url_with_wrong_identity(self) -> None:
        client = GitHubClient()
        source = GitHubRepository(
            "source/project",
            "https://example.com/source/project",
            False,
            "main",
        )
        with (
            mock.patch.object(client.local_git, "materialize") as materialize,
            self.assertRaisesRegex(GitHubError, "invalid clone URL"),
        ):
            client.ensure_repository_clone(source)
        materialize.assert_not_called()

    def test_repository_clone_uses_source_origin_without_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary)
            root = allowed / "src"
            client = GitHubClient(clone_root=root, allowed_root=allowed)
            source = _repository("source/project")
            expected_checkout = (root / "source" / "project").resolve()

            with (
                mock.patch.object(
                    client.local_git,
                    "materialize",
                    return_value=expected_checkout,
                ) as clone,
                mock.patch.object(
                    client.local_git,
                    "ensure_remote",
                ) as upstream,
            ):
                checkout = client.ensure_repository_clone(source)

            self.assertEqual(checkout, expected_checkout)
            self.assertEqual(clone.call_args.args[0], Path("source/project"))
            self.assertEqual(clone.call_args.kwargs["clone_url"], source.url)
            self.assertEqual(
                clone.call_args.kwargs["clone_command"],
                ("gh", "repo", "clone", source.name_with_owner),
            )
            self.assertNotIn("finalize", clone.call_args.kwargs)
            upstream.assert_not_called()

    def test_provision_pull_request_leaves_shared_clone_unchanged(self) -> None:
        client = GitHubClient()
        source = _repository("source/project")
        checkout = Path("/tmp/src/source/project")
        with (
            mock.patch.object(
                client, "inspect_repository", return_value=source
            ) as inspect,
            mock.patch.object(
                client, "ensure_repository_clone", return_value=checkout
            ) as clone,
            mock.patch.object(client, "checkout_pull_request") as checkout_pr,
        ):
            provisioned = client.provision_pull_request("source/project", 42)

        inspect.assert_called_once_with("source/project")
        clone.assert_called_once_with(source)
        checkout_pr.assert_not_called()
        self.assertEqual(provisioned.checkout, checkout)
        self.assertEqual(provisioned.number, 42)
        self.assertIsNone(provisioned.branch)

    def test_provision_pull_request_rejects_non_positive_number(self) -> None:
        client = GitHubClient()
        with self.assertRaisesRegex(GitHubError, "positive"):
            client.provision_pull_request("source/project", 0)

    def test_checkout_pull_request_runs_gh_in_checkout(self) -> None:
        client = GitHubClient()
        checkout = Path("/tmp/src/source/project")
        calls: list[tuple[list[str], Path | None]] = []

        def run(
            command: list[str],
            *,
            timeout: float = 30,
            check: bool = True,
            cwd: Path | None = None,
        ) -> subprocess.CompletedProcess[str]:
            calls.append((command, cwd))
            if command[1:3] == ["pr", "checkout"]:
                return _completed()
            return _completed("feature/cache\n")

        with (
            mock.patch.object(client, "_run", side_effect=run),
            mock.patch.object(
                client.local_git,
                "current_branch",
                return_value="feature/cache",
            ) as current_branch,
        ):
            branch = client.checkout_pull_request(
                checkout, 42, branch="voice/github-pr-123456789abc"
            )

        self.assertEqual(branch, "feature/cache")
        current_branch.assert_called_once_with(checkout)
        pr_command, pr_cwd = next(call for call in calls if "checkout" in call[0])
        self.assertEqual(
            pr_command,
            [
                "gh",
                "pr",
                "checkout",
                "42",
                "--branch",
                "voice/github-pr-123456789abc",
                "--force",
            ],
        )
        self.assertEqual(pr_cwd, checkout)

    def test_repository_for_checkout_reads_and_validates_github_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            client = GitHubClient(allowed_root=root)
            with (
                mock.patch.object(client.local_git, "verify_checkout") as verify,
                mock.patch.object(
                    client.local_git,
                    "git",
                    return_value=_completed("git@github.com:Example/Project.git\n"),
                ) as git,
            ):
                self.assertEqual(
                    client.repository_for_checkout(checkout),
                    "example/project",
                )

        verify.assert_called_once_with(checkout.resolve())
        git.assert_called_once_with(checkout.resolve(), "remote", "get-url", "origin")

    def test_repository_for_checkout_rejects_invalid_checkout_or_remote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            checkout.mkdir()
            client = GitHubClient(allowed_root=root)
            with self.assertRaisesRegex(GitHubError, "not a Git repository"):
                client.repository_for_checkout(checkout)

            (checkout / ".git").mkdir()
            with (
                mock.patch.object(
                    client.local_git,
                    "git",
                    return_value=_completed("https://example.com/owner/project\n"),
                ),
                self.assertRaisesRegex(GitHubError, "not a GitHub repository"),
            ):
                client.repository_for_checkout(checkout)

            outside = root.parent / "outside"
            with self.assertRaisesRegex(GitHubError, "outside"):
                client.repository_for_checkout(outside)

    def test_local_git_ambiguous_failure_is_preserved(self) -> None:
        client = GitHubClient()
        with (
            mock.patch.object(
                client.local_git,
                "materialize",
                side_effect=LocalGitOperationAmbiguous("ambiguous clone"),
            ),
            self.assertRaisesRegex(
                GitHubOperationAmbiguous,
                "ambiguous clone",
            ) as raised,
        ):
            client.ensure_repository_clone(_repository("source/project"))
        self.assertIsInstance(
            raised.exception.__cause__,
            LocalGitOperationAmbiguous,
        )


if __name__ == "__main__":
    unittest.main()
