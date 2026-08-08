from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.integrations.github import (
    GitHubClient,
    GitHubError,
    GitHubIssue,
    GitHubRepository,
)


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
                "local_voice_harness.integrations.github.subprocess.run",
                side_effect=OSError("missing"),
            ),
            self.assertRaisesRegex(GitHubError, "missing"),
        ):
            client._run(["gh", "status"])

        with (
            mock.patch(
                "local_voice_harness.integrations.github.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["gh"], 1),
            ),
            self.assertRaisesRegex(GitHubError, "timed out"),
        ):
            client._run(["gh", "status"])

        failed = _completed(stdout="fallback error", returncode=2)
        with (
            mock.patch(
                "local_voice_harness.integrations.github.subprocess.run",
                return_value=failed,
            ),
            self.assertRaisesRegex(GitHubError, "fallback error"),
        ):
            client._run(["gh", "status"])
        with mock.patch(
            "local_voice_harness.integrations.github.subprocess.run",
            return_value=failed,
        ):
            self.assertIs(client._run(["gh", "status"], check=False), failed)

    def test_repository_identity_validation_rejects_paths(self) -> None:
        self.assertEqual(
            GitHubClient.validate_repository("Example/project.git"),
            "Example/project",
        )
        for invalid in ("project", "../project", "owner/../repo", "owner/repo/extra"):
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
        client = GitHubClient()
        source = _repository("source/project")
        fork = _repository("me/project", parent="source/project")
        events: list[str] = []

        def submit(
            command: list[str], *, timeout: float = 30
        ) -> subprocess.CompletedProcess[str]:
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
        )

    def test_public_fork_provisioning_requires_confirmation(self) -> None:
        client = GitHubClient()
        with (
            mock.patch.object(client, "inspect_public_repository") as inspect,
            self.assertRaisesRegex(GitHubError, "confirmation"),
        ):
            client.provision_public_fork("source/project", confirmed=False)
        inspect.assert_not_called()

    def test_clone_uses_temporary_owner_qualified_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary)
            root = allowed / "src"
            client = GitHubClient(clone_root=root, allowed_root=allowed)
            source = _repository("source/project")
            fork = _repository("me/project", parent="source/project")
            commands: list[list[str]] = []

            def run(
                command: list[str], *, timeout: float = 30, check: bool = True
            ) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                if command[1:3] == ["repo", "clone"]:
                    Path(command[-1]).mkdir(parents=True)
                return _completed()

            with (
                mock.patch.object(client, "_run", side_effect=run),
                mock.patch.object(client, "_verify_checkout") as verify,
                mock.patch.object(client, "_ensure_upstream") as upstream,
            ):
                checkout = client.ensure_clone(source, fork)

            self.assertEqual(checkout, (root / "source" / "project").resolve())
            self.assertTrue(checkout.is_dir())
            clone = next(command for command in commands if "clone" in command)
            self.assertEqual(clone[:4], ["gh", "repo", "clone", "me/project"])
            self.assertTrue(Path(clone[-1]).name.startswith(".project.clone-"))
            verify.assert_called_once()
            upstream.assert_called_once()

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

    def test_repository_clone_uses_source_origin_without_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary)
            root = allowed / "src"
            client = GitHubClient(clone_root=root, allowed_root=allowed)
            source = _repository("source/project")
            commands: list[list[str]] = []

            def run(
                command: list[str],
                *,
                timeout: float = 30,
                check: bool = True,
                cwd: Path | None = None,
            ) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                if command[1:3] == ["repo", "clone"]:
                    Path(command[-1]).mkdir(parents=True)
                return _completed()

            with (
                mock.patch.object(client, "_run", side_effect=run),
                mock.patch.object(client, "_verify_checkout") as verify,
                mock.patch.object(client, "_ensure_upstream") as upstream,
            ):
                checkout = client.ensure_repository_clone(source)

            self.assertEqual(checkout, (root / "source" / "project").resolve())
            clone = next(command for command in commands if "clone" in command)
            self.assertEqual(clone[:4], ["gh", "repo", "clone", "source/project"])
            verify.assert_called_once()
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

        with mock.patch.object(client, "_run", side_effect=run):
            branch = client.checkout_pull_request(
                checkout, 42, branch="voice/github-pr-123456789abc"
            )

        self.assertEqual(branch, "feature/cache")
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

    def test_remote_repository_supports_https_and_ssh(self) -> None:
        self.assertEqual(
            GitHubClient._remote_repository("https://github.com/example/project.git"),
            "example/project",
        )
        self.assertEqual(
            GitHubClient._remote_repository("git@github.com:example/project.git"),
            "example/project",
        )
        self.assertIsNone(
            GitHubClient._remote_repository("https://example.com/example/project")
        )

    def test_checkout_verification_and_upstream_configuration(self) -> None:
        client = GitHubClient()
        repository = _repository("source/project")
        fork = _repository("me/project", parent="source/project")
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            with self.assertRaisesRegex(GitHubError, "not a Git repository"):
                client._verify_checkout(checkout, repository)
            (checkout / ".git").mkdir()

            with mock.patch.object(
                client,
                "_git",
                return_value=_completed("git@github.com:source/project.git\n"),
            ):
                client._verify_checkout(checkout, repository)
            with (
                mock.patch.object(
                    client,
                    "_git",
                    return_value=_completed("https://github.com/other/project.git\n"),
                ),
                self.assertRaisesRegex(GitHubError, "origin is not"),
            ):
                client._verify_checkout(checkout, repository)

            with mock.patch.object(client, "_run") as run:
                client._ensure_upstream(checkout, repository, repository)
            run.assert_not_called()

            for returncode, action in ((1, "add"), (0, "set-url")):
                with (
                    self.subTest(action=action),
                    mock.patch.object(
                        client, "_run", return_value=_completed(returncode=returncode)
                    ),
                    mock.patch.object(client, "_git") as git,
                ):
                    client._ensure_upstream(checkout, repository, fork)
                git.assert_called_once_with(
                    checkout, "remote", action, "upstream", repository.url
                )


if __name__ == "__main__":
    unittest.main()
