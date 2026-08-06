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
    def test_repository_identity_validation_rejects_paths(self) -> None:
        self.assertEqual(
            GitHubClient.validate_repository("Example/project.git"),
            "Example/project",
        )
        for invalid in ("project", "../project", "owner/../repo", "owner/repo/extra"):
            with self.subTest(invalid=invalid), self.assertRaises(GitHubError):
                GitHubClient.validate_repository(invalid)

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
        with (
            mock.patch.object(client, "_repo_view", side_effect=[None, fork]) as view,
            mock.patch.object(client, "_run", return_value=_completed()) as run,
        ):
            self.assertEqual(client.ensure_fork(source, "me"), fork)
        self.assertEqual(view.call_count, 2)
        run.assert_called_once_with(
            ["gh", "repo", "fork", "source/project", "--clone=false"],
            timeout=120,
        )

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


if __name__ == "__main__":
    unittest.main()
