from __future__ import annotations

import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.local_git import (
    ExpectedRemote,
    LocalGitError,
    LocalGitOperationAmbiguous,
    LocalGitRepository,
    remote_identity,
)


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class Cancelled(RuntimeError):
    pass


class LocalGitRepositoryTests(unittest.TestCase):
    def test_remote_identity_is_protocol_independent(self) -> None:
        expected = "github.com/example/project"
        for remote in (
            "https://github.com/Example/Project.git",
            "ssh://git@github.com/Example/Project.git",
            "git@github.com:Example/Project.git",
        ):
            with self.subTest(remote=remote):
                self.assertEqual(remote_identity(remote), expected)
        self.assertEqual(
            remote_identity("https://example.com/example/project"),
            "example.com/example/project",
        )
        self.assertIsNone(remote_identity("file:///tmp/project"))

    def test_checkout_verification_find_remote_and_branch_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            checkout.mkdir()
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            expected = ExpectedRemote.from_url("https://github.com/source/project")

            with self.assertRaisesRegex(LocalGitError, "not a Git repository"):
                repository.verify_checkout(checkout, expected)
            (checkout / ".git").mkdir()

            responses = [
                _completed("git@github.com:source/project.git\n"),
                _completed("https://github.com/other/project.git\n"),
                _completed(returncode=1),
                _completed(),
                _completed("main\n"),
            ]
            with mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=responses,
            ) as run:
                repository.verify_checkout(checkout, expected)
                self.assertIsNone(
                    repository.find_checkout(
                        [checkout],
                        expected,
                        expected_label="source/project",
                    )
                )
                repository.ensure_remote(
                    checkout,
                    "upstream",
                    "https://github.com/source/project",
                )
                self.assertEqual(repository.current_branch(checkout), "main")

            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "remote",
                    "add",
                    "upstream",
                    "https://github.com/source/project",
                ],
                commands,
            )

    def test_checkout_remote_ref_verifies_oid_before_switching_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            oid = "a" * 40
            responses = [
                _completed("https://github.com/source/project\n"),
                _completed(),
                _completed(),
                _completed(f"{oid}\n"),
                _completed(),
                _completed(f"{oid}\n"),
                _completed("voice/github-pr-job\n"),
            ]

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=responses,
            ) as run:
                branch = repository.checkout_remote_ref(
                    checkout,
                    remote_url="https://github.com/source/project",
                    remote_ref="refs/pull/42/head",
                    branch="voice/github-pr-job",
                    expected_oid=oid,
                )

            self.assertEqual(branch, "voice/github-pr-job")
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "fetch",
                    "--force",
                    "--no-tags",
                    "--",
                    "https://github.com/source/project",
                    "refs/pull/42/head",
                ],
                commands,
            )
            self.assertIn(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "checkout",
                    "-B",
                    "voice/github-pr-job",
                    "FETCH_HEAD",
                ],
                commands,
            )

    def test_checkout_remote_ref_refuses_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=[
                    _completed("https://github.com/source/project\n"),
                    _completed(),
                    _completed(),
                    _completed(f"{'b' * 40}\n"),
                ],
            ) as run:
                with self.assertRaisesRegex(LocalGitError, "does not match"):
                    repository.checkout_remote_ref(
                        checkout,
                        remote_url="https://github.com/source/project",
                        remote_ref="refs/pull/42/head",
                        branch="voice/github-pr-job",
                        expected_oid="a" * 40,
                    )

            self.assertFalse(
                any("checkout" in call.args[0] for call in run.call_args_list)
            )

    def test_checkout_remote_ref_must_be_inside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "allowed"
            checkout = root / "outside"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(
                clone_root=allowed / "src",
                allowed_root=allowed,
            )

            with self.assertRaisesRegex(LocalGitError, "escapes"):
                repository.checkout_remote_ref(
                    checkout,
                    remote_url="https://github.com/source/project",
                    remote_ref="refs/pull/42/head",
                    branch="voice/github-pr-job",
                    expected_oid="a" * 40,
                )

    def test_checkout_remote_ref_rejects_invalid_inputs_before_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            invalid = (
                ("file:///tmp/project", "refs/pull/42/head", "a" * 40),
                (
                    "https://github.com/source/project",
                    "refs/pull/42 bad/head",
                    "a" * 40,
                ),
                (
                    "https://github.com/source/project",
                    "refs/pull/42/head",
                    "not-an-oid",
                ),
            )
            for remote_url, remote_ref, expected_oid in invalid:
                with (
                    self.subTest(remote_ref=remote_ref),
                    self.assertRaises(LocalGitError),
                ):
                    repository.checkout_remote_ref(
                        checkout,
                        branch="voice/github-pr-job",
                        remote_url=remote_url,
                        remote_ref=remote_ref,
                        expected_oid=expected_oid,
                    )

    def test_checkout_remote_ref_detects_post_checkout_oid_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            oid = "a" * 40

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=[
                    _completed("https://github.com/source/project\n"),
                    _completed(),
                    _completed(),
                    _completed(f"{oid}\n"),
                    _completed(),
                    _completed(f"{'b' * 40}\n"),
                ],
            ):
                with self.assertRaisesRegex(LocalGitError, "changed unexpectedly"):
                    repository.checkout_remote_ref(
                        checkout,
                        remote_url="https://github.com/source/project",
                        remote_ref="refs/pull/42/head",
                        branch="voice/github-pr-job",
                        expected_oid=oid,
                    )

    def test_checkout_remote_ref_rejects_wrong_repository_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                return_value=_completed("https://github.com/other/project\n"),
            ) as run:
                with self.assertRaisesRegex(LocalGitError, "origin"):
                    repository.checkout_remote_ref(
                        checkout,
                        remote_url="https://github.com/source/project",
                        remote_ref="refs/pull/42/head",
                        branch="voice/github-pr-job",
                        expected_oid="a" * 40,
                    )

            self.assertEqual(run.call_count, 1)

    def test_materialize_is_contained_temporary_verified_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary)
            root = allowed / "src"
            destination = root / "source" / "project"
            expected = ExpectedRemote.from_url("https://github.com/me/project")
            repository = LocalGitRepository(
                clone_root=root,
                allowed_root=allowed,
                lock_name=".compat.lock",
            )
            commands: list[list[str]] = []

            def run(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                if command[1:3] == ["clone", "--"]:
                    staging = Path(command[-1])
                    self.assertFalse(destination.exists())
                    self.assertTrue(staging.name.startswith(".project.clone-"))
                    staging.mkdir()
                    (staging / ".git").mkdir()
                    return _completed()
                return _completed("git@github.com:me/project.git\n")

            finalized: list[Path] = []
            with mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=run,
            ):
                checkout = repository.materialize(
                    Path("source/project"),
                    clone_url="https://github.com/me/project",
                    expected=expected,
                    expected_label="me/project",
                    finalize=finalized.append,
                )

            self.assertEqual(checkout, destination.resolve())
            self.assertTrue((checkout / ".git").is_dir())
            self.assertEqual(finalized[0].parent, destination.parent)
            self.assertTrue(finalized[0].name.startswith(".project.clone-"))
            self.assertEqual(
                commands[0][:4],
                ["git", "clone", "--", "https://github.com/me/project"],
            )
            self.assertTrue((root / ".compat.lock").exists())

    def test_existing_destination_is_verified_reused_and_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "source" / "project"
            (destination / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            finalized: list[Path] = []

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                return_value=_completed("https://github.com/source/project.git\n"),
            ) as run:
                checkout = repository.materialize(
                    Path("source/project"),
                    clone_url="https://github.com/source/project",
                    expected=ExpectedRemote.from_url(
                        "https://github.com/source/project"
                    ),
                    finalize=finalized.append,
                )

            self.assertEqual(checkout, destination)
            self.assertEqual(finalized, [destination])
            self.assertFalse(
                any("clone" in call.args[0] for call in run.call_args_list)
            )

    def test_observe_materialized_never_clones_and_rejects_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "source" / "project"
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            expected = ExpectedRemote.from_url("https://example.com/source/project")

            with mock.patch("local_voice_harness.local_git.run_command") as run:
                self.assertIsNone(
                    repository.observe_materialized(
                        Path("source/project"),
                        expected=expected,
                    )
                )
            run.assert_not_called()
            self.assertFalse(destination.exists())

            (destination / ".git").mkdir(parents=True)
            with (
                mock.patch(
                    "local_voice_harness.local_git.run_command",
                    return_value=_completed("https://example.com/other/project\n"),
                ) as run,
                self.assertRaisesRegex(LocalGitError, "origin is not"),
            ):
                repository.observe_materialized(
                    Path("source/project"),
                    expected=expected,
                )
            self.assertFalse(
                any("clone" in call.args[0] for call in run.call_args_list)
            )

    def test_observe_materialized_binds_generic_remote_across_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "source" / "project"
            (destination / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            expected = ExpectedRemote.from_url("https://example.com/source/project")

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                return_value=_completed("git@example.com:source/project.git\n"),
            ) as run:
                observed = repository.observe_materialized(
                    Path("source/project"),
                    expected=expected,
                )

            self.assertEqual(observed, destination.resolve())
            self.assertEqual(run.call_count, 1)
            self.assertNotIn("clone", run.call_args.args[0])

    def test_observe_materialized_rejects_escaping_destination_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary)
            root = allowed / "src"
            outside = allowed / "outside"
            (outside / ".git").mkdir(parents=True)
            destination = root / "source" / "project"
            destination.parent.mkdir(parents=True)
            destination.symlink_to(outside, target_is_directory=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=allowed)
            expected = ExpectedRemote.from_url("https://example.com/source/project")

            with (
                mock.patch("local_voice_harness.local_git.run_command") as run,
                self.assertRaisesRegex(LocalGitError, "escapes its root"),
            ):
                repository.observe_materialized(
                    Path("source/project"),
                    expected=expected,
                )

            run.assert_not_called()

    def test_observe_materialized_rejects_escaping_destination_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary)
            root = allowed / "src"
            outside = allowed / "outside"
            outside.mkdir()
            root.mkdir()
            (root / "source").symlink_to(outside, target_is_directory=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=allowed)
            expected = ExpectedRemote.from_url("https://example.com/source/project")

            with (
                mock.patch("local_voice_harness.local_git.run_command") as run,
                self.assertRaisesRegex(LocalGitError, "escapes its root"),
            ):
                repository.observe_materialized(
                    Path("source/project"),
                    expected=expected,
                )

            run.assert_not_called()

    def test_materialize_accepts_adapter_clone_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)

            def run(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                staging = Path(command[-1])
                staging.mkdir()
                (staging / ".git").mkdir()
                return _completed()

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=run,
            ) as command:
                repository.materialize(
                    Path("project"),
                    clone_url="https://github.com/source/project",
                    clone_command=(
                        "gh",
                        "repo",
                        "clone",
                        "https://github.com/source/project",
                    ),
                )

            self.assertEqual(
                command.call_args.args[0][:-1],
                [
                    "gh",
                    "repo",
                    "clone",
                    "https://github.com/source/project",
                ],
            )

    def test_invalid_roots_and_resolved_parent_escape_before_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            outside = base / "outside"
            repository = LocalGitRepository(
                clone_root=outside,
                allowed_root=base / "allowed",
            )
            with (
                mock.patch("local_voice_harness.local_git.run_command") as run,
                self.assertRaisesRegex(LocalGitError, "must be inside"),
            ):
                repository.materialize(Path("project"), clone_url="https://example/x")
            run.assert_not_called()
            self.assertFalse(outside.exists())

            root = base / "root"
            root.mkdir()
            escaped = base / "escaped"
            escaped.mkdir()
            (root / "owner").symlink_to(escaped, target_is_directory=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=base)
            with (
                mock.patch("local_voice_harness.local_git.run_command") as run,
                self.assertRaisesRegex(LocalGitError, "escapes"),
            ):
                repository.materialize(
                    Path("owner/project"),
                    clone_url="https://example/project",
                )
            run.assert_not_called()

    def test_existing_destination_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            outside = base / "outside"
            (outside / ".git").mkdir(parents=True)
            (root / "project").symlink_to(outside, target_is_directory=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=base)

            with (
                mock.patch("local_voice_harness.local_git.run_command") as run,
                self.assertRaisesRegex(LocalGitError, "escapes"),
            ):
                repository.materialize(
                    Path("project"),
                    clone_url="https://example.com/project",
                )
            run.assert_not_called()

    def test_contained_destination_symlink_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "existing"
            (target / ".git").mkdir(parents=True)
            destination = root / "project"
            destination.symlink_to(target, target_is_directory=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)

            with mock.patch("local_voice_harness.local_git.run_command") as run:
                checkout = repository.materialize(
                    Path("project"),
                    clone_url="https://example.com/project",
                )
            self.assertEqual(checkout, target)
            run.assert_not_called()

    def test_filesystem_failures_are_local_git_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.write_text("not a directory")
            repository = LocalGitRepository(clone_root=root, allowed_root=root.parent)
            with self.assertRaisesRegex(
                LocalGitError,
                "Could not materialize repository",
            ) as raised:
                repository.materialize(
                    Path("project"),
                    clone_url="https://example.com/project",
                )
            self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_failed_verification_cleans_staging_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)

            def run(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if command[1:3] == ["clone", "--"]:
                    staging = Path(command[-1])
                    staging.mkdir()
                    (staging / ".git").mkdir()
                    return _completed()
                return _completed("https://github.com/other/project\n")

            with (
                mock.patch(
                    "local_voice_harness.local_git.run_command",
                    side_effect=run,
                ),
                self.assertRaisesRegex(LocalGitError, "origin is not"),
            ):
                repository.materialize(
                    Path("source/project"),
                    clone_url="https://github.com/source/project",
                    expected=ExpectedRemote.from_url(
                        "https://github.com/source/project"
                    ),
                )

            owner = root / "source"
            self.assertFalse((owner / "project").exists())
            self.assertEqual(
                [path for path in owner.iterdir() if ".clone-" in path.name],
                [],
            )

    def test_post_replace_cancellation_recovers_by_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "project"
            repository = LocalGitRepository(clone_root=root, allowed_root=root)

            def run(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if command[1:3] == ["clone", "--"]:
                    staging = Path(command[-1])
                    staging.mkdir()
                    (staging / ".git").mkdir()
                return _completed()

            def checkpoint() -> None:
                if destination.exists():
                    raise Cancelled

            with (
                mock.patch(
                    "local_voice_harness.local_git.run_command",
                    side_effect=run,
                ),
                self.assertRaises(Cancelled),
            ):
                repository.materialize(
                    Path("project"),
                    clone_url="https://example.com/project",
                    checkpoint=checkpoint,
                )

            self.assertTrue((destination / ".git").exists())
            with mock.patch("local_voice_harness.local_git.run_command") as next_run:
                self.assertEqual(
                    repository.materialize(
                        Path("project"),
                        clone_url="https://example.com/project",
                    ),
                    destination,
                )
            next_run.assert_not_called()

    def test_timeout_is_ambiguous_and_staging_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            with (
                mock.patch(
                    "local_voice_harness.local_git.run_command",
                    side_effect=subprocess.TimeoutExpired(["git"], 300),
                ),
                self.assertRaisesRegex(
                    LocalGitOperationAmbiguous,
                    "outcome is ambiguous",
                ),
            ):
                repository.materialize(
                    Path("project"),
                    clone_url="https://example.com/project",
                )
            self.assertFalse((root / "project").exists())
            self.assertFalse(any(".clone-" in path.name for path in root.iterdir()))

    def test_lock_serializes_clone_and_reuses_materialized_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = LocalGitRepository(clone_root=root, allowed_root=root)
            second = LocalGitRepository(clone_root=root, allowed_root=root)
            clone_started = threading.Event()
            release_clone = threading.Event()
            calls: list[list[str]] = []
            errors: list[BaseException] = []

            def run(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                if command[1:3] == ["clone", "--"]:
                    clone_started.set()
                    release_clone.wait(timeout=2)
                    staging = Path(command[-1])
                    staging.mkdir()
                    (staging / ".git").mkdir()
                return _completed()

            def materialize(repository: LocalGitRepository) -> None:
                try:
                    repository.materialize(
                        Path("project"),
                        clone_url="https://example.com/project",
                    )
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch(
                "local_voice_harness.local_git.run_command",
                side_effect=run,
            ):
                first_thread = threading.Thread(target=materialize, args=(first,))
                second_thread = threading.Thread(target=materialize, args=(second,))
                first_thread.start()
                self.assertTrue(clone_started.wait(timeout=1))
                second_thread.start()
                time.sleep(0.05)
                self.assertEqual(
                    sum(command[1:3] == ["clone", "--"] for command in calls),
                    1,
                )
                release_clone.set()
                first_thread.join(timeout=2)
                second_thread.join(timeout=2)

            self.assertEqual(errors, [])
            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(
                sum(command[1:3] == ["clone", "--"] for command in calls),
                1,
            )

    def test_waiting_for_lock_remains_cancellable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            holder = LocalGitRepository(clone_root=root, allowed_root=root)
            waiter = LocalGitRepository(clone_root=root, allowed_root=root)
            waiting = threading.Event()
            cancel = threading.Event()
            errors: list[BaseException] = []
            checkpoints = 0

            def checkpoint() -> None:
                nonlocal checkpoints
                checkpoints += 1
                if checkpoints >= 4:
                    waiting.set()
                if cancel.is_set():
                    raise Cancelled

            def materialize() -> None:
                try:
                    waiter.materialize(
                        Path("project"),
                        clone_url="https://example.com/project",
                        checkpoint=checkpoint,
                    )
                except BaseException as exc:
                    errors.append(exc)

            with holder._provisioning_lock():
                thread = threading.Thread(target=materialize)
                thread.start()
                self.assertTrue(waiting.wait(timeout=1))
                cancel.set()
                thread.join(timeout=1)

            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], Cancelled)

    def test_commit_and_push_require_confirmation_and_leave_clean_checkout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            with self.assertRaisesRegex(LocalGitError, "confirmation"):
                repository.commit_unpublished_changes(
                    checkout, "Subject", confirmed=False
                )
            with self.assertRaisesRegex(LocalGitError, "confirmation"):
                repository.push_current_branch(checkout, confirmed=False)

            with mock.patch.object(
                repository,
                "git",
                side_effect=[
                    _completed(""),
                    _completed("voice/job\n"),
                    _completed(),
                    _completed(),
                ],
            ) as git:
                self.assertIsNone(
                    repository.commit_unpublished_changes(
                        checkout, "Subject", confirmed=True
                    )
                )
                self.assertEqual(
                    repository.push_current_branch(checkout, confirmed=True),
                    "voice/job",
                )
            commands = [call.args[1:] for call in git.call_args_list]
            self.assertIn(("status", "--porcelain"), commands)
            self.assertNotIn(("add", "-A"), commands)
            self.assertIn(("push", "-u", "origin", "voice/job"), commands)

    def test_commit_unpublished_changes_adds_and_returns_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            with mock.patch.object(
                repository,
                "git",
                side_effect=[
                    _completed(" M file.py\n"),
                    _completed(),
                    _completed(),
                    _completed("abc123\n"),
                ],
            ) as git:
                oid = repository.commit_unpublished_changes(
                    checkout, "Fix the reader", confirmed=True
                )
            self.assertEqual(oid, "abc123")
            commands = [call.args[1:] for call in git.call_args_list]
            self.assertEqual(commands[1], ("add", "-A"))
            self.assertEqual(commands[2], ("commit", "-m", "Fix the reader"))

    def test_commit_and_push_reject_invalid_checkout_and_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)
            outside = Path(temporary).resolve().parent / "outside-checkout"
            with self.assertRaisesRegex(LocalGitError, "allowed project root"):
                repository.has_unpublished_changes(outside)
            with self.assertRaisesRegex(LocalGitError, "non-empty subject"):
                repository.commit_unpublished_changes(checkout, "   ", confirmed=True)
            with self.assertRaisesRegex(LocalGitError, "too long"):
                repository.commit_unpublished_changes(
                    checkout, "x" * 73, confirmed=True
                )
            with mock.patch.object(
                repository,
                "git",
                return_value=_completed("HEAD\n"),
            ):
                with self.assertRaisesRegex(LocalGitError, "named branch"):
                    repository.push_current_branch(checkout, confirmed=True)

    def test_push_timeout_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "project"
            (checkout / ".git").mkdir(parents=True)
            repository = LocalGitRepository(clone_root=root, allowed_root=root)

            def git(*_args: object, **kwargs: object) -> object:
                if kwargs.get("timeout") == 180:
                    raise LocalGitOperationAmbiguous("timed out")
                return _completed("voice/job\n")

            with (
                mock.patch.object(repository, "git", side_effect=git),
                self.assertRaises(LocalGitOperationAmbiguous),
            ):
                repository.push_current_branch(checkout, confirmed=True)
