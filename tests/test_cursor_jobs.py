from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest import mock

from local_voice_harness.cursor import jobs
from local_voice_harness.cursor.model import CURRENT_SCHEMA_VERSION
from local_voice_harness.cursor.store import JobQuarantineWarning
from local_voice_harness.integrations.github import (
    GitHubIssue,
    GitHubRepository,
    ProvisionedIssue,
    ProvisionedPullRequest,
)
from local_voice_harness.integrations.herdr import (
    AgentSelection,
    HerdrError,
    PromptOutcome,
)


class CursorJobStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.jobs_patch = mock.patch.object(jobs, "JOBS_DIR", Path(self.temporary.name))
        self.jobs_patch.start()

    def tearDown(self) -> None:
        self.jobs_patch.stop()
        self.temporary.cleanup()

    def test_malformed_job_file_is_quarantined_from_collection_read(self) -> None:
        (Path(self.temporary.name) / "123456789abc.json").write_text("{not json")

        with self.assertWarnsRegex(
            JobQuarantineWarning, "123456789abc.json: job is quarantined"
        ):
            self.assertEqual(jobs.active_jobs(), [])

    def test_invalid_job_file_is_not_silently_skipped(self) -> None:
        jobs.write_job(
            {
                "id": "aaaaaaaaaaaa",
                "status": "queued",
                "request": "valid",
                "delivered": False,
            }
        )
        (Path(self.temporary.name) / "bbbbbbbbbbbb.json").write_text(
            json.dumps({"id": "bbbbbbbbbbbb", "status": "unknown"})
        )

        with self.assertWarnsRegex(
            JobQuarantineWarning, "bbbbbbbbbbbb.json: job is quarantined"
        ):
            self.assertEqual(
                [job["id"] for job in jobs.active_jobs()], ["aaaaaaaaaaaa"]
            )

    def test_repository_reply_preserves_original_task(self) -> None:
        job: dict[str, object] = {
            "id": "123456789abc",
            "request": "Fix APP-42",
            "status": "awaiting_user",
            "clarification_kind": "repository",
        }
        jobs.write_job(job)
        with mock.patch.object(jobs, "launch_worker") as launch:
            jobs.reply_job("123456789abc", "example-repo")
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["request"], "Fix APP-42")
        self.assertEqual(updated["repository_hint"], "example-repo")
        launch.assert_called_once_with("123456789abc")

    def test_github_repository_reply_preserves_explicit_fork_intent(self) -> None:
        job: dict[str, object] = {
            "id": "123456789abc",
            "request": "Fork this repo and add a feature",
            "fork_requested": True,
            "status": "awaiting_user",
            "clarification_kind": "github_repository",
        }
        jobs.write_job(job)
        with mock.patch.object(jobs, "launch_worker"):
            jobs.reply_job("123456789abc", "example/project")
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["github_repository"], "example/project")
        self.assertTrue(updated["fork_requested"])
        self.assertEqual(updated["request"], "Fork this repo and add a feature")

    def test_pull_request_job_gets_unique_worktree_identity(self) -> None:
        with mock.patch.object(jobs, "launch_worker"):
            job_id = jobs.start_job(
                "review this pull request",
                github_repository="source/project",
                github_pull_request=42,
            )

        job = jobs.read_job(job_id)
        self.assertEqual(job["worktree_branch"], f"voice/github-pr-{job_id}")
        self.assertEqual(job["worktree_label"], "pr-42")
        self.assertEqual(job["pull_request_worktree_state"], "pending")

    def test_worker_checks_out_pull_request_only_in_reserved_worktree(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        worktree = Path(self.temporary.name) / "worktrees" / "pr-42"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: shared/worktrees/pr-42\n")
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "review pull request 42",
                "github_repository": "source/project",
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "worktree_label": "pr-42",
                "pull_request_worktree_state": "pending",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.provision_pull_request.return_value = ProvisionedPullRequest(
            source, repository, 42
        )
        github.checkout_pull_request.return_value = "voice/github-pr-123456789abc"
        client = mock.Mock()
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(worktree),
            "agent",
            str(worktree),
        )
        client.prompt_and_wait.return_value = PromptOutcome(
            "idle",
            "done",
            None,
            "VOICE_SUMMARY[123456789abc-1]: done",
        )

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            jobs.run_worker("123456789abc")

        github.provision_pull_request.assert_called_once_with(
            "source/project", 42, checkpoint=mock.ANY
        )
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-pr-123456789abc",
            worktree_label="pr-42",
            checkpoint=mock.ANY,
            reserve=mock.ANY,
            settle=mock.ANY,
            fail_agent=mock.ANY,
            reserve_worktree=mock.ANY,
            settle_worktree=mock.ANY,
            fail_worktree=mock.ANY,
        )
        github.checkout_pull_request.assert_called_once_with(
            worktree,
            42,
            branch="voice/github-pr-123456789abc",
            checkpoint=mock.ANY,
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["worktree_path"], str(worktree))
        self.assertEqual(updated["pull_request_worktree_state"], "retained")

    def test_pull_request_shared_clone_is_quarantined_without_checkout(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        repository.mkdir(parents=True)
        (repository / ".git").mkdir()
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "repository": str(repository),
                "worktree_path": str(repository),
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "pull_request_worktree_state": "provisioning",
            }
        )
        github = mock.Mock()

        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            self.assertRaisesRegex(jobs.HarnessError, "shared repository clone"),
        ):
            jobs._prepare_pull_request_checkout(
                "123456789abc",
                "worker",
                jobs.read_job("123456789abc"),
            )

        github.checkout_pull_request.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["pull_request_worktree_state"], "quarantined")

    def test_concurrent_pull_requests_prepare_distinct_worktrees(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        barrier = threading.Barrier(2)
        calls: list[tuple[Path, int, str]] = []
        for job_id, number in (("aaaaaaaaaaaa", 41), ("bbbbbbbbbbbb", 42)):
            worktree = Path(self.temporary.name) / "worktrees" / job_id
            worktree.mkdir(parents=True)
            (worktree / ".git").write_text("gitdir: shared\n")
            jobs.write_job(
                {
                    "id": job_id,
                    "status": "routing",
                    "worker_token": job_id,
                    "worker_pid": 42,
                    "worker_process_start": f"start-{job_id}",
                    "repository": str(repository),
                    "worktree_path": str(worktree),
                    "github_pull_request": number,
                    "worktree_branch": f"voice/github-pr-{job_id}",
                    "pull_request_worktree_state": "provisioning",
                }
            )
        github = mock.Mock()

        def checkout(path: Path, number: int, *, branch: str) -> str:
            calls.append((path, number, branch))
            barrier.wait()
            return branch

        github.checkout_pull_request.side_effect = checkout

        def prepare(job_id: str) -> None:
            jobs._prepare_pull_request_checkout(
                job_id,
                job_id,
                jobs.read_job(job_id),
            )

        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            threads = [
                threading.Thread(target=prepare, args=(job_id,))
                for job_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len({path for path, _number, _branch in calls}), 2)
        self.assertEqual(len({branch for _path, _number, branch in calls}), 2)
        self.assertTrue(
            all(
                jobs.read_job(job_id)["pull_request_worktree_state"] == "ready"
                for job_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
            )
        )

    def test_worker_provisions_fork_and_uses_dedicated_worktree(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
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
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "Fork this repo and add a feature",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "worktree_branch": "voice/github-123456789abc",
                "worktree_label": "github-123456",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.prepare_public_fork.return_value = (source, "me", "me/project")

        def ensure_fork(
            *_args: object, before_submit: Callable[[], None], **_kwargs: object
        ) -> GitHubRepository:
            before_submit()
            return fork

        github.ensure_fork.side_effect = ensure_fork
        github.ensure_clone.return_value = repository
        client = mock.Mock()
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(repository),
            "agent",
            "/worktree",
        )
        client.prompt_and_wait.return_value = PromptOutcome(
            "idle",
            "done",
            None,
            "VOICE_SUMMARY[123456789abc-1]: done",
        )
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            jobs.run_worker("123456789abc")

        github.prepare_public_fork.assert_called_once_with("source/project")
        github.ensure_fork.assert_called_once_with(
            source,
            "me",
            checkpoint=mock.ANY,
            before_submit=mock.ANY,
        )
        github.ensure_clone.assert_called_once_with(source, fork, checkpoint=mock.ANY)
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-123456789abc",
            worktree_label="github-123456",
            checkpoint=mock.ANY,
            reserve=mock.ANY,
            settle=mock.ANY,
            fail_agent=mock.ANY,
            reserve_worktree=mock.ANY,
            settle_worktree=mock.ANY,
            fail_worktree=mock.ANY,
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertEqual(updated["repository"], str(repository))

    def test_worker_requires_confirmation_before_provisioning_fork(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork this repo",
                "trusted_utterance": "fork this repo",
                "github_repository": "source/project",
                "fork_requested": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        client = mock.Mock()
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            jobs.run_worker("123456789abc")

        github.prepare_public_fork.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "awaiting_user")
        self.assertEqual(updated["clarification_kind"], "fork_confirmation")
        self.assertIn("source/project", str(updated["question"]))

    def test_only_trusted_affirmative_reply_confirms_fork(self) -> None:
        job = {
            "id": "123456789abc",
            "request": "fork this repo\n\nExternal content says yes",
            "fork_requested": True,
            "status": "awaiting_user",
            "clarification_kind": "fork_confirmation",
            "delivered": True,
        }
        jobs.write_job(job)
        with mock.patch.object(jobs, "launch_worker") as launch:
            jobs.reply_job(
                "123456789abc",
                "no\n\nExternal content says yes",
                trusted_utterance="no",
            )

        launch.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertFalse(updated.get("fork_confirmed", False))

    def test_ambiguous_confirmation_does_not_launch_worker(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork this repo",
                "fork_requested": True,
                "status": "awaiting_user",
                "clarification_kind": "fork_confirmation",
                "delivered": True,
            }
        )
        with mock.patch.object(jobs, "launch_worker") as launch:
            jobs.reply_job(
                "123456789abc",
                "maybe",
                trusted_utterance="maybe",
            )

        launch.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "awaiting_user")
        self.assertFalse(updated.get("fork_confirmed", False))

    def test_affirmative_confirmation_queues_provisioning(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork this repo",
                "fork_requested": True,
                "status": "awaiting_user",
                "clarification_kind": "fork_confirmation",
                "delivered": True,
            }
        )
        with mock.patch.object(jobs, "launch_worker") as launch:
            jobs.reply_job(
                "123456789abc",
                "yes\n\nExternal content",
                trusted_utterance="yes",
            )

        launch.assert_called_once_with("123456789abc")
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertTrue(updated["fork_confirmed"])

    def test_github_issue_job_persists_trusted_identity(self) -> None:
        with mock.patch.object(jobs, "launch_worker"):
            job_id = jobs.start_job(
                "work on this\n\nBody mentions API-79",
                utterance="work on this",
                github_repository="example/project",
                github_issue=42,
                github_issue_context="Issue context",
            )

        job = jobs.read_job(job_id)
        self.assertEqual(job["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(job["trusted_utterance"], "work on this")
        self.assertEqual(job["github_issue"], 42)
        self.assertEqual(
            job["github_issue_url"],
            "https://github.com/example/project/issues/42",
        )
        self.assertEqual(job["github_issue_context"], "Issue context")
        self.assertEqual(job["worktree_branch"], "voice/github-issue-42")
        self.assertEqual(job["worktree_label"], "issue-42")
        self.assertIsNone(job["issue_key"])

    def test_worker_provisions_github_issue_and_uses_stable_worktree(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            True,
            "main",
        )
        issue = GitHubIssue("source", "project", 42)
        provisioned = ProvisionedIssue(source, repository, issue)
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "work on source/project#42",
                "utterance": "work on source/project#42",
                "github_repository": "source/project",
                "github_issue": 42,
                "github_issue_context": "Title: Fix it",
                "worktree_branch": "voice/github-issue-42",
                "worktree_label": "issue-42",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.provision_issue.return_value = provisioned
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.ensure_agent.return_value = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(repository),
            "agent",
            "/worktree/issue-42",
        )
        client.prompt_and_wait.return_value = PromptOutcome(
            "idle",
            "done",
            None,
            "VOICE_SUMMARY[123456789abc-1]: done",
        )
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            jobs.run_worker("123456789abc")

        github.provision_issue.assert_called_once_with(
            issue, candidates=[repository], checkpoint=mock.ANY
        )
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-issue-42",
            worktree_label="issue-42",
            checkpoint=mock.ANY,
            reserve=mock.ANY,
            settle=mock.ANY,
            fail_agent=mock.ANY,
            reserve_worktree=mock.ANY,
            settle_worktree=mock.ANY,
            fail_worktree=mock.ANY,
        )
        prompt = client.prompt_and_wait.call_args.args[1]
        self.assertIn("Title: Fix it", prompt)
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["github_issue_url"], issue.url)

    def test_reservation_rejects_an_active_shared_worktree(self) -> None:
        jobs.write_job(
            {
                "id": "aaaaaaaaaaaa",
                "status": "running",
                "worker_token": "first-worker",
                "worker_pid": 41,
                "worker_process_start": "first-start",
                "herdr_target": "first-agent",
                "worktree_path": "/worktree/issue-42",
            }
        )
        jobs.write_job(
            {
                "id": "bbbbbbbbbbbb",
                "status": "routing",
                "worker_token": "worker-token",
                "worker_pid": 42,
                "worker_process_start": "second-start",
            }
        )
        selection = AgentSelection(
            "second-agent",
            "pane",
            "workspace",
            "/repo",
            "second-agent",
            "/worktree/issue-42",
        )

        reserved = jobs._reserve_worker_target(
            "bbbbbbbbbbbb",
            "worker-token",
            selection,
            Path("/repo"),
            None,
        )

        self.assertIsNone(reserved)
        self.assertNotIn("herdr_target", jobs.read_job("bbbbbbbbbbbb"))

    def test_submit_starts_new_job_despite_existing_session(self) -> None:
        with (
            mock.patch.object(jobs, "start_job", return_value="newjob123456") as start,
            mock.patch.object(
                jobs,
                "read_job",
                return_value={"status": "completed", "result": "done"},
            ),
            mock.patch.object(jobs, "mark_delivered"),
            mock.patch.object(jobs, "reply_job") as reply,
        ):
            result, session = jobs.cursor_turn(
                "Work on APP-43", session_id="oldjob123456"
            )

        start.assert_called_once_with(
            "Work on APP-43",
            repository=None,
            github_repository=None,
            github_issue=None,
            github_issue_context=None,
            fork_requested=False,
            github_pull_request=None,
            agent=None,
            utterance=None,
            context_repository=None,
        )
        reply.assert_not_called()
        self.assertEqual(result, "done")
        self.assertIsNone(session)

    def test_spoken_repository_precedes_focused_context_fallback(self) -> None:
        spoken = Path("/repos/spoken")
        focused = Path("/repos/focused")
        client = mock.Mock()
        client.resolve_repository.return_value = (spoken, [spoken])
        job: dict[str, object] = {
            "request": "work on this\n\nRepository: owner/focused",
            "utterance": "work on the spoken repository",
            "context_repository": "owner/focused",
        }

        repository, candidates = jobs.resolve_job_repository(
            client, job, [spoken, focused]
        )

        self.assertEqual(repository, spoken)
        self.assertEqual(candidates, [spoken])
        client.resolve_repository.assert_called_once_with(
            None, "work on the spoken repository", [spoken, focused]
        )

    def test_focused_repository_is_used_when_utterance_has_no_match(self) -> None:
        focused = Path("/repos/focused")
        client = mock.Mock()
        client.resolve_repository.side_effect = [
            (None, []),
            (focused, [focused]),
        ]
        job: dict[str, object] = {
            "request": "work on this\n\nRepository: owner/focused",
            "utterance": "work on this task",
            "context_repository": "owner/focused",
        }

        repository, candidates = jobs.resolve_job_repository(client, job, [focused])

        self.assertEqual(repository, focused)
        self.assertEqual(candidates, [focused])
        self.assertEqual(
            client.resolve_repository.call_args_list,
            [
                mock.call(None, "work on this task", [focused]),
                mock.call("owner/focused", "", [focused]),
            ],
        )

    def test_latest_voice_marker_controls_terminal_state(self) -> None:
        job: dict[str, object] = {"id": "123456789abc", "turn_token": "token"}
        jobs.complete_from_output(
            job,
            output=(
                "VOICE_QUESTION[token]: old question\n"
                "VOICE_SUMMARY[token]: finished successfully"
            ),
            agent_status="idle",
        )
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"], "finished successfully")

    def test_blocked_without_marker_requires_attention(self) -> None:
        job: dict[str, object] = {
            "id": "123456789abc",
            "turn_token": "token",
            "herdr_target": "agent",
        }
        jobs.complete_from_output(job, output="plain output", agent_status="idle")
        self.assertEqual(job["status"], "blocked")
        self.assertIn("needs attention", str(job["result"]))

    def test_worker_spawns_package_module(self) -> None:
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 1234
        process.wait.return_value = 0
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        with (
            mock.patch("subprocess.Popen", return_value=process) as popen,
            mock.patch("threading.Thread") as thread,
            mock.patch.object(jobs, "_process_identity", return_value="worker-start"),
        ):
            jobs.launch_worker("123456789abc")
        command = popen.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "local_voice_harness.cursor.worker"])
        thread.assert_called_once()

    def test_worker_uses_rofi_repository_selection_before_prompting(self) -> None:
        repository = Path(self.temporary.name) / "cloned-project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "Use Cursor to fix the bug",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.repository_roots.return_value = []
        client.resolve_repository.return_value = (None, [])
        client.choose_or_clone_repository.return_value = (repository, "")
        client.ensure_agent.return_value = AgentSelection(
            target="cursor-agent",
            pane_id="pane",
            workspace_id="workspace",
            cwd=str(repository),
            name="cursor-agent",
            worktree_path=str(repository),
        )
        client.prompt_and_wait.return_value = PromptOutcome(
            status="idle",
            summary="done",
            question=None,
            output="VOICE_SUMMARY[123456789abc-1]: done",
        )

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            jobs.run_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["repository"], str(repository))
        client.choose_or_clone_repository.assert_called_once_with(
            [], checkpoint=mock.ANY
        )

    def test_dead_worker_reconciles_existing_agent(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "created_at": 1,
                "worker_pid": 999999,
                "worker_process_start": "old-worker",
                "worker_token": "old-claim",
                "herdr_target": "cursor-agent",
                "delivered": False,
            }
        )
        with (
            mock.patch("time.time", return_value=100),
            mock.patch.object(jobs, "_process_identity", return_value=None),
            mock.patch.object(jobs, "launch_worker") as launch,
        ):
            self.assertEqual(jobs.pending_results(), [])
        updated = json.loads(
            (Path(self.temporary.name) / "123456789abc.json").read_text()
        )
        self.assertTrue(updated["reconcile"])
        launch.assert_called_once_with("123456789abc")

    def test_dispatch_crash_retries_same_reserved_agent_identity(self) -> None:
        repository = Path(self.temporary.name) / "project"
        worktree = Path(self.temporary.name) / "worktree"
        selection = AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(worktree),
            "planned-agent",
            str(worktree),
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "status": "queued",
                "repository": str(repository),
                "worktree_path": str(worktree),
                "herdr_target": "planned-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_name": "planned-agent",
                "agent_dispatch_state": "dispatching",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("not found", code="agent_not_found")
        client.start_agent.return_value = selection
        client.prompt_and_wait.return_value = PromptOutcome(
            "idle",
            "done",
            None,
            "VOICE_SUMMARY[123456789abc-1]: done",
        )

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            jobs.run_worker("123456789abc")

        client.start_agent.assert_called_once_with(
            worktree,
            "project",
            "pane",
            "workspace",
            name="planned-agent",
            checkpoint=mock.ANY,
        )
        client.ensure_agent.assert_not_called()
        self.assertEqual(jobs.read_job("123456789abc")["status"], "completed")

    def test_dispatch_reconciliation_defers_on_transient_lookup_failure(self) -> None:
        repository = Path(self.temporary.name) / "project"
        worktree = Path(self.temporary.name) / "worktree"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "status": "queued",
                "repository": str(repository),
                "worktree_path": str(worktree),
                "herdr_target": "planned-agent",
                "herdr_pane_id": "pane",
                "herdr_workspace_id": "workspace",
                "agent_dispatch_state": "dispatching",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("server unavailable")

        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            jobs.run_worker("123456789abc")

        client.start_agent.assert_not_called()
        client.prompt_and_wait.assert_not_called()
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertEqual(updated["agent_dispatch_state"], "dispatching")

    def test_stale_worker_cannot_overwrite_cancellation(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "turn_token": "turn",
                "worker_token": "worker-claim",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "delivered": False,
            }
        )

        jobs.cancel_job("123456789abc")
        jobs._worker_complete(
            "123456789abc",
            "worker-claim",
            output="VOICE_SUMMARY[turn]: stale success",
            agent_status="idle",
        )

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "cancelled")
        self.assertNotEqual(updated["result"], "stale success")

    def test_cancellation_during_repository_resolution_stops_routing(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix the bug",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        client = mock.Mock()

        def repository_roots() -> list[Path]:
            entered.set()
            self.assertTrue(release.wait(2))
            return []

        client.repository_roots.side_effect = repository_roots
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            worker.start()
            self.assertTrue(entered.wait(2))
            jobs.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        client.resolve_repository.assert_not_called()
        client.choose_or_clone_repository.assert_not_called()
        client.ensure_agent.assert_not_called()

    def test_fork_commit_wins_before_cancellation(self) -> None:
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
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork it",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        settled = threading.Event()
        continue_after_settle = threading.Event()
        github = mock.Mock()

        github.prepare_public_fork.return_value = (source, "me", "me/project")

        def ensure_fork(
            *_args: object, before_submit: Callable[[], None], **_kwargs: object
        ) -> GitHubRepository:
            before_submit()
            entered.set()
            self.assertTrue(release.wait(2))
            return fork

        github.ensure_fork.side_effect = ensure_fork
        original_settle = jobs._settle_fork_operation

        def settle_fork(
            job_id: str,
            token: str,
            value: GitHubRepository | None,
            *,
            ambiguous: bool = False,
            failed_observing: bool = False,
        ) -> dict[str, object] | None:
            result = original_settle(
                job_id,
                token,
                value,
                ambiguous=ambiguous,
                failed_observing=failed_observing,
            )
            if value is fork:
                settled.set()
                self.assertTrue(continue_after_settle.wait(2))
            return result

        client = mock.Mock()
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_settle_fork_operation", side_effect=settle_fork),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(jobs.HarnessError, "committed fork submission"):
                jobs.cancel_job("123456789abc")
            committed = jobs.read_job("123456789abc")
            self.assertEqual(committed["status"], "routing")
            self.assertEqual(committed["fork_operation_state"], "submitted")
            release.set()
            self.assertTrue(settled.wait(2))
            self.assertEqual(
                jobs.read_job("123456789abc")["fork_operation_state"], "exists"
            )
            jobs.cancel_job("123456789abc")
            continue_after_settle.set()
            worker.join(2)
            with mock.patch.object(jobs, "_worker_is_alive", return_value=False):
                jobs.recover_jobs()

        self.assertFalse(worker.is_alive())
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["fork_operation_state"], "exists")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertEqual(updated["status"], "cancelled")
        self.assertFalse(updated["target_release_pending"])
        github.ensure_clone.assert_not_called()
        client.ensure_agent.assert_not_called()

    def test_cancellation_wins_before_fork_commit(self) -> None:
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork it",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        submitted = threading.Event()
        github = mock.Mock()
        client = mock.Mock()
        github.prepare_public_fork.return_value = (source, "me", "me/project")

        def ensure_fork(
            *_args: object, before_submit: Callable[[], None], **_kwargs: object
        ) -> GitHubRepository:
            entered.set()
            self.assertTrue(release.wait(2))
            before_submit()
            submitted.set()
            raise AssertionError("fork submission should not run")

        github.ensure_fork.side_effect = ensure_fork
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
            mock.patch(
                "local_voice_harness.integrations.herdr.subprocess.run",
                side_effect=AssertionError("real service operation attempted"),
            ) as service_operation,
        ):
            worker.start()
            try:
                self.assertTrue(
                    entered.wait(2),
                    f"worker stopped before fork routing: {jobs.read_job('123456789abc')}",
                )
                client.ensure_server.assert_called_once_with()
                jobs.cancel_job("123456789abc")
                self.assertEqual(jobs.read_job("123456789abc")["status"], "cancelled")
            finally:
                release.set()
                worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(submitted.is_set())
        self.assertEqual(jobs.read_job("123456789abc")["status"], "cancelled")
        github.ensure_clone.assert_not_called()
        client.ensure_agent.assert_not_called()
        service_operation.assert_not_called()

    def test_worker_persists_herdr_failure_before_fork_routing(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fork it",
                "github_repository": "source/project",
                "fork_requested": True,
                "fork_confirmed": True,
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.ensure_server.side_effect = HerdrError(
            "Herdr command failed: executable not found",
            code="operation_spawn_failed",
        )
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "GitHubClient") as github_client,
        ):
            jobs.run_worker("123456789abc")

        failed = jobs.read_job("123456789abc")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "Herdr command failed: executable not found")
        self.assertNotIn("fork_operation_state", failed)
        github_client.assert_not_called()

    def test_post_create_cancellation_retains_reserved_worktree(self) -> None:
        repository = Path(self.temporary.name) / "project"
        checkout = Path(self.temporary.name) / "worktrees" / "task"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "worktree_branch": "voice/task",
                "worktree_label": "task",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve_worktree = cast(
                Callable[[Path, str, Path, str], None],
                kwargs["reserve_worktree"],
            )
            settle_worktree = cast(
                Callable[[Path, str | None, str | None], None],
                kwargs["settle_worktree"],
            )
            checkpoint = cast(Callable[[], None], kwargs["checkpoint"])
            reserve_worktree(repository, "voice/task", checkout, "planned")
            checkpoint()
            reserve_worktree(repository, "voice/task", checkout, "dispatching")
            entered.set()
            self.assertTrue(release.wait(2))
            settle_worktree(checkout, "workspace", "pane")
            checkpoint()
            raise AssertionError("cancelled worker continued after worktree settle")

        client.ensure_agent.side_effect = ensure_agent
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            dispatching = jobs.read_job("123456789abc")
            self.assertEqual(dispatching["worktree_path"], str(checkout))
            self.assertEqual(dispatching["worktree_provision_state"], "dispatching")
            jobs.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        retained = jobs.read_job("123456789abc")
        self.assertEqual(retained["worktree_provision_state"], "retained")
        self.assertTrue(retained["target_release_pending"])
        client.prompt_and_wait.assert_not_called()
        with mock.patch.object(jobs, "_worker_is_alive", return_value=False):
            jobs.recover_jobs()
        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])

    def test_cancellation_before_agent_dispatch_uses_planned_reservation(self) -> None:
        repository = Path(self.temporary.name) / "project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        dispatched = threading.Event()
        selection = AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(repository),
            "planned-agent",
            str(repository),
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            checkpoint = cast(Callable[[], None], kwargs["checkpoint"])
            reserve(selection, True)
            entered.set()
            self.assertTrue(release.wait(2))
            checkpoint()
            dispatched.set()
            return selection

        client.ensure_agent.side_effect = ensure_agent
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            worker.start()
            self.assertTrue(entered.wait(2))
            self.assertEqual(
                jobs.read_job("123456789abc")["agent_dispatch_state"],
                "dispatching",
            )
            jobs.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(dispatched.is_set())
        client.cancel_agent.assert_called_once_with("planned-agent")
        client.prompt_and_wait.assert_not_called()

    def test_late_agent_visibility_after_cancelled_startup(self) -> None:
        repository = Path(self.temporary.name) / "project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        selection = AgentSelection(
            "planned-agent",
            "pane",
            "workspace",
            str(repository),
            "planned-agent",
            str(repository),
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])
        client.cancel_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            None,
        ]
        client.get_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            {
                "name": "planned-agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
            },
        ]

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            fail_agent = cast(Callable[[HerdrError], None], kwargs["fail_agent"])
            reserve(selection, True)
            entered.set()
            self.assertTrue(release.wait(2))
            error = HerdrError("startup timed out", code="operation_timeout")
            fail_agent(error)
            raise error

        client.ensure_agent.side_effect = ensure_agent
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_stop_worker", return_value=False),
            mock.patch("time.time", return_value=90),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            jobs.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(
            jobs.read_job("123456789abc")["agent_dispatch_state"], "ambiguous"
        )
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_worker_is_alive", return_value=False),
        ):
            with mock.patch("time.time", return_value=100):
                jobs.recover_jobs()
            with mock.patch("time.time", return_value=105):
                jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["agent_dispatch_state"], "ready")
        self.assertFalse(updated["target_release_pending"])
        self.assertEqual(client.cancel_agent.call_count, 2)

    def test_cancellation_during_pr_checkout_prevents_prompt_submission(self) -> None:
        repository = Path(self.temporary.name) / "source" / "project"
        worktree = Path(self.temporary.name) / "worktrees" / "pr-42"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: shared\n")
        source = GitHubRepository(
            "source/project",
            "https://github.com/source/project",
            False,
            "main",
        )
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "review pull request 42",
                "github_repository": "source/project",
                "github_pull_request": 42,
                "worktree_branch": "voice/github-pr-123456789abc",
                "pull_request_worktree_state": "pending",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        github = mock.Mock()
        github.provision_pull_request.return_value = ProvisionedPullRequest(
            source, repository, 42
        )

        def checkout(*_args: object, **_kwargs: object) -> str:
            entered.set()
            self.assertTrue(release.wait(2))
            return "voice/github-pr-123456789abc"

        github.checkout_pull_request.side_effect = checkout
        selection = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(worktree),
            "agent",
            str(worktree),
        )
        client = mock.Mock()

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            reserve(selection, False)
            return selection

        client.ensure_agent.side_effect = ensure_agent
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            worker.start()
            self.assertTrue(entered.wait(2))
            jobs.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        client.prompt_and_wait.assert_not_called()
        self.assertEqual(
            jobs.read_job("123456789abc")["pull_request_worktree_state"],
            "retained",
        )

    def test_cancellation_before_prompt_submission_stops_submission(self) -> None:
        repository = Path(self.temporary.name) / "project"
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "fix it",
                "repository_hint": str(repository),
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        entered = threading.Event()
        release = threading.Event()
        submitted = threading.Event()
        selection = AgentSelection(
            "agent",
            "pane",
            "workspace",
            str(repository),
            "agent",
            str(repository),
        )
        client = mock.Mock()
        client.repository_roots.return_value = [repository]
        client.resolve_repository.return_value = (repository, [repository])

        def ensure_agent(*_args: object, **kwargs: object) -> AgentSelection:
            reserve = cast(Callable[[AgentSelection, bool], None], kwargs["reserve"])
            reserve(selection, False)
            return selection

        def prompt(*_args: object, **kwargs: object) -> PromptOutcome:
            entered.set()
            self.assertTrue(release.wait(2))
            checkpoint = cast(Callable[[], None], kwargs["checkpoint"])
            checkpoint()
            submitted.set()
            return PromptOutcome("idle", "done", None, "")

        client.ensure_agent.side_effect = ensure_agent
        client.prompt_and_wait.side_effect = prompt
        worker = threading.Thread(target=jobs.run_worker, args=("123456789abc",))
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            worker.start()
            self.assertTrue(entered.wait(2))
            jobs.cancel_job("123456789abc")
            release.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(submitted.is_set())
        client.cancel_agent.assert_called_once_with("agent")

    def test_worker_signal_requires_full_process_identity(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_pid": 42,
            "worker_process_start": "start",
            "worker_token": "claim",
        }
        wrong_command = b"\0".join(
            (
                b"/usr/bin/python",
                b"-m",
                b"local_voice_harness.cursor.worker",
                b"123456789abc",
                b"--claim",
                b"different",
                b"",
            )
        )
        with (
            mock.patch.object(jobs, "_process_identity", return_value="start"),
            mock.patch.object(Path, "read_bytes", return_value=wrong_command),
            mock.patch("os.killpg") as kill_group,
            mock.patch("os.kill") as kill_process,
        ):
            self.assertFalse(jobs._signal_worker(job, jobs.signal.SIGTERM))

        kill_group.assert_not_called()
        kill_process.assert_not_called()

    def test_critical_operation_never_escalates_to_process_group_kill(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_pid": 42,
            "worker_process_start": "start",
            "worker_token": "claim",
            "worker_operation": "fork_create",
        }
        with (
            mock.patch.object(jobs, "_worker_is_alive", side_effect=[True, True, True]),
            mock.patch.object(
                jobs, "_signal_worker", return_value=True
            ) as signal_worker,
        ):
            self.assertFalse(jobs._stop_worker(job, timeout=0))

        signal_worker.assert_called_once_with(
            job,
            jobs.signal.SIGTERM,
            include_process_group=False,
        )

    def test_only_one_concurrent_reply_transitions_job(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "original",
                "status": "awaiting_user",
                "clarification_kind": "agent",
                "delivered": True,
            }
        )
        barrier = threading.Barrier(2)
        successes: list[str] = []
        errors: list[Exception] = []

        def reply(text: str) -> None:
            barrier.wait()
            try:
                jobs.reply_job("123456789abc", text)
                successes.append(text)
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(jobs, "launch_worker"):
            threads = [
                threading.Thread(target=reply, args=("first",)),
                threading.Thread(target=reply, args=("second",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len(successes), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(jobs.read_job("123456789abc")["status"], "queued")

    def test_delivery_claim_has_single_concurrent_winner(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "completed",
                "result": "done",
                "completed_at": 1,
                "delivered": False,
            }
        )
        barrier = threading.Barrier(2)
        claims: list[dict[str, object] | None] = []

        def claim() -> None:
            barrier.wait()
            claims.append(jobs.claim_delivery())

        with mock.patch("time.time", return_value=100):
            threads = [threading.Thread(target=claim) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(sum(claim is not None for claim in claims), 1)

    def test_launch_failure_is_persisted_for_delivery(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        with (
            mock.patch("subprocess.Popen", side_effect=OSError("spawn failed")),
            self.assertRaisesRegex(OSError, "spawn failed"),
        ):
            jobs.launch_worker("123456789abc")

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "failed")
        self.assertFalse(updated["delivered"])
        self.assertEqual(updated["error"], "spawn failed")

    def test_abandoned_queued_job_is_launched_only_once(self) -> None:
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 4321
        process.wait.return_value = 0
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        with (
            mock.patch("subprocess.Popen", return_value=process) as popen,
            mock.patch("threading.Thread"),
            mock.patch.object(jobs, "_process_identity", return_value="worker-start"),
        ):
            jobs.recover_jobs()
            jobs.recover_jobs()

        popen.assert_called_once()
        updated = jobs.read_job("123456789abc")
        self.assertTrue(updated["worker_token"])

    def test_interrupted_routing_worker_restarts_before_dispatch(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "old",
                "worker_pid": 999999,
                "worker_process_start": "old-worker",
                "created_at": 1,
                "delivered": False,
            }
        )
        with (
            mock.patch.object(jobs, "_process_identity", return_value=None),
            mock.patch.object(jobs, "launch_worker") as launch,
        ):
            jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "queued")
        self.assertNotIn("reconcile", updated)
        launch.assert_called_once_with("123456789abc")

    def test_legacy_worker_pid_is_verified_by_command_line(self) -> None:
        job = {
            "id": "123456789abc",
            "worker_pid": 42,
        }
        command = b"\0".join(
            (
                b"/usr/bin/python",
                b"-m",
                b"local_voice_harness.cursor.worker",
                b"123456789abc",
                b"",
            )
        )
        with mock.patch.object(Path, "read_bytes", return_value=command):
            self.assertTrue(jobs._worker_is_alive(job))

    def test_cancel_keeps_target_reserved_until_interrupt_finishes(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "herdr_target": "cursor-agent",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "github_pull_request": 42,
                "worktree_path": "/worktree/pr-42",
                "pull_request_worktree_state": "provisioning",
                "delivered": False,
            }
        )
        client = mock.Mock()
        reserved_during_cancel: set[str] = set()

        def inspect_reservation(_target: str) -> None:
            reserved_during_cancel.update(jobs.reserved_targets())

        client.cancel_agent.side_effect = inspect_reservation
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            jobs.cancel_job("123456789abc")
            jobs.cancel_job("123456789abc")

        self.assertIn("cursor-agent", reserved_during_cancel)
        self.assertNotIn("cursor-agent", jobs.reserved_targets())
        client.cancel_agent.assert_called_once_with("cursor-agent")
        self.assertEqual(
            jobs.read_job("123456789abc")["pull_request_worktree_state"],
            "retained",
        )

    def test_failed_agent_interrupt_keeps_reservation_for_recovery(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "herdr_target": "cursor-agent",
                "worktree_path": "/worktree/task",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "delivered": False,
            }
        )
        failing = mock.Mock()
        failing.cancel_agent.side_effect = HerdrError("still working")
        recovered = mock.Mock()
        with (
            mock.patch.object(jobs, "_stop_worker", return_value=True),
            mock.patch.object(jobs, "HerdrClient", side_effect=[failing, recovered]),
        ):
            jobs.cancel_job("123456789abc")
            cancelled = jobs.read_job("123456789abc")
            self.assertTrue(cancelled["target_release_pending"])
            self.assertIn("cursor-agent", jobs.reserved_targets())
            jobs.recover_jobs()

        recovered.cancel_agent.assert_called_once_with("cursor-agent")
        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])

    def test_provisional_agent_fence_survives_first_not_found(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "herdr_target": "planned-agent",
                "worktree_path": "/worktree/task",
                "agent_dispatch_state": "dispatching",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "old-worker",
                "delivered": False,
            }
        )
        client = mock.Mock()
        client.cancel_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            None,
        ]
        client.get_agent.side_effect = [
            HerdrError("not found", code="agent_not_found"),
            {
                "name": "planned-agent",
                "pane_id": "pane",
                "workspace_id": "workspace",
            },
        ]
        with (
            mock.patch.object(jobs, "_stop_worker", return_value=True),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            jobs.cancel_job("123456789abc")
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            with mock.patch("time.time", return_value=100):
                jobs.recover_jobs()
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            with mock.patch("time.time", return_value=104):
                jobs.recover_jobs()
            self.assertEqual(client.get_agent.call_count, 1)
            with mock.patch("time.time", return_value=105):
                jobs.recover_jobs()

        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])
        self.assertEqual(client.get_agent.call_count, 2)
        self.assertEqual(client.cancel_agent.call_count, 2)

    def test_truly_absent_agent_releases_after_bounded_backoff(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "herdr_target": "missing-agent",
                "agent_dispatch_state": "failed_observing",
                "agent_dispatch_exited": True,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("not found", code="agent_not_found")
        client.cancel_agent.side_effect = HerdrError(
            "not found", code="agent_not_found"
        )
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            for now in (100, 104, 105, 114):
                with mock.patch("time.time", return_value=now):
                    jobs.recover_jobs()
            pending = jobs.read_job("123456789abc")
            self.assertEqual(pending["agent_reconcile_attempts"], 2)
            self.assertEqual(pending["agent_next_reconcile_at"], 115)
            with mock.patch("time.time", return_value=115):
                jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["agent_dispatch_state"], "confirmed_absent")
        self.assertEqual(updated["agent_reconcile_attempts"], 3)
        self.assertFalse(updated["target_release_pending"])

    def test_truly_absent_fork_releases_after_bounded_backoff(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "fork_operation_state": "failed_observing",
                "fork_dispatch_exited": True,
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_source_private": False,
                "fork_operation_target": "me/project",
                "target_release_pending": True,
                "delivered": True,
            }
        )
        github = mock.Mock()
        github.reconcile_fork.return_value = None
        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            for now in (100, 105, 115):
                with mock.patch("time.time", return_value=now):
                    jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["fork_operation_state"], "confirmed_absent")
        self.assertEqual(updated["fork_reconcile_attempts"], 3)
        self.assertFalse(updated["target_release_pending"])

    def test_failed_fork_becomes_visible_after_first_observation(self) -> None:
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
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "fork_operation_state": "failed_observing",
                "fork_dispatch_exited": True,
                "fork_operation_source": source.name_with_owner,
                "fork_operation_source_url": source.url,
                "fork_operation_source_default_branch": source.default_branch,
                "fork_operation_source_private": False,
                "fork_operation_target": fork.name_with_owner,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        github = mock.Mock()
        github.reconcile_fork.side_effect = [None, fork]
        with mock.patch.object(jobs, "GitHubClient", return_value=github):
            with mock.patch("time.time", return_value=100):
                jobs.recover_jobs()
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            with mock.patch("time.time", return_value=105):
                jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["fork_operation_state"], "exists")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertFalse(updated["target_release_pending"])

    def test_truly_absent_worktree_releases_after_bounded_backoff(self) -> None:
        checkout = Path(self.temporary.name) / "missing-worktree"
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "repository": str(Path(self.temporary.name) / "repository"),
                "worktree_branch": "voice/task",
                "worktree_path": str(checkout),
                "worktree_provision_state": "failed_observing",
                "worktree_dispatch_exited": True,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.run_json.return_value = {"worktrees": []}
        with mock.patch.object(jobs, "HerdrClient", return_value=client):
            for now in (100, 105, 115):
                with mock.patch("time.time", return_value=now):
                    jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["worktree_provision_state"], "confirmed_absent")
        self.assertEqual(updated["worktree_reconcile_attempts"], 3)
        self.assertFalse(updated["target_release_pending"])

    def test_dispatching_agent_backoff_is_capped_then_requires_manual_review(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "error": "agent startup failed; external operation reconciliation is pending",
                "result": "agent startup failed; external operation reconciliation is pending",
                "herdr_target": "late-agent",
                "agent_dispatch_state": "dispatching",
                "agent_reconcile_attempts": 4,
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.get_agent.side_effect = HerdrError("not found", code="agent_not_found")
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_worker_is_alive", return_value=False),
        ):
            with mock.patch("time.time", return_value=100):
                jobs.recover_jobs()
            capped = jobs.read_job("123456789abc")
            self.assertEqual(capped["agent_reconcile_attempts"], 5)
            self.assertEqual(capped["agent_next_reconcile_at"], 160)

            with mock.patch("time.time", return_value=160):
                jobs.recover_jobs()
            manual = jobs.read_job("123456789abc")
            self.assertEqual(manual["agent_dispatch_state"], "manual_required")
            self.assertEqual(manual["agent_reconcile_attempts"], 6)
            self.assertTrue(manual["target_release_pending"])
            self.assertIn(
                "manual reconciliation required for Herdr agent late-agent",
                str(manual["error"]),
            )
            self.assertNotIn("reconciliation is pending", str(manual["result"]))

            with mock.patch("time.time", return_value=10_000):
                jobs.recover_jobs()

        self.assertEqual(client.get_agent.call_count, 2)
        client.cancel_agent.assert_not_called()

    def test_ambiguous_fork_requires_manual_review_and_can_confirm_absence(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "error": "fork outcome unknown; external operation reconciliation is pending",
                "result": "fork outcome unknown; external operation reconciliation is pending",
                "fork_operation_state": "ambiguous",
                "fork_operation_source": "source/project",
                "fork_operation_source_url": "https://github.com/source/project",
                "fork_operation_source_default_branch": "main",
                "fork_operation_source_private": False,
                "fork_operation_target": "me/project",
                "fork_reconcile_attempts": 5,
                "herdr_target": "unrelated-agent",
                "target_release_pending": True,
                "delivered": True,
            }
        )
        github = mock.Mock()
        github.reconcile_fork.return_value = None
        client = mock.Mock()
        with (
            mock.patch.object(jobs, "GitHubClient", return_value=github),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch.object(jobs, "_worker_is_alive", return_value=False),
            mock.patch("time.time", return_value=100),
        ):
            jobs.recover_jobs()

        manual = jobs.read_job("123456789abc")
        self.assertEqual(manual["fork_operation_state"], "manual_required")
        self.assertFalse(manual["target_release_pending"])
        client.cancel_agent.assert_called_once_with("unrelated-agent")
        token = str(manual["manual_reconcile_token"])
        with self.assertRaisesRegex(jobs.HarnessError, "fence is stale"):
            jobs.resolve_manual_reconciliation(
                "123456789abc", "fork", "stale-token", "confirmed_absent"
            )
        with self.assertRaisesRegex(jobs.HarnessError, "fence is stale"):
            jobs.resolve_manual_reconciliation(
                "123456789abc", "worktree", token, "confirmed_absent"
            )
        resolved = jobs.resolve_manual_reconciliation(
            "123456789abc", "fork", token, "confirmed_absent"
        )
        self.assertEqual(resolved["fork_operation_state"], "confirmed_absent")
        self.assertEqual(resolved["error"], "fork outcome unknown")
        self.assertEqual(github.reconcile_fork.call_count, 1)

    def test_manual_materialized_agent_is_retained_without_external_action(
        self,
    ) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "failed",
                "error": (
                    "agent startup failed; manual reconciliation required for "
                    "Herdr agent retained-agent"
                ),
                "result": (
                    "agent startup failed; manual reconciliation required for "
                    "Herdr agent retained-agent"
                ),
                "reconciliation_base_error": "agent startup failed",
                "herdr_target": "retained-agent",
                "agent_dispatch_state": "manual_required",
                "manual_reconcile_operation": "agent",
                "manual_reconcile_token": "manual-token",
                "target_release_pending": True,
                "cancellation_reconciliation_pending": True,
                "delivered": True,
            }
        )

        resolved = jobs.resolve_manual_reconciliation(
            "123456789abc", "agent", "manual-token", "materialized"
        )

        self.assertEqual(resolved["agent_dispatch_state"], "retained")
        self.assertEqual(resolved["herdr_target"], "retained-agent")
        self.assertFalse(resolved["target_release_pending"])
        self.assertFalse(resolved["cancellation_reconciliation_pending"])
        self.assertEqual(resolved["error"], "agent startup failed")
        self.assertEqual(resolved["result"], "agent startup failed")

    def test_quarantined_worktree_releases_target_but_blocks_path(self) -> None:
        repository = Path(self.temporary.name) / "repository"
        checkout = Path(self.temporary.name) / "unexpected-worktree"
        checkout.mkdir()
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "repository": str(repository),
                "herdr_target": "agent",
                "worktree_branch": "voice/task",
                "worktree_path": str(checkout),
                "worktree_provision_state": "ambiguous",
                "target_release_pending": True,
                "delivered": True,
            }
        )
        client = mock.Mock()
        client.run_json.return_value = {"worktrees": []}
        with (
            mock.patch.object(jobs, "HerdrClient", return_value=client),
            mock.patch("time.time", return_value=100),
        ):
            jobs.recover_jobs()

        quarantined = jobs.read_job("123456789abc")
        self.assertEqual(quarantined["worktree_provision_state"], "quarantined")
        self.assertTrue(quarantined["worktree_manual_inspection_required"])
        self.assertFalse(quarantined["target_release_pending"])
        client.cancel_agent.assert_called_once_with("agent")

        jobs.write_job(
            {
                "id": "bbbbbbbbbbbb",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
            }
        )
        self.assertIsNone(
            jobs._reserve_worker_worktree(
                "bbbbbbbbbbbb",
                "worker",
                repository,
                "voice/task",
                checkout,
                state="planned",
            )
        )

        jobs.acknowledge_worktree_quarantine("123456789abc")
        self.assertEqual(
            jobs.read_job("123456789abc")["worktree_provision_state"],
            "retained",
        )
        self.assertIsNotNone(
            jobs._reserve_worker_worktree(
                "bbbbbbbbbbbb",
                "worker",
                repository,
                "voice/task",
                checkout,
                state="planned",
            )
        )

    def test_recovery_retries_worker_stop_without_target_reservation(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "worker",
                "worker_pid": 42,
                "worker_process_start": "worker-start",
                "delivered": False,
            }
        )
        alive = True
        stops = 0

        def stop_worker(_job: dict[str, object]) -> bool:
            nonlocal alive, stops
            stops += 1
            if stops == 2:
                alive = False
                return True
            return False

        with (
            mock.patch.object(jobs, "_worker_is_alive", side_effect=lambda _job: alive),
            mock.patch.object(jobs, "_stop_worker", side_effect=stop_worker),
        ):
            jobs.cancel_job("123456789abc")
            self.assertTrue(jobs.read_job("123456789abc")["target_release_pending"])
            jobs.recover_jobs()

        self.assertEqual(stops, 2)
        self.assertFalse(jobs.read_job("123456789abc")["target_release_pending"])

    def test_cancel_retires_unfenced_legacy_worker_first(self) -> None:
        (Path(self.temporary.name) / "123456789abc.json").write_text(
            json.dumps(
                {
                    "id": "123456789abc",
                    "status": "running",
                    "worker_pid": 42,
                    "created_at": 1,
                    "delivered": False,
                }
            )
        )
        with (
            mock.patch.object(jobs, "_worker_is_alive", return_value=True),
            mock.patch.object(
                jobs, "_stop_legacy_worker", return_value=True
            ) as stop_legacy,
        ):
            jobs.cancel_job("123456789abc")

        stop_legacy.assert_called_once_with("123456789abc")
        self.assertEqual(jobs.read_job("123456789abc")["status"], "cancelled")

    def test_recovery_releases_abandoned_cancellation_fence(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "cancelled",
                "herdr_target": "cursor-agent",
                "target_release_pending": True,
                "target_release_owner_pid": 999999,
                "target_release_owner_start": "old-process",
                "completed_at": 1,
                "delivered": True,
            }
        )
        client = mock.Mock()
        with (
            mock.patch.object(jobs, "_process_identity", return_value=None),
            mock.patch.object(jobs, "HerdrClient", return_value=client),
        ):
            jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertFalse(updated["target_release_pending"])
        client.cancel_agent.assert_called_once_with("cursor-agent")

    def test_foreground_delivery_is_acknowledged_explicitly(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "completed",
                "result": "done",
                "completed_at": 1,
                "delivered": False,
            }
        )
        claims: jobs.DeliveryClaims = []
        with mock.patch.object(jobs, "start_job", return_value="123456789abc"):
            result, session = jobs.cursor_turn("do it", delivery_claims=claims)

        self.assertEqual(result, "done")
        self.assertIsNone(session)
        self.assertFalse(jobs.read_job("123456789abc")["delivered"])
        self.assertEqual(len(claims), 1)
        jobs.acknowledge_deliveries(claims)
        self.assertTrue(jobs.read_job("123456789abc")["delivered"])


if __name__ == "__main__":
    unittest.main()
