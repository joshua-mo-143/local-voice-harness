from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor import jobs
from local_voice_harness.integrations.github import (
    GitHubIssue,
    GitHubRepository,
    ProvisionedIssue,
    ProvisionedRepository,
)
from local_voice_harness.integrations.herdr import AgentSelection, PromptOutcome


class CursorJobStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.jobs_patch = mock.patch.object(jobs, "JOBS_DIR", Path(self.temporary.name))
        self.jobs_patch.start()

    def tearDown(self) -> None:
        self.jobs_patch.stop()
        self.temporary.cleanup()

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
        provisioned = ProvisionedRepository(source, fork, repository)
        jobs.write_job(
            {
                "id": "123456789abc",
                "request": "Fork this repo and add a feature",
                "github_repository": "source/project",
                "fork_requested": True,
                "worktree_branch": "voice/github-123456789abc",
                "worktree_label": "github-123456",
                "status": "queued",
                "created_at": 1,
                "delivered": False,
            }
        )
        github = mock.Mock()
        github.provision_public_fork.return_value = provisioned
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

        github.provision_public_fork.assert_called_once_with("source/project")
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-123456789abc",
            worktree_label="github-123456",
        )
        updated = jobs.read_job("123456789abc")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["fork_repository"], "me/project")
        self.assertEqual(updated["repository"], str(repository))

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
        self.assertEqual(job["schema_version"], 3)
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

        github.provision_issue.assert_called_once_with(issue, candidates=[repository])
        client.ensure_agent.assert_called_once_with(
            repository,
            issue_key=None,
            agent_hint=None,
            reserved=set(),
            worktree_branch="voice/github-issue-42",
            worktree_label="issue-42",
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
                "herdr_target": "first-agent",
                "worktree_path": "/worktree/issue-42",
            }
        )
        jobs.write_job(
            {
                "id": "bbbbbbbbbbbb",
                "status": "routing",
                "worker_token": "worker-token",
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
        client.choose_or_clone_repository.assert_called_once_with([])

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

        self.assertIn("cursor-agent", reserved_during_cancel)
        self.assertNotIn("cursor-agent", jobs.reserved_targets())

    def test_cancel_retires_unfenced_legacy_worker_first(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "worker_pid": 42,
                "created_at": 1,
                "delivered": False,
            }
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
        with mock.patch.object(jobs, "_process_identity", return_value=None):
            jobs.recover_jobs()

        updated = jobs.read_job("123456789abc")
        self.assertFalse(updated["target_release_pending"])

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
