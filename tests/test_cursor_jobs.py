from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor import jobs


class CursorJobStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.jobs_patch = mock.patch.object(jobs, "JOBS_DIR", Path(self.temporary.name))
        self.jobs_patch.start()

    def tearDown(self) -> None:
        self.jobs_patch.stop()
        self.temporary.cleanup()

    def test_repository_reply_preserves_original_task(self) -> None:
        job = {
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

        start.assert_called_once_with("Work on APP-43", repository=None, agent=None)
        reply.assert_not_called()
        self.assertEqual(result, "done")
        self.assertIsNone(session)

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
        process.wait.return_value = 0
        with (
            mock.patch("subprocess.Popen", return_value=process) as popen,
            mock.patch("threading.Thread") as thread,
        ):
            jobs.launch_worker("123456789abc")
        command = popen.call_args.args[0]
        self.assertEqual(
            command[1:3], ["-m", "local_voice_harness.cursor.worker"]
        )
        thread.assert_called_once()

    def test_dead_worker_reconciles_existing_agent(self) -> None:
        jobs.write_job(
            {
                "id": "123456789abc",
                "status": "running",
                "created_at": 1,
                "worker_pid": 999999,
                "herdr_target": "cursor-agent",
                "delivered": False,
            }
        )
        with (
            mock.patch("time.time", return_value=100),
            mock.patch("os.kill", side_effect=ProcessLookupError),
            mock.patch.object(jobs, "launch_worker") as launch,
        ):
            self.assertEqual(jobs.pending_results(), [])
        updated = json.loads(
            (Path(self.temporary.name) / "123456789abc.json").read_text()
        )
        self.assertTrue(updated["reconcile"])
        launch.assert_called_once_with("123456789abc")


if __name__ == "__main__":
    unittest.main()
