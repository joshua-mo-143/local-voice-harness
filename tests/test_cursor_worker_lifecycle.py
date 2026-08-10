from __future__ import annotations

import signal
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.cursor import provisioning
from local_voice_harness.cursor.model import CursorJob, JobStatus, transition
from local_voice_harness.cursor.store import JobStore, MaintenanceLease
from local_voice_harness.cursor.worker_lifecycle import (
    WorkerCancelled,
    WorkerContext,
    begin_worker,
    inspect_and_stop_legacy_worker,
    launch_worker,
    legacy_worker_command_matches,
    process_owner_alive,
    signal_worker,
    worker_is_alive,
)
from local_voice_harness.integrations.herdr import AgentSelection


class CursorWorkerLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = JobStore(self.root / "jobs", self.root / "legacy")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_queued(self, job_id: str = "123456789abc") -> CursorJob:
        return self.store.create(
            CursorJob.from_dict(
                {
                    "id": job_id,
                    "status": "queued",
                    "request": "test",
                    "created_at": 1,
                    "queued_at": 1,
                    "delivered": False,
                }
            )
        )

    def test_boot_identity_fences_reused_pid_after_reboot(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "routing",
                "worker_token": "claim",
                "worker_pid": 42,
                "worker_boot_id": "old-boot",
                "worker_process_start": "start",
            }
        )

        self.assertFalse(
            worker_is_alive(
                job,
                get_boot_identity=lambda: "new-boot",
                get_process_identity=lambda _pid: "start",
            )
        )
        self.assertTrue(
            worker_is_alive(
                job,
                get_boot_identity=lambda: "old-boot",
                get_process_identity=lambda _pid: "start",
            )
        )

    def test_begin_worker_records_typed_owner_and_stale_context_is_fenced(
        self,
    ) -> None:
        self.create_queued()
        claimed = begin_worker(
            self.store,
            "123456789abc",
            None,
            pid=42,
            now=10,
            get_boot_identity=lambda: "boot",
            get_process_identity=lambda _pid: "start",
        )

        assert claimed is not None
        job, token = claimed
        self.assertEqual(job.status, JobStatus.ROUTING)
        self.assertEqual(job.worker_pid, 42)
        context = WorkerContext(self.store, job, token, threading.Event())
        self.assertEqual(context.checkpoint().id, job.id)
        self.store.update(
            job.id,
            lambda current: transition(
                current,
                JobStatus.QUEUED,
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
                queued_at=11,
            ),
        )
        with self.assertRaises(WorkerCancelled):
            context.checkpoint()

    def test_signal_requires_verified_command_and_claim(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "running",
                "worker_token": "claim",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
            }
        )
        with mock.patch("os.killpg") as kill_group, mock.patch("os.kill") as kill:
            self.assertFalse(
                signal_worker(
                    job,
                    signal.SIGTERM,
                    is_alive=lambda _job: True,
                    command_matches=lambda _job: False,
                )
            )
        kill_group.assert_not_called()
        kill.assert_not_called()

    def test_process_owner_identity_distinguishes_reuse_from_uncertainty(self) -> None:
        self.assertFalse(
            process_owner_alive(
                42,
                "boot",
                "old-start",
                get_boot_identity=lambda: "boot",
                get_process_identity=lambda _pid: "replacement-start",
            )
        )
        with mock.patch("os.kill", return_value=None):
            self.assertIsNone(
                process_owner_alive(
                    42,
                    "boot",
                    "start",
                    get_boot_identity=lambda: "boot",
                    get_process_identity=lambda _pid: None,
                )
            )

    def test_signal_rechecks_identity_during_process_reuse_race(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "running",
                "worker_token": "claim",
                "worker_pid": 42,
                "worker_boot_id": "boot",
                "worker_process_start": "start",
            }
        )
        second_check = threading.Event()
        replacement_ready = threading.Event()
        calls = 0

        def is_alive(_job: CursorJob) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                return True
            second_check.set()
            self.assertTrue(replacement_ready.wait(2))
            return False

        result: list[bool] = []
        with (
            mock.patch("os.getpgid", return_value=42),
            mock.patch("os.killpg") as kill_group,
            mock.patch("os.kill") as kill,
        ):
            thread = threading.Thread(
                target=lambda: result.append(
                    signal_worker(
                        job,
                        signal.SIGTERM,
                        is_alive=is_alive,
                        command_matches=lambda _job: True,
                    )
                )
            )
            thread.start()
            self.assertTrue(second_check.wait(2))
            replacement_ready.set()
            thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])
        kill_group.assert_not_called()
        kill.assert_not_called()

    def test_legacy_worker_command_requires_exact_module_and_job_id(self) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "running",
                "worker_pid": 42,
                "worker_process_start": "start",
            }
        )
        exact = b"\0".join(
            (
                b"/usr/bin/python",
                b"-m",
                b"local_voice_harness.cursor.worker",
                b"123456789abc",
                b"",
            )
        )
        with mock.patch.object(Path, "read_bytes", return_value=exact):
            self.assertTrue(legacy_worker_command_matches(job))
        with mock.patch.object(
            Path,
            "read_bytes",
            return_value=exact.replace(b"123456789abc", b"aaaaaaaaaaaa"),
        ):
            self.assertFalse(legacy_worker_command_matches(job))

    def test_legacy_worker_is_stopped_only_after_identity_and_command_match(
        self,
    ) -> None:
        job = CursorJob.from_dict(
            {
                "id": "123456789abc",
                "status": "running",
                "worker_pid": 42,
                "worker_process_start": "start",
            }
        )
        identities = iter(("start", "start", None))
        with mock.patch("os.getpgid", return_value=42), mock.patch("os.killpg") as kill:
            outcome = inspect_and_stop_legacy_worker(
                job,
                timeout=0.1,
                get_process_identity=lambda _pid: next(identities),
                command_matches=lambda _job: True,
            )
        self.assertEqual(outcome, "stopped")
        kill.assert_called_once_with(42, signal.SIGTERM)

        with mock.patch("os.killpg") as unsafe_kill:
            self.assertEqual(
                inspect_and_stop_legacy_worker(
                    job,
                    get_process_identity=lambda _pid: "start",
                    command_matches=lambda _job: False,
                ),
                "unsafe",
            )
        unsafe_kill.assert_not_called()

    def test_concurrent_launch_reservation_prevents_duplicate_spawn(self) -> None:
        self.create_queued()
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 43
        process.wait.return_value = 0

        def prepare_failure(
            job: CursorJob, message: str, failed_at: float
        ) -> CursorJob:
            return transition(
                job,
                JobStatus.FAILED,
                error=message,
                result=message,
                completed_at=failed_at,
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
            )

        with (
            mock.patch(
                "local_voice_harness.cursor.worker_lifecycle.subprocess.Popen",
                return_value=process,
            ) as popen,
            mock.patch("threading.Thread"),
        ):
            for _ in range(2):
                launch_worker(
                    self.store,
                    self.root / "logs",
                    "123456789abc",
                    prepare_failure=prepare_failure,
                    get_boot_identity=lambda: "boot",
                    get_process_identity=lambda pid: f"start-{pid}",
                )

        popen.assert_called_once()
        self.assertEqual(self.store.get("123456789abc").worker_pid, 43)

    def test_maintenance_wins_spawn_handoff_and_reaps_unadopted_child(self) -> None:
        self.create_queued()
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 43
        process.poll.return_value = None
        process.wait.return_value = 0
        child_identity_requested = threading.Event()
        allow_identity = threading.Event()
        failures: list[BaseException] = []

        def process_identity(pid: int) -> str | None:
            if pid != 43:
                return f"start-{pid}"
            child_identity_requested.set()
            self.assertTrue(allow_identity.wait(2))
            return "start-43"

        def prepare_failure(
            job: CursorJob, message: str, failed_at: float
        ) -> CursorJob:
            return transition(
                job,
                JobStatus.FAILED,
                error=message,
                result=message,
                completed_at=failed_at,
                delivered=False,
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
            )

        def launch() -> None:
            try:
                launch_worker(
                    self.store,
                    self.root / "logs",
                    "123456789abc",
                    prepare_failure=prepare_failure,
                    get_boot_identity=lambda: "boot",
                    get_process_identity=process_identity,
                )
            except BaseException as exc:
                failures.append(exc)

        with mock.patch(
            "local_voice_harness.cursor.worker_lifecycle.subprocess.Popen",
            return_value=process,
        ):
            launcher = threading.Thread(target=launch)
            launcher.start()
            self.assertTrue(child_identity_requested.wait(2))
            lease = MaintenanceLease("maintenance", 1, 99, "boot", "owner-start")
            self.store.begin_maintenance(
                lease,
                lambda _job: None,
                owner_alive=lambda _lease: False,
            )
            allow_identity.set()
            launcher.join(2)

        self.assertFalse(launcher.is_alive())
        self.assertEqual(failures, [])
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=0.5)
        self.assertIsNone(self.store.get("123456789abc").worker_token)
        self.assertTrue(self.store.abort_maintenance(lease.token))

    def test_maintenance_fences_concurrent_critical_operation_commit(self) -> None:
        self.create_queued()
        claimed = begin_worker(
            self.store,
            "123456789abc",
            None,
            pid=42,
            get_boot_identity=lambda: "boot",
            get_process_identity=lambda _pid: "start",
        )
        assert claimed is not None
        _job, token = claimed
        staging = threading.Event()
        release = threading.Event()
        operation_finished = threading.Event()
        result: list[CursorJob | None] = []
        lease = MaintenanceLease("maintenance", 1, 99, "boot", "owner-start")

        def stage(_job: CursorJob) -> None:
            staging.set()
            self.assertTrue(release.wait(2))

        def maintain() -> None:
            self.store.begin_maintenance(
                lease,
                stage,
                owner_alive=lambda _lease: False,
            )

        selection = AgentSelection(
            "agent",
            "pane",
            "workspace",
            "/repo",
            "agent",
            "/repo/worktree",
        )

        def reserve() -> None:
            result.append(
                provisioning._reserve_worker_target(
                    self.store,
                    "123456789abc",
                    token,
                    selection,
                    Path("/repo"),
                    None,
                    dispatching=True,
                )
            )
            operation_finished.set()

        maintenance_thread = threading.Thread(target=maintain)
        maintenance_thread.start()
        self.assertTrue(staging.wait(2))
        operation_thread = threading.Thread(target=reserve)
        operation_thread.start()
        self.assertFalse(operation_finished.wait(0.05))
        release.set()
        maintenance_thread.join(2)
        operation_thread.join(2)

        self.assertFalse(maintenance_thread.is_alive())
        self.assertFalse(operation_thread.is_alive())
        self.assertEqual(result, [None])
        self.assertIsNone(self.store.get("123456789abc").herdr_target)
        self.assertTrue(self.store.abort_maintenance(lease.token))

    def test_fast_child_exit_persists_failure_and_clears_launcher_owner(
        self,
    ) -> None:
        self.create_queued()
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 43
        process.poll.return_value = 1

        def prepare_failure(
            job: CursorJob, message: str, failed_at: float
        ) -> CursorJob:
            return transition(
                job,
                JobStatus.FAILED,
                error=message,
                result=message,
                completed_at=failed_at,
                delivered=False,
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
            )

        with (
            mock.patch(
                "local_voice_harness.cursor.worker_lifecycle.subprocess.Popen",
                return_value=process,
            ),
            mock.patch("threading.Thread") as waiter,
        ):
            launch_worker(
                self.store,
                self.root / "logs",
                "123456789abc",
                prepare_failure=prepare_failure,
                get_boot_identity=lambda: "boot",
                get_process_identity=lambda pid: (
                    "launcher-start" if pid == __import__("os").getpid() else None
                ),
            )

        failed = self.store.get("123456789abc")
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertIsNone(failed.worker_token)
        self.assertIsNone(failed.worker_pid)
        self.assertIn("identity handoff", failed.error or "")
        waiter.assert_not_called()

    def test_child_exit_during_terminate_still_clears_launcher_owner(self) -> None:
        self.create_queued()
        process = mock.Mock(spec=subprocess.Popen)
        process.pid = 43
        process.poll.return_value = None
        process.terminate.side_effect = ProcessLookupError
        process.wait.side_effect = ChildProcessError

        def prepare_failure(
            job: CursorJob, message: str, failed_at: float
        ) -> CursorJob:
            return transition(
                job,
                JobStatus.FAILED,
                error=message,
                result=message,
                completed_at=failed_at,
                delivered=False,
                worker_token=None,
                worker_pid=None,
                worker_boot_id=None,
                worker_process_start=None,
            )

        with (
            mock.patch(
                "local_voice_harness.cursor.worker_lifecycle.subprocess.Popen",
                return_value=process,
            ),
            mock.patch("local_voice_harness.cursor.worker_lifecycle.time.sleep"),
            mock.patch("threading.Thread") as waiter,
        ):
            launch_worker(
                self.store,
                self.root / "logs",
                "123456789abc",
                prepare_failure=prepare_failure,
                get_boot_identity=lambda: "boot",
                get_process_identity=lambda pid: (
                    "launcher-start" if pid == __import__("os").getpid() else None
                ),
            )

        failed = self.store.get("123456789abc")
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertIsNone(failed.worker_token)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=0.5)
        waiter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
