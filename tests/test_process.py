from __future__ import annotations

import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import process
from local_voice_harness.errors import HarnessError


class ProcessIdentityTests(unittest.TestCase):
    def test_process_identity_is_available_for_current_process(self) -> None:
        identity = process.process_identity(os.getpid())

        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertTrue(identity.isdigit())

    def test_boot_identity_is_available(self) -> None:
        self.assertIsNotNone(process.boot_identity())

    def test_process_owner_alive_rejects_reused_pid_after_reboot(self) -> None:
        self.assertFalse(
            process.process_owner_alive(
                42,
                "old-boot",
                "start",
                get_boot_identity=lambda: "new-boot",
                get_process_identity=lambda _pid: "start",
            )
        )

    def test_process_owner_alive_matches_exact_owner(self) -> None:
        self.assertTrue(
            process.process_owner_alive(
                42,
                "boot",
                "start",
                get_boot_identity=lambda: "boot",
                get_process_identity=lambda _pid: "start",
            )
        )


class ProcessHandleTests(unittest.TestCase):
    def test_open_rejects_pid_reuse(self) -> None:
        with mock.patch.object(process, "process_identity", return_value="replacement"):
            handle = process.ProcessHandle.open(42, expected_start="original")

        self.assertIsNone(handle)

    def test_open_accepts_matching_owner(self) -> None:
        from local_voice_harness.process import linux as process_linux

        with (
            mock.patch.object(
                process_linux,
                "capabilities",
                return_value=process.ProcessCapabilities(True, True, True, True),
            ),
            mock.patch.object(os, "pidfd_open", return_value=7),
            mock.patch.object(process_linux, "process_identity", return_value="start"),
        ):
            handle = process.ProcessHandle.open(42, expected_start="start")

        assert handle is not None
        self.assertEqual(handle.pid, 42)
        handle.close()

    def test_terminate_pidfd_escalates_to_sigkill(self) -> None:
        from local_voice_harness.process import linux as process_linux

        with (
            mock.patch.object(process_linux, "pidfd_send") as send,
            mock.patch.object(process_linux, "pidfd_exited", side_effect=[False, True]),
        ):
            stopped = process.terminate_pidfd(3)

        self.assertTrue(stopped)
        self.assertEqual(
            [call.args[1] for call in send.call_args_list],
            [signal.SIGTERM, signal.SIGKILL],
        )


class WakeStateTests(unittest.TestCase):
    def test_legacy_plain_pid_files_are_not_signalled(self) -> None:
        from local_voice_harness.wake import daemon as wake_daemon

        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "wake.pid"
            pid_path.write_text("4321")
            with (
                mock.patch.object(wake_daemon, "WAKE_PID_PATH", pid_path),
                self.assertRaisesRegex(HarnessError, "not running"),
            ):
                wake_daemon.request_listen()


if __name__ == "__main__":
    unittest.main()
