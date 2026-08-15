from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from local_voice_harness import notifications
from local_voice_harness.notifications import NotificationResult, notify


class NotificationTests(unittest.TestCase):
    def test_availability_requires_working_notification_service(self) -> None:
        with mock.patch.object(notifications.shutil, "which", return_value=None):
            self.assertFalse(notifications.notification_service_available())
            self.assertFalse(notifications.NotifySendService().available())
        with (
            mock.patch.object(
                notifications.shutil, "which", return_value="/usr/bin/notify-send"
            ),
            mock.patch.object(
                notifications.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
        ):
            self.assertTrue(notifications.notification_service_available())
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            notifications.CAPABILITY_PROBE_TIMEOUT_SECONDS,
        )

    def test_availability_probe_timeout_is_unavailable(self) -> None:
        with (
            mock.patch.object(
                notifications.shutil, "which", return_value="/usr/bin/notify-send"
            ),
            mock.patch.object(
                notifications.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("notify-send", 2),
            ),
        ):
            self.assertFalse(notifications.notification_service_available())

    @mock.patch("local_voice_harness.notifications.subprocess.run")
    def test_success_requires_zero_exit_status(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)

        result = notify("Done")

        self.assertEqual(result, NotificationResult.SUCCEEDED)
        run.assert_called_once_with(
            [
                "notify-send",
                "-u",
                "normal",
                "-t",
                "5000",
                "Voice harness",
                "Done",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @mock.patch("local_voice_harness.notifications.subprocess.run")
    def test_nonzero_exit_status_is_failure(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 1)

        self.assertEqual(notify("No delivery"), NotificationResult.FAILED)

    @mock.patch("local_voice_harness.notifications.subprocess.run")
    def test_missing_executable_is_failure(self, run: mock.Mock) -> None:
        run.side_effect = FileNotFoundError("notify-send")

        self.assertEqual(notify("No backend"), NotificationResult.FAILED)
