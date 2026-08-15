from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from local_voice_harness import notifications
from local_voice_harness.notifications import NotificationResult, notify


class NotificationTests(unittest.TestCase):
    def test_availability_is_capability_not_invocation(self) -> None:
        with mock.patch.object(notifications.shutil, "which", return_value=None):
            self.assertFalse(notifications.notification_service_available())
            self.assertFalse(notifications.NotifySendService().available())
        with mock.patch.object(
            notifications.shutil, "which", return_value="/usr/bin/notify-send"
        ):
            self.assertTrue(notifications.notification_service_available())

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
