from __future__ import annotations

import subprocess
from enum import StrEnum


class NotificationResult(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def notify(message: str, *, error: bool = False) -> NotificationResult:
    urgency = "critical" if error else "normal"
    try:
        result = subprocess.run(
            ["notify-send", "-u", urgency, "-t", "5000", "Voice harness", message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return NotificationResult.FAILED
    if result.returncode != 0:
        return NotificationResult.FAILED
    return NotificationResult.SUCCEEDED
