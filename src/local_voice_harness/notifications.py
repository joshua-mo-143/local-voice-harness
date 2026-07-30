from __future__ import annotations

import subprocess


def notify(message: str, *, error: bool = False) -> None:
    urgency = "critical" if error else "normal"
    subprocess.run(
        ["notify-send", "-u", urgency, "-t", "5000", "Voice harness", message],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
