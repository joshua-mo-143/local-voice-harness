from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

ROFI_TIMEOUT_SECONDS = 300


def _dmenu(
    options: Sequence[str],
    *,
    prompt: str,
    message: str,
    allow_custom: bool,
) -> str | None:
    executable = shutil.which("rofi")
    if executable is None:
        return None
    command = [
        executable,
        "-dmenu",
        "-i",
        "-p",
        prompt,
        "-mesg",
        message,
    ]
    if not allow_custom:
        command.append("-no-custom")
    if not options:
        command.extend(("-theme-str", "listview { enabled: false; }"))
    try:
        process = subprocess.run(
            command,
            input="\n".join(options),
            capture_output=True,
            text=True,
            timeout=ROFI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode:
        return None
    value = process.stdout.strip()
    return value or None


def choose_repository(repositories: Sequence[str]) -> str | None:
    return _dmenu(
        repositories,
        prompt="Repository",
        message="Select a local repository or paste a Git HTTPS/SSH URL",
        allow_custom=True,
    )


def confirm_clone(url: str) -> bool:
    choice = _dmenu(
        ("Clone", "Cancel"),
        prompt="Clone repository?",
        message=url,
        allow_custom=False,
    )
    return choice == "Clone"
