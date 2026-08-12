from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

TERMINATE_GRACE_SECONDS = 1.0
GROUP_EXIT_POLL_SECONDS = 0.01


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group(
    process: subprocess.Popen[str],
    *,
    deadline: float | None,
) -> bool:
    while _process_group_exists(process.pid):
        process.poll()
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(GROUP_EXIT_POLL_SECONDS)
    process.poll()
    return True


def _stop_process_group(
    process: subprocess.Popen[str],
    *,
    terminate_grace: float,
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return

    if _wait_for_process_group(
        process,
        deadline=time.monotonic() + max(terminate_grace, 0),
    ):
        return

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.poll()
        return

    # Do not return control to cleanup, lock release, or failure persistence until
    # the complete process group has gone.
    _wait_for_process_group(process, deadline=None)


def run_command(
    command: Sequence[str],
    *,
    timeout: float,
    cwd: Path | None = None,
    stdin: str | None = None,
    terminate_grace: float = TERMINATE_GRACE_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run a command in an isolated session and stop all descendants on timeout."""
    arguments = list(command)
    process = subprocess.Popen(
        arguments,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _stop_process_group(process, terminate_grace=terminate_grace)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            arguments,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(
        arguments,
        process.returncode,
        stdout,
        stderr,
    )
