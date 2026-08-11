from __future__ import annotations

import os
import select
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..errors import HarnessError


@dataclass(frozen=True, slots=True)
class ProcessCapabilities:
    process_start_identity: bool
    boot_identity: bool
    pidfd_open: bool
    pidfd_send_signal: bool

    @property
    def strong_ownership(self) -> bool:
        return self.pidfd_open and self.pidfd_send_signal


def process_identity(pid: int) -> str | None:
    """Return the Linux process start tick, which remains stable for its lifetime."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    return fields[19] if closing >= 0 and len(fields) > 19 else None


def boot_identity() -> str | None:
    try:
        identity = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None
    return identity or None


@lru_cache(maxsize=1)
def capabilities() -> ProcessCapabilities:
    process_start = process_identity(os.getpid()) is not None
    boot = boot_identity() is not None
    pidfd_open = hasattr(os, "pidfd_open")
    pidfd_send = hasattr(signal, "pidfd_send_signal")
    return ProcessCapabilities(
        process_start_identity=process_start,
        boot_identity=boot,
        pidfd_open=pidfd_open,
        pidfd_send_signal=pidfd_send,
    )


def process_owner_alive(
    pid: int,
    boot_id: str,
    process_start: str,
    *,
    get_boot_identity: Callable[[], str | None] = boot_identity,
    get_process_identity: Callable[[int], str | None] = process_identity,
) -> bool | None:
    """Return whether an exact process owner lives, or ``None`` if unknowable."""

    current_boot = get_boot_identity()
    if current_boot is None:
        return None
    if current_boot != boot_id:
        return False
    current_start = get_process_identity(pid)
    if current_start is not None:
        return current_start == process_start
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    return None


def pidfd_send(pidfd: int, sig: signal.Signals) -> None:
    signal.pidfd_send_signal(pidfd, sig)


def pidfd_exited(pidfd: int, timeout: float) -> bool:
    readable, _, _ = select.select([pidfd], [], [], timeout)
    return bool(readable)


@dataclass(slots=True)
class ProcessHandle:
    pid: int
    pidfd: int
    process_start: str | None = None

    @classmethod
    def open(
        cls,
        pid: int,
        *,
        expected_start: str | None = None,
        require_identity: bool = True,
    ) -> ProcessHandle | None:
        if pid <= 0:
            return None
        if not capabilities().pidfd_open:
            if expected_start is None:
                return None
            current = process_identity(pid)
            if current is None:
                return None
            if current != expected_start:
                return None
            raise HarnessError(
                "pidfd is unavailable; refusing to signal without strong ownership"
            )
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            return None
        start = process_identity(pid)
        if require_identity and start is None:
            os.close(pidfd)
            return None
        if expected_start is not None and start != expected_start:
            os.close(pidfd)
            return None
        return cls(pid, pidfd, start)

    def close(self) -> None:
        os.close(self.pidfd)

    def __enter__(self) -> ProcessHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def send_signal(self, sig: signal.Signals) -> None:
        pidfd_send(self.pidfd, sig)

    def wait_exit(self, timeout: float) -> bool:
        return pidfd_exited(self.pidfd, timeout)


def terminate_pidfd(
    pidfd: int,
    *,
    sigint_first: bool = False,
    sigterm_timeout: float = 2.0,
    sigkill_timeout: float = 2.0,
) -> bool:
    """Signal a pidfd-backed process and wait for exit; return True when it exits."""

    try:
        if sigint_first:
            pidfd_send(pidfd, signal.SIGINT)
            if pidfd_exited(pidfd, sigterm_timeout):
                return True
        pidfd_send(pidfd, signal.SIGTERM)
        if pidfd_exited(pidfd, sigterm_timeout):
            return True
        pidfd_send(pidfd, signal.SIGKILL)
        return pidfd_exited(pidfd, sigkill_timeout)
    except ProcessLookupError:
        return pidfd_exited(pidfd, 0)


def capability_diagnostics() -> list[str]:
    caps = capabilities()
    issues: list[str] = []
    if not caps.process_start_identity:
        issues.append("/proc start identity is unavailable")
    if not caps.boot_identity:
        issues.append("boot identity is unavailable")
    if sys.platform.startswith("linux") and not caps.strong_ownership:
        missing = []
        if not caps.pidfd_open:
            missing.append("pidfd_open")
        if not caps.pidfd_send_signal:
            missing.append("pidfd_send_signal")
        issues.append(
            "strong process ownership is unavailable"
            + (f" ({', '.join(missing)})" if missing else "")
        )
    return issues
