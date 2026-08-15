"""Platform service interfaces and the Linux systemd / desktop implementations."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .credentials import CredentialStore, SecretServiceStore
from .notifications import NotificationService, NotifySendService


class ServiceSupervisor(Protocol):
    """User-session service supervision used by core orchestration."""

    def available(self) -> bool: ...

    def run(
        self, *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]: ...

    def start(
        self, *units: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]: ...

    def stop(
        self, *units: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]: ...

    def enable(
        self, *units: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]: ...

    def disable(
        self, *units: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]: ...

    def reload(self, *, check: bool = True) -> subprocess.CompletedProcess[str]: ...

    def is_active(self, unit: str) -> str: ...

    def show(self, unit: str, properties: Sequence[str]) -> dict[str, str]: ...

    def user_environment(self) -> dict[str, str]: ...

    def restart(
        self, unit: str, *, check: bool = False
    ) -> subprocess.CompletedProcess[str]: ...

    def try_restart(self, unit: str) -> subprocess.CompletedProcess[str]: ...

    def start_transient(
        self, unit: str, command: Sequence[str]
    ) -> subprocess.CompletedProcess[str]: ...


class SystemdUserSupervisor:
    """Linux implementation that keeps systemd command construction here."""

    def __init__(
        self,
        *,
        systemctl: str = "systemctl",
        systemd_run: str = "systemd-run",
    ) -> None:
        self.systemctl = systemctl
        self.systemd_run = systemd_run

    def available(self) -> bool:
        return shutil.which(self.systemctl) is not None

    def run(
        self, *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.systemctl, "--user", *arguments],
            capture_output=True,
            text=True,
            check=check,
        )

    def start(
        self, *units: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self.run("start", *units, check=check)

    def stop(
        self, *units: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self.run("stop", *units, check=check)

    def enable(
        self, *units: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self.run("enable", *units, check=check)

    def disable(
        self, *units: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self.run("disable", *units, check=check)

    def reload(self, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.run("daemon-reload", check=check)

    def is_active(self, unit: str) -> str:
        process = self.run("is-active", unit, check=False)
        return process.stdout.strip() or "inactive"

    def show(self, unit: str, properties: Sequence[str]) -> dict[str, str]:
        arguments = ["show", unit, "--no-pager"]
        arguments.extend(f"--property={name}" for name in properties)
        process = self.run(*arguments, check=False)
        values: dict[str, str] = {}
        if process.returncode:
            return values
        for line in process.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        return values

    def user_environment(self) -> dict[str, str]:
        process = self.run("show-environment", check=False)
        values: dict[str, str] = {}
        if process.returncode:
            return values
        for line in process.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and key and value:
                values[key] = value
        return values

    def restart(
        self, unit: str, *, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self.run("restart", unit, check=check)

    def try_restart(self, unit: str) -> subprocess.CompletedProcess[str]:
        return self.run("try-restart", unit, check=False)

    def start_transient(
        self, unit: str, command: Sequence[str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.systemd_run, "--user", f"--unit={unit}", "--collect", *command],
            capture_output=True,
            text=True,
            check=False,
        )


@dataclass(frozen=True)
class LinuxPlatform:
    """Bundled Linux implementations for supervision, secrets, and notifications."""

    services: ServiceSupervisor
    credentials: CredentialStore
    notifications: NotificationService


def user_services() -> ServiceSupervisor:
    return SystemdUserSupervisor()


def linux_platform() -> LinuxPlatform:
    return LinuxPlatform(
        services=user_services(),
        credentials=SecretServiceStore(),
        notifications=NotifySendService(),
    )
