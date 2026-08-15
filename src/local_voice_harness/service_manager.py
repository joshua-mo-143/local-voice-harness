from __future__ import annotations

import importlib.resources
import subprocess
from dataclasses import dataclass

from .config import (
    PROJECT_ROOT,
    SERVICE_FILES,
    START_SERVICES,
    STOP_SERVICES,
    SYSTEMD_USER_DIR,
)
from .errors import HarnessError
from .integrations.herdr import HerdrError
from .integrations.registry import IntegrationRegistry, build_integration_registry
from .platform_services import user_services
from .service_units import audit_installed
from .user_config import UserConfig, load_user_config


@dataclass(frozen=True)
class ServiceManagementSnapshot:
    """One resolved configuration and client registry for a management action."""

    config: UserConfig
    registry: IntegrationRegistry

    @classmethod
    def load(cls) -> ServiceManagementSnapshot:
        config = load_user_config()
        return cls(config=config, registry=build_integration_registry(config))


def _resolved(
    snapshot: ServiceManagementSnapshot | None,
) -> ServiceManagementSnapshot:
    return snapshot if snapshot is not None else ServiceManagementSnapshot.load()


def systemctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Delegate user-service commands through the platform supervisor."""

    return user_services().run(*arguments, check=check)


def unit_text(name: str) -> str:
    source = PROJECT_ROOT / "systemd" / "user" / name
    if source.is_file():
        return source.read_text()
    resource = importlib.resources.files("local_voice_harness").joinpath(
        "data", "systemd", name
    )
    return resource.read_text()


def install_services(
    *,
    force: bool,
    replace_dictation: bool = False,
    snapshot: ServiceManagementSnapshot | None = None,
) -> None:
    _resolved(snapshot)
    SYSTEMD_USER_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
    for name in SERVICE_FILES:
        destination = SYSTEMD_USER_DIR / name
        expected = unit_text(name)
        if destination.exists() or destination.is_symlink():
            try:
                if destination.read_text() == expected:
                    continue
            except OSError:
                pass
            if name == "dictation.service" and not replace_dictation:
                print(
                    f"Preserved existing standalone {destination}; it may not "
                    "include shipped hardening. Rerun with --force "
                    "--replace-dictation only after reviewing its customizations."
                )
                continue
            if not force:
                raise HarnessError(
                    f"{destination} differs; rerun services install --force to replace it"
                )
            if destination.is_dir() and not destination.is_symlink():
                raise HarnessError(f"refusing to replace directory {destination}")
            destination.unlink()
        destination.write_text(expected)
    systemctl("daemon-reload")
    systemctl("enable", *START_SERVICES)
    print("Installed voice harness services. Run `voice-harness services start`.")


def audit_services(snapshot: ServiceManagementSnapshot | None = None) -> int:
    """Read effective installed units and runtime state without changing them."""

    resolved = _resolved(snapshot)
    return audit_installed(config=resolved.config)


def start_services(snapshot: ServiceManagementSnapshot | None = None) -> None:
    _resolved(snapshot)
    systemctl("start", *START_SERVICES)
    print("Voice harness listener started.")


def stop_herdr(snapshot: ServiceManagementSnapshot | None = None) -> None:
    resolved = _resolved(snapshot)
    client = resolved.registry.herdr_client()
    if not client.is_running():
        return
    process = client.run("server", "stop", check=False)
    if process.returncode:
        raise HarnessError(
            process.stderr.strip() or process.stdout.strip() or "Herdr stop failed"
        )


def stop_services(
    *,
    include_herdr: bool,
    snapshot: ServiceManagementSnapshot | None = None,
) -> None:
    resolved = _resolved(snapshot)
    systemctl("stop", *STOP_SERVICES, check=False)
    if include_herdr:
        stop_herdr(resolved)
    print(
        "Voice harness services stopped"
        + (" including Herdr." if include_herdr else "; Herdr was left running.")
    )


def restart_services(
    *,
    include_herdr: bool,
    snapshot: ServiceManagementSnapshot | None = None,
) -> None:
    resolved = _resolved(snapshot)
    stop_services(include_herdr=include_herdr, snapshot=resolved)
    start_services(resolved)


def status(snapshot: ServiceManagementSnapshot | None = None) -> None:
    resolved = _resolved(snapshot)
    rows = []
    for name in SERVICE_FILES:
        process = systemctl("is-active", name, check=False)
        rows.append((name, process.stdout.strip() or "inactive"))
    try:
        herdr_state = (
            "running" if resolved.registry.herdr_client().is_running() else "stopped"
        )
    except HerdrError:
        herdr_state = "unavailable"
    width = max(len(name) for name, _state in rows)
    for name, state in rows:
        print(f"{name:<{width}}  {state}")
    print(f"{'herdr':<{width}}  {herdr_state}")


def logs(*, follow: bool, lines: int) -> None:
    command = ["journalctl", "--user"]
    for name in SERVICE_FILES:
        command.extend(("-u", name))
    command.extend(("-n", str(lines)))
    if follow:
        command.append("-f")
    raise SystemExit(subprocess.run(command, check=False).returncode)


def uninstall_services(
    *,
    include_herdr: bool,
    snapshot: ServiceManagementSnapshot | None = None,
) -> None:
    resolved = _resolved(snapshot)
    stop_services(include_herdr=include_herdr, snapshot=resolved)
    systemctl("disable", *START_SERVICES, check=False)
    for name in SERVICE_FILES:
        destination = SYSTEMD_USER_DIR / name
        if destination.is_file() or destination.is_symlink():
            try:
                if destination.read_text() == unit_text(name):
                    destination.unlink()
            except OSError:
                continue
    systemctl("daemon-reload")
    print("Uninstalled voice harness units.")
