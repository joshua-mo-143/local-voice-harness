"""Distro adapters and path discovery for the one-shot installer.

This module uses only the standard library so ``scripts/install.sh`` can
resolve packages and paths before the project environment exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .install_profile import InstallationPlan, resolve_installation_plan

GENERIC_PACKAGES = (
    "pipewire",
    "pipewire-tools",
    "wireplumber",
    "libnotify",
    "git",
    "curl",
    "github-cli",
    "xdotool",
    "xclip",
    "wl-clipboard",
    "wtype",
    "uv",
    "libsndfile",
    "ffmpeg",
    "libsecret",
    "oo7",
    "cuda",
    "llama.cpp-cuda",
)


class DistroFamily(StrEnum):
    ARCH = "arch"
    DEBIAN = "debian"
    FEDORA = "fedora"


class DistroError(ValueError):
    """The host distribution cannot be used for installation."""


@dataclass(frozen=True)
class InstallPaths:
    """Discovered checkout, executable, and user-service locations."""

    checkout: Path
    home: Path
    user_bin: Path
    voice_harness: Path
    systemd_user_dir: Path
    chatterbox_dir: Path
    config_dir: Path


@dataclass(frozen=True)
class DistroPlan:
    """Package-manager command and mapped packages for one distro family."""

    family: str
    distro_id: str
    package_manager: str
    install_command: tuple[str, ...]
    packages: tuple[str, ...]
    cuda_packages: tuple[str, ...]
    skipped_packages: tuple[str, ...]
    uv_bootstrap: bool
    paths: InstallPaths

    def nvidia_packages(self) -> tuple[str, ...]:
        markers = ("cuda", "nvidia")
        return tuple(
            package
            for package in (*self.packages, *self.cuda_packages)
            if any(marker in package.casefold() for marker in markers)
        )


_ARCH_PACKAGES = {
    **{name: name for name in GENERIC_PACKAGES},
    "pipewire-tools": "pipewire-audio",
}

_DEBIAN_PACKAGES = {
    "pipewire": "pipewire",
    "pipewire-tools": "pipewire-bin",
    "wireplumber": "wireplumber",
    "libnotify": "libnotify-bin",
    "git": "git",
    "curl": "curl",
    "github-cli": "gh",
    "xdotool": "xdotool",
    "xclip": "xclip",
    "wl-clipboard": "wl-clipboard",
    "wtype": "wtype",
    "uv": None,
    "libsndfile": "libsndfile1",
    "ffmpeg": "ffmpeg",
    "libsecret": "libsecret-tools",
    "oo7": "gnome-keyring",
    "cuda": None,
    "llama.cpp-cuda": None,
}

_FEDORA_PACKAGES = {
    "pipewire": "pipewire",
    "pipewire-tools": "pipewire-utils",
    "wireplumber": "wireplumber",
    "libnotify": "libnotify",
    "git": "git",
    "curl": "curl",
    "github-cli": "gh",
    "xdotool": "xdotool",
    "xclip": "xclip",
    "wl-clipboard": "wl-clipboard",
    "wtype": "wtype",
    "uv": None,
    "libsndfile": "libsndfile",
    "ffmpeg": "ffmpeg",
    "libsecret": "libsecret",
    "oo7": "gnome-keyring",
    "cuda": None,
    "llama.cpp-cuda": None,
}

_PACKAGE_MAPS = {
    DistroFamily.ARCH: _ARCH_PACKAGES,
    DistroFamily.DEBIAN: _DEBIAN_PACKAGES,
    DistroFamily.FEDORA: _FEDORA_PACKAGES,
}

_INSTALL_COMMANDS = {
    DistroFamily.ARCH: ("paru", "-S", "--needed", "--noconfirm"),
    DistroFamily.DEBIAN: (
        "sudo",
        "apt-get",
        "install",
        "-y",
        "--no-install-recommends",
    ),
    DistroFamily.FEDORA: ("sudo", "dnf", "install", "-y"),
}

_PACKAGE_MANAGERS = {
    DistroFamily.ARCH: "paru",
    DistroFamily.DEBIAN: "apt-get",
    DistroFamily.FEDORA: "dnf",
}


def parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition("=")
        if not separator:
            continue
        values[name] = value.strip().strip('"')
    return values


def detect_distro_family(
    os_release: Mapping[str, str] | None = None,
) -> DistroFamily:
    values = dict(os_release or {})
    identity = " ".join(
        (
            values.get("ID", ""),
            values.get("ID_LIKE", ""),
            values.get("NAME", ""),
        )
    ).casefold()
    if any(
        token in identity for token in ("arch", "cachyos", "manjaro", "endeavouros")
    ):
        return DistroFamily.ARCH
    if any(
        token in identity
        for token in ("ubuntu", "debian", "pop", "linuxmint", "elementary")
    ):
        return DistroFamily.DEBIAN
    if any(token in identity for token in ("fedora", "rhel", "centos", "nobara")):
        return DistroFamily.FEDORA
    raise DistroError(
        "unsupported distribution; installation supports Arch, Ubuntu/Debian, "
        "and Fedora"
    )


def discover_install_paths(
    *,
    checkout: Path,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    chatterbox_dir: Path | None = None,
) -> InstallPaths:
    env = environment if environment is not None else os.environ
    resolved_home = home or Path(env.get("HOME") or Path.home())
    config_home = env.get("XDG_CONFIG_HOME")
    if config_home and Path(config_home).is_absolute():
        config_root = Path(config_home)
    else:
        config_root = resolved_home / ".config"
    chatterbox = chatterbox_dir or resolved_home / "chatterbox-audition"
    user_bin = resolved_home / ".local" / "bin"
    return InstallPaths(
        checkout=checkout.resolve(),
        home=resolved_home,
        user_bin=user_bin,
        voice_harness=user_bin / "voice-harness",
        systemd_user_dir=config_root / "systemd" / "user",
        chatterbox_dir=chatterbox,
        config_dir=config_root / "voice-harness",
    )


def _map_packages(
    family: DistroFamily, names: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapping = _PACKAGE_MAPS[family]
    mapped: list[str] = []
    skipped: list[str] = []
    for name in names:
        package = mapping.get(name, name if family == DistroFamily.ARCH else None)
        if package is None:
            skipped.append(name)
            continue
        if package not in mapped:
            mapped.append(package)
    return tuple(mapped), tuple(skipped)


def resolve_distro_plan(
    install_plan: InstallationPlan,
    *,
    checkout: Path,
    os_release: Mapping[str, str] | None = None,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    chatterbox_dir: Path | None = None,
) -> DistroPlan:
    family = detect_distro_family(os_release)
    packages, skipped_system = _map_packages(family, install_plan.system_packages)
    cuda_packages, skipped_cuda = _map_packages(family, install_plan.cuda_packages)
    skipped = tuple(dict.fromkeys((*skipped_system, *skipped_cuda)))
    required_unmapped = tuple(name for name in skipped if name != "uv")
    if required_unmapped:
        raise DistroError(
            f"{family.value} has no supported package adapter for required "
            f"packages: {', '.join(required_unmapped)}; use the supported Arch "
            "adapter or select an install profile this distro can satisfy"
        )
    return DistroPlan(
        family=family,
        distro_id=(os_release or {}).get("ID", family),
        package_manager=_PACKAGE_MANAGERS[family],
        install_command=_INSTALL_COMMANDS[family],
        packages=packages,
        cuda_packages=cuda_packages,
        skipped_packages=skipped,
        uv_bootstrap="uv" in skipped,
        paths=discover_install_paths(
            checkout=checkout,
            home=home,
            environment=environment,
            chatterbox_dir=chatterbox_dir,
        ),
    )


def plan_to_env(plan: DistroPlan) -> str:
    values: Mapping[str, object] = {
        "INSTALL_DISTRO_FAMILY": plan.family,
        "INSTALL_DISTRO_ID": plan.distro_id,
        "INSTALL_PACKAGE_MANAGER": plan.package_manager,
        "INSTALL_PACKAGE_COMMAND": " ".join(plan.install_command),
        "INSTALL_DISTRO_PACKAGES": " ".join(plan.packages),
        "INSTALL_DISTRO_CUDA_PACKAGES": " ".join(plan.cuda_packages),
        "INSTALL_SKIPPED_PACKAGES": " ".join(plan.skipped_packages),
        "INSTALL_UV_BOOTSTRAP": "1" if plan.uv_bootstrap else "0",
        "INSTALL_CHECKOUT": str(plan.paths.checkout),
        "INSTALL_USER_BIN": str(plan.paths.user_bin),
        "INSTALL_VOICE_HARNESS": str(plan.paths.voice_harness),
        "INSTALL_SYSTEMD_USER_DIR": str(plan.paths.systemd_user_dir),
        "INSTALL_CHATTERBOX_DIR": str(plan.paths.chatterbox_dir),
        "INSTALL_CONFIG_DIR": str(plan.paths.config_dir),
    }
    return "".join(f"{name}={_shell_value(value)}\n" for name, value in values.items())


def _shell_value(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _read_os_release(path: Path) -> dict[str, str]:
    try:
        return parse_os_release(path.read_text())
    except OSError as exc:
        raise DistroError(f"could not read {path}: {exc}") from exc


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve distro packages and install paths"
    )
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--os-release", default="/etc/os-release")
    parser.add_argument("--profile", default="")
    parser.add_argument("--llm", dest="llm_provider", default="")
    parser.add_argument("--tts", dest="tts_provider", default="")
    parser.add_argument("--llm-device", dest="llm_device", default="")
    parser.add_argument("--tts-device", dest="tts_device", default="")
    parser.add_argument("--dictation-device", dest="dictation_device", default="")
    parser.add_argument(
        "--cuda-available",
        action="store_true",
        help="a bounded runtime probe confirmed usable CUDA on this host",
    )
    parser.add_argument("--chatterbox-dir", default="")
    parser.add_argument("--format", choices=("env", "json"), default="env")
    options = parser.parse_args(arguments)
    try:
        install_plan = resolve_installation_plan(
            profile=options.profile or None,
            llm_provider=options.llm_provider or None,
            tts_provider=options.tts_provider or None,
            llm_device=options.llm_device or None,
            tts_device=options.tts_device or None,
            dictation_device=options.dictation_device or None,
            cuda_available=options.cuda_available,
        )
        distro_plan = resolve_distro_plan(
            install_plan,
            checkout=Path(options.checkout),
            os_release=_read_os_release(Path(options.os_release)),
            chatterbox_dir=(
                Path(options.chatterbox_dir) if options.chatterbox_dir else None
            ),
        )
    except (DistroError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if options.format == "json":
        payload = asdict(distro_plan)
        payload["paths"] = {key: str(value) for key, value in payload["paths"].items()}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(plan_to_env(distro_plan), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
