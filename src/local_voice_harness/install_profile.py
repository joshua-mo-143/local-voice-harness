"""Testable installation profile decisions.

The one-shot installer asks this module which packages, extras, models, and
local services a selected provider/compute profile requires. The module uses
only the standard library so ``scripts/install.sh`` can resolve a plan before
the project environment exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

PROVIDERS = frozenset({"local", "venice"})
PROFILES = frozenset({"showcase", "local-cuda"})
NVIDIA_PACKAGE_MARKERS = ("cuda", "nvidia-", "llama.cpp-cuda")
BASE_PACKAGES = (
    "pipewire",
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
)
VENICE_PACKAGES = ("libsecret", "oo7")
CUDA_PACKAGES = ("cuda", "llama.cpp-cuda")
LOCAL_LLM_SERVICES = ("voice-harness-llm.service",)
LOCAL_TTS_EXTRAS = ("tts",)


class InstallProfile(StrEnum):
    """Named installation shapes the installer can apply without GPU hardware."""

    SHOWCASE = "showcase"
    LOCAL_CUDA = "local-cuda"


class InstallProfileError(ValueError):
    """A profile or provider selector is missing or invalid."""


@dataclass(frozen=True)
class InstallationPlan:
    """Concrete packages, extras, models, and services for one install."""

    profile: str
    llm_provider: str
    tts_provider: str
    dictation_extra: str
    dictation_device: str
    system_packages: tuple[str, ...]
    cuda_packages: tuple[str, ...]
    venice_packages: tuple[str, ...]
    python_extras: tuple[str, ...]
    download_qwen: bool
    download_chatterbox: bool
    install_llm_service: bool
    install_tts_extra: bool

    def nvidia_packages(self) -> tuple[str, ...]:
        return tuple(
            package
            for package in (*self.system_packages, *self.cuda_packages)
            if any(marker in package.casefold() for marker in NVIDIA_PACKAGE_MARKERS)
        )


def _provider(value: str | None, *, label: str) -> str:
    selected = (value or "").strip().casefold()
    if selected not in PROVIDERS:
        raise InstallProfileError(f"{label} provider must be local or venice")
    return selected


def _profile(value: str | None) -> str | None:
    if value is None:
        return None
    selected = value.strip().casefold()
    if not selected:
        return None
    if selected not in PROFILES:
        raise InstallProfileError(
            f"installation profile must be one of: {', '.join(sorted(PROFILES))}"
        )
    return selected


def resolve_installation_plan(
    *,
    profile: str | None = None,
    llm_provider: str | None = None,
    tts_provider: str | None = None,
) -> InstallationPlan:
    """Return the packages and local assets required by one install choice."""

    selected_profile = _profile(profile)
    if selected_profile == InstallProfile.SHOWCASE:
        llm = "venice"
        tts = "venice"
        dictation_extra = "dictation"
        dictation_device = "cpu"
        cuda_packages: tuple[str, ...] = ()
    else:
        llm = _provider(llm_provider or "local", label="LLM")
        tts = _provider(tts_provider or "local", label="TTS")
        local_compute = llm == "local" or tts == "local"
        if selected_profile == InstallProfile.LOCAL_CUDA or local_compute:
            dictation_extra = "dictation-cuda"
            dictation_device = "cuda"
            cuda_packages = CUDA_PACKAGES
        else:
            dictation_extra = "dictation"
            dictation_device = "cpu"
            cuda_packages = ()
        if selected_profile is None:
            selected_profile = (
                InstallProfile.SHOWCASE
                if not local_compute
                else InstallProfile.LOCAL_CUDA
            )

    venice_packages = VENICE_PACKAGES if "venice" in {llm, tts} else ()
    install_tts_extra = tts == "local"
    python_extras = ("wake", dictation_extra)
    if install_tts_extra:
        python_extras += LOCAL_TTS_EXTRAS
    return InstallationPlan(
        profile=selected_profile,
        llm_provider=llm,
        tts_provider=tts,
        dictation_extra=dictation_extra,
        dictation_device=dictation_device,
        system_packages=BASE_PACKAGES + venice_packages,
        cuda_packages=cuda_packages,
        venice_packages=venice_packages,
        python_extras=python_extras,
        download_qwen=llm == "local",
        download_chatterbox=install_tts_extra,
        install_llm_service=llm == "local",
        install_tts_extra=install_tts_extra,
    )


def plan_to_env(plan: InstallationPlan) -> str:
    """Render shell assignments for ``scripts/install.sh``."""

    values: Mapping[str, object] = {
        "INSTALL_PROFILE": plan.profile,
        "INSTALL_LLM_PROVIDER": plan.llm_provider,
        "INSTALL_TTS_PROVIDER": plan.tts_provider,
        "INSTALL_DICTATION_EXTRA": plan.dictation_extra,
        "INSTALL_DICTATION_DEVICE": plan.dictation_device,
        "INSTALL_SYSTEM_PACKAGES": " ".join(plan.system_packages),
        "INSTALL_CUDA_PACKAGES": " ".join(plan.cuda_packages),
        "INSTALL_VENICE_PACKAGES": " ".join(plan.venice_packages),
        "INSTALL_PYTHON_EXTRAS": " ".join(plan.python_extras),
        "INSTALL_DOWNLOAD_QWEN": "1" if plan.download_qwen else "0",
        "INSTALL_DOWNLOAD_CHATTERBOX": "1" if plan.download_chatterbox else "0",
        "INSTALL_LLM_SERVICE": "1" if plan.install_llm_service else "0",
        "INSTALL_TTS_EXTRA": "1" if plan.install_tts_extra else "0",
    }
    return "".join(f"{name}={_shell_value(value)}\n" for name, value in values.items())


def _shell_value(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve installation packages and extras for a provider profile"
    )
    parser.add_argument("--profile", default="")
    parser.add_argument("--llm", dest="llm_provider", default="")
    parser.add_argument("--tts", dest="tts_provider", default="")
    parser.add_argument(
        "--format",
        choices=("env", "json"),
        default="env",
        help="env assignments for the installer, or JSON for tests",
    )
    options = parser.parse_args(arguments)
    try:
        plan = resolve_installation_plan(
            profile=options.profile or None,
            llm_provider=options.llm_provider or None,
            tts_provider=options.tts_provider or None,
        )
    except InstallProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if options.format == "json":
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    else:
        print(plan_to_env(plan), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
