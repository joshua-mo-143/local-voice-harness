from __future__ import annotations

import argparse
import importlib.resources
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import (
    PROJECT_ROOT,
    RUNTIME,
    SERVICE_FILES,
    STT_SOCKET,
    TTS_SOCKET,
)
from .user_config import UserConfig, load_user_config

SOURCE_RELATIVE = Path("systemd/user")
PACKAGED_RELATIVE = ("data", "systemd")
INSTALL_ROOT = "%h/local-voice-harness"
EXTERNAL_UNIT_STUBS = {
    "default.target",
    "graphical-session.target",
    "pipewire.service",
    "wireplumber.service",
}
COMMON_SECURITY_POLICY = {
    "Type": "simple",
    "UMask": "0077",
    "Restart": "on-failure",
    "RestartSec": "5",
    "LimitNOFILE": "4096",
    "LimitCORE": "0",
    "MemorySwapMax": "infinity",
    "OOMPolicy": "stop",
    "OOMScoreAdjust": "200",
    "NoNewPrivileges": "true",
    "CapabilityBoundingSet": "",
    "AmbientCapabilities": "",
    "ProtectSystem": "strict",
    "ProtectControlGroups": "true",
    "ProtectKernelModules": "true",
    "ProtectKernelTunables": "true",
    "ProtectKernelLogs": "true",
    "ProtectClock": "true",
    "ProtectHostname": "true",
    "ProtectProc": "invisible",
    "LockPersonality": "true",
    "RestrictNamespaces": "true",
    "RestrictSUIDSGID": "true",
    "SystemCallArchitectures": "native",
    "StartLimitIntervalSec": "60",
    "StartLimitBurst": "5",
    "RuntimeDirectoryMode": "0700",
}
SERVICE_SECURITY_POLICY = {
    "dictation.service": {
        "WorkingDirectory": INSTALL_ROOT,
        "TimeoutStopSec": "15",
        "PrivateTmp": "true",
        "ProtectHome": "read-only",
        "ReadWritePaths": "%t",
        "RestrictRealtime": "true",
        "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "RuntimeDirectory": "dictation",
        "RuntimeDirectoryPreserve": "yes",
        "CacheDirectory": "huggingface",
        "CacheDirectoryMode": "0700",
        "TasksMax": "512",
        "MemoryHigh": "4G",
        "MemoryMax": "6G",
    },
    "voice-harness-llm.service": {
        "WorkingDirectory": INSTALL_ROOT,
        "TimeoutStopSec": "15",
        "PrivateTmp": "true",
        "ProtectHome": "read-only",
        "RestrictRealtime": "true",
        "RestrictAddressFamilies": "AF_UNIX AF_INET AF_NETLINK",
        "RuntimeDirectory": "voice-harness-llm",
        "TasksMax": "256",
        "MemoryHigh": "6G",
        "MemoryMax": "8G",
    },
    "voice-harness-tts.service": {
        "WorkingDirectory": INSTALL_ROOT,
        "TimeoutStopSec": "15",
        "PrivateTmp": "true",
        "ProtectHome": "read-only",
        "ReadWritePaths": "%t",
        "RestrictRealtime": "true",
        "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "RuntimeDirectory": "voice-harness voice-harness-tts",
        "RuntimeDirectoryPreserve": "yes",
        "TasksMax": "512",
        "MemoryHigh": "8G",
        "MemoryMax": "10G",
    },
    "voice-harness-wake.service": {
        "WorkingDirectory": INSTALL_ROOT,
        "TimeoutStopSec": "10",
        "PrivateTmp": "false",
        "PrivateDevices": "true",
        "ProtectHome": "false",
        "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "RuntimeDirectory": "voice-harness",
        "RuntimeDirectoryPreserve": "yes",
        "StateDirectory": "voice-harness",
        "StateDirectoryMode": "0700",
        "TasksMax": "1024",
        "MemoryHigh": "3G",
        "MemoryMax": "4G",
    },
}
INTENTIONALLY_UNUSED_DIRECTIVES = {
    "DevicePolicy",
    "IPAddressDeny",
    "MemoryDenyWriteExecute",
    "SystemCallFilter",
}
GPU_SERVICES = {
    "dictation.service",
    "voice-harness-llm.service",
    "voice-harness-tts.service",
}
GPU_DEVICE_RESTRICTIONS = {
    "DeviceAllow",
    "DeviceDeny",
    "DevicePolicy",
    "PrivateDevices",
}
EXPECTED_EXECUTABLES = {
    "dictation.service": (
        "%h/local-voice-harness/.venv-dictation/bin/voice-harness-dictation"
    ),
    "voice-harness-llm.service": ("%h/local-voice-harness/.venv/bin/voice-harness-llm"),
    "voice-harness-tts.service": ("%h/chatterbox-audition/.venv/bin/voice-harness-tts"),
    "voice-harness-wake.service": (
        "%h/local-voice-harness/.venv/bin/voice-harness-wake"
    ),
}
EXPECTED_EXECSTART = {
    "dictation.service": EXPECTED_EXECUTABLES["dictation.service"],
    "voice-harness-llm.service": EXPECTED_EXECUTABLES["voice-harness-llm.service"],
    "voice-harness-tts.service": EXPECTED_EXECUTABLES["voice-harness-tts.service"],
    "voice-harness-wake.service": EXPECTED_EXECUTABLES["voice-harness-wake.service"],
}
REQUIRED_ENVIRONMENT = {
    "dictation.service": {
        "DICTATION_SOCKET": "%t/dictation.sock",
        "CUDA_CACHE_PATH": "%t/dictation/cuda-cache",
        "HF_HOME": "%h/.cache/huggingface",
        "TMPDIR": "%t/dictation",
    },
    "voice-harness-llm.service": {
        "CUDA_CACHE_PATH": "%t/voice-harness-llm/cuda-cache",
    },
    "voice-harness-tts.service": {
        "HF_HUB_OFFLINE": "1",
        "CUDA_CACHE_PATH": "%t/voice-harness-tts/cuda-cache",
    },
    "voice-harness-wake.service": {
        "PYTHONUNBUFFERED": "1",
    },
}
OPTIONAL_ENVIRONMENT_POLICY: dict[str, dict[str, str]] = {
    "dictation.service": {},
    "voice-harness-llm.service": {},
    "voice-harness-tts.service": {},
    "voice-harness-wake.service": {},
}
SERVICE_OWNED_ENVIRONMENT: dict[str, set[str]] = {
    "dictation.service": set(),
    "voice-harness-llm.service": set(),
    "voice-harness-tts.service": set(),
    "voice-harness-wake.service": {"STATE_DIRECTORY"},
}
CUDA_RUNTIME_DIRECTORIES = {
    "dictation.service": "dictation",
    "voice-harness-llm.service": "voice-harness-llm",
    "voice-harness-tts.service": "voice-harness-tts",
}
UNIT_DIRECTIVES = {"StartLimitIntervalSec", "StartLimitBurst"}
AUDIT_SHOW_PROPERTIES = (
    "ActiveState",
    "SubState",
    "Result",
    "NRestarts",
    "ExecMainStatus",
    "MemoryCurrent",
    "TasksCurrent",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
    "TasksMax",
    "LimitNOFILE",
    "LimitCORE",
    "UMask",
    "Restart",
    "RestartUSec",
    "TimeoutStopUSec",
    "WorkingDirectory",
    "Type",
    "PrivateTmp",
    "NoNewPrivileges",
    "ProtectSystem",
    "ProtectHome",
    "ProtectProc",
    "ProcSubset",
    "ProtectControlGroups",
    "ProtectKernelModules",
    "ProtectKernelTunables",
    "ProtectKernelLogs",
    "ProtectClock",
    "ProtectHostname",
    "PrivateDevices",
    "DevicePolicy",
    "CapabilityBoundingSet",
    "AmbientCapabilities",
    "LockPersonality",
    "MemoryDenyWriteExecute",
    "RestrictNamespaces",
    "RestrictSUIDSGID",
    "RestrictRealtime",
    "RestrictAddressFamilies",
    "SystemCallArchitectures",
    "RuntimeDirectory",
    "RuntimeDirectoryMode",
    "RuntimeDirectoryPreserve",
    "StateDirectory",
    "StateDirectoryMode",
    "CacheDirectory",
    "CacheDirectoryMode",
    "ReadWritePaths",
    "InaccessiblePaths",
    "OOMPolicy",
    "OOMScoreAdjust",
    "StartLimitIntervalUSec",
    "StartLimitBurst",
    "Environment",
    "EnvironmentFiles",
    "ExecStart",
)


@dataclass(frozen=True)
class StagedVerification:
    process: subprocess.CompletedProcess[str]
    context: str
    rooted_user_context: str
    reason: str | None = None

    @property
    def returncode(self) -> int:
        return self.process.returncode

    @property
    def stdout(self) -> str:
        return self.process.stdout

    @property
    def stderr(self) -> str:
        return self.process.stderr


def packaged_unit_text(name: str) -> str:
    resource = importlib.resources.files("local_voice_harness").joinpath(
        *PACKAGED_RELATIVE, name
    )
    return resource.read_text()


def _service_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.service")}


def parity_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Check that source and wheel unit inventories and contents are identical."""

    source_dir = project_root / SOURCE_RELATIVE
    expected = set(SERVICE_FILES)
    source_names = _service_names(source_dir)
    packaged_names = {
        entry.name
        for entry in importlib.resources.files("local_voice_harness")
        .joinpath(*PACKAGED_RELATIVE)
        .iterdir()
        if entry.name.endswith(".service")
    }
    errors: list[str] = []
    for label, actual in (("source", source_names), ("packaged", packaged_names)):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{label} units missing: {', '.join(missing)}")
        if extra:
            errors.append(f"{label} units unexpected: {', '.join(extra)}")
    for name in sorted(expected & source_names & packaged_names):
        source_text = (source_dir / name).read_text()
        packaged_text = packaged_unit_text(name)
        if source_text != packaged_text:
            errors.append(f"source and packaged units differ: {name}")
    return errors


def _directive_values(text: str, name: str, *, section: str | None = None) -> list[str]:
    """Collect assignments in file order, optionally within one unit section."""

    current_section: str | None = None
    values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue
        if section is not None and current_section != section:
            continue
        key, separator, value = line.partition("=")
        if separator and key == name:
            values.append(value.strip())
    return values


def _directive(text: str, name: str, *, section: str | None = None) -> str | None:
    """Return the effective value for a scalar systemd assignment."""

    values = _directive_values(text, name, section=section)
    return values[-1] if values else None


def _environment_values(text: str, variable: str) -> list[str]:
    """Collect every explicit assignment of one Environment= variable."""

    values: list[str] = []
    for directive in _directive_values(text, "Environment", section="Service"):
        if not directive:
            values.append("<reset>")
            continue
        for assignment in shlex.split(directive):
            key, separator, value = assignment.partition("=")
            if separator and key == variable:
                values.append(value)
    return values


def _environment_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for directive in _directive_values(text, "Environment", section="Service"):
        for assignment in shlex.split(directive):
            key, separator, _value = assignment.partition("=")
            if separator:
                keys.add(key)
    return keys


def _effective_list_values(text: str, name: str, *, section: str) -> list[str]:
    """Apply systemd's empty-assignment reset semantics for list directives."""

    effective: list[str] = []
    for value in _directive_values(text, name, section=section):
        if value:
            effective.append(value)
        else:
            effective.clear()
    return effective


def consistency_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Validate service paths and model defaults against package configuration/docs."""

    source_dir = project_root / SOURCE_RELATIVE
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    errors: list[str] = []

    llm = (source_dir / "voice-harness-llm.service").read_text()
    if "--model" in (_directive(llm, "ExecStart", section="Service") or ""):
        errors.append("LLM service must defer model and device choices to its launcher")

    dictation = (source_dir / "dictation.service").read_text()
    for variable in (
        "DICTATION_BACKEND",
        "DICTATION_DEVICE",
        "DICTATION_MODEL",
        "DICTATION_LANGUAGE",
        "DICTATION_COMPUTE",
        "DICTATION_QUANTIZATION",
    ):
        if _environment_values(dictation, variable):
            errors.append(
                f"dictation service must defer {variable} to the typed launcher"
            )
    if not any(
        str(item).startswith("onnx-asr") for item in extras.get("dictation", [])
    ):
        errors.append("dictation extra does not install the default parakeet backend")

    for name in (
        "dictation.service",
        "voice-harness-llm.service",
        "voice-harness-tts.service",
        "voice-harness-wake.service",
    ):
        text = (source_dir / name).read_text()
        working_directory = _directive(text, "WorkingDirectory", section="Service")
        if working_directory != INSTALL_ROOT:
            errors.append(f"{name} WorkingDirectory must be {INSTALL_ROOT}")
    if STT_SOCKET != RUNTIME / "dictation.sock":
        errors.append(
            "configured dictation socket must retain standalone compatibility"
        )
    if TTS_SOCKET != RUNTIME / "voice-harness-tts.sock":
        errors.append("configured TTS socket must retain client/server compatibility")
    return errors


def security_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Validate controls without ignoring later systemd assignments or resets."""

    source_dir = project_root / SOURCE_RELATIVE
    errors: list[str] = []
    for name in SERVICE_FILES:
        text = (source_dir / name).read_text()
        policy = {**COMMON_SECURITY_POLICY, **SERVICE_SECURITY_POLICY[name]}
        for directive, expected in policy.items():
            section = "Unit" if directive in UNIT_DIRECTIVES else "Service"
            values = _directive_values(text, directive, section=section)
            if len(values) != 1:
                errors.append(
                    f"{name} {directive} must have exactly one assignment, "
                    f"got {values!r}"
                )
                continue
            actual = values[0]
            if actual != expected:
                errors.append(
                    f"{name} {directive} must be {expected!r}, got {actual!r}"
                )
        for directive in INTENTIONALLY_UNUSED_DIRECTIVES:
            if _directive_values(text, directive, section="Service"):
                errors.append(
                    f"{name} must not set compatibility-sensitive {directive}"
                )
        for variable, expected in REQUIRED_ENVIRONMENT[name].items():
            values = _environment_values(text, variable)
            if values != [expected]:
                errors.append(
                    f"{name} Environment={variable} must be assigned exactly once "
                    f"to {expected!r}, got {values!r}"
                )
        environment_keys = _environment_keys(text)
        if environment_keys != set(REQUIRED_ENVIRONMENT[name]):
            errors.append(
                f"{name} Environment keys must be exactly "
                f"{sorted(REQUIRED_ENVIRONMENT[name])!r}, got "
                f"{sorted(environment_keys)!r}"
            )
        if _directive_values(text, "EnvironmentFile", section="Service"):
            errors.append(f"{name} must not import unrestricted EnvironmentFile values")
        if name in GPU_SERVICES:
            for directive in GPU_DEVICE_RESTRICTIONS:
                if _directive_values(text, directive, section="Service"):
                    errors.append(
                        f"{name} must not restrict CUDA devices with {directive}"
                    )
            runtime_directory = CUDA_RUNTIME_DIRECTORIES[name]
            cache_path = REQUIRED_ENVIRONMENT[name]["CUDA_CACHE_PATH"]
            if not cache_path.startswith(f"%t/{runtime_directory}/"):
                errors.append(
                    f"{name} CUDA_CACHE_PATH must be below RuntimeDirectory="
                    f"{runtime_directory}"
                )

    tts = (source_dir / "voice-harness-tts.service").read_text()
    if _environment_values(tts, "VOICE_HARNESS_TTS_SOCKET"):
        errors.append("voice-harness-tts.service must use the fixed compatible socket")
    errors.extend(executable_errors(project_root))
    return errors


def executable_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Require one exact ExecStart executable for every shipped service."""

    source_dir = project_root / SOURCE_RELATIVE
    errors: list[str] = []
    for name, expected in EXPECTED_EXECUTABLES.items():
        text = (source_dir / name).read_text()
        commands = _effective_list_values(text, "ExecStart", section="Service")
        if commands != [EXPECTED_EXECSTART[name]]:
            errors.append(
                f"{name} effective ExecStart must be "
                f"{EXPECTED_EXECSTART[name]!r}, got {commands!r}"
            )
            continue
        try:
            arguments = shlex.split(commands[0])
        except ValueError as exc:
            errors.append(f"{name} has invalid ExecStart quoting: {exc}")
            continue
        executable = arguments[0].lstrip("-+!:@") if arguments else ""
        if executable != expected:
            errors.append(
                f"{name} ExecStart executable must be {expected!r}, got {executable!r}"
            )
    return errors


def stage_executable_stub(root: Path, executable: str) -> Path:
    """Create an executable stub without permitting traversal outside ``root``."""

    resolved_root = root.resolve(strict=True)
    components = executable.split("/")
    if (
        not executable.startswith("/")
        or any(component in {"", ".", ".."} for component in components[1:])
        or Path(executable).as_posix() != executable
    ):
        raise ValueError(f"unsafe staged executable path: {executable!r}")

    destination = resolved_root.joinpath(*components[1:])
    current = resolved_root
    for component in components[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"staged executable path contains a symlink: {executable!r}"
            )

    resolved_destination = destination.resolve(strict=False)
    if not resolved_destination.is_relative_to(resolved_root):
        raise ValueError(f"staged executable escapes verification root: {executable!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"staged executable destination is a symlink: {executable!r}")
    destination.touch()
    destination.chmod(0o755)
    return destination


def systemd_analyze(
    paths: Iterable[Path], *, executable: str | None = None
) -> StagedVerification | None:
    """Verify units in an isolated root with deterministic dependency/command stubs."""

    command = executable or shutil.which("systemd-analyze")
    if command is None:
        return None
    unit_paths = tuple(paths)
    with tempfile.TemporaryDirectory(prefix="voice-harness-systemd-") as temporary:
        root = Path(temporary)
        unit_dir = root / "etc/systemd/user"
        unit_dir.mkdir(parents=True)
        system_unit_dir = root / "etc/systemd/system"
        system_unit_dir.mkdir(parents=True)
        staged_paths: list[Path] = []
        command_paths = {"/bin/true"}
        for source in unit_paths:
            text = source.read_text()
            staged = unit_dir / source.name
            staged_text = f"{text}\n[Unit]\nDefaultDependencies=no\n"
            staged.write_text(staged_text)
            (system_unit_dir / source.name).write_text(staged_text)
            staged_paths.append(staged)
            for match in re.finditer(
                r"^Exec(?:Condition|StartPre|Start|StartPost|Reload|Stop|StopPost)=(.*)$",
                text,
                re.MULTILINE,
            ):
                arguments = shlex.split(match.group(1))
                if not arguments:
                    continue
                configured = arguments[0].lstrip("-+!:@")
                expected = EXPECTED_EXECUTABLES.get(source.name)
                if configured == expected:
                    executable_path = configured.replace("%h", str(Path.home()), 1)
                    if Path(executable_path).is_absolute():
                        command_paths.add(executable_path)

        for name in EXTERNAL_UNIT_STUBS:
            if name.endswith(".service"):
                stub_text = (
                    "[Unit]\n"
                    f"Description=Verification stub for {name}\n"
                    "DefaultDependencies=no\n"
                    "[Service]\n"
                    "Type=oneshot\n"
                    "ExecStart=/bin/true\n"
                )
            else:
                stub_text = (
                    "[Unit]\n"
                    f"Description=Verification stub for {name}\n"
                    "DefaultDependencies=no\n"
                )
            for directory in (unit_dir, system_unit_dir):
                (directory / name).write_text(stub_text)
        for path in command_paths:
            stage_executable_stub(root, path)

        common_arguments = [
            f"--root={root}",
            "--man=no",
            "--generators=no",
            "verify",
            *(path.name for path in staged_paths),
        ]
        result = subprocess.run(
            [command, "--user", *common_arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode and re.search(
            r"Failed to initialize unit search paths.*Invalid argument",
            result.stderr,
        ):
            user_error = result.stderr.strip()
            fallback = subprocess.run(
                [command, *common_arguments],
                capture_output=True,
                text=True,
                check=False,
            )
            return StagedVerification(
                fallback,
                context="staged-system",
                rooted_user_context="unsupported",
                reason=user_error,
            )
        return StagedVerification(
            result,
            context="staged-user",
            rooted_user_context="supported",
        )


def _memory_bytes(value: str) -> str:
    match = re.fullmatch(r"(\d+)([KMGT])", value)
    if match is None:
        return value
    powers = {"K": 1, "M": 2, "G": 3, "T": 4}
    return str(int(match.group(1)) * 1024 ** powers[match.group(2)])


def _audit_expected_properties(name: str) -> dict[str, str]:
    policy = {**COMMON_SECURITY_POLICY, **SERVICE_SECURITY_POLICY[name]}
    protect_home = policy["ProtectHome"]
    service_policy = SERVICE_SECURITY_POLICY[name]

    def yes_no(directive: str, default: str = "false") -> str:
        return "yes" if policy.get(directive, default) == "true" else "no"

    expected = {
        "Type": policy["Type"],
        "MemoryHigh": _memory_bytes(policy["MemoryHigh"]),
        "MemoryMax": _memory_bytes(policy["MemoryMax"]),
        "MemorySwapMax": policy["MemorySwapMax"],
        "TasksMax": policy["TasksMax"],
        "LimitNOFILE": policy["LimitNOFILE"],
        "LimitCORE": policy["LimitCORE"],
        "UMask": policy["UMask"],
        "Restart": policy["Restart"],
        "RestartUSec": f"{policy['RestartSec']}s",
        "TimeoutStopUSec": f"{policy['TimeoutStopSec']}s",
        "OOMPolicy": policy["OOMPolicy"],
        "OOMScoreAdjust": policy["OOMScoreAdjust"],
        "PrivateTmp": yes_no("PrivateTmp"),
        "NoNewPrivileges": "yes",
        "ProtectSystem": policy["ProtectSystem"],
        "ProtectHome": "no" if protect_home == "false" else protect_home,
        "ProtectProc": policy["ProtectProc"],
        "ProcSubset": "all",
        "ProtectControlGroups": yes_no("ProtectControlGroups"),
        "ProtectKernelModules": yes_no("ProtectKernelModules"),
        "ProtectKernelTunables": yes_no("ProtectKernelTunables"),
        "ProtectKernelLogs": yes_no("ProtectKernelLogs"),
        "ProtectClock": yes_no("ProtectClock"),
        "ProtectHostname": yes_no("ProtectHostname"),
        "PrivateDevices": (
            "yes" if service_policy.get("PrivateDevices") == "true" else "no"
        ),
        "DevicePolicy": "auto",
        "CapabilityBoundingSet": "",
        "AmbientCapabilities": "",
        "LockPersonality": yes_no("LockPersonality"),
        "MemoryDenyWriteExecute": "no",
        "RestrictNamespaces": yes_no("RestrictNamespaces"),
        "RestrictSUIDSGID": yes_no("RestrictSUIDSGID"),
        "RestrictRealtime": yes_no("RestrictRealtime"),
        "RestrictAddressFamilies": policy["RestrictAddressFamilies"],
        "SystemCallArchitectures": policy["SystemCallArchitectures"],
        "RuntimeDirectory": policy["RuntimeDirectory"],
        "RuntimeDirectoryMode": policy["RuntimeDirectoryMode"],
        "RuntimeDirectoryPreserve": (
            "yes" if service_policy.get("RuntimeDirectoryPreserve") == "yes" else "no"
        ),
        "StateDirectory": service_policy.get("StateDirectory", ""),
        "StateDirectoryMode": service_policy.get("StateDirectoryMode", "0755"),
        "CacheDirectory": service_policy.get("CacheDirectory", ""),
        "CacheDirectoryMode": service_policy.get("CacheDirectoryMode", "0755"),
        "ReadWritePaths": _expanded_unit_value(
            service_policy.get("ReadWritePaths", "")
        ),
        "InaccessiblePaths": "",
        "EnvironmentFiles": "",
        "StartLimitIntervalUSec": "1min",
        "StartLimitBurst": policy["StartLimitBurst"],
    }
    expected["WorkingDirectory"] = _expanded_unit_value(
        policy.get("WorkingDirectory", "")
    )
    return expected


def _show_properties(text: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            properties[name] = value
    return properties


def _same_address_families(actual: str, expected: str) -> bool:
    return set(actual.split()) == set(expected.split())


def _show_exec_argv(value: str) -> list[str] | None:
    match = re.search(r"(?:^|;\s*)argv\[\]=(.*?)(?:\s*;\s*ignore_errors=|$)", value)
    if match is None:
        return None
    return shlex.split(match.group(1))


def _expanded_unit_value(value: str) -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return value.replace("%h", str(Path.home())).replace("%t", runtime)


def _optional_environment_errors(name: str, environment: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for variable, validator in OPTIONAL_ENVIRONMENT_POLICY[name].items():
        if variable not in environment:
            continue
        value = environment[variable]
        problem: str | None = None
        if "\0" in value or "\n" in value or "\r" in value:
            problem = "must not contain control characters"
        elif validator == "nonempty" and not value.strip():
            problem = "must not be empty"
        elif (
            validator == "absolute_path"
            and not Path(_expanded_unit_value(value)).is_absolute()
        ):
            problem = "must be an absolute path"
        elif validator in {"nonnegative_float", "positive_float", "probability"}:
            try:
                number = float(value)
            except ValueError:
                problem = "must be a finite number"
            else:
                if not math.isfinite(number) or number < 0:
                    problem = "must be a finite non-negative number"
                elif validator == "positive_float" and number <= 0:
                    problem = "must be a finite positive number"
                elif validator == "probability" and number > 1:
                    problem = "must be between 0 and 1"
        elif validator == "positive_int":
            try:
                valid = int(value) >= 1
            except ValueError:
                valid = False
            if not valid:
                problem = "must be an integer of at least 1"
        elif (
            validator == "duration"
            and re.fullmatch(r"(?:0|[1-9]\d*)(?:\.\d+)?(?:us|ms|s)", value) is None
        ):
            problem = "must be a non-negative duration ending in us, ms, or s"
        elif validator == "flag" and value.strip().casefold() not in {
            "0",
            "1",
            "true",
            "false",
            "yes",
            "no",
            "on",
            "off",
        }:
            problem = "must be a boolean flag (0/1, true/false, yes/no, on/off)"
        elif validator == "barge_in_mode" and value not in {"wake", "vad", "off"}:
            problem = "must be wake, vad, or off"
        elif validator == "dictation_inject" and value not in {
            "auto",
            "paste",
            "type",
            "stdout",
        }:
            problem = "must be auto, paste, type, or stdout"
        if problem is not None:
            errors.append(f"{name}: optional environment {variable} {problem}")

    project_root = Path(
        _expanded_unit_value(
            environment.get("VOICE_HARNESS_PROJECT_ROOT", str(Path.home()))
        )
    ).resolve()
    github_root = Path(
        _expanded_unit_value(
            environment.get("VOICE_HARNESS_GITHUB_ROOT", str(Path.home() / "src"))
        )
    ).resolve()
    if (
        (
            "VOICE_HARNESS_PROJECT_ROOT" in environment
            or "VOICE_HARNESS_GITHUB_ROOT" in environment
        )
        and github_root.is_absolute()
        and project_root.is_absolute()
    ):
        try:
            github_root.relative_to(project_root)
        except ValueError:
            errors.append(
                f"{name}: VOICE_HARNESS_GITHUB_ROOT must be inside "
                "VOICE_HARNESS_PROJECT_ROOT"
            )
    return errors


def audit_installed(
    service_names: Iterable[str] = SERVICE_FILES,
    *,
    systemctl: str = "systemctl",
    config: UserConfig | None = None,
) -> int:
    """Read effective installed units and state without changing the user manager."""

    resolved_config = config if config is not None else load_user_config()
    errors: list[str] = []
    for name in service_names:
        cat = subprocess.run(
            [systemctl, "--user", "cat", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if cat.returncode:
            errors.append(f"{name}: systemctl cat failed: {cat.stderr.strip()}")
            continue
        commands = _effective_list_values(cat.stdout, "ExecStart", section="Service")
        if commands != [EXPECTED_EXECSTART[name]]:
            errors.append(
                f"{name}: effective ExecStart must be "
                f"{EXPECTED_EXECSTART[name]!r}, got {commands!r}"
            )
        if _effective_list_values(cat.stdout, "EnvironmentFile", section="Service"):
            errors.append(
                f"{name}: effective EnvironmentFile must be empty; "
                "shipped services do not allow environment files"
            )
        for directive in {"IPAddressDeny", "SystemCallFilter"}:
            if _effective_list_values(cat.stdout, directive, section="Service"):
                errors.append(f"{name}: effective {directive} must be empty")
        if name in GPU_SERVICES:
            for directive in {"DeviceAllow", "DeviceDeny"}:
                if _effective_list_values(cat.stdout, directive, section="Service"):
                    errors.append(
                        f"{name}: effective {directive} must not restrict CUDA"
                    )

        show_command = [systemctl, "--user", "show", name]
        for prop in AUDIT_SHOW_PROPERTIES:
            show_command.append(f"--property={prop}")
        shown = subprocess.run(
            show_command,
            capture_output=True,
            text=True,
            check=False,
        )
        if shown.returncode:
            errors.append(f"{name}: systemctl show failed: {shown.stderr.strip()}")
            continue
        properties = _show_properties(shown.stdout)
        for prop, expected in _audit_expected_properties(name).items():
            actual = properties.get(prop)
            matches = (
                _same_address_families(actual or "", expected)
                if prop
                in {
                    "RestrictAddressFamilies",
                    "RuntimeDirectory",
                    "ReadWritePaths",
                    "InaccessiblePaths",
                    "CapabilityBoundingSet",
                    "AmbientCapabilities",
                }
                else actual == expected
            )
            if not matches:
                errors.append(
                    f"{name}: effective {prop} must be {expected!r}, got {actual!r}"
                )
        expected_argv = shlex.split(_expanded_unit_value(EXPECTED_EXECSTART[name]))
        actual_argv = _show_exec_argv(properties.get("ExecStart", ""))
        if actual_argv != expected_argv:
            errors.append(
                f"{name}: effective ExecStart argv must be {expected_argv!r}, "
                f"got {actual_argv!r}"
            )
        environment = properties.get("Environment", "")
        assignments = shlex.split(environment)
        actual_environment = {
            key: value
            for assignment in assignments
            for key, separator, value in [assignment.partition("=")]
            if separator
        }
        allowed_environment = (
            set(REQUIRED_ENVIRONMENT[name])
            | set(OPTIONAL_ENVIRONMENT_POLICY[name])
            | SERVICE_OWNED_ENVIRONMENT[name]
        )
        unknown_environment = set(actual_environment) - allowed_environment
        missing_environment = set(REQUIRED_ENVIRONMENT[name]) - set(actual_environment)
        if unknown_environment:
            errors.append(
                f"{name}: effective Environment has unsupported keys "
                f"{sorted(unknown_environment)!r}"
            )
        if missing_environment:
            errors.append(
                f"{name}: effective Environment lacks required keys "
                f"{sorted(missing_environment)!r}"
            )
        for variable, expected in REQUIRED_ENVIRONMENT[name].items():
            if actual_environment.get(variable) not in {
                expected,
                _expanded_unit_value(expected),
            }:
                errors.append(
                    f"{name}: effective environment lacks {variable}={expected}"
                )
        if state_directory := actual_environment.get("STATE_DIRECTORY"):
            expected_state = (
                Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
                / "voice-harness"
            )
            if state_directory != str(expected_state):
                errors.append(
                    f"{name}: STATE_DIRECTORY is service-owned and must be "
                    f"{str(expected_state)!r}, got {state_directory!r}"
                )
        errors.extend(_optional_environment_errors(name, actual_environment))
        active = properties.get("ActiveState", "unknown")
        result = properties.get("Result", "unknown")
        restarts = properties.get("NRestarts", "unknown")
        if active == "failed" or result not in {"success", ""}:
            errors.append(
                f"{name}: unhealthy state ActiveState={active} Result={result}"
            )
        try:
            restart_count = int(restarts)
            if restart_count >= int(
                COMMON_SECURITY_POLICY["StartLimitBurst"]
            ) and active in {"activating", "failed"}:
                errors.append(
                    f"{name}: restart count {restarts} indicates an active crash loop"
                )
        except ValueError:
            errors.append(f"{name}: invalid NRestarts={restarts!r}")
        print(
            f"{name}: active={active} sub={properties.get('SubState', 'unknown')} "
            f"result={result} restarts={restarts} "
            f"memory={properties.get('MemoryCurrent', 'unknown')} "
            f"tasks={properties.get('TasksCurrent', 'unknown')}"
        )

    print(
        "configuration: "
        f"llm={resolved_config.providers.llm_provider} "
        f"tts={resolved_config.providers.tts_provider} "
        f"cuda={resolved_config.compute.cuda_device}"
    )
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate voice harness systemd units")
    parser.add_argument(
        "--require-systemd-analyze",
        action="store_true",
        help="fail instead of skipping when systemd-analyze is unavailable",
    )
    parser.add_argument(
        "--audit-installed",
        action="store_true",
        help="read and validate effective installed user units without changing them",
    )
    parser.add_argument(
        "--require-user-context",
        action="store_true",
        help="fail unless rooted staged verification ran in systemd user context",
    )
    options = parser.parse_args(arguments)

    if options.audit_installed:
        return audit_installed()

    errors = [*parity_errors(), *consistency_errors(), *security_errors()]
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    paths = [PROJECT_ROOT / SOURCE_RELATIVE / name for name in SERVICE_FILES]
    analyzed = systemd_analyze(paths)
    if analyzed is None:
        if options.require_systemd_analyze or options.require_user_context:
            print("error: systemd-analyze is unavailable", file=sys.stderr)
            return 1
        print("systemd-analyze unavailable; syntax verification skipped")
        return 0
    print(
        json.dumps(
            {
                "systemd_verification": {
                    "context": analyzed.context,
                    "rooted_user_context": analyzed.rooted_user_context,
                    "reason": analyzed.reason,
                }
            },
            sort_keys=True,
        )
    )
    if options.require_user_context and analyzed.rooted_user_context != "supported":
        print(
            "error: rooted staged systemd user-context verification is unsupported",
            file=sys.stderr,
        )
        return 1
    if analyzed.returncode:
        sys.stderr.write(analyzed.stdout)
        sys.stderr.write(analyzed.stderr)
        return analyzed.returncode
    print(
        f"validated {len(paths)} source/packaged systemd units "
        f"in {analyzed.context} context"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
