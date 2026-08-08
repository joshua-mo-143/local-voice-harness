from __future__ import annotations

import argparse
import importlib.resources
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable
from pathlib import Path

from .config import DEFAULT_LLM_MODEL, PROJECT_ROOT, SERVICE_FILES

SOURCE_RELATIVE = Path("systemd/user")
PACKAGED_RELATIVE = ("data", "systemd")
INSTALL_ROOT = "%h/local-voice-harness"
EXTERNAL_UNIT_STUBS = {
    "default.target",
    "graphical-session.target",
    "pipewire.service",
    "wireplumber.service",
}


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


def _directive(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def consistency_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Validate service paths and model defaults against package configuration/docs."""

    source_dir = project_root / SOURCE_RELATIVE
    readme = (project_root / "README.md").read_text()
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    errors: list[str] = []

    llm = (source_dir / "voice-harness-llm.service").read_text()
    llm_command = _directive(llm, "ExecStart") or ""
    model_match = re.search(r"(?:^|\s)--model\s+(\S+)", llm_command)
    alias_match = re.search(r"(?:^|\s)--alias\s+(\S+)", llm_command)
    if model_match is None:
        errors.append("voice-harness-llm.service has no --model path")
    else:
        model_path = model_match.group(1)
        expected_prefix = f"{INSTALL_ROOT}/models/"
        if not model_path.startswith(expected_prefix):
            errors.append(f"LLM model path must be below {expected_prefix}")
        documented_path = model_path.replace("%h", "~", 1)
        if documented_path not in readme:
            errors.append(f"README does not document LLM model path {documented_path}")
    if alias_match is None or alias_match.group(1) != DEFAULT_LLM_MODEL:
        errors.append(
            f"LLM service alias must match configured default {DEFAULT_LLM_MODEL}"
        )

    dictation = (source_dir / "dictation.service").read_text()
    backend = _directive(dictation, "Environment")
    model = re.search(r"^Environment=DICTATION_MODEL=(\S+)$", dictation, re.MULTILINE)
    if backend != "DICTATION_BACKEND=parakeet":
        errors.append("dictation service default backend must be parakeet")
    if model is None:
        errors.append("dictation service has no default model")
    elif model.group(1) not in readme:
        errors.append(
            f"README does not document dictation model default {model.group(1)}"
        )
    if not any(
        str(item).startswith("onnx-asr") for item in extras.get("dictation", [])
    ):
        errors.append("dictation extra does not install the default parakeet backend")

    for name in (
        "dictation.service",
        "voice-harness-tts.service",
        "voice-harness-wake.service",
    ):
        text = (source_dir / name).read_text()
        working_directory = _directive(text, "WorkingDirectory")
        if working_directory != INSTALL_ROOT:
            errors.append(f"{name} WorkingDirectory must be {INSTALL_ROOT}")
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
) -> subprocess.CompletedProcess[str] | None:
    """Verify units in an isolated root with deterministic dependency/command stubs."""

    command = executable or shutil.which("systemd-analyze")
    if command is None:
        return None
    unit_paths = tuple(paths)
    with tempfile.TemporaryDirectory(prefix="voice-harness-systemd-") as temporary:
        root = Path(temporary)
        unit_dir = root / "etc/systemd/system"
        unit_dir.mkdir(parents=True)
        staged_paths: list[Path] = []
        command_paths = {"/bin/true"}
        for source in unit_paths:
            text = source.read_text()
            staged = unit_dir / source.name
            staged.write_text(f"{text}\n[Unit]\nDefaultDependencies=no\n")
            staged_paths.append(staged)
            for match in re.finditer(
                r"^Exec(?:Condition|StartPre|Start|StartPost|Reload|Stop|StopPost)=(.*)$",
                text,
                re.MULTILINE,
            ):
                arguments = shlex.split(match.group(1))
                if not arguments:
                    continue
                executable_path = arguments[0].lstrip("-+!:@")
                executable_path = executable_path.replace("%h", str(Path.home()), 1)
                candidate = Path(executable_path)
                if candidate.is_absolute():
                    command_paths.add(executable_path)

        for name in EXTERNAL_UNIT_STUBS:
            stub = unit_dir / name
            if name.endswith(".service"):
                stub.write_text(
                    "[Unit]\n"
                    f"Description=Verification stub for {name}\n"
                    "DefaultDependencies=no\n"
                    "[Service]\n"
                    "Type=oneshot\n"
                    "ExecStart=/bin/true\n"
                )
            else:
                stub.write_text(
                    "[Unit]\n"
                    f"Description=Verification stub for {name}\n"
                    "DefaultDependencies=no\n"
                )
        for path in command_paths:
            stage_executable_stub(root, path)

        return subprocess.run(
            [
                command,
                f"--root={root}",
                "--man=no",
                "--generators=no",
                "verify",
                *(path.name for path in staged_paths),
            ],
            capture_output=True,
            text=True,
            check=False,
        )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate voice harness systemd units")
    parser.add_argument(
        "--require-systemd-analyze",
        action="store_true",
        help="fail instead of skipping when systemd-analyze is unavailable",
    )
    options = parser.parse_args(arguments)

    errors = [*parity_errors(), *consistency_errors()]
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    paths = [PROJECT_ROOT / SOURCE_RELATIVE / name for name in SERVICE_FILES]
    analyzed = systemd_analyze(paths)
    if analyzed is None:
        if options.require_systemd_analyze:
            print("error: systemd-analyze is unavailable", file=sys.stderr)
            return 1
        print("systemd-analyze unavailable; syntax verification skipped")
        return 0
    if analyzed.returncode:
        sys.stderr.write(analyzed.stdout)
        sys.stderr.write(analyzed.stderr)
        return analyzed.returncode
    print(f"validated {len(paths)} source/packaged systemd units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
