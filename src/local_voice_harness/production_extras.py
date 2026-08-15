"""Install declared production extras into clean environments and smoke-check them."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT


@dataclass(frozen=True)
class ExtraSpec:
    """How to prove one optional-dependency extra installed and is callable."""

    imports: tuple[str, ...]
    commands: tuple[tuple[str, tuple[str, ...]], ...]
    required_distributions: tuple[str, ...] = ()


CORE_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("voice-harness", ("--help",)),
    ("voice-harness-dictate", ("--help",)),
    ("voice-harness-llm", ("--check",)),
    ("voice-harness-cursor-worker", ("--help",)),
)

EXTRA_SPECS: dict[str, ExtraSpec] = {
    "wake": ExtraSpec(
        imports=("openwakeword", "numpy"),
        commands=(("voice-harness-wake", ("--check",)),),
    ),
    "dictation": ExtraSpec(
        imports=("onnx_asr", "onnxruntime"),
        commands=(("voice-harness-dictation", ("--check",)),),
    ),
    "dictation-cuda": ExtraSpec(
        imports=("onnx_asr",),
        commands=(("voice-harness-dictation", ("--check",)),),
        required_distributions=("onnxruntime-gpu",),
    ),
    "dictation-whisper": ExtraSpec(
        imports=("faster_whisper",),
        commands=(("voice-harness-dictation", ("--check",)),),
    ),
    "tts": ExtraSpec(
        imports=("chatterbox", "soundfile"),
        commands=(("voice-harness-tts", ("--check",)),),
    ),
}


def declared_extras(pyproject: Path) -> tuple[str, ...]:
    metadata = tomllib.loads(pyproject.read_text())
    extras = metadata["project"]["optional-dependencies"]
    if not isinstance(extras, dict):
        raise ValueError("project.optional-dependencies must be a table")
    return tuple(sorted(str(name) for name in extras))


def missing_extra_specs(pyproject: Path) -> tuple[str, ...]:
    return tuple(
        extra for extra in declared_extras(pyproject) if extra not in EXTRA_SPECS
    )


def export_command(extra: str | None) -> list[str]:
    command = [
        "uv",
        "export",
        "--locked",
        "--no-dev",
        "--no-emit-project",
    ]
    if extra:
        command.extend(("--extra", extra))
    return command


def build_wheel_command(destination: Path) -> list[str]:
    return ["uv", "build", "--wheel", "--out-dir", str(destination)]


def venv_command(python: str, destination: Path) -> list[str]:
    return ["uv", "venv", "--python", python, str(destination)]


def install_commands(
    python_bin: Path, wheel: Path, requirements: Path | None
) -> list[list[str]]:
    commands = [["uv", "pip", "install", "--python", str(python_bin), str(wheel)]]
    if requirements is not None:
        commands.append(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_bin),
                "-r",
                str(requirements),
            ]
        )
    return commands


def import_command(python_bin: Path, modules: Sequence[str]) -> list[str]:
    statement = "; ".join(f"import {module}" for module in modules)
    return [str(python_bin), "-c", statement]


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"error: command failed ({result.returncode}): {' '.join(command)}"
        )
    return result


def _require_uv() -> None:
    if shutil.which("uv") is None:
        raise SystemExit("error: uv is required to check production extras")


def _wheel_path(directory: Path) -> Path:
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"error: expected one wheel in {directory}, found {len(wheels)}"
        )
    return wheels[0]


def _export_requirements(project: Path, extra: str | None, destination: Path) -> Path:
    path = destination / ("core.txt" if extra is None else f"{extra}.txt")
    _run([*export_command(extra), "-o", str(path)], cwd=project)
    return path


def _install_environment(
    *,
    python: str,
    wheel: Path,
    requirements: Path,
    destination: Path,
    project: Path,
) -> Path:
    _run(venv_command(python, destination), cwd=project)
    python_bin = destination / "bin" / "python"
    for command in install_commands(python_bin, wheel, requirements):
        _run(command, cwd=project)
    return python_bin


def _check_commands(
    venv: Path, commands: Sequence[tuple[str, tuple[str, ...]]]
) -> None:
    for name, arguments in commands:
        executable = venv / "bin" / name
        if not executable.is_file():
            raise SystemExit(f"error: entry point {name} is missing from {venv}")
        _run([str(executable), *arguments])


def show_distribution_command(python_bin: Path, name: str) -> list[str]:
    return ["uv", "pip", "show", "--python", str(python_bin), name]


def _check_distributions(python_bin: Path, names: Sequence[str]) -> None:
    for name in names:
        _run(show_distribution_command(python_bin, name))


def check_environments(
    *,
    project: Path,
    python: str,
    extras: Sequence[str],
    work: Path,
) -> None:
    """Build the wheel, then install core and each extra into a clean venv."""

    _require_uv()
    missing = tuple(extra for extra in extras if extra not in EXTRA_SPECS)
    if missing:
        raise SystemExit(f"error: no extra spec for {', '.join(missing)}")
    undeclared = missing_extra_specs(project / "pyproject.toml")
    if undeclared:
        raise SystemExit(
            "error: production extras are missing check specs: " + ", ".join(undeclared)
        )

    dist = work / "dist"
    dist.mkdir()
    _run(build_wheel_command(dist), cwd=project)
    wheel = _wheel_path(dist)

    core = work / "core"
    core_python = _install_environment(
        python=python,
        wheel=wheel,
        requirements=_export_requirements(project, None, work),
        destination=core,
        project=project,
    )
    _run(import_command(core_python, ("local_voice_harness",)))
    _check_commands(core, CORE_COMMANDS)
    print("core wheel entry points ok")

    for extra in extras:
        spec = EXTRA_SPECS[extra]
        environment = work / extra
        python_bin = _install_environment(
            python=python,
            wheel=wheel,
            requirements=_export_requirements(project, extra, work),
            destination=environment,
            project=project,
        )
        _run(import_command(python_bin, spec.imports))
        _check_distributions(python_bin, spec.required_distributions)
        _check_commands(environment, spec.commands)
        print(f"{extra} extra entry points ok")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install production extras into clean environments and smoke-check them"
    )
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--python", default=sys.version.split()[0])
    parser.add_argument("--extra", action="append", dest="extras")
    options = parser.parse_args(arguments)
    extras = (
        tuple(options.extras)
        if options.extras
        else declared_extras(options.project / "pyproject.toml")
    )
    with tempfile.TemporaryDirectory(prefix="voice-harness-extras-") as temporary:
        check_environments(
            project=options.project,
            python=options.python,
            extras=extras,
            work=Path(temporary),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
