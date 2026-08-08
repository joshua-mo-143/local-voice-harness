from __future__ import annotations

import os
import pwd
import shlex
import site
import sys
from pathlib import Path

BACKEND_ENVIRONMENT_KEYS = frozenset(
    {
        "DICTATION_BACKEND",
        "DICTATION_MODEL",
        "DICTATION_LANGUAGE",
        "DICTATION_COMPUTE",
        "DICTATION_QUANTIZATION",
    }
)
PROTECTED_ENVIRONMENT_KEYS = frozenset(
    {
        "DICTATION_SOCKET",
        "CUDA_CACHE_PATH",
        "HF_HOME",
        "TMPDIR",
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
    }
)


def load_backend_environment(path: Path) -> dict[str, str]:
    """Parse only documented backend selectors from backend.env."""

    if not path.exists():
        return {}
    environment: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        fields = shlex.split(raw_line, comments=True, posix=True)
        if not fields:
            continue
        if len(fields) != 1 or "=" not in fields[0]:
            raise ValueError(f"{path}:{line_number}: invalid environment assignment")
        key, value = fields[0].split("=", 1)
        if key not in BACKEND_ENVIRONMENT_KEYS:
            raise ValueError(
                f"{path}:{line_number}: unsupported backend environment key {key!r}"
            )
        environment[key] = value
    return environment


def protected_environment(*, uid: int | None = None) -> dict[str, str]:
    """Return service-owned paths that backend configuration cannot replace."""

    resolved_uid = os.getuid() if uid is None else uid
    home = Path(pwd.getpwuid(resolved_uid).pw_dir)
    runtime = Path(f"/run/user/{resolved_uid}")
    return {
        "DICTATION_SOCKET": str(runtime / "dictation.sock"),
        "CUDA_CACHE_PATH": str(runtime / "dictation/cuda-cache"),
        "HF_HOME": str(home / ".cache/huggingface"),
        "TMPDIR": str(runtime / "dictation"),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_RUNTIME_DIR": str(runtime),
    }


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-dictation: ok")
        return
    environment = os.environ.copy()
    owned = protected_environment()
    backend_file = Path(owned["HOME"]) / ".config/dictation/backend.env"
    environment.update(load_backend_environment(backend_file))
    environment.update(owned)
    libraries = [
        str(path)
        for packages in map(Path, site.getsitepackages())
        for path in (packages / "nvidia").glob("*/lib")
        if path.is_dir()
    ]
    existing = environment.get("LD_LIBRARY_PATH", "")
    if libraries:
        environment["LD_LIBRARY_PATH"] = ":".join(
            [*libraries, *([existing] if existing else [])]
        )
    os.execve(
        sys.executable,
        [sys.executable, "-m", "local_voice_harness.stt.server"],
        environment,
    )


if __name__ == "__main__":
    main()
