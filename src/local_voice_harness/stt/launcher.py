from __future__ import annotations

import os
import pwd
import site
import sys
from pathlib import Path

from ..user_config import load_backend_environment as load_backend_environment
from ..user_config import load_user_config

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
LEGACY_UNIT_DEFAULTS = {
    "DICTATION_BACKEND": "parakeet",
    "DICTATION_MODEL": "nemo-parakeet-tdt-0.6b-v2",
    "DICTATION_QUANTIZATION": "int8",
}


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


def resolver_environment(environment: dict[str, str]) -> dict[str, str]:
    """Demote selector defaults embedded in previously installed systemd units."""

    resolved = environment.copy()
    if resolved.get("INVOCATION_ID"):
        for key, default in LEGACY_UNIT_DEFAULTS.items():
            if resolved.get(key) == default:
                resolved.pop(key)
    return resolved


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-dictation: ok")
        return
    environment = os.environ.copy()
    owned = protected_environment()
    snapshot = load_user_config(
        resolver_environment(environment), home=Path(owned["HOME"])
    )
    compute = snapshot.compute
    dictation = snapshot.dictation
    environment.update(
        {
            "DICTATION_BACKEND": compute.dictation_backend,
            "DICTATION_MODEL": compute.dictation_model,
            "DICTATION_LANGUAGE": compute.dictation_language,
            "DICTATION_COMPUTE": compute.dictation_compute,
            "DICTATION_QUANTIZATION": compute.dictation_quantization,
            "DICTATION_PROMPT": dictation.prompt,
            "DICTATION_REPLACEMENTS": ";".join(
                f"{source}:{target}" for source, target in dictation.replacements
            ),
        }
    )
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
