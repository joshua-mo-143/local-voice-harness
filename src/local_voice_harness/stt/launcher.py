from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-dictation: ok")
        return
    environment = os.environ.copy()
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
