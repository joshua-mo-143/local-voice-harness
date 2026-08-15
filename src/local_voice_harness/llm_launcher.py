from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from .config import PROJECT_ROOT
from .user_config import (
    ComputeDevice,
    ComputeSettings,
    UserConfig,
    load_user_config,
    resolve_local_compute,
)

LLAMA_SERVER = Path("/usr/sbin/llama-server")
MODEL_FILE = PROJECT_ROOT / "models" / "Qwen3.5-4B-Q4_K_M.gguf"


def llama_cuda_available(
    cuda_device: str, *, llama_server: Path = LLAMA_SERVER
) -> bool:
    """Return whether llama.cpp reports the configured CUDA device."""

    try:
        process = subprocess.run(
            [str(llama_server), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if process.returncode:
        return False
    return cuda_device.casefold() in process.stdout.casefold()


def resolve_llm_device(
    compute: ComputeSettings,
    *,
    cuda_available: bool | None = None,
) -> Literal["cpu", "cuda"]:
    """Resolve local LLM compute without probing CUDA on the CPU path."""

    if compute.llm_device is ComputeDevice.CPU:
        return "cpu"
    available = (
        llama_cuda_available(compute.cuda_device)
        if cuda_available is None
        else cuda_available
    )
    return resolve_local_compute(
        compute.llm_device, cuda_available=available, label="LLM"
    )


def command(
    snapshot: UserConfig | None = None,
    *,
    cuda_available: bool | None = None,
) -> list[str]:
    """Build the local LLM command from one validated user-config snapshot."""

    resolved = snapshot if snapshot is not None else load_user_config()
    if resolved.providers.llm_provider != "local":
        raise RuntimeError(
            "voice-harness-llm.service is only used when providers.llm.provider=local"
        )
    device = resolve_llm_device(resolved.compute, cuda_available=cuda_available)
    arguments = [
        str(LLAMA_SERVER),
        "--model",
        str(MODEL_FILE),
        "--alias",
        resolved.providers.llm_model,
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        "--n-gpu-layers",
        "99" if device == "cuda" else "0",
        "--flash-attn",
        "on" if device == "cuda" else "off",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    if device == "cuda":
        arguments.extend(("--device", resolved.compute.cuda_device))
    return arguments


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-llm: ok")
        return
    arguments = command()
    os.execv(arguments[0], arguments)


if __name__ == "__main__":
    main()
