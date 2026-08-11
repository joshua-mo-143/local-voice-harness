from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import PROJECT_ROOT
from .user_config import load_user_config

LLAMA_SERVER = Path("/usr/sbin/llama-server")
MODEL_FILE = PROJECT_ROOT / "models" / "Qwen3.5-4B-Q4_K_M.gguf"


def command() -> list[str]:
    """Build the local LLM command from one validated user-config snapshot."""

    snapshot = load_user_config()
    if snapshot.providers.llm_provider != "local":
        raise RuntimeError(
            "voice-harness-llm.service is only used when providers.llm.provider=local"
        )
    return [
        str(LLAMA_SERVER),
        "--model",
        str(MODEL_FILE),
        "--alias",
        snapshot.providers.llm_model,
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        "--n-gpu-layers",
        "99",
        "--device",
        snapshot.compute.cuda_device,
        "--flash-attn",
        "on",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-llm: ok")
        return
    arguments = command()
    os.execv(arguments[0], arguments)


if __name__ == "__main__":
    main()
