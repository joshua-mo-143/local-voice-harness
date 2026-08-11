from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from local_voice_harness.integrations.registry import (
    build_integration_registry,
    enabled_integrations,
)
from local_voice_harness.stt.server import runtime_settings
from local_voice_harness.user_config import load_user_config


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def main() -> None:
    config = load_user_config()
    registry = build_integration_registry(config)
    github = registry.github_client()
    herdr = registry.herdr_client()
    snapshot = {
        "config": _json_value(asdict(config)),
        "stt": _json_value(asdict(runtime_settings(config))),
        "enabled_integrations": [
            str(getattr(integration, "name", ""))
            for integration in enabled_integrations(registry)
        ],
        "clients": {
            "github_root": str(github.clone_root),
            "git_bin": github.local_git.git_executable,
            "herdr_bin": herdr.executable,
            "herdr_worktree_root": str(herdr.workspace.worktree_root),
        },
    }
    payload = json.dumps(snapshot, sort_keys=True)
    print(payload, flush=True)
    if sys.argv[1:] == ["watch"]:
        for _line in sys.stdin:
            print(payload, flush=True)


if __name__ == "__main__":
    main()
