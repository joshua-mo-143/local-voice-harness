from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from local_voice_harness import config_activation
from local_voice_harness.user_config import default_user_config

WAKE = "voice-harness-wake.service"


class FileController:
    def __init__(self, path: Path, *, crash: bool) -> None:
        self.path = path
        self.crash = crash

    def _read(self) -> dict[str, object]:
        value = json.loads(self.path.read_text())
        assert isinstance(value, dict)
        return value

    def snapshot(self, service: str) -> config_activation.ServiceSnapshot:
        assert service == WAKE
        value = self._read()["snapshot"]
        assert isinstance(value, dict)
        return config_activation.ServiceSnapshot(**value)

    def restart(self, service: str) -> subprocess.CompletedProcess[str]:
        value = self._read()
        snapshot = self.snapshot(service)
        restarts = value["restarts"]
        assert isinstance(restarts, int) and not isinstance(restarts, bool)
        value["restarts"] = restarts + 1
        value["snapshot"] = asdict(
            replace(
                snapshot,
                invocation_id=f"{snapshot.invocation_id}-next",
                process_start=f"{snapshot.process_start}-next",
            )
        )
        self.path.write_text(json.dumps(value))
        if self.crash:
            os._exit(91)
        return subprocess.CompletedProcess([], 0, "", "")


def updated_config():
    config = default_user_config(home=Path("/home/example"))
    return replace(config, audio=replace(config.audio, voice="new_voice"))


def original_config():
    return default_user_config(home=Path("/home/example"))


def main() -> None:
    command = sys.argv[1]
    store = config_activation.ActivationStore(Path(sys.argv[2]))
    if command == "current":
        current = store.current()
        print(json.dumps({"record": current.id if current else None}))
        return
    if command == "delivery":
        delivery = store.next_delivery()
        print(
            json.dumps(
                {
                    "kind": delivery.kind.value if delivery else None,
                    "id": delivery.record.id if delivery else None,
                }
            )
        )
        return
    if command == "ack":
        store.acknowledge(sys.argv[3])
        return
    if command == "execute":
        state_path = Path(sys.argv[3])
        record = store.current()
        assert record is not None
        final = config_activation.execute_activation(
            record.id,
            store=store,
            controller=FileController(
                state_path,
                crash=os.environ.get("CRASH_AFTER_RESTART") == "1",
            ),
            load_config=updated_config,
            voice_validator=lambda _voice: config_activation.VoiceValidationResult(
                True,
                "accepted",
            ),
            restore_voice=lambda _record: original_config(),
            observation_timeout=0,
        )
        print(json.dumps({"status": final.status.value}))
        return
    raise AssertionError(f"unknown command {command}")


if __name__ == "__main__":
    main()
