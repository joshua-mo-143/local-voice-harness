from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from .types import HERDR_BIN, HERDR_UNIT, HerdrError


class HerdrTransport:
    """Low-level Herdr CLI transport."""

    def __init__(self, executable: str = HERDR_BIN) -> None:
        self.executable = executable

    def command(self, *args: str) -> list[str]:
        return [self.executable, *args]

    @staticmethod
    def decode(text: str) -> dict[str, Any]:
        try:
            envelope = json.loads(text.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise HerdrError("Herdr returned malformed JSON") from exc
        if "error" in envelope:
            error = envelope["error"]
            raise HerdrError(
                str(error.get("message") or "Herdr command failed"),
                code=str(error.get("code") or ""),
            )
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise HerdrError("Herdr response did not include a result")
        return result

    def run(
        self, *args: str, timeout: float = 30, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        try:
            process = subprocess.run(
                self.command(*args),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except OSError as exc:
            raise HerdrError(
                f"Herdr command failed: {exc}", code="operation_spawn_failed"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise HerdrError(
                f"Herdr command failed: {exc}", code="operation_timeout"
            ) from exc
        if check and process.returncode:
            text = process.stdout.strip() or process.stderr.strip()
            try:
                self.decode(text)
            except HerdrError as exc:
                raise exc
            raise HerdrError(text or f"Herdr exited with status {process.returncode}")
        return process

    def run_json(self, *args: str, timeout: float = 30) -> dict[str, Any]:
        return self.decode(self.run(*args, timeout=timeout).stdout)

    def run_text(self, *args: str, timeout: float = 30) -> str:
        return self.run(*args, timeout=timeout).stdout

    def is_running(self) -> bool:
        process = self.run("status", "server", check=False)
        return process.returncode == 0 and "status: running" in process.stdout

    def ensure_server(self, timeout: float = 15) -> None:
        if self.is_running():
            return
        subprocess.run(
            ["systemctl", "--user", "start", HERDR_UNIT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if not self.is_running():
            process = subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    "--unit=voice-harness-herdr",
                    "--collect",
                    self.executable,
                    "server",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if (
                process.returncode
                and "already exists" not in (process.stdout + process.stderr).casefold()
            ):
                raise HerdrError(process.stderr.strip() or "Could not start Herdr")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_running():
                return
            time.sleep(0.2)
        raise HerdrError("Herdr did not become ready")
