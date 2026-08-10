from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "dev.sh"


class DevLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.test_root = Path(self.temporary.name)
        self.bin_dir = self.test_root / "bin"
        self.bin_dir.mkdir()
        self.uv_record = self.test_root / "uv.json"
        self.systemctl_record = self.test_root / "systemctl.json"
        self.ready = self.test_root / "ready"
        self._write_fake_commands()

    def _write_executable(self, path: Path, source: str) -> None:
        path.write_text(f"#!{sys.executable}\n{source}")
        path.chmod(0o755)

    def _write_fake_commands(self) -> None:
        self._write_executable(
            self.bin_dir / "uv",
            """
import json
import os
import signal
import sys
from pathlib import Path

record = {
    "arguments": sys.argv[1:],
    "environment": {
        name: os.environ.get(name)
        for name in (
            "GH_CONFIG_DIR",
            "XDG_CONFIG_HOME",
            "XDG_STATE_HOME",
            "XDG_RUNTIME_DIR",
            "VOICE_HARNESS_WAKE_THRESHOLD",
        )
    },
}
Path(os.environ["FAKE_UV_RECORD"]).write_text(json.dumps(record))
if os.environ.get("FAKE_UV_MODE") == "wait":
    Path(os.environ["FAKE_READY"]).write_text(str(os.getpid()))
    signal.pause()
raise SystemExit(int(os.environ.get("FAKE_UV_EXIT", "0")))
""",
        )
        self._write_executable(
            self.bin_dir / "systemctl",
            """
import json
import os
import sys
from pathlib import Path

Path(os.environ["FAKE_SYSTEMCTL_RECORD"]).write_text(json.dumps(sys.argv[1:]))
raise SystemExit(int(os.environ.get("FAKE_SYSTEMCTL_EXIT", "3")))
""",
        )

    def _environment(self, **overrides: str) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin_dir}{os.pathsep}{environment['PATH']}",
                "FAKE_UV_RECORD": str(self.uv_record),
                "FAKE_SYSTEMCTL_RECORD": str(self.systemctl_record),
                "FAKE_READY": str(self.ready),
                "FAKE_SYSTEMCTL_EXIT": "3",
                "XDG_CONFIG_HOME": str(self.test_root / "official-config"),
                "XDG_STATE_HOME": str(self.test_root / "official-state"),
                "XDG_RUNTIME_DIR": str(self.test_root / "shared-runtime"),
                "VOICE_HARNESS_WAKE_THRESHOLD": "0.73",
            }
        )
        environment.update(overrides)
        return environment

    def _run(
        self, *arguments: str, environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(LAUNCHER), *arguments],
            cwd=self.test_root,
            env=environment or self._environment(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def _uv_invocation(self) -> dict[str, object]:
        return json.loads(self.uv_record.read_text())

    def _uv_environment(self) -> dict[str, str | None]:
        return cast(dict[str, str | None], self._uv_invocation()["environment"])

    def test_text_uses_checkout_with_isolated_homes_and_inherited_overrides(
        self,
    ) -> None:
        process = self._run(
            "text",
            "request with spaces",
            "",
            "--literal",
            environment=self._environment(FAKE_UV_EXIT="23"),
        )

        self.assertEqual(process.returncode, 23)
        invocation = self._uv_invocation()
        self.assertEqual(
            invocation["arguments"],
            [
                "run",
                "--project",
                str(PROJECT_ROOT),
                "voice-harness",
                "text",
                "request with spaces",
                "",
                "--literal",
            ],
        )
        self.assertEqual(
            invocation["environment"],
            {
                "GH_CONFIG_DIR": str(self.test_root / "official-config" / "gh"),
                "XDG_CONFIG_HOME": str(PROJECT_ROOT / ".dev" / "config"),
                "XDG_STATE_HOME": str(PROJECT_ROOT / ".dev" / "state"),
                "XDG_RUNTIME_DIR": str(self.test_root / "shared-runtime"),
                "VOICE_HARNESS_WAKE_THRESHOLD": "0.73",
            },
        )
        self.assertTrue((PROJECT_ROOT / ".dev" / "config").is_dir())
        self.assertTrue((PROJECT_ROOT / ".dev" / "state").is_dir())
        self.assertFalse(self.systemctl_record.exists())

    def test_preserves_explicit_github_cli_config_directory(self) -> None:
        github_config = self.test_root / "github-config"

        process = self._run(
            "text",
            "request",
            environment=self._environment(GH_CONFIG_DIR=str(github_config)),
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            self._uv_environment()["GH_CONFIG_DIR"],
            str(github_config),
        )

    def test_uses_home_github_cli_config_without_xdg_override(self) -> None:
        home = self.test_root / "home"
        environment = self._environment(HOME=str(home))
        environment.pop("XDG_CONFIG_HOME")

        process = self._run("text", "request", environment=environment)

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            self._uv_environment()["GH_CONFIG_DIR"],
            str(home / ".config" / "gh"),
        )

    def test_wake_checks_inactive_service_before_running_foreground_daemon(
        self,
    ) -> None:
        process = self._run("wake")

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            json.loads(self.systemctl_record.read_text()),
            [
                "--user",
                "is-active",
                "--quiet",
                "voice-harness-wake.service",
            ],
        )
        self.assertEqual(
            self._uv_invocation()["arguments"],
            [
                "run",
                "--project",
                str(PROJECT_ROOT),
                "--extra",
                "wake",
                "voice-harness-wake",
            ],
        )

    def test_wake_refuses_to_run_while_installed_listener_is_active(self) -> None:
        process = self._run(
            "wake", environment=self._environment(FAKE_SYSTEMCTL_EXIT="0")
        )

        self.assertEqual(process.returncode, 1)
        self.assertIn(
            "systemctl --user stop voice-harness-wake.service", process.stderr
        )
        self.assertIn(
            "systemctl --user start voice-harness-wake.service", process.stderr
        )
        self.assertIn("No services were changed.", process.stderr)
        self.assertFalse(self.uv_record.exists())
        self.assertEqual(
            json.loads(self.systemctl_record.read_text()).count(
                "voice-harness-wake.service"
            ),
            1,
        )

    def test_wake_fails_safe_when_service_state_cannot_be_checked(self) -> None:
        process = self._run(
            "wake", environment=self._environment(FAKE_SYSTEMCTL_EXIT="1")
        )

        self.assertEqual(process.returncode, 1)
        self.assertIn("unable to determine", process.stderr)
        self.assertIn("No services were changed.", process.stderr)
        self.assertFalse(self.uv_record.exists())

    def test_rejects_commands_outside_the_narrow_surface(self) -> None:
        for arguments in (("services", "stop"), ("text",), ("wake", "--check")):
            with self.subTest(arguments=arguments):
                process = self._run(*arguments)
                self.assertEqual(process.returncode, 2)
                self.assertIn("scripts/dev.sh text <request>", process.stderr)
                self.assertFalse(self.uv_record.exists())
                self.assertFalse(self.systemctl_record.exists())

    def test_exec_preserves_process_identity_and_signals(self) -> None:
        process = subprocess.Popen(
            [str(LAUNCHER), "text", "wait"],
            cwd=self.test_root,
            env=self._environment(FAKE_UV_MODE="wait"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while not self.ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(self.ready.exists(), "fake uv did not start")
            self.assertEqual(int(self.ready.read_text()), process.pid)

            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
            self.assertEqual(process.returncode, -signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
