from __future__ import annotations

import json
import os
import shutil
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
import shutil
import signal
import sys
from pathlib import Path

project = Path(sys.argv[sys.argv.index("--project") + 1]).resolve()
sys.path.insert(0, str(project / "src"))
from local_voice_harness import config as application_config

record = {
    "arguments": sys.argv[1:],
    "project": str(project),
    "application_source": str(Path(application_config.__file__).resolve()),
    "checkout_marker": getattr(application_config, "CHECKOUT_MARKER", None),
    "jobs_database": str(application_config.JOBS_DB),
    "job_logs_dir": str(application_config.JOB_LOGS_DIR),
    "environment": {
        name: os.environ.get(name)
        for name in (
            "GH_CONFIG_DIR",
            "STATE_DIRECTORY",
            "TMPDIR",
            "XDG_CONFIG_HOME",
            "XDG_STATE_HOME",
            "XDG_RUNTIME_DIR",
            "UV_PROJECT_ENVIRONMENT",
            "VOICE_HARNESS_BRANCH_RUNTIME",
            "VOICE_HARNESS_WAKE_THRESHOLD",
        )
    },
}
Path(os.environ["FAKE_UV_RECORD"]).write_text(json.dumps(record))
if os.environ.get("FAKE_UV_RECREATE_ENV") == "1":
    environment = Path(
        os.environ.get("UV_PROJECT_ENVIRONMENT", str(project / ".venv"))
    )
    shutil.rmtree(environment, ignore_errors=True)
    environment.mkdir(parents=True)
    (environment / "synced").write_text("launcher")
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
                "STATE_DIRECTORY": str(
                    self.test_root / "production-state" / "voice-harness"
                ),
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

    def _write_checkout_application(self, checkout: Path, marker: str) -> Path:
        package = checkout / "src" / "local_voice_harness"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        config = package / "config.py"
        config.write_text(
            "\n".join(
                (
                    "import os",
                    "from pathlib import Path",
                    f"CHECKOUT_MARKER = {marker!r}",
                    'JOBS_DB = Path(os.environ["XDG_STATE_HOME"])',
                    'JOBS_DB /= "voice-harness/jobs/jobs.sqlite3"',
                    'JOB_LOGS_DIR = Path(os.environ["VOICE_HARNESS_BRANCH_RUNTIME"])',
                    'JOB_LOGS_DIR /= "jobs"',
                    "",
                )
            )
        )
        return config

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
                "--python",
                "3.11",
                "--extra",
                "wake",
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
                "STATE_DIRECTORY": None,
                "TMPDIR": str(PROJECT_ROOT / ".dev" / "runtime" / "tmp"),
                "XDG_CONFIG_HOME": str(PROJECT_ROOT / ".dev" / "config"),
                "XDG_STATE_HOME": str(PROJECT_ROOT / ".dev" / "state"),
                "XDG_RUNTIME_DIR": str(self.test_root / "shared-runtime"),
                "UV_PROJECT_ENVIRONMENT": str(PROJECT_ROOT / ".dev" / "venv"),
                "VOICE_HARNESS_BRANCH_RUNTIME": str(PROJECT_ROOT / ".dev" / "runtime"),
                "VOICE_HARNESS_WAKE_THRESHOLD": "0.73",
            },
        )
        self.assertEqual(
            invocation["jobs_database"],
            str(
                PROJECT_ROOT
                / ".dev"
                / "state"
                / "voice-harness"
                / "jobs"
                / "jobs.sqlite3"
            ),
        )
        self.assertTrue((PROJECT_ROOT / ".dev" / "config").is_dir())
        self.assertTrue((PROJECT_ROOT / ".dev" / "state").is_dir())
        self.assertTrue((PROJECT_ROOT / ".dev" / "runtime").is_dir())
        self.assertEqual(
            invocation["job_logs_dir"],
            str(PROJECT_ROOT / ".dev" / "runtime" / "jobs"),
        )
        self.assertFalse(self.systemctl_record.exists())

    def test_commands_cannot_inherit_production_durable_state(self) -> None:
        production_database = (
            self.test_root
            / "production-state"
            / "voice-harness"
            / "jobs"
            / "jobs.sqlite3"
        )
        production_database.parent.mkdir(parents=True)
        production_database.write_text("production sentinel")
        expected_database = (
            PROJECT_ROOT / ".dev" / "state" / "voice-harness" / "jobs" / "jobs.sqlite3"
        )
        resolved_databases: set[str] = set()

        for arguments in (
            ("text", "request"),
            ("wake",),
            ("setup", "--defaults"),
            ("config", "show", "audio.wake_threshold"),
            ("integrations", "list"),
        ):
            with self.subTest(arguments=arguments):
                process = self._run(*arguments)

                self.assertEqual(process.returncode, 0, process.stderr)
                invocation = self._uv_invocation()
                resolved_databases.add(cast(str, invocation["jobs_database"]))
                self.assertEqual(
                    invocation["jobs_database"],
                    str(expected_database),
                )
                self.assertIsNone(
                    cast(dict[str, str | None], invocation["environment"])[
                        "STATE_DIRECTORY"
                    ]
                )
                self.assertEqual(production_database.read_text(), "production sentinel")

        self.assertEqual(resolved_databases, {str(expected_database)})

    def test_copied_launchers_use_distinct_checkout_databases(self) -> None:
        resolved_databases: list[str] = []
        application_sources: list[str] = []

        for name in ("checkout-one", "checkout-two"):
            checkout = self.test_root / name
            scripts = checkout / "scripts"
            scripts.mkdir(parents=True)
            config = self._write_checkout_application(checkout, name)
            launcher = scripts / "dev.sh"
            shutil.copy2(LAUNCHER, launcher)
            record = self.test_root / f"{name}-uv.json"

            process = subprocess.run(
                [str(launcher), "text", "request"],
                cwd=self.test_root,
                env=self._environment(FAKE_UV_RECORD=str(record)),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            invocation = json.loads(record.read_text())
            expected_database = (
                checkout / ".dev" / "state" / "voice-harness" / "jobs" / "jobs.sqlite3"
            )
            self.assertEqual(invocation["jobs_database"], str(expected_database))
            self.assertEqual(invocation["project"], str(checkout))
            self.assertEqual(invocation["checkout_marker"], name)
            self.assertEqual(
                invocation["application_source"],
                str(config.resolve()),
            )
            resolved_databases.append(cast(str, invocation["jobs_database"]))
            application_sources.append(cast(str, invocation["application_source"]))

        self.assertNotEqual(*resolved_databases)
        self.assertNotEqual(*application_sources)

    def test_concurrent_worktree_launchers_isolate_branch_owned_files(self) -> None:
        processes: list[subprocess.Popen[str]] = []
        records: list[Path] = []
        checkouts: list[Path] = []
        shared_runtime = self.test_root / "shared-runtime"
        shared_runtime.mkdir(parents=True, exist_ok=True)
        (shared_runtime / "dictation.sock").write_text("shared-stt")
        (shared_runtime / "voice-harness-tts.sock").write_text("shared-tts")

        try:
            for name in ("worktree-one", "worktree-two"):
                checkout = self.test_root / name
                scripts = checkout / "scripts"
                scripts.mkdir(parents=True)
                self._write_checkout_application(checkout, name)
                shutil.copy2(LAUNCHER, scripts / "dev.sh")
                record = self.test_root / f"{name}-uv.json"
                ready = self.test_root / f"{name}-ready"
                checkouts.append(checkout)
                records.append(record)
                processes.append(
                    subprocess.Popen(
                        [str(scripts / "dev.sh"), "text", "request"],
                        cwd=self.test_root,
                        env=self._environment(
                            FAKE_UV_RECORD=str(record),
                            FAKE_UV_MODE="wait",
                            FAKE_READY=str(ready),
                        ),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not all(
                (self.test_root / f"{name}-ready").exists()
                for name in ("worktree-one", "worktree-two")
            ):
                time.sleep(0.01)
            self.assertTrue(
                all(
                    (self.test_root / f"{name}-ready").exists()
                    for name in ("worktree-one", "worktree-two")
                ),
                "concurrent launchers did not start",
            )

            invocations = [json.loads(record.read_text()) for record in records]
            environments = [
                cast(dict[str, str | None], invocation["environment"])
                for invocation in invocations
            ]
            owned_roots = [
                {
                    environments[index]["XDG_CONFIG_HOME"],
                    environments[index]["XDG_STATE_HOME"],
                    environments[index]["UV_PROJECT_ENVIRONMENT"],
                    environments[index]["VOICE_HARNESS_BRANCH_RUNTIME"],
                    environments[index]["TMPDIR"],
                    invocations[index]["jobs_database"],
                    invocations[index]["job_logs_dir"],
                }
                for index in range(2)
            ]
            for index, checkout in enumerate(checkouts):
                other = checkouts[1 - index]
                self.assertEqual(invocations[index]["project"], str(checkout))
                self.assertEqual(
                    invocations[index]["checkout_marker"],
                    checkout.name,
                )
                application_source = cast(str, invocations[index]["application_source"])
                self.assertTrue(
                    application_source.startswith(str(checkout / "src")),
                    application_source,
                )
                self.assertFalse(
                    application_source.startswith(str(other / "src")),
                    application_source,
                )
                self.assertEqual(
                    environments[index]["XDG_RUNTIME_DIR"],
                    str(shared_runtime),
                )
                self.assertEqual(
                    environments[index]["VOICE_HARNESS_BRANCH_RUNTIME"],
                    str(checkout / ".dev" / "runtime"),
                )
                self.assertEqual(
                    invocations[index]["job_logs_dir"],
                    str(checkout / ".dev" / "runtime" / "jobs"),
                )
                self.assertTrue((checkout / ".dev" / "config").is_dir())
                self.assertTrue((checkout / ".dev" / "state").is_dir())
                self.assertTrue((checkout / ".dev" / "runtime").is_dir())
                for path in owned_roots[index]:
                    self.assertIsNotNone(path)
                    assert path is not None
                    self.assertTrue(
                        path.startswith(str(checkout / ".dev")),
                        path,
                    )
                    self.assertFalse(path.startswith(str(other / ".dev")), path)
            self.assertTrue(owned_roots[0].isdisjoint(owned_roots[1]))
            self.assertEqual(
                (shared_runtime / "dictation.sock").read_text(), "shared-stt"
            )
            self.assertEqual(
                (shared_runtime / "voice-harness-tts.sock").read_text(),
                "shared-tts",
            )
        finally:
            for process in processes:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=5)

    def test_primary_checkout_launcher_preserves_installed_wake_environment(
        self,
    ) -> None:
        checkout = self.test_root / "primary-checkout"
        scripts = checkout / "scripts"
        scripts.mkdir(parents=True)
        launcher = scripts / "dev.sh"
        shutil.copy2(LAUNCHER, launcher)
        installed = checkout / ".venv"
        interpreter = installed / "bin" / "python"
        package = installed / "lib" / "openwakeword" / "__init__.py"
        model = (
            installed
            / "lib"
            / "openwakeword"
            / "resources"
            / "models"
            / "hey_jarvis_v0.1.onnx"
        )
        for path, content in (
            (interpreter, "installed interpreter"),
            (package, "installed package"),
            (model, "installed model"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        process = subprocess.run(
            [str(launcher), "text", "request"],
            cwd=self.test_root,
            env=self._environment(FAKE_UV_RECREATE_ENV="1"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(interpreter.read_text(), "installed interpreter")
        self.assertEqual(package.read_text(), "installed package")
        self.assertEqual(model.read_text(), "installed model")
        self.assertEqual(
            self._uv_environment()["UV_PROJECT_ENVIRONMENT"],
            str(checkout / ".dev" / "venv"),
        )
        self.assertEqual(
            (checkout / ".dev" / "venv" / "synced").read_text(), "launcher"
        )

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

    def test_pronounce_uses_checkout_without_touching_services(self) -> None:
        process = self._run("pronounce", "PR #128")

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            self._uv_invocation()["arguments"],
            [
                "run",
                "--project",
                str(PROJECT_ROOT),
                "--python",
                "3.11",
                "--extra",
                "wake",
                "voice-harness",
                "pronounce",
                "PR #128",
            ],
        )
        self.assertFalse(self.systemctl_record.exists())

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
                "--python",
                "3.11",
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

    def test_config_uses_checkout_with_isolated_homes(self) -> None:
        process = self._run(
            "config",
            "show",
            "audio.wake_threshold",
            environment=self._environment(FAKE_UV_EXIT="0"),
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        invocation = self._uv_invocation()
        self.assertEqual(
            invocation["arguments"],
            [
                "run",
                "--project",
                str(PROJECT_ROOT),
                "--python",
                "3.11",
                "--extra",
                "wake",
                "voice-harness",
                "config",
                "show",
                "audio.wake_threshold",
            ],
        )
        self.assertEqual(
            self._uv_environment()["XDG_CONFIG_HOME"],
            str(PROJECT_ROOT / ".dev" / "config"),
        )
        self.assertFalse(self.systemctl_record.exists())

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
