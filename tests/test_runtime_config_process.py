from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

CHILD = Path(__file__).with_name("runtime_config_child.py")


def _environment(config_home: Path, overrides: dict[str, str] | None = None):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VOICE_HARNESS_", "DICTATION_"))
    }
    environment["XDG_CONFIG_HOME"] = str(config_home)
    environment.update(overrides or {})
    return environment


def _snapshot(
    config_home: Path,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(CHILD)],
        env=_environment(config_home, overrides),
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(process.stdout)
    assert isinstance(value, dict)
    return value


class RuntimeConfigProcessTests(unittest.TestCase):
    def test_legacy_channels_are_resolver_inputs_only(self) -> None:
        source = Path(__file__).parents[1] / "src" / "local_voice_harness"
        direct_setting_read = re.compile(
            r'os\.environ\.get\("(?:VOICE_HARNESS_|DICTATION_)'
        )
        direct_reads: list[str] = []
        backend_environment_calls: list[str] = []
        backend_settings_calls: list[str] = []
        for path in source.rglob("*.py"):
            text = path.read_text()
            for match in direct_setting_read.finditer(text):
                line = text[: match.start()].count("\n") + 1
                if match.group() == 'os.environ.get("VOICE_HARNESS_':
                    if (
                        path.name == "config.py"
                        and "VOICE_HARNESS_ROOT" in text.splitlines()[line - 1]
                    ):
                        continue
                direct_reads.append(f"{path.relative_to(source)}:{line}")
            backend_environment_calls.extend(
                str(path.relative_to(source))
                for line in text.splitlines()
                if "load_backend_environment(" in line
            )
            backend_settings_calls.extend(
                str(path.relative_to(source))
                for line in text.splitlines()
                if "load_backend_settings(" in line
            )

        self.assertEqual(direct_reads, [])
        self.assertEqual(
            backend_environment_calls,
            ["user_config.py", "user_config.py"],
        )
        self.assertEqual(backend_settings_calls, ["config.py"])

    def test_unified_legacy_and_environment_inputs_match_every_process_view(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unified = root / "unified"
            legacy = root / "legacy"
            environment_only = root / "environment"
            (unified / "voice-harness").mkdir(parents=True)
            (legacy / "voice-harness").mkdir(parents=True)
            (legacy / "dictation").mkdir(parents=True)
            environment_only.mkdir()
            (unified / "voice-harness" / "config.toml").write_text(
                "[providers.llm]\n"
                'model = "parity-model"\n'
                "[providers.tts]\n"
                'voice = "parity-voice"\n'
                "[compute]\n"
                'dictation_backend = "whisper"\n'
                'dictation_model = "small.en"\n'
                'dictation_quantization = "none"\n'
                'dictation_compute = "int8_float16"\n'
                'dictation_language = "english"\n'
            )
            (legacy / "voice-harness" / "backends.toml").write_text(
                '[llm]\nmodel = "parity-model"\n[tts]\nvoice = "parity-voice"\n'
            )
            (legacy / "dictation" / "backend.env").write_text(
                "DICTATION_BACKEND=whisper\n"
                "DICTATION_MODEL=small.en\n"
                "DICTATION_QUANTIZATION=none\n"
                "DICTATION_COMPUTE=int8_float16\n"
                "DICTATION_LANGUAGE=english\n"
            )
            overrides = {
                "VOICE_HARNESS_LLM_MODEL": "parity-model",
                "VOICE_HARNESS_TTS_VOICE": "parity-voice",
                "DICTATION_BACKEND": "whisper",
                "DICTATION_MODEL": "small.en",
                "DICTATION_QUANTIZATION": "none",
                "DICTATION_COMPUTE": "int8_float16",
                "DICTATION_LANGUAGE": "english",
            }

            unified_snapshot = _snapshot(unified)
            legacy_snapshot = _snapshot(legacy)
            environment_snapshot = _snapshot(environment_only, overrides)

        self.assertEqual(unified_snapshot, legacy_snapshot)
        self.assertEqual(unified_snapshot, environment_snapshot)

    def test_running_process_keeps_snapshot_and_restart_observes_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_home = Path(temporary)
            config_dir = config_home / "voice-harness"
            config_dir.mkdir()
            path = config_dir / "config.toml"
            path.write_text('[providers.llm]\nmodel = "startup-model"\n')
            process = subprocess.Popen(
                [sys.executable, str(CHILD), "watch"],
                env=_environment(config_home),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                startup = json.loads(process.stdout.readline())
                path.write_text('[providers.llm]\nmodel = "restart-model"\n')
                process.stdin.write("snapshot\n")
                process.stdin.flush()
                unchanged = json.loads(process.stdout.readline())
            finally:
                process.stdin.close()
                process.wait(timeout=5)

            restarted = _snapshot(config_home)

        self.assertEqual(startup, unchanged)
        self.assertEqual(
            startup["config"]["providers"]["llm_model"],
            "startup-model",
        )
        self.assertEqual(
            restarted["config"]["providers"]["llm_model"],
            "restart-model",
        )

    def test_malformed_config_fails_at_process_boundary_with_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_home = Path(temporary)
            config_dir = config_home / "voice-harness"
            config_dir.mkdir()
            path = config_dir / "config.toml"
            path.write_text("[providers.llm\n")

            process = subprocess.run(
                [sys.executable, str(CHILD)],
                env=_environment(config_home),
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(process.returncode, 0)
        self.assertIn(f"user configuration {path}", process.stderr)

    def test_disabled_integrations_construct_no_runtime_providers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_home = Path(temporary)
            config_dir = config_home / "voice-harness"
            config_dir.mkdir()
            (config_dir / "config.toml").write_text(
                "[integrations]\ngithub = false\nzendesk = false\nlinear = false\n"
            )

            snapshot = _snapshot(config_home)

        self.assertEqual(snapshot["enabled_integrations"], [])


if __name__ == "__main__":
    unittest.main()
