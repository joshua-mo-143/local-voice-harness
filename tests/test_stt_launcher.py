from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.stt import launcher


class DictationLauncherTests(unittest.TestCase):
    def test_legacy_systemd_selector_defaults_are_resolver_baselines(self) -> None:
        environment = {
            "INVOCATION_ID": "service-start",
            **launcher.LEGACY_UNIT_DEFAULTS,
            "DICTATION_LANGUAGE": "en",
        }

        resolved = launcher.resolver_environment(environment)

        self.assertNotIn("DICTATION_BACKEND", resolved)
        self.assertNotIn("DICTATION_MODEL", resolved)
        self.assertNotIn("DICTATION_QUANTIZATION", resolved)
        self.assertEqual(resolved["DICTATION_LANGUAGE"], "en")

    def test_explicit_nondefault_environment_override_is_preserved(self) -> None:
        resolved = launcher.resolver_environment(
            {
                "INVOCATION_ID": "service-start",
                "DICTATION_BACKEND": "whisper",
                "DICTATION_MODEL": "large-v3-turbo",
            }
        )

        self.assertEqual(resolved["DICTATION_BACKEND"], "whisper")
        self.assertEqual(resolved["DICTATION_MODEL"], "large-v3-turbo")

    def test_backend_environment_accepts_only_documented_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "backend.env"
            path.write_text(
                "# backend selection\n"
                "DICTATION_BACKEND=whisper\n"
                "DICTATION_MODEL='large-v3-turbo'\n"
                "DICTATION_LANGUAGE=en\n"
            )

            environment = launcher.load_backend_environment(path)

        self.assertEqual(
            environment,
            {
                "DICTATION_BACKEND": "whisper",
                "DICTATION_MODEL": "large-v3-turbo",
                "DICTATION_LANGUAGE": "en",
            },
        )

    def test_backend_environment_rejects_protected_and_unknown_keys(self) -> None:
        for assignment in (
            "DICTATION_SOCKET=/tmp/attacker.sock",
            "CUDA_CACHE_PATH=/home/user/cache",
            "HF_HOME=/tmp/hf",
            "TMPDIR=/tmp",
            "XDG_RUNTIME_DIR=/tmp/runtime",
            "LD_PRELOAD=/tmp/attack.so",
        ):
            with (
                self.subTest(assignment=assignment),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "backend.env"
                path.write_text(f"{assignment}\n")
                with self.assertRaisesRegex(
                    ValueError, "unsupported backend environment key"
                ):
                    launcher.load_backend_environment(path)

    def test_launcher_overwrites_inherited_protected_paths_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            config = home / ".config/dictation"
            config.mkdir(parents=True)
            (config / "backend.env").write_text(
                "DICTATION_BACKEND=whisper\nDICTATION_COMPUTE=float16\n"
            )
            owned = {
                "DICTATION_SOCKET": "/run/user/1000/dictation.sock",
                "CUDA_CACHE_PATH": "/run/user/1000/dictation/cuda-cache",
                "HF_HOME": f"{home}/.cache/huggingface",
                "TMPDIR": "/run/user/1000/dictation",
                "HOME": str(home),
                "XDG_CACHE_HOME": f"{home}/.cache",
                "XDG_RUNTIME_DIR": "/run/user/1000",
            }
            inherited = {
                key: f"/attacker/{key.casefold()}"
                for key in launcher.PROTECTED_ENVIRONMENT_KEYS
            }
            captured: dict[str, str] = {}

            def inspect_execve(
                _executable: str, _arguments: list[str], environment: dict[str, str]
            ) -> None:
                captured.update(environment)
                raise RuntimeError("exec inspected")

            with (
                mock.patch.dict(os.environ, inherited, clear=True),
                mock.patch.object(
                    launcher, "protected_environment", return_value=owned
                ),
                mock.patch.object(launcher.site, "getsitepackages", return_value=[]),
                mock.patch.object(launcher.os, "execve", side_effect=inspect_execve),
                mock.patch.object(launcher.sys, "argv", ["voice-harness-dictation"]),
                self.assertRaisesRegex(RuntimeError, "exec inspected"),
            ):
                launcher.main()

        for key, expected in owned.items():
            self.assertEqual(captured[key], expected)
        self.assertEqual(captured["DICTATION_BACKEND"], "whisper")
        self.assertEqual(captured["DICTATION_COMPUTE"], "float16")


if __name__ == "__main__":
    unittest.main()
