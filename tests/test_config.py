from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_voice_harness import config
from local_voice_harness.user_config import load_user_config


class ConfigPathTests(unittest.TestCase):
    def test_job_state_is_durable_but_worker_logs_are_session_only(self) -> None:
        self.assertEqual(config.JOBS_DIR, config.DURABLE_STATE_DIR / "jobs")
        self.assertEqual(config.JOB_LOGS_DIR, config.STATE_DIR / "jobs")
        self.assertEqual(config.LEGACY_JOBS_DIR, config.STATE_DIR / "jobs")
        self.assertNotEqual(config.JOBS_DIR, config.JOB_LOGS_DIR)

    def test_branch_runtime_isolates_job_logs_without_moving_shared_sockets(
        self,
    ) -> None:
        branch_runtime = Path("/tmp/worktree/.dev/runtime")
        environment = {
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "VOICE_HARNESS_BRANCH_RUNTIME": str(branch_runtime),
        }

        self.assertEqual(config.branch_runtime_dir(environment), branch_runtime)
        self.assertIsNone(config.branch_runtime_dir({}))
        self.assertIsNone(
            config.branch_runtime_dir({"VOICE_HARNESS_BRANCH_RUNTIME": "relative"})
        )
        self.assertEqual(config.STT_SOCKET, config.RUNTIME / "dictation.sock")
        self.assertEqual(config.TTS_SOCKET, config.RUNTIME / "voice-harness-tts.sock")
        self.assertEqual(config.WAKE_LOCK, config.RUNTIME / "voice-harness-wake.lock")

    def test_xdg_state_home_uses_standard_fallback(self) -> None:
        home = Path("/home/example")

        self.assertEqual(
            config.xdg_state_home({"XDG_STATE_HOME": "/custom/state"}, home=home),
            Path("/custom/state"),
        )
        self.assertEqual(config.xdg_state_home({}, home=home), home / ".local/state")
        self.assertEqual(
            config.xdg_state_home({"XDG_STATE_HOME": "relative/state"}, home=home),
            home / ".local/state",
        )

    def test_systemd_state_directory_selects_absolute_voice_harness_entry(
        self,
    ) -> None:
        environment = {
            "STATE_DIRECTORY": (
                "/var/lib/other:/home/example/.local/state/voice-harness"
            ),
            "XDG_STATE_HOME": "/ignored",
        }

        self.assertEqual(
            config.systemd_state_directory(environment),
            Path("/home/example/.local/state/voice-harness"),
        )
        self.assertEqual(
            config.durable_state_dir(environment, home=Path("/home/example")),
            Path("/home/example/.local/state/voice-harness"),
        )

    def test_invalid_or_ambiguous_state_directory_uses_xdg_fallback(self) -> None:
        home = Path("/home/example")
        self.assertEqual(
            config.durable_state_dir(
                {
                    "STATE_DIRECTORY": "relative:/absolute/one:/absolute/two",
                    "XDG_STATE_HOME": "/custom/state",
                },
                home=home,
            ),
            Path("/custom/state/voice-harness"),
        )

    def test_backend_settings_default_to_venice_and_support_venice_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.toml"
            defaults = config.load_backend_settings({}, path=missing, home=root)
            self.assertEqual(
                (defaults.llm_provider, defaults.tts_provider), ("venice", "venice")
            )
            self.assertEqual(defaults.llm_model, "venice-uncensored")
            self.assertEqual(defaults.tts_model, "tts-kokoro")
            self.assertEqual(defaults.tts_speed, 1.25)

            backend_file = root / "backends.toml"
            backend_file.write_text(
                "[llm]\n"
                'provider = "venice"\n'
                'model = "venice-uncensored"\n'
                "timeout = 15\n"
                "[tts]\n"
                'provider = "venice"\n'
                'model = "tts-kokoro"\n'
                'voice = "af_sky"\n'
                "speed = 1.25\n"
            )

            settings = config.load_backend_settings({}, path=backend_file, home=root)

        self.assertEqual(settings.llm_provider, "venice")
        self.assertEqual(settings.llm_timeout, 15)
        self.assertEqual(settings.tts_provider, "venice")
        self.assertEqual(settings.tts_voice, "af_sky")
        self.assertEqual(settings.tts_speed, 1.25)

    def test_venice_tts_defaults_to_quarter_faster_speed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "backends.toml"
            path.write_text('[tts]\nprovider = "venice"\n')

            settings = config.load_backend_settings({}, path=path)

        self.assertEqual(settings.tts_speed, 1.25)

    def test_backend_environment_overrides_toml_and_rejects_unknown_provider(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "backends.toml"
            path.write_text('[llm]\nprovider = "local"\n')
            settings = config.load_backend_settings(
                {
                    "VOICE_HARNESS_LLM_PROVIDER": "venice",
                    "VOICE_HARNESS_LLM_MODEL": "custom-model",
                },
                path=path,
                home=Path(temporary),
            )
            self.assertEqual(settings.llm_provider, "venice")
            self.assertEqual(settings.llm_model, "custom-model")

            with self.assertRaisesRegex(
                config.BackendConfigurationError, "local or venice"
            ):
                config.load_backend_settings(
                    {"VOICE_HARNESS_TTS_PROVIDER": "unknown"},
                    path=path,
                    home=Path(temporary),
                )
            with self.assertRaisesRegex(
                config.BackendConfigurationError, "positive number"
            ):
                config.load_backend_settings(
                    {"VOICE_HARNESS_LLM_TIMEOUT": "nan"},
                    path=path,
                    home=Path(temporary),
                )
            with self.assertRaisesRegex(
                config.BackendConfigurationError, "between 0.25 and 4"
            ):
                config.load_backend_settings(
                    {"VOICE_HARNESS_TTS_SPEED": "4.1"},
                    path=path,
                    home=Path(temporary),
                )

    def test_rejects_legacy_file_based_venice_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "backends.toml"
            path.write_text('[venice]\napi_key_file = "/tmp/legacy"\n')
            with self.assertRaisesRegex(
                config.BackendConfigurationError, "credentials set"
            ):
                config.load_backend_settings({}, path=path)

    def test_followup_window_rejects_non_finite_and_negative_values(self) -> None:
        name = "VOICE_HARNESS_CURSOR_FOLLOWUP_WINDOW_SECONDS"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "path": root / "config.toml",
                "backends_path": root / "backends.toml",
                "backend_env_path": root / "backend.env",
                "home": root,
            }
            for value in ("invalid", "-1", "nan", "inf", "-inf"):
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(ValueError, "finite non-negative"),
                ):
                    load_user_config({name: value}, **paths)

            self.assertEqual(
                load_user_config(
                    {name: "0"}, **paths
                ).platform.cursor_followup_window_seconds,
                0,
            )


if __name__ == "__main__":
    unittest.main()
