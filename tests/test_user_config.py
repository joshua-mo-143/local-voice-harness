from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import user_config
from local_voice_harness.user_config import UserConfigurationError


class UserConfigDefaultsTests(unittest.TestCase):
    HOME = Path("/home/example")

    def _missing(self, root: Path) -> dict[str, Path]:
        return {
            "path": root / "config.toml",
            "backends_path": root / "backends.toml",
        }

    def test_defaults_disable_optional_integrations(self) -> None:
        config = user_config.default_user_config(home=self.HOME)

        self.assertTrue(config.integrations.github_enabled)
        self.assertFalse(config.integrations.zendesk_enabled)
        self.assertFalse(config.integrations.linear_enabled)

    def test_defaults_use_local_providers_and_standard_paths(self) -> None:
        config = user_config.default_user_config(home=self.HOME)

        self.assertEqual(config.providers.llm_provider, "local")
        self.assertEqual(config.providers.tts_provider, "local")
        self.assertEqual(config.audio.wake_threshold, 0.55)
        self.assertEqual(config.compute.dictation_backend, "parakeet")
        self.assertEqual(config.platform.project_root, self.HOME)
        self.assertEqual(config.platform.github_root, self.HOME / "src")
        self.assertEqual(
            config.platform.herdr_bin, self.HOME / ".local" / "bin" / "herdr"
        )
        self.assertEqual(config.platform.cursor_agent_inactivity_seconds, 900)
        self.assertEqual(config.platform.cursor_agent_max_runtime_seconds, 3600)
        self.assertEqual(config.platform.agent_job_start_concurrency, 3)

    def test_config_path_lives_under_xdg_config_home(self) -> None:
        path = user_config.user_config_path({"XDG_CONFIG_HOME": "/cfg"}, home=self.HOME)
        self.assertEqual(path, Path("/cfg/voice-harness/config.toml"))


class UserConfigPrecedenceTests(unittest.TestCase):
    HOME = Path("/home/example")

    def test_backends_override_config_and_env_overrides_both(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.toml"
            backends_path = root / "backends.toml"
            config_path.write_text(
                "[providers.llm]\n"
                'provider = "local"\n'
                'model = "config-model"\n'
                "[audio]\n"
                "wake_threshold = 0.4\n"
            )
            backends_path.write_text('[llm]\nmodel = "backends-model"\n')

            from_files = user_config.load_user_config(
                {}, path=config_path, backends_path=backends_path, home=self.HOME
            )
            self.assertEqual(from_files.providers.llm_model, "backends-model")
            self.assertEqual(from_files.audio.wake_threshold, 0.4)

            with_env = user_config.load_user_config(
                {
                    "VOICE_HARNESS_LLM_MODEL": "env-model",
                    "VOICE_HARNESS_WAKE_THRESHOLD": "0.9",
                },
                path=config_path,
                backends_path=backends_path,
                home=self.HOME,
            )
            self.assertEqual(with_env.providers.llm_model, "env-model")
            self.assertEqual(with_env.audio.wake_threshold, 0.9)

    def test_backend_env_overrides_config_and_environment_overrides_legacy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.toml"
            backend_env = root / "backend.env"
            config_path.write_text(
                '[compute]\ndictation_backend = "parakeet"\n'
                'dictation_model = "config-model"\n'
            )
            backend_env.write_text(
                "DICTATION_BACKEND=whisper\nDICTATION_MODEL=legacy-model\n"
            )

            legacy = user_config.load_user_config(
                {},
                path=config_path,
                backends_path=root / "missing.toml",
                backend_env_path=backend_env,
                home=self.HOME,
            )
            overridden = user_config.load_user_config(
                {
                    "DICTATION_BACKEND": "parakeet",
                    "DICTATION_MODEL": "environment-model",
                },
                path=config_path,
                backends_path=root / "missing.toml",
                backend_env_path=backend_env,
                home=self.HOME,
            )

        self.assertEqual(legacy.compute.dictation_backend, "whisper")
        self.assertEqual(legacy.compute.dictation_model, "legacy-model")
        self.assertEqual(overridden.compute.dictation_backend, "parakeet")
        self.assertEqual(overridden.compute.dictation_model, "environment-model")

    def test_dictation_settings_use_typed_config_and_environment_overrides(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.toml"
            path.write_text(
                "[dictation]\n"
                'source = "config-source"\n'
                'inject = "paste"\n'
                'prompt = "config prompt"\n'
                'replacements = "spoken:written"\n'
                "vad_end_silence_ms = 750\n"
            )
            config = user_config.load_user_config(
                {
                    "DICTATION_INJECT": "stdout",
                    "DICTATION_PROMPT": "environment prompt",
                },
                path=path,
                backends_path=root / "missing.toml",
                backend_env_path=root / "missing.env",
                home=self.HOME,
            )

        self.assertEqual(config.dictation.source, "config-source")
        self.assertEqual(config.dictation.inject, "stdout")
        self.assertEqual(config.dictation.prompt, "environment prompt")
        self.assertEqual(config.dictation.replacements, (("spoken", "written"),))
        self.assertEqual(config.dictation.vad_end_silence_ms, 750)

    def test_empty_legacy_replacements_disable_static_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = user_config.load_user_config(
                {"DICTATION_REPLACEMENTS": ""},
                path=root / "missing.toml",
                backends_path=root / "missing-backends.toml",
                backend_env_path=root / "missing.env",
                home=self.HOME,
            )

        self.assertEqual(config.dictation.replacements, ())

    def test_config_file_overrides_defaults_for_non_provider_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.toml"
            config_path.write_text(
                "[integrations]\nzendesk = true\nlinear = true\n"
                '[compute]\ndictation_backend = "whisper"\n'
                "[platform]\nfocused_app_context = false\n"
            )

            config = user_config.load_user_config(
                {},
                path=config_path,
                backends_path=root / "missing.toml",
                home=self.HOME,
            )

        self.assertTrue(config.integrations.zendesk_enabled)
        self.assertTrue(config.integrations.linear_enabled)
        self.assertEqual(config.compute.dictation_backend, "whisper")
        self.assertFalse(config.platform.focused_app_context_enabled)

    def test_config_selects_provider_without_backends_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.toml"
            config_path.write_text('[providers.llm]\nprovider = "venice"\n')

            config = user_config.load_user_config(
                {},
                path=config_path,
                backends_path=root / "missing.toml",
                home=self.HOME,
            )

        self.assertEqual(config.providers.llm_provider, "venice")
        self.assertEqual(config.providers.llm_model, "venice-uncensored")

    def test_default_paths_resolve_from_xdg_config_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "voice-harness"
            config_dir.mkdir()
            (config_dir / "config.toml").write_text("[audio]\nwake_threshold = 0.7\n")

            config = user_config.load_user_config(
                {"XDG_CONFIG_HOME": str(root)}, home=self.HOME
            )

        self.assertEqual(config.audio.wake_threshold, 0.7)

    def test_running_snapshot_retains_values_and_restart_observes_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.toml"
            path.write_text('[providers.llm]\nmodel = "startup-model"\n')
            running = user_config.load_user_config(
                {},
                path=path,
                backends_path=root / "missing.toml",
                backend_env_path=root / "missing.env",
                home=self.HOME,
            )

            path.write_text('[providers.llm]\nmodel = "restart-model"\n')
            restarted = user_config.load_user_config(
                {},
                path=path,
                backends_path=root / "missing.toml",
                backend_env_path=root / "missing.env",
                home=self.HOME,
            )

        self.assertEqual(running.providers.llm_model, "startup-model")
        self.assertEqual(restarted.providers.llm_model, "restart-model")

    def test_integration_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = user_config.load_user_config(
                {"VOICE_HARNESS_INTEGRATION_ZENDESK": "on"},
                path=root / "config.toml",
                backends_path=root / "backends.toml",
                home=self.HOME,
            )
        self.assertTrue(config.integrations.zendesk_enabled)

    def test_cursor_watchdog_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = user_config.load_user_config(
                {
                    "VOICE_HARNESS_CURSOR_AGENT_INACTIVITY_SECONDS": "120",
                    "VOICE_HARNESS_CURSOR_AGENT_MAX_RUNTIME_SECONDS": "600",
                    "VOICE_HARNESS_AGENT_JOB_START_CONCURRENCY": "5",
                },
                path=root / "config.toml",
                backends_path=root / "backends.toml",
                home=self.HOME,
            )

        self.assertEqual(config.platform.cursor_agent_inactivity_seconds, 120)
        self.assertEqual(config.platform.cursor_agent_max_runtime_seconds, 600)
        self.assertEqual(config.platform.agent_job_start_concurrency, 5)


class UserConfigValidationTests(unittest.TestCase):
    HOME = Path("/home/example")

    def _load(self, body: str, environment: dict[str, str] | None = None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.toml"
            config_path.write_text(body)
            return user_config.load_user_config(
                environment or {},
                path=config_path,
                backends_path=root / "missing.toml",
                home=self.HOME,
            )

    def test_unknown_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "bogus"):
            self._load("[bogus]\nx = 1\n")

    def test_unknown_key_within_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "mystery"):
            self._load("[audio]\nmystery = 1\n")

    def test_invalid_provider_produces_actionable_error(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "local or venice"):
            self._load('[providers.llm]\nprovider = "hosted"\n')

    def test_non_finite_timeout_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "positive number"):
            self._load("", {"VOICE_HARNESS_LLM_TIMEOUT": "nan"})

    def test_out_of_range_tts_speed_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "between 0.25 and 4"):
            self._load("[providers.tts]\nspeed = 9\n")

    def test_wake_threshold_out_of_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "between 0.0 and 1.0"):
            self._load("[audio]\nwake_threshold = 5\n")

    def test_invalid_barge_in_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "off, vad, wake"):
            self._load('[audio]\nbarge_in_mode = "loud"\n')

    def test_empty_dictation_compute_values_are_rejected(self) -> None:
        for key in (
            "dictation_model",
            "dictation_quantization",
            "dictation_compute",
            "dictation_language",
        ):
            with (
                self.subTest(key=key),
                self.assertRaises(UserConfigurationError),
            ):
                self._load(f'[compute]\n{key} = ""\n')

    def test_empty_legacy_dictation_compute_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend_env = root / "backend.env"
            backend_env.write_text("DICTATION_MODEL=\n")

            with self.assertRaisesRegex(
                UserConfigurationError,
                "compute.dictation_model must not be empty",
            ):
                user_config.load_user_config(
                    {},
                    path=root / "missing.toml",
                    backends_path=root / "missing-backends.toml",
                    backend_env_path=backend_env,
                    home=self.HOME,
                )

    def test_invalid_playback_latency_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "us, ms, or s"):
            self._load('[audio]\nplayback_latency = "fast"\n')

    def test_non_positive_frame_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "positive integer"):
            self._load("[audio]\nbarge_in_speech_frames = 0\n")

    def test_non_positive_cursor_watchdog_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "positive number"):
            self._load("[platform]\ncursor_agent_inactivity_seconds = 0\n")

    def test_non_positive_agent_job_start_concurrency_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "positive integer"):
            self._load("[platform]\nagent_job_start_concurrency = 0\n")

    def test_file_based_credentials_are_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "credentials set"):
            self._load('[providers.venice]\napi_key = "secret"\n')

    def test_env_file_based_venice_credentials_are_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "credentials set"):
            self._load("", {"VOICE_HARNESS_VENICE_API_KEY_FILE": "/tmp/key"})

    def test_section_must_be_a_table(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "TOML table"):
            self._load("audio = 1\n")

    def test_malformed_config_names_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.toml"
            config_path.write_text("[audio\n")

            with self.assertRaisesRegex(
                UserConfigurationError,
                rf"user configuration {config_path}",
            ):
                user_config.load_user_config(
                    {},
                    path=config_path,
                    backends_path=root / "missing.toml",
                    home=self.HOME,
                )

    def test_malformed_backends_config_names_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backends_path = root / "backends.toml"
            backends_path.write_text("[llm\n")

            with self.assertRaisesRegex(
                UserConfigurationError,
                rf"backend configuration {backends_path}",
            ):
                user_config.load_user_config(
                    {},
                    path=root / "missing.toml",
                    backends_path=backends_path,
                    home=self.HOME,
                )

    def test_malformed_backend_environment_names_source_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend_env = root / "backend.env"
            backend_env.write_text("DICTATION_MODEL='unterminated\n")

            with self.assertRaisesRegex(
                UserConfigurationError,
                rf"{backend_env}:1: invalid environment assignment",
            ):
                user_config.load_user_config(
                    {},
                    path=root / "missing.toml",
                    backends_path=root / "missing-backends.toml",
                    backend_env_path=backend_env,
                    home=self.HOME,
                )

    def test_backend_environment_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend_env = root / "backend.env"
            backend_env.write_text("LD_PRELOAD=/tmp/attack.so\n")

            with self.assertRaisesRegex(
                UserConfigurationError,
                "unsupported backend environment key",
            ):
                user_config.load_user_config(
                    {},
                    path=root / "missing.toml",
                    backends_path=root / "missing-backends.toml",
                    backend_env_path=backend_env,
                    home=self.HOME,
                )

    def test_invalid_utf8_config_names_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.toml"
            config_path.write_bytes(b"\xff")

            with self.assertRaisesRegex(
                UserConfigurationError,
                rf"user configuration {config_path}",
            ):
                user_config.load_user_config(
                    {},
                    path=config_path,
                    backends_path=root / "missing.toml",
                    home=self.HOME,
                )

    def test_unreadable_config_names_source_path(self) -> None:
        config_path = Path("/config/config.toml")
        with (
            mock.patch.object(
                Path,
                "read_text",
                side_effect=PermissionError("permission denied"),
            ),
            self.assertRaisesRegex(
                UserConfigurationError,
                rf"user configuration {config_path}",
            ),
        ):
            user_config.load_user_config(
                {},
                path=config_path,
                backends_path=Path("/config/backends.toml"),
                home=self.HOME,
            )


class UserConfigWriteTests(unittest.TestCase):
    HOME = Path("/home/example")

    def test_atomic_write_round_trips_and_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = user_config.default_user_config(home=self.HOME)
            path = root / "nested" / "config.toml"

            user_config.write_user_config(config, path)

            self.assertTrue(path.exists())
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, 0o600)

            reloaded = user_config.load_user_config(
                {}, path=path, backends_path=root / "missing.toml", home=self.HOME
            )
            self.assertEqual(reloaded, config)

    def test_write_preserves_venice_providers_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = user_config.load_user_config(
                {
                    "VOICE_HARNESS_LLM_PROVIDER": "venice",
                    "VOICE_HARNESS_TTS_PROVIDER": "venice",
                },
                path=root / "config.toml",
                backends_path=root / "backends.toml",
                home=self.HOME,
            )
            path = root / "config.toml"
            user_config.write_user_config(source, path)
            text = path.read_text()

            self.assertNotIn("api_key", text)
            reloaded = user_config.load_user_config(
                {}, path=path, backends_path=root / "missing.toml", home=self.HOME
            )
            self.assertEqual(reloaded.providers.llm_provider, "venice")
            self.assertEqual(reloaded.providers.tts_provider, "venice")


class PlanApprovalPreferenceTests(unittest.TestCase):
    def test_default_is_ask_and_environment_override_selects_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "approval.json"
            selected = user_config.plan_approval_preferences_path(
                {"VOICE_HARNESS_PLAN_APPROVAL_FILE": str(path)}
            )

            self.assertEqual(selected, path)
            preferences = user_config.load_plan_approval_preferences(path)
            self.assertEqual(
                preferences.mode,
                user_config.PlanApprovalMode.ASK,
            )
            self.assertEqual(preferences.explicit_approval_count, 0)

    def test_explicit_approvals_are_idempotent_and_offer_once_at_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "approval.json"
            user_config.record_explicit_plan_approval("one", path=path)
            user_config.record_explicit_plan_approval("one", path=path)
            user_config.record_explicit_plan_approval("two", path=path)
            preferences = user_config.record_explicit_plan_approval(
                "three",
                path=path,
            )

            self.assertEqual(preferences.explicit_approval_count, 3)
            self.assertEqual(preferences.offer_pending_id, "three")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_offer_enables_auto_and_cli_reset_state_returns_to_ask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "approval.json"
            for approval_id in ("one", "two", "three"):
                user_config.record_explicit_plan_approval(approval_id, path=path)

            automatic = user_config.resolve_plan_approval_offer(
                "three",
                approved=True,
                path=path,
            )
            asking = user_config.set_plan_approval_mode(
                user_config.PlanApprovalMode.ASK,
                path=path,
            )

            self.assertEqual(automatic.mode, user_config.PlanApprovalMode.AUTO)
            self.assertTrue(automatic.offer_completed)
            self.assertEqual(asking.mode, user_config.PlanApprovalMode.ASK)
            self.assertEqual(asking.explicit_approval_count, 3)

    def test_stale_offer_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "approval.json"
            for approval_id in ("one", "two", "three"):
                user_config.record_explicit_plan_approval(approval_id, path=path)

            with self.assertRaisesRegex(
                UserConfigurationError,
                "no longer pending",
            ):
                user_config.resolve_plan_approval_offer(
                    "stale",
                    approved=True,
                    path=path,
                )


if __name__ == "__main__":
    unittest.main()
