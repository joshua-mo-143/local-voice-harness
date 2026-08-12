from __future__ import annotations

import json
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import config_management
from local_voice_harness.user_config import DictationDevice, UserConfigurationError


class ConfigManagementTests(unittest.TestCase):
    HOME = Path("/home/example")

    def paths(self, root: Path) -> config_management.ConfigPaths:
        return config_management.ConfigPaths(
            config=root / "voice-harness" / "config.toml",
            backends=root / "voice-harness" / "backends.toml",
            backend_env=root / "dictation" / "backend.env",
            home=self.HOME,
        )

    def test_set_validates_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            paths.config.parent.mkdir(parents=True)

            with self.assertRaisesRegex(UserConfigurationError, "between 0.0 and 1.0"):
                config_management.commit_config_change(
                    {"audio.wake_threshold": "5"},
                    paths=paths,
                )

            self.assertFalse(paths.config.exists())

    def test_set_writes_owner_only_config_without_touching_legacy_backend_env(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            paths.backend_env.parent.mkdir(parents=True)
            legacy = "DICTATION_MODEL=legacy-model\n"
            paths.backend_env.write_text(legacy)

            result = config_management.commit_config_change(
                {
                    "audio.wake_threshold": "0.42",
                    "compute.dictation_backend": "whisper",
                    "compute.dictation_device": "cpu",
                },
                paths=paths,
            )

            self.assertEqual(result.config.audio.wake_threshold, 0.42)
            self.assertEqual(result.config.compute.dictation_backend, "whisper")
            self.assertIs(result.config.compute.dictation_device, DictationDevice.CPU)
            self.assertEqual(stat.S_IMODE(paths.config.stat().st_mode), 0o600)
            self.assertEqual(paths.backend_env.read_text(), legacy)

    def test_reset_does_not_create_or_update_legacy_backend_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            config_management.commit_config_change(
                {"compute.dictation_backend": "whisper"},
                paths=paths,
            )
            self.assertFalse(paths.backend_env.exists())

            paths.backend_env.parent.mkdir(parents=True)
            legacy = "DICTATION_BACKEND=whisper\n"
            paths.backend_env.write_text(legacy)
            config_management.reset_config(section="compute", paths=paths)

            self.assertEqual(paths.backend_env.read_text(), legacy)

    def test_show_single_key_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            config_management.commit_config_change(
                {"integrations.zendesk": "true"},
                paths=paths,
            )

            self.assertEqual(
                config_management.show_config(key="integrations.zendesk", paths=paths),
                "true",
            )
            payload = json.loads(
                config_management.show_config(
                    key="integrations.zendesk",
                    json_output=True,
                    paths=paths,
                )
            )
            self.assertTrue(payload["integrations.zendesk"])

    def test_reset_section_restores_defaults_without_touching_other_sections(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            config_management.commit_config_change(
                {
                    "audio.wake_threshold": "0.2",
                    "integrations.linear": "true",
                },
                paths=paths,
            )

            result = config_management.reset_config(section="audio", paths=paths)

            config = config_management.load_managed_config(paths)
            self.assertEqual(config.audio.wake_threshold, 0.55)
            self.assertTrue(config.integrations.linear_enabled)
            self.assertIn("audio.wake_threshold", result.changed_keys)

    def test_commit_config_change_reports_only_active_services(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            with mock.patch.object(
                config_management,
                "active_services",
                return_value=("voice-harness-wake.service",),
            ) as active:
                result = config_management.commit_config_change(
                    {
                        "audio.wake_threshold": "0.42",
                        "compute.dictation_backend": "whisper",
                    },
                    paths=paths,
                )

            active.assert_called_once()
            self.assertEqual(result.restart_services, ("voice-harness-wake.service",))
            notice = config_management.format_restart_notice(result.restart_services)
            self.assertIn("voice-harness-wake.service", notice)
            self.assertNotIn("dictation.service", notice)

    def test_expected_value_fences_stale_confirmed_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            config_management.commit_config_change(
                {"audio.voice": "original"},
                paths=paths,
            )
            config_management.commit_config_change(
                {"audio.voice": "intervening"},
                paths=paths,
            )

            with self.assertRaisesRegex(
                config_management.StaleConfigChangeError,
                "changed after confirmation",
            ):
                config_management.commit_config_change(
                    {"audio.voice": "confirmed"},
                    paths=paths,
                    expected_values={"audio.voice": "original"},
                )

            current = config_management.load_managed_config(paths)
            self.assertEqual(current.audio.voice, "intervening")

    def test_switching_to_venice_still_stops_an_active_local_llm(self) -> None:
        config = config_management.default_user_config()
        config = replace(
            config,
            providers=replace(config.providers, llm_provider="venice"),
        )

        services = config_management.restart_services_for_keys(
            config, ["providers.llm.provider"]
        )

        self.assertIn("voice-harness-llm.service", services)
        notice = config_management.format_restart_notice(services)
        self.assertIn("active on-demand services restart on their next use", notice)

    def test_integration_doctor_inspects_enabled_integrations_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            config_management.commit_config_change(
                {
                    "integrations.github": "true",
                    "integrations.linear": "true",
                    "integrations.zendesk": "false",
                },
                paths=paths,
            )
            with mock.patch.object(
                config_management,
                "capability_statuses",
                return_value=(
                    ("linear", mock.Mock(available=True, detail="ok", suggestion=None)),
                ),
            ) as statuses:
                diagnostics = config_management.integration_diagnostics(
                    config_management.load_managed_config(paths).integrations
                )

            statuses.assert_called_once()
            names = {item.name for item in diagnostics}
            self.assertIn("linear", names)
            self.assertIn("github", names)
            self.assertNotIn("zendesk", names)

    def test_config_paths_honor_xdg_config_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = config_management.config_paths(
                {"XDG_CONFIG_HOME": str(root / "cfg")},
                home=self.HOME,
            )

        self.assertEqual(
            paths.config,
            root / "cfg" / "voice-harness" / "config.toml",
        )
        self.assertEqual(
            paths.backend_env,
            root / "cfg" / "dictation" / "backend.env",
        )

    def test_setup_defaults_is_non_interactive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)

            result = config_management.run_setup(
                defaults_only=True,
                paths=paths,
                print_fn=lambda _message: None,
            )

            self.assertEqual(result.config.providers.llm_provider, "venice")
            self.assertEqual(result.config.providers.tts_provider, "venice")
            self.assertTrue(result.config.integrations.github_enabled)
            self.assertFalse(result.config.integrations.zendesk_enabled)

    def test_interactive_setup_prefers_venice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            with (
                mock.patch.object(
                    config_management,
                    "_available_llm_providers",
                    return_value=("local", "venice"),
                ),
                mock.patch.object(
                    config_management,
                    "_available_tts_providers",
                    return_value=("local", "venice"),
                ),
            ):
                result = config_management.run_setup(
                    paths=paths,
                    input_fn=lambda _prompt: "",
                    print_fn=lambda _message: None,
                )

            self.assertEqual(result.config.providers.llm_provider, "venice")
            self.assertEqual(result.config.providers.tts_provider, "venice")

    def test_showcase_profile_is_non_interactive_and_uses_venice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)

            result = config_management.run_setup(
                profile="showcase",
                paths=paths,
                input_fn=mock.Mock(side_effect=AssertionError("unexpected prompt")),
                print_fn=lambda _message: None,
            )

            self.assertEqual(result.config.providers.llm_provider, "venice")
            self.assertEqual(result.config.providers.tts_provider, "venice")
            self.assertEqual(result.config.compute.dictation_backend, "parakeet")
            self.assertTrue(result.config.integrations.github_enabled)
            self.assertFalse(result.config.integrations.zendesk_enabled)
            self.assertFalse(result.config.integrations.linear_enabled)

    def test_showcase_profile_migrates_legacy_provider_config_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            paths.backends.parent.mkdir(parents=True)
            paths.backends.write_text(
                "[llm]\n"
                'provider = "local"\n'
                'model = "legacy-model"\n'
                "\n"
                "[tts]\n"
                'provider = "local"\n'
            )
            output: list[str] = []

            first = config_management.run_setup(
                profile="showcase",
                paths=paths,
                print_fn=output.append,
            )
            second = config_management.run_setup(
                profile="showcase",
                paths=paths,
                print_fn=lambda _message: None,
            )

            backup = paths.backends.with_name("backends.toml.migrated")
            self.assertFalse(paths.backends.exists())
            self.assertTrue(backup.is_file())
            self.assertEqual(first.legacy_backup, backup)
            self.assertIsNone(second.legacy_backup)
            self.assertEqual(first.config.providers.llm_provider, "venice")
            self.assertEqual(first.config.providers.tts_provider, "venice")
            self.assertEqual(first.config.providers.llm_model, "legacy-model")
            self.assertEqual(
                config_management.load_managed_config(paths).providers.llm_provider,
                "venice",
            )
            self.assertTrue(any(str(backup) in message for message in output))

    def test_showcase_profile_restores_legacy_config_when_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)
            paths.backends.parent.mkdir(parents=True)
            legacy = '[llm]\nprovider = "local"\n'
            paths.backends.write_text(legacy)

            with (
                mock.patch.object(
                    config_management,
                    "validate_managed_config",
                    side_effect=OSError("write failed"),
                ),
                self.assertRaisesRegex(OSError, "write failed"),
            ):
                config_management.run_setup(
                    profile="showcase",
                    paths=paths,
                    print_fn=lambda _message: None,
                )

            self.assertEqual(paths.backends.read_text(), legacy)
            self.assertFalse(
                paths.backends.with_name("backends.toml.migrated").exists()
            )


if __name__ == "__main__":
    unittest.main()
