from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import config_management
from local_voice_harness.user_config import UserConfigurationError


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

    def test_set_writes_owner_only_config_and_syncs_backend_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)

            result = config_management.commit_config_change(
                {
                    "audio.wake_threshold": "0.42",
                    "compute.dictation_backend": "whisper",
                },
                paths=paths,
            )

            self.assertEqual(result.config.audio.wake_threshold, 0.42)
            self.assertEqual(result.config.compute.dictation_backend, "whisper")
            self.assertEqual(stat.S_IMODE(paths.config.stat().st_mode), 0o600)
            self.assertIn("DICTATION_BACKEND=whisper", paths.backend_env.read_text())

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

    def test_setup_defaults_is_non_interactive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.paths(root)

            result = config_management.run_setup(
                defaults_only=True,
                paths=paths,
                print_fn=lambda _message: None,
            )

            self.assertEqual(result.config.providers.llm_provider, "local")
            self.assertTrue(result.config.integrations.github_enabled)
            self.assertFalse(result.config.integrations.zendesk_enabled)


if __name__ == "__main__":
    unittest.main()
