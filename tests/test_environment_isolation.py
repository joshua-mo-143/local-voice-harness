from __future__ import annotations

import os
import unittest
from pathlib import Path

from local_voice_harness.user_config import load_user_config


class IsolatedTestEnvironmentTests(unittest.TestCase):
    def test_default_tests_use_empty_home_and_xdg_paths(self) -> None:
        home = Path(os.environ["HOME"])
        config_home = Path(os.environ["XDG_CONFIG_HOME"])
        self.assertEqual(home, Path.home())
        self.assertTrue(home.is_dir())
        self.assertTrue(config_home.is_dir())
        self.assertIn("isolated-home", str(home))
        self.assertFalse((config_home / "voice-harness" / "config.toml").exists())
        self.assertIsNone(os.environ.get("STATE_DIRECTORY"))
        self.assertFalse(any(key.startswith("VOICE_HARNESS_") for key in os.environ))
        self.assertFalse(any(key.startswith("DICTATION_") for key in os.environ))

        config = load_user_config()

        self.assertEqual(config.providers.llm_provider, "venice")
        self.assertEqual(config.providers.tts_provider, "venice")
        self.assertTrue(config.integrations.github_enabled)
        self.assertFalse(config.integrations.zendesk_enabled)
        self.assertFalse(config.integrations.linear_enabled)

    def test_isolated_config_files_are_visible_and_do_not_need_ambient_env(
        self,
    ) -> None:
        config_dir = Path(os.environ["XDG_CONFIG_HOME"]) / "voice-harness"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[providers.llm]\nprovider = "local"\n[providers.tts]\nprovider = "local"\n'
        )

        config = load_user_config()

        self.assertEqual(config.providers.llm_provider, "local")
        self.assertEqual(config.providers.tts_provider, "local")

    def test_credential_lookup_is_fail_closed_in_the_isolated_path(self) -> None:
        from local_voice_harness.credentials import (
            CredentialError,
            get_venice_api_key,
            secret_service_available,
            secret_service_binary_available,
        )

        self.assertTrue(secret_service_binary_available())
        self.assertFalse(secret_service_available())
        with self.assertRaises(CredentialError):
            get_venice_api_key()


if __name__ == "__main__":
    unittest.main()
