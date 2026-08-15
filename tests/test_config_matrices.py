from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness import components, llm_launcher
from local_voice_harness.credentials import CredentialError
from local_voice_harness.diagnostics import checks
from local_voice_harness.integrations.registry import (
    build_integration_registry,
    enabled_integrations,
    integration_enabled,
)
from local_voice_harness.user_config import load_user_config


def _write_user_config(text: str) -> None:
    config_dir = Path(os.environ["XDG_CONFIG_HOME"]) / "voice-harness"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(text)


def _snapshot():
    config = load_user_config()
    return checks.DiagnosticSnapshot(
        config=config,
        registry=checks.build_integration_registry(config),
    )


class ConfiguredBackendMatrixTests(unittest.TestCase):
    def test_hosted_defaults_skip_local_llm_and_require_credentials(self) -> None:
        config = load_user_config()
        self.assertEqual(config.providers.llm_provider, "venice")
        self.assertEqual(config.providers.tts_provider, "venice")
        with self.assertRaisesRegex(RuntimeError, "providers.llm.provider=local"):
            llm_launcher.command(config)

        names = {
            result.name for result in checks.check_required_executables(_snapshot())
        }
        self.assertNotIn("executable:llama-server", names)

        supervisor = mock.Mock()
        with (
            mock.patch.object(components, "user_services", return_value=supervisor),
            mock.patch.object(
                components, "get_venice_api_key", side_effect=CredentialError("missing")
            ),
        ):
            with self.assertRaises(CredentialError):
                components.start_components(config.providers, timeout=0.01)
            supervisor.start.assert_not_called()

        with (
            mock.patch.object(components, "user_services", return_value=supervisor),
            mock.patch.object(components, "get_venice_api_key", return_value="token"),
            mock.patch.object(components, "socket_ready", return_value=True),
        ):
            components.start_components(config.providers, timeout=0.01)

        supervisor.start.assert_called_once_with("voice-harness-tts.service")

    def test_local_backends_start_llm_and_do_not_touch_venice_credentials(
        self,
    ) -> None:
        _write_user_config(
            '[providers.llm]\nprovider = "local"\n[providers.tts]\nprovider = "local"\n'
        )
        config = load_user_config()
        self.assertEqual(config.providers.llm_provider, "local")
        self.assertEqual(config.providers.tts_provider, "local")
        command = llm_launcher.command(config, cuda_available=False)
        self.assertIn("--n-gpu-layers", command)
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "0")

        names = {
            result.name for result in checks.check_required_executables(_snapshot())
        }
        self.assertIn("executable:llama-server", names)

        supervisor = mock.Mock()
        with (
            mock.patch.object(components, "user_services", return_value=supervisor),
            mock.patch.object(components, "get_venice_api_key") as credentials,
            mock.patch.object(components, "llm_ready", return_value=True),
            mock.patch.object(components, "socket_ready", return_value=True),
        ):
            components.start_components(config.providers, timeout=0.01)

        credentials.assert_not_called()
        supervisor.start.assert_called_once_with(
            "voice-harness-llm.service", "voice-harness-tts.service"
        )


class ConfiguredIntegrationMatrixTests(unittest.TestCase):
    def test_default_integrations_enable_only_github(self) -> None:
        registry = build_integration_registry(load_user_config())
        self.assertEqual(len(enabled_integrations(registry)), 1)
        self.assertTrue(integration_enabled("github", registry))
        self.assertFalse(integration_enabled("zendesk", registry))
        self.assertFalse(integration_enabled("linear", registry))

    def test_enabled_integration_matrix_constructs_each_provider(self) -> None:
        _write_user_config(
            "[integrations]\ngithub = true\nzendesk = true\nlinear = true\n"
        )
        config = load_user_config()
        self.assertTrue(config.integrations.github_enabled)
        self.assertTrue(config.integrations.zendesk_enabled)
        self.assertTrue(config.integrations.linear_enabled)
        registry = build_integration_registry(config)
        self.assertEqual(len(enabled_integrations(registry)), 3)
        self.assertTrue(integration_enabled("github", registry))
        self.assertTrue(integration_enabled("zendesk", registry))
        self.assertTrue(integration_enabled("linear", registry))


if __name__ == "__main__":
    unittest.main()
