from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from local_voice_harness import self_management
from local_voice_harness.user_config import default_user_config


class ConfigInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        config = default_user_config(home=Path("/home/example"))
        self.config = replace(
            config,
            audio=replace(config.audio, voice="af_sky", barge_in_mode="vad"),
            integrations=replace(
                config.integrations,
                github_enabled=True,
                linear_enabled=True,
                zendesk_enabled=False,
            ),
        )

    def inspect(self, utterance: str) -> self_management.InspectionResult:
        return self_management.inspect_config(
            self_management.SelfManagementRequest(
                self_management.SelfManagementIntent.INSPECT_CONFIG,
                utterance,
            ),
            self.config,
        )

    def test_supported_aliases_read_typed_snapshot_values(self) -> None:
        cases = (
            (
                "What voice are you using?",
                self_management.SettingKey.VOICE,
                "af_sky",
            ),
            (
                "Which interruption mode is configured?",
                self_management.SettingKey.BARGE_IN_MODE,
                "vad",
            ),
            (
                "Is Git Hub enabled?",
                self_management.SettingKey.GITHUB,
                True,
            ),
            (
                "Is Linear enabled?",
                self_management.SettingKey.LINEAR,
                True,
            ),
            (
                "Do I have Zendesk enabled?",
                self_management.SettingKey.ZENDESK,
                False,
            ),
        )
        for utterance, setting, value in cases:
            with self.subTest(utterance=utterance):
                result = self.inspect(utterance)
                self.assertEqual(result.status, self_management.InspectionStatus.OK)
                self.assertEqual(result.setting, setting)
                self.assertEqual(result.value, value)

    def test_ambiguous_aliases_are_rejected_without_guessing(self) -> None:
        result = self.inspect("What voice and Linear settings are active?")

        self.assertEqual(result.status, self_management.InspectionStatus.AMBIGUOUS)
        self.assertIsNone(result.setting)
        response = self_management.render_inspection_result(result)
        self.assertEqual(
            response.spoken_text,
            self_management.AMBIGUOUS_INSPECTION_RESPONSE,
        )

    def test_unknown_sensitive_path_and_executable_requests_are_unsupported(
        self,
    ) -> None:
        for utterance in (
            "What is my Venice API credential?",
            "What project path are you using?",
            "Which Git executable do you run?",
            "What is the wake threshold?",
        ):
            with self.subTest(utterance=utterance):
                result = self.inspect(utterance)
                self.assertEqual(
                    result.status,
                    self_management.InspectionStatus.UNSUPPORTED,
                )
                self.assertEqual(
                    self_management.render_inspection_result(result).spoken_text,
                    self_management.UNSUPPORTED_INSPECTION_RESPONSE,
                )

    def test_rendering_is_bounded_and_uses_separate_response_channels(self) -> None:
        config = replace(
            self.config,
            audio=replace(self.config.audio, voice="v" * 200),
        )

        response = self_management.inspect_config_utterance(
            "What is the assistant voice?",
            config,
        )

        self.assertLess(len(response.spoken_text), 120)
        self.assertTrue(response.spoken_text.endswith("…."))
        self.assertTrue(response.display_text.startswith("audio.voice: "))

    def test_empty_voice_is_rendered_as_the_configured_default(self) -> None:
        config = replace(
            self.config,
            audio=replace(self.config.audio, voice=""),
        )

        response = self_management.inspect_config_utterance(
            "What voice are you using?",
            config,
        )

        self.assertEqual(
            response.spoken_text,
            "My configured voice is the configured default.",
        )


if __name__ == "__main__":
    unittest.main()
