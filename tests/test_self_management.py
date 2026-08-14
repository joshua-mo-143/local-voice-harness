from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import config_management, self_management
from local_voice_harness.user_config import default_user_config, write_user_config


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
            (
                "What announcement policy is configured?",
                self_management.SettingKey.ANNOUNCEMENT_MODE,
                "all",
            ),
            (
                "What is the speaking speed?",
                self_management.SettingKey.TTS_SPEED,
                "1.25",
            ),
            (
                "What TTS speed is configured?",
                self_management.SettingKey.TTS_SPEED,
                "1.25",
            ),
            (
                "What speech speed are you using?",
                self_management.SettingKey.TTS_SPEED,
                "1.25",
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

    def test_tts_speed_inspection_speaks_the_stored_value(self) -> None:
        response = self_management.inspect_config_utterance(
            "What is the current speaking speed?",
            self.config,
        )

        self.assertEqual(response.spoken_text, "Speaking speed is 1.25.")
        self.assertEqual(response.display_text, "providers.tts.speed: 1.25")


class ConfigChangeTests(unittest.TestCase):
    HOME = Path("/home/example")

    def setUp(self) -> None:
        config = default_user_config(home=self.HOME)
        self.config = replace(
            config,
            audio=replace(config.audio, voice="old_voice", barge_in_mode="wake"),
            integrations=replace(
                config.integrations,
                github_enabled=True,
                linear_enabled=True,
                zendesk_enabled=False,
            ),
        )

    def paths(self, root: Path) -> config_management.ConfigPaths:
        return config_management.ConfigPaths(
            config=root / "voice-harness" / "config.toml",
            backends=root / "voice-harness" / "backends.toml",
            backend_env=root / "dictation" / "backend.env",
            home=self.HOME,
        )

    def prepare(
        self,
        utterance: str,
        raw_value: str | None,
        *,
        active_config=None,
    ) -> self_management.ChangePreparation:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.paths(Path(temporary))
            write_user_config(self.config, paths.config)
            return self_management.prepare_config_change(
                self_management.ConfigChangeRequest(utterance, raw_value),
                active_config or self.config,
                paths=paths,
            )

    def test_initial_allow_list_preflights_exact_old_and_new_values(self) -> None:
        cases = (
            ("Set the voice to new_voice", "new_voice", "old_voice", "new_voice"),
            ("Set barge-in mode to off", "off", "wake", "off"),
            (
                "Set the announcement policy to quiet",
                "quiet",
                "all",
                "quiet",
            ),
            ("Disable GitHub", "disabled", True, False),
            ("Disable Linear", "false", True, False),
            ("Enable Zendesk", "enabled", False, True),
        )
        for utterance, raw_value, old_value, new_value in cases:
            with self.subTest(utterance=utterance):
                preparation = self.prepare(utterance, raw_value)

                self.assertEqual(
                    preparation.status,
                    self_management.ChangePreparationStatus.READY,
                )
                pending = preparation.pending
                assert pending is not None
                self.assertEqual(pending.old_value, old_value)
                self.assertEqual(pending.new_value, new_value)
                self.assertEqual(
                    pending.affected_services,
                    ("voice-harness-wake.service",),
                )
                response = self_management.render_change_preparation(preparation)
                old_text = (
                    "enabled"
                    if old_value is True
                    else "disabled"
                    if old_value is False
                    else str(old_value)
                )
                new_text = (
                    "enabled"
                    if new_value is True
                    else "disabled"
                    if new_value is False
                    else str(new_value)
                )
                self.assertIn(old_text, response.spoken_text)
                self.assertIn(new_text, response.spoken_text)

    def test_tts_speed_change_preflights_exact_old_and_new_values(self) -> None:
        cases = (
            ("Set the speaking speed to 1.2", "1.2", "1.25", "1.2"),
            ("Set TTS speed to 2", "2", "1.25", "2"),
            ("Set speech speed to 0.25", "0.25", "1.25", "0.25"),
            ("Set speed to 4", "4", "1.25", "4"),
        )
        for utterance, raw_value, old_value, new_value in cases:
            with self.subTest(utterance=utterance):
                preparation = self.prepare(utterance, raw_value)

                self.assertEqual(
                    preparation.status,
                    self_management.ChangePreparationStatus.READY,
                )
                pending = preparation.pending
                assert pending is not None
                self.assertEqual(pending.setting, self_management.SettingKey.TTS_SPEED)
                self.assertEqual(pending.old_value, old_value)
                self.assertEqual(pending.new_value, new_value)
                self.assertEqual(
                    pending.affected_services,
                    (
                        "voice-harness-wake.service",
                        "voice-harness-tts.service",
                    ),
                )
                response = self_management.render_change_preparation(preparation)
                self.assertIn(old_value, response.spoken_text)
                self.assertIn(new_value, response.spoken_text)
                self.assertIn("Say yes to confirm", response.spoken_text)

    def test_malformed_unsupported_ambiguous_and_unsafe_changes_fail_closed(
        self,
    ) -> None:
        cases = (
            (
                "Set the wake threshold to point seven",
                "0.7",
                self_management.ChangePreparationStatus.UNSUPPORTED,
            ),
            (
                "Set voice and Linear to something",
                "something",
                self_management.ChangePreparationStatus.AMBIGUOUS,
            ),
            (
                "Change the voice",
                None,
                self_management.ChangePreparationStatus.MALFORMED,
            ),
            (
                "Set the voice to the API key credential",
                "secret",
                self_management.ChangePreparationStatus.MALFORMED,
            ),
            (
                "Set the voice executable path to bash",
                "/bin/bash",
                self_management.ChangePreparationStatus.MALFORMED,
            ),
            (
                "Set the voice to slash",
                "../voice",
                self_management.ChangePreparationStatus.INVALID,
            ),
            (
                "Set barge-in mode to loud",
                "loud",
                self_management.ChangePreparationStatus.INVALID,
            ),
            (
                "Set the speaking speed to 5",
                "5",
                self_management.ChangePreparationStatus.INVALID,
            ),
            (
                "Set TTS speed to 0.1",
                "0.1",
                self_management.ChangePreparationStatus.INVALID,
            ),
            (
                "Set speed to faster",
                "faster",
                self_management.ChangePreparationStatus.INVALID,
            ),
            (
                "Change the speaking speed",
                None,
                self_management.ChangePreparationStatus.MALFORMED,
            ),
            (
                "Set voice and speaking speed to 1.2",
                "1.2",
                self_management.ChangePreparationStatus.AMBIGUOUS,
            ),
            (
                "Speak faster",
                "faster",
                self_management.ChangePreparationStatus.UNSUPPORTED,
            ),
        )
        for utterance, raw_value, status in cases:
            with self.subTest(utterance=utterance):
                preparation = self.prepare(utterance, raw_value)
                self.assertEqual(preparation.status, status)
                self.assertIsNone(preparation.pending)

    def test_active_and_stored_snapshot_mismatch_fails_closed(self) -> None:
        active = replace(
            self.config,
            audio=replace(self.config.audio, voice="environment_override"),
        )

        preparation = self.prepare(
            "Set the voice to new_voice",
            "new_voice",
            active_config=active,
        )

        self.assertEqual(
            preparation.status,
            self_management.ChangePreparationStatus.CONFLICT,
        )
        self.assertIsNone(preparation.pending)

    def test_strict_confirmation_parser_rejects_partial_or_combined_answers(
        self,
    ) -> None:
        for utterance in ("yes", "confirm", "go ahead", "save it"):
            with self.subTest(utterance=utterance):
                self.assertEqual(
                    self_management.resolve_confirmation(utterance),
                    self_management.ConfirmationDecision.CONFIRM,
                )
        for utterance in ("no", "cancel", "never mind", "do not"):
            with self.subTest(utterance=utterance):
                self.assertEqual(
                    self_management.resolve_confirmation(utterance),
                    self_management.ConfirmationDecision.CANCEL,
                )
        for utterance in ("maybe", "yes but change Linear", "the ticket says yes"):
            with self.subTest(utterance=utterance):
                self.assertEqual(
                    self_management.resolve_confirmation(utterance),
                    self_management.ConfirmationDecision.AMBIGUOUS,
                )

    def test_commit_preserves_runtime_snapshot_and_reports_manual_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.paths(Path(temporary))
            write_user_config(self.config, paths.config)
            preparation = self_management.prepare_config_change(
                self_management.ConfigChangeRequest(
                    "Set the voice to new_voice",
                    "new_voice",
                ),
                self.config,
                paths=paths,
            )
            pending = preparation.pending
            assert pending is not None
            with mock.patch.object(
                config_management,
                "active_services",
                return_value=("voice-harness-wake.service",),
            ):
                result = self_management.commit_pending_change(pending, paths=paths)

            stored = config_management.load_managed_config(paths)

        self.assertEqual(self.config.audio.voice, "old_voice")
        self.assertEqual(stored.audio.voice, "new_voice")
        response = self_management.render_change_committed(pending, result)
        self.assertIn(
            "running configuration snapshot is unchanged", response.spoken_text
        )
        self.assertIn(
            "Restart voice-harness-wake.service manually", response.spoken_text
        )
        self.assertIn("Active affected services", response.spoken_text)

    def test_tts_speed_commit_preserves_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.paths(Path(temporary))
            write_user_config(self.config, paths.config)
            preparation = self_management.prepare_config_change(
                self_management.ConfigChangeRequest(
                    "Set the speaking speed to 1.2",
                    "1.2",
                ),
                self.config,
                paths=paths,
            )
            pending = preparation.pending
            assert pending is not None
            with mock.patch.object(
                config_management,
                "active_services",
                return_value=(
                    "voice-harness-wake.service",
                    "voice-harness-tts.service",
                ),
            ):
                result = self_management.commit_pending_change(pending, paths=paths)

            stored = config_management.load_managed_config(paths)

        self.assertEqual(self.config.providers.tts_speed, 1.25)
        self.assertEqual(stored.providers.tts_speed, 1.2)
        response = self_management.render_change_committed(pending, result)
        self.assertIn(
            "running configuration snapshot is unchanged", response.spoken_text
        )
        self.assertIn("providers.tts.speed", response.spoken_text)
        self.assertIn("1.2", response.spoken_text)

    def test_tts_speed_commit_fences_the_typed_stored_float(self) -> None:
        precise = 1.23456789
        self.assertNotEqual(float(format(precise, "g")), precise)
        config = replace(
            self.config,
            providers=replace(self.config.providers, tts_speed=precise),
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.paths(Path(temporary))
            write_user_config(config, paths.config)
            preparation = self_management.prepare_config_change(
                self_management.ConfigChangeRequest(
                    "Set the speaking speed to 1.2",
                    "1.2",
                ),
                config,
                paths=paths,
            )
            pending = preparation.pending
            assert pending is not None
            self.assertEqual(pending.stored_value, precise)
            result = self_management.commit_pending_change(pending, paths=paths)
            stored = config_management.load_managed_config(paths)

        self.assertEqual(stored.providers.tts_speed, 1.2)
        self.assertEqual(result.changed_keys, ("providers.tts.speed",))


if __name__ == "__main__":
    unittest.main()
