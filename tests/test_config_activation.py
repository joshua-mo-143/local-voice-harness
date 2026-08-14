from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

from local_voice_harness import config_activation
from local_voice_harness.config_management import ConfigChangeResult
from local_voice_harness.self_management import PendingConfigChange, SettingKey
from local_voice_harness.user_config import default_user_config

WAKE = "voice-harness-wake.service"
TTS = "voice-harness-tts.service"


class ControlledServices:
    def __init__(
        self,
        snapshots: dict[str, config_activation.ServiceSnapshot],
        *,
        fail: set[str] | None = None,
        crash_after: set[str] | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.fail = fail or set()
        self.crash_after = crash_after or set()
        self.restarts: list[str] = []

    def snapshot(self, service: str) -> config_activation.ServiceSnapshot:
        return self.snapshots[service]

    def restart(self, service: str) -> subprocess.CompletedProcess[str]:
        self.restarts.append(service)
        before = self.snapshots[service]
        if service not in self.fail:
            self.snapshots[service] = replace(
                before,
                invocation_id=f"{before.invocation_id}-next",
                process_start=f"{before.process_start}-next",
            )
        if service in self.crash_after:
            raise SystemExit("simulated process crash")
        return subprocess.CompletedProcess(
            ["systemctl", "--user", "restart", service],
            1 if service in self.fail else 0,
            "",
            "failed" if service in self.fail else "",
        )


class ConfigActivationTests(unittest.TestCase):
    def setUp(self) -> None:
        original = default_user_config(home=Path("/home/example"))
        self.original = original
        self.updated = replace(
            original,
            audio=replace(original.audio, voice="new_voice"),
        )
        self.pending = PendingConfigChange(
            trusted_utterance="Set voice to new_voice",
            setting=SettingKey.VOICE,
            raw_value="new_voice",
            old_value=original.audio.voice,
            new_value="new_voice",
            stored_value=original.audio.voice,
            affected_services=(WAKE,),
        )
        self.result = ConfigChangeResult(
            config=self.updated,
            changed_keys=("audio.voice",),
            restart_services=(WAKE,),
        )

    def store(self, root: Path) -> config_activation.ActivationStore:
        return config_activation.ActivationStore(root / "activation.json")

    def ready(
        self,
        store: config_activation.ActivationStore,
        *,
        pending: PendingConfigChange | None = None,
        result: ConfigChangeResult | None = None,
    ) -> config_activation.ActivationRecord:
        offer = store.create_offer(
            pending or self.pending,
            result or self.result,
            expected_config=self.updated,
        )
        assert offer is not None
        store.mark_offer_delivered(offer.id)
        accepted = store.accept(offer.id)
        return store.mark_pre_restart_delivered(accepted.id)

    def active_snapshot(
        self,
        service: str,
        *,
        digest: str | None = None,
        voice: str = "",
    ) -> config_activation.ServiceSnapshot:
        return config_activation.ServiceSnapshot(
            installed=True,
            active_state="active",
            sub_state="running",
            invocation_id=f"{service}-invocation",
            process_start=f"{service}-process",
            config_digest=digest or config_activation.config_digest(self.updated),
            voice=voice,
        )

    def execute(
        self,
        record_id: str,
        **kwargs: Any,
    ) -> config_activation.ActivationRecord:
        return config_activation.execute_activation(
            record_id,
            voice_validator=lambda _voice: config_activation.VoiceValidationResult(
                True,
                "accepted",
            ),
            restore_voice=lambda _record: self.original,
            **kwargs,
        )

    def test_offer_targets_exact_reported_allowlisted_services(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            offer = store.create_offer(
                self.pending,
                self.result,
                expected_config=self.updated,
            )

            assert offer is not None
            self.assertEqual(offer.targets, (WAKE,))
            self.assertEqual(offer.status, config_activation.ActivationStatus.OFFERED)

            store.acknowledge(
                store.decline(
                    store.mark_offer_delivered(offer.id).id,
                ).id
            )
            unsafe = replace(self.result, restart_services=("herdr",))
            with self.assertRaises(config_activation.ActivationStateError):
                store.create_offer(self.pending, unsafe)

    def test_acceptance_and_pre_restart_delivery_do_not_execute_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            controller = ControlledServices({WAKE: self.active_snapshot(WAKE)})
            ready = self.ready(store)

            self.assertEqual(ready.status, config_activation.ActivationStatus.READY)
            self.assertEqual(controller.restarts, [])

            final = self.execute(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.SUCCEEDED)
        self.assertEqual(controller.restarts, [WAKE])

    def test_wake_activation_requires_distinct_exact_confirmation(self) -> None:
        for utterance in ("yes", "confirm", "go ahead", "activate it"):
            with self.subTest(utterance=utterance):
                self.assertEqual(
                    config_activation.resolve_activation_decision(utterance),
                    config_activation.ActivationDecision.NONE,
                )
        self.assertEqual(
            config_activation.resolve_activation_decision("activate now"),
            config_activation.ActivationDecision.ACTIVATE,
        )
        self.assertEqual(
            config_activation.resolve_activation_decision("not now"),
            config_activation.ActivationDecision.DECLINE,
        )

    def test_success_is_idempotent_and_requires_new_identity_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices({WAKE: self.active_snapshot(WAKE)})

            first = self.execute(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )
            second = self.execute(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )

        self.assertEqual(first.status, config_activation.ActivationStatus.SUCCEEDED)
        self.assertEqual(second.status, config_activation.ActivationStatus.SUCCEEDED)
        self.assertEqual(controller.restarts, [WAKE])

    def test_unrelated_edit_after_voice_load_preserves_success_and_edit(self) -> None:
        state = {"config": self.updated}

        class ConcurrentEditServices(ControlledServices):
            def restart(self, service: str) -> subprocess.CompletedProcess[str]:
                result = super().restart(service)
                self.snapshots[service] = replace(
                    self.snapshots[service],
                    voice="new_voice",
                )
                current = state["config"]
                state["config"] = replace(
                    current,
                    audio=replace(current.audio, barge_in_mode="off"),
                )
                return result

        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ConcurrentEditServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        voice=str(self.pending.old_value),
                    )
                }
            )
            restore = mock.Mock()

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=lambda _voice: config_activation.VoiceValidationResult(
                    True, "accepted"
                ),
                restore_voice=restore,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.SUCCEEDED)
        self.assertEqual(state["config"].audio.voice, "new_voice")
        self.assertEqual(state["config"].audio.barge_in_mode, "off")
        self.assertEqual(controller.restarts, [WAKE])
        after = final.outcomes[0].after
        assert after is not None
        self.assertEqual(after.voice, "new_voice")
        restore.assert_not_called()

    def test_unrelated_edit_during_restart_uses_effective_voice_snapshot(self) -> None:
        state = {"config": self.updated}

        class MergedConfigServices(ControlledServices):
            def restart(self, service: str) -> subprocess.CompletedProcess[str]:
                result = super().restart(service)
                current = state["config"]
                state["config"] = replace(
                    current,
                    audio=replace(current.audio, barge_in_mode="off"),
                )
                self.snapshots[service] = replace(
                    self.snapshots[service],
                    config_digest=config_activation.config_digest(state["config"]),
                    voice="new_voice",
                )
                return result

        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = MergedConfigServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        voice=str(self.pending.old_value),
                    )
                }
            )

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=lambda _voice: config_activation.VoiceValidationResult(
                    True, "accepted"
                ),
                restore_voice=mock.Mock(),
                observation_timeout=0,
            )

        merged_digest = config_activation.config_digest(state["config"])
        self.assertEqual(final.status, config_activation.ActivationStatus.SUCCEEDED)
        self.assertEqual(
            final.expected_config_digest,
            config_activation.config_digest(self.updated),
        )
        self.assertNotEqual(final.expected_config_digest, merged_digest)
        after = final.outcomes[0].after
        assert after is not None
        self.assertEqual(after.config_digest, merged_digest)
        self.assertEqual(state["config"].audio.barge_in_mode, "off")

    def test_concurrent_voice_replacement_after_load_is_not_activated_or_reverted(
        self,
    ) -> None:
        state = {"config": self.updated}

        class VoiceReplacementServices(ControlledServices):
            def restart(self, service: str) -> subprocess.CompletedProcess[str]:
                result = super().restart(service)
                self.snapshots[service] = replace(
                    self.snapshots[service],
                    voice="new_voice",
                )
                current = state["config"]
                state["config"] = replace(
                    current,
                    audio=replace(current.audio, voice="replacement_voice"),
                )
                return result

        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = VoiceReplacementServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        voice=str(self.pending.old_value),
                    )
                }
            )
            restore = mock.Mock()

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=lambda _voice: config_activation.VoiceValidationResult(
                    True, "accepted"
                ),
                restore_voice=restore,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        self.assertEqual(state["config"].audio.voice, "replacement_voice")
        self.assertEqual(controller.restarts, [WAKE])
        self.assertFalse(final.rollback_config_restored)
        self.assertIn("stored voice ownership was lost", final.detail)
        restore.assert_not_called()

    def test_provider_rejected_voice_restores_previous_config_without_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            previous_digest = config_activation.config_digest(self.original)
            controller = ControlledServices(
                {WAKE: self.active_snapshot(WAKE, digest=previous_digest)}
            )
            state = {"config": self.updated}
            validations: list[str] = []
            restorations: list[str] = []

            def reject(voice: str) -> config_activation.VoiceValidationResult:
                validations.append(voice)
                return config_activation.VoiceValidationResult(
                    False,
                    "provider rejected unknown voice",
                )

            def restore(
                record: config_activation.ActivationRecord,
            ):
                restorations.append(str(record.old_value))
                state["config"] = self.original
                return self.original

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=reject,
                restore_voice=restore,
                observation_timeout=0,
            )
            reopened = config_activation.ActivationStore(store.path).current()

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        assert reopened is not None
        self.assertEqual(reopened.rollback_config_digest, final.rollback_config_digest)
        self.assertEqual(validations, ["new_voice"])
        self.assertEqual(restorations, [str(self.pending.old_value)])
        self.assertEqual(state["config"].audio.voice, self.pending.old_value)
        self.assertEqual(controller.restarts, [])
        self.assertIn("not usable", final.detail)
        self.assertIn("service snapshot were restored", final.detail)
        response = config_activation.render_activation_delivery(
            config_activation.ActivationDelivery(
                final,
                config_activation.ActivationDeliveryKind.RESULT,
            )
        )
        self.assertIn("Voice 'new_voice' was not usable", response.display_text)
        self.assertEqual(response.spoken_text, response.display_text)

    def test_transient_voice_validation_failure_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        digest=config_activation.config_digest(self.original),
                    )
                }
            )
            state = {"config": self.updated}
            validator = mock.Mock(side_effect=TimeoutError("provider timed out"))

            def restore(_record: config_activation.ActivationRecord):
                state["config"] = self.original
                return self.original

            first = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=validator,
                restore_voice=restore,
                observation_timeout=0,
            )
            second = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=validator,
                restore_voice=restore,
                observation_timeout=0,
            )

        self.assertEqual(first.status, config_activation.ActivationStatus.FAILED)
        self.assertEqual(second.status, config_activation.ActivationStatus.FAILED)
        validator.assert_called_once_with("new_voice")
        self.assertEqual(controller.restarts, [])

    def test_validation_success_lost_before_journal_write_rolls_back_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        digest=config_activation.config_digest(self.original),
                    )
                }
            )
            state = {"config": self.updated}
            validator = mock.Mock(
                return_value=config_activation.VoiceValidationResult(
                    True,
                    "provider accepted candidate",
                )
            )

            with (
                mock.patch.object(
                    store,
                    "mark_voice_validated",
                    side_effect=SystemExit("crashed before journal write"),
                ),
                self.assertRaisesRegex(SystemExit, "before journal write"),
            ):
                config_activation.execute_activation(
                    ready.id,
                    store=store,
                    controller=controller,
                    load_config=lambda: state["config"],
                    voice_validator=validator,
                    restore_voice=mock.Mock(),
                    observation_timeout=0,
                )
            reopened = config_activation.ActivationStore(store.path)
            during = reopened.current()

            def restore(_record: config_activation.ActivationRecord):
                state["config"] = self.original
                return self.original

            recovered = config_activation.execute_activation(
                ready.id,
                store=reopened,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=validator,
                restore_voice=restore,
                observation_timeout=0,
            )

        assert during is not None
        self.assertEqual(during.status, config_activation.ActivationStatus.VALIDATING)
        self.assertTrue(during.voice_validation_attempted)
        self.assertFalse(during.voice_validated)
        self.assertEqual(recovered.status, config_activation.ActivationStatus.FAILED)
        validator.assert_called_once_with("new_voice")
        self.assertEqual(state["config"].audio.voice, self.pending.old_value)
        self.assertEqual(controller.restarts, [])
        self.assertIn("result was not durably recorded", recovered.detail)
        self.assertIn("Validation was not repeated", recovered.detail)

    def test_validation_call_crash_is_ambiguous_and_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        digest=config_activation.config_digest(self.original),
                    )
                }
            )
            state = {"config": self.updated}
            crashing_validator = mock.Mock(
                side_effect=SystemExit("provider outcome unknown")
            )

            with self.assertRaisesRegex(SystemExit, "outcome unknown"):
                config_activation.execute_activation(
                    ready.id,
                    store=store,
                    controller=controller,
                    load_config=lambda: state["config"],
                    voice_validator=crashing_validator,
                    restore_voice=mock.Mock(),
                    observation_timeout=0,
                )
            retry_validator = mock.Mock(
                side_effect=AssertionError("provider validation repeated")
            )

            def restore(_record: config_activation.ActivationRecord):
                state["config"] = self.original
                return self.original

            recovered = config_activation.execute_activation(
                ready.id,
                store=config_activation.ActivationStore(store.path),
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=retry_validator,
                restore_voice=restore,
                observation_timeout=0,
            )

        crashing_validator.assert_called_once_with("new_voice")
        retry_validator.assert_not_called()
        self.assertEqual(recovered.status, config_activation.ActivationStatus.FAILED)
        self.assertEqual(state["config"].audio.voice, self.pending.old_value)
        self.assertEqual(controller.restarts, [])
        self.assertIn("result was not durably recorded", recovered.detail)

    def test_ambiguous_validation_recovery_rolls_back_without_revalidation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            validating = store.begin_voice_validation(ready.id)
            controller = ControlledServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        digest=config_activation.config_digest(self.original),
                    )
                }
            )
            state = {"config": self.updated}
            validator = mock.Mock()

            def restore(_record: config_activation.ActivationRecord):
                state["config"] = self.original
                return self.original

            final = config_activation.execute_activation(
                validating.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=validator,
                restore_voice=restore,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        validator.assert_not_called()
        self.assertIn("result was not durably recorded", final.detail)
        self.assertEqual(state["config"].audio.voice, self.pending.old_value)

    def test_restart_failure_restores_previous_voice_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        digest=config_activation.config_digest(self.original),
                    )
                },
                fail={WAKE},
            )
            state = {"config": self.updated}

            def restore(_record: config_activation.ActivationRecord):
                state["config"] = self.original
                return self.original

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=lambda _voice: config_activation.VoiceValidationResult(
                    True, "accepted"
                ),
                restore_voice=restore,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        self.assertEqual(controller.restarts, [WAKE])
        self.assertEqual(state["config"].audio.voice, self.pending.old_value)
        self.assertEqual(
            controller.snapshot(WAKE).config_digest,
            config_activation.config_digest(self.original),
        )
        self.assertIn("service snapshot were restored", final.detail)

    def test_inactive_or_uninstalled_service_is_never_started(self) -> None:
        cases = (
            replace(self.active_snapshot(WAKE), installed=False),
            replace(self.active_snapshot(WAKE), active_state="inactive"),
        )
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                with tempfile.TemporaryDirectory() as temporary:
                    store = self.store(Path(temporary))
                    ready = self.ready(store)
                    controller = ControlledServices({WAKE: snapshot})

                    final = self.execute(
                        ready.id,
                        store=store,
                        controller=controller,
                        load_config=lambda: self.updated,
                        observation_timeout=0,
                    )

                self.assertEqual(
                    final.status,
                    config_activation.ActivationStatus.FAILED,
                )
                self.assertEqual(controller.restarts, [])

    def test_partial_result_remains_visible(self) -> None:
        pending = replace(
            self.pending,
            setting=SettingKey.BARGE_IN_MODE,
            affected_services=(WAKE, TTS),
        )
        result = replace(self.result, restart_services=(WAKE, TTS))
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store, pending=pending, result=result)
            controller = ControlledServices(
                {
                    WAKE: self.active_snapshot(WAKE),
                    TTS: self.active_snapshot(TTS),
                },
                fail={TTS},
            )

            final = self.execute(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )
            delivery = config_activation.ActivationDelivery(
                final,
                config_activation.ActivationDeliveryKind.RESULT,
            )
            response = config_activation.render_activation_delivery(delivery)

        self.assertEqual(final.status, config_activation.ActivationStatus.PARTIAL)
        self.assertIn(TTS, final.detail)
        self.assertIn("not repeat", response.spoken_text)

    def test_crash_after_restart_reconciles_without_repeating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices(
                {WAKE: self.active_snapshot(WAKE)},
                crash_after={WAKE},
            )

            with self.assertRaisesRegex(SystemExit, "simulated process crash"):
                self.execute(
                    ready.id,
                    store=store,
                    controller=controller,
                    load_config=lambda: self.updated,
                    observation_timeout=0,
                )
            controller.crash_after.clear()
            recovered = self.execute(
                ready.id,
                store=config_activation.ActivationStore(store.path),
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )

        self.assertEqual(
            recovered.status,
            config_activation.ActivationStatus.SUCCEEDED,
        )
        self.assertEqual(controller.restarts, [WAKE])

    def test_restarting_digest_mismatch_resumes_into_voice_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            validating = store.begin_voice_validation(ready.id)
            validated = store.mark_voice_validated(validating.id)
            restarting = store.begin_restart(
                validated.id,
                (
                    config_activation.ServiceOutcome(
                        WAKE,
                        attempted=True,
                        before=self.active_snapshot(WAKE),
                    ),
                ),
            )
            controller = ControlledServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        digest=config_activation.config_digest(self.original),
                    )
                }
            )
            validator = mock.Mock()
            restore = mock.Mock()

            recovered = config_activation.execute_activation(
                restarting.id,
                store=store,
                controller=controller,
                load_config=lambda: self.original,
                voice_validator=validator,
                restore_voice=restore,
                observation_timeout=0,
            )

        self.assertEqual(recovered.status, config_activation.ActivationStatus.FAILED)
        self.assertTrue(recovered.rollback_config_restored)
        self.assertTrue(
            all(
                outcome.status == config_activation.ServiceOutcomeStatus.SUCCEEDED
                for outcome in recovered.rollback_outcomes
            )
        )
        self.assertEqual(controller.restarts, [])
        validator.assert_not_called()
        restore.assert_not_called()
        self.assertIn("service snapshot were restored", recovered.detail)

    def test_crash_before_restart_call_is_failed_without_blind_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(
                store,
                pending=replace(
                    self.pending,
                    setting=SettingKey.BARGE_IN_MODE,
                ),
            )
            before = self.active_snapshot(WAKE)
            restarting = store.begin_restart(
                ready.id,
                (
                    config_activation.ServiceOutcome(
                        WAKE,
                        attempted=True,
                        before=before,
                    ),
                ),
            )
            controller = ControlledServices({WAKE: before})

            recovered = self.execute(
                restarting.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )

        self.assertEqual(recovered.status, config_activation.ActivationStatus.FAILED)
        self.assertEqual(controller.restarts, [])
        self.assertIn("not observed", recovered.outcomes[0].detail)

    def test_expected_stored_config_mismatch_blocks_all_restarts(self) -> None:
        different = replace(
            self.updated,
            audio=replace(self.updated.audio, voice="intervening"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices({WAKE: self.active_snapshot(WAKE)})

            final = self.execute(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: different,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        self.assertEqual(controller.restarts, [])
        self.assertIn("stored voice changed", final.detail)

    def test_voice_rollback_preserves_unrelated_concurrent_config_edit(self) -> None:
        concurrent = replace(
            self.updated,
            audio=replace(self.updated.audio, barge_in_mode="off"),
        )
        state = {"config": concurrent}

        class RestoringServices(ControlledServices):
            def restart(self, service: str) -> subprocess.CompletedProcess[str]:
                result = super().restart(service)
                self.snapshots[service] = replace(
                    self.snapshots[service],
                    config_digest=config_activation.config_digest(state["config"]),
                )
                return result

        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = RestoringServices(
                {
                    WAKE: self.active_snapshot(
                        WAKE,
                        digest=config_activation.config_digest(self.original),
                    )
                }
            )

            def restore(record: config_activation.ActivationRecord):
                current = state["config"]
                state["config"] = replace(
                    current,
                    audio=replace(current.audio, voice=str(record.old_value)),
                )
                return state["config"]

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: state["config"],
                voice_validator=mock.Mock(
                    side_effect=AssertionError("validation must not repeat")
                ),
                restore_voice=restore,
                observation_timeout=0,
            )
            reopened = config_activation.ActivationStore(store.path).current()

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        assert reopened is not None
        self.assertEqual(reopened.rollback_config_digest, final.rollback_config_digest)
        self.assertEqual(state["config"].audio.voice, self.pending.old_value)
        self.assertEqual(state["config"].audio.barge_in_mode, "off")
        self.assertEqual(
            final.rollback_config_digest,
            config_activation.config_digest(state["config"]),
        )
        self.assertEqual(controller.restarts, [WAKE])
        self.assertIn("service snapshot were restored", final.detail)

    def test_voice_rollback_does_not_overwrite_genuinely_replaced_voice(self) -> None:
        intervening = replace(
            self.updated,
            audio=replace(
                self.updated.audio,
                voice="replacement_voice",
                barge_in_mode="off",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices({WAKE: self.active_snapshot(WAKE)})
            restore = mock.Mock()

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: intervening,
                voice_validator=mock.Mock(
                    side_effect=AssertionError("validation must not repeat")
                ),
                restore_voice=restore,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        self.assertFalse(final.rollback_config_restored)
        self.assertEqual(intervening.audio.voice, "replacement_voice")
        self.assertEqual(intervening.audio.barge_in_mode, "off")
        restore.assert_not_called()
        self.assertEqual(controller.restarts, [])
        self.assertIn("replacement voice was not overwritten", final.detail)

    def test_result_delivery_survives_store_reopen_until_acknowledged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices({WAKE: self.active_snapshot(WAKE)})
            final = self.execute(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )

            reopened = config_activation.ActivationStore(store.path)
            first = reopened.next_delivery()
            second = config_activation.ActivationStore(store.path).next_delivery()
            assert first is not None and second is not None
            reopened.acknowledge(final.id)

            self.assertEqual(
                first.kind, config_activation.ActivationDeliveryKind.RESULT
            )
            self.assertEqual(second.record.id, final.id)
            self.assertIsNone(
                config_activation.ActivationStore(store.path).next_delivery()
            )

    def test_systemd_controller_uses_exact_user_restart_without_herdr(self) -> None:
        controller = config_activation.SystemdUserServiceController()
        completed = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(
            config_activation.subprocess,
            "run",
            return_value=completed,
        ) as run:
            controller.restart(WAKE)

        run.assert_called_once_with(
            ["systemctl", "--user", "try-restart", WAKE],
            capture_output=True,
            text=True,
            check=False,
        )
        with self.assertRaises(config_activation.ActivationStateError):
            controller.restart("herdr")

    def test_voice_snapshot_requires_a_valid_effective_config_digest(self) -> None:
        before = self.active_snapshot(
            WAKE,
            digest=config_activation.config_digest(self.original),
            voice=str(self.pending.old_value),
        )

        class InvalidMarkerServices(ControlledServices):
            def restart(self, service: str) -> subprocess.CompletedProcess[str]:
                result = super().restart(service)
                self.snapshots[service] = replace(
                    self.snapshots[service],
                    config_digest="",
                    voice="new_voice",
                )
                return result

        controller = InvalidMarkerServices({WAKE: before})
        controller.restart(WAKE)
        observed = config_activation._observe_restart(
            controller,
            config_activation.ServiceOutcome(WAKE, before=before, attempted=True),
            config_activation.config_digest(self.updated),
            expected_voice="new_voice",
            timeout=0,
        )

        self.assertEqual(
            observed.status,
            config_activation.ServiceOutcomeStatus.FAILED,
        )
        self.assertIn("configuration or voice snapshot", observed.detail)

    def test_systemd_controller_ignores_snapshot_from_stale_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            config_activation.publish_service_snapshot(
                WAKE,
                self.updated,
                pid=42,
                process_start="start-42",
                state_dir=state_dir,
            )
            completed = subprocess.CompletedProcess(
                [],
                0,
                (
                    "LoadState=loaded\n"
                    "ActiveState=active\n"
                    "SubState=running\n"
                    "InvocationID=current\n"
                    "MainPID=43\n"
                ),
                "",
            )
            controller = config_activation.SystemdUserServiceController(
                state_dir=state_dir
            )
            with mock.patch.object(
                config_activation.subprocess,
                "run",
                return_value=completed,
            ):
                snapshot = controller.snapshot(WAKE)

        self.assertEqual(snapshot.process_start, "")
        self.assertEqual(snapshot.config_digest, "")

    def test_worker_uses_transient_scope_without_install_or_service_expansion(
        self,
    ) -> None:
        record_id = "a" * 32
        with mock.patch.object(config_activation.subprocess, "Popen") as popen:
            config_activation.launch_activation_worker(record_id)

        command = popen.call_args.args[0]
        self.assertEqual(command[:4], ["systemd-run", "--user", "--scope", "--collect"])
        self.assertIn("local_voice_harness.config_activation", command)
        self.assertNotIn("install", command)
        self.assertNotIn("uninstall", command)
        self.assertFalse(
            any(
                argument.casefold() == "herdr" or "include-herdr" in argument.casefold()
                for argument in command
            )
        )

    def test_publish_snapshot_records_process_identity_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_activation.publish_service_snapshot(
                WAKE,
                self.updated,
                pid=42,
                process_start="start-42",
                state_dir=root,
            )
            value = json.loads(
                config_activation.service_snapshot_path(
                    WAKE,
                    state_dir=root,
                ).read_text()
            )

        self.assertEqual(value["pid"], 42)
        self.assertEqual(value["process_start"], "start-42")
        self.assertEqual(value["voice"], "new_voice")
        self.assertEqual(
            value["config_digest"],
            config_activation.config_digest(self.updated),
        )


if __name__ == "__main__":
    unittest.main()
