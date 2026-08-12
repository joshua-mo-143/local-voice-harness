from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
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
    ) -> config_activation.ServiceSnapshot:
        return config_activation.ServiceSnapshot(
            installed=True,
            active_state="active",
            sub_state="running",
            invocation_id=f"{service}-invocation",
            process_start=f"{service}-process",
            config_digest=digest or config_activation.config_digest(self.updated),
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

            final = config_activation.execute_activation(
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

            first = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )
            second = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: self.updated,
                observation_timeout=0,
            )

        self.assertEqual(first.status, config_activation.ActivationStatus.SUCCEEDED)
        self.assertEqual(second.status, config_activation.ActivationStatus.SUCCEEDED)
        self.assertEqual(controller.restarts, [WAKE])

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

                    final = config_activation.execute_activation(
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
        pending = replace(self.pending, affected_services=(WAKE, TTS))
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

            final = config_activation.execute_activation(
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
                config_activation.execute_activation(
                    ready.id,
                    store=store,
                    controller=controller,
                    load_config=lambda: self.updated,
                    observation_timeout=0,
                )
            controller.crash_after.clear()
            recovered = config_activation.execute_activation(
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

    def test_crash_before_restart_call_is_failed_without_blind_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
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

            recovered = config_activation.execute_activation(
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

            final = config_activation.execute_activation(
                ready.id,
                store=store,
                controller=controller,
                load_config=lambda: different,
                observation_timeout=0,
            )

        self.assertEqual(final.status, config_activation.ActivationStatus.FAILED)
        self.assertEqual(controller.restarts, [])
        self.assertIn("stored configuration changed", final.detail)

    def test_result_delivery_survives_store_reopen_until_acknowledged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self.store(Path(temporary))
            ready = self.ready(store)
            controller = ControlledServices({WAKE: self.active_snapshot(WAKE)})
            final = config_activation.execute_activation(
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
        self.assertEqual(
            value["config_digest"],
            config_activation.config_digest(self.updated),
        )


if __name__ == "__main__":
    unittest.main()
