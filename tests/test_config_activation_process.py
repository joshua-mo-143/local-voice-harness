from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from local_voice_harness import config_activation
from local_voice_harness.config_management import ConfigChangeResult
from local_voice_harness.self_management import PendingConfigChange, SettingKey
from local_voice_harness.user_config import default_user_config

CHILD = Path(__file__).with_name("config_activation_process_child.py")
WAKE = "voice-harness-wake.service"


class ConfigActivationProcessTests(unittest.TestCase):
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

    def run_child(
        self,
        *arguments: object,
        crash: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        if crash:
            environment["CRASH_AFTER_RESTART"] = "1"
        return subprocess.run(
            [sys.executable, str(CHILD), *(str(value) for value in arguments)],
            env=environment,
            capture_output=True,
            text=True,
            check=check,
        )

    def ready(
        self,
        path: Path,
    ) -> tuple[config_activation.ActivationStore, config_activation.ActivationRecord]:
        store = config_activation.ActivationStore(path)
        offer = store.create_offer(
            self.pending,
            ConfigChangeResult(
                config=self.updated,
                changed_keys=("audio.voice",),
                restart_services=(WAKE,),
            ),
            expected_config=self.updated,
        )
        assert offer is not None
        store.mark_offer_delivered(offer.id)
        store.accept(offer.id)
        return store, store.mark_pre_restart_delivered(offer.id)

    def service_state(self, path: Path) -> None:
        snapshot = config_activation.ServiceSnapshot(
            installed=True,
            active_state="active",
            sub_state="running",
            invocation_id="invocation-1",
            process_start="process-1",
            config_digest=config_activation.config_digest(self.updated),
        )
        path.write_text(json.dumps({"snapshot": asdict(snapshot), "restarts": 0}))

    def test_empty_recovery_before_request_has_no_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "activation.json"
            result = self.run_child("current", journal)

        self.assertEqual(json.loads(result.stdout), {"record": None})

    def test_offer_and_pre_restart_delivery_survive_without_executing_request(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "activation.json"
            state = root / "services.json"
            self.service_state(state)
            store = config_activation.ActivationStore(journal)
            offer = store.create_offer(
                self.pending,
                ConfigChangeResult(
                    config=self.updated,
                    changed_keys=("audio.voice",),
                    restart_services=(WAKE,),
                ),
                expected_config=self.updated,
            )
            assert offer is not None

            offered_process = self.run_child("current", journal)
            store.mark_offer_delivered(offer.id)
            store.accept(offer.id)
            first_delivery = self.run_child("delivery", journal)
            second_delivery = self.run_child("delivery", journal)
            persisted = json.loads(state.read_text())

        self.assertEqual(json.loads(offered_process.stdout), {"record": offer.id})
        self.assertEqual(json.loads(first_delivery.stdout)["kind"], "pre_restart")
        self.assertEqual(json.loads(second_delivery.stdout)["id"], offer.id)
        self.assertEqual(persisted["restarts"], 0)

    def test_process_crash_during_restart_reconciles_without_duplicate_effect(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "activation.json"
            state = root / "services.json"
            store, ready = self.ready(journal)
            self.service_state(state)

            crashed = self.run_child(
                "execute",
                journal,
                state,
                crash=True,
                check=False,
            )
            self.assertEqual(crashed.returncode, 91)
            during = store.current()
            assert during is not None
            self.assertEqual(
                during.status,
                config_activation.ActivationStatus.RESTARTING,
            )

            recovered = self.run_child("execute", journal, state)
            persisted = json.loads(state.read_text())

        self.assertEqual(json.loads(recovered.stdout), {"status": "succeeded"})
        self.assertEqual(persisted["restarts"], 1)
        self.assertEqual(ready.id, during.id)

    def test_post_restart_result_survives_process_exit_until_acknowledgement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "activation.json"
            state = root / "services.json"
            _store, ready = self.ready(journal)
            self.service_state(state)

            completed = self.run_child("execute", journal, state)
            self.assertEqual(json.loads(completed.stdout), {"status": "succeeded"})
            first = self.run_child("delivery", journal)
            second = self.run_child("delivery", journal)
            self.run_child("ack", journal, ready.id)
            after_ack = self.run_child("delivery", journal)
            persisted = json.loads(state.read_text())

        self.assertEqual(json.loads(first.stdout)["kind"], "result")
        self.assertEqual(json.loads(second.stdout)["id"], ready.id)
        self.assertEqual(json.loads(after_ack.stdout), {"kind": None, "id": None})
        self.assertEqual(persisted["restarts"], 1)


if __name__ == "__main__":
    unittest.main()
