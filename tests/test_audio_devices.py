from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import recorder
from local_voice_harness.diagnostics import checks
from local_voice_harness.diagnostics.model import Severity
from local_voice_harness.tts import client as tts_client
from local_voice_harness.tts import queue as tts_queue
from local_voice_harness.user_config import ComputeDevice, default_user_config
from tests.test_diagnostics import _WPCTL_STATUS, _snapshot
from tests.test_recorder import _paths


class SystemAudioDeviceTests(unittest.TestCase):
    def test_defaults_use_pipewire_system_devices(self) -> None:
        config = default_user_config()
        self.assertEqual(config.audio.source, "")
        self.assertEqual(config.audio.sink, "")

    def test_recorder_omits_target_for_system_default_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            process = mock.Mock(pid=os.getpid(), returncode=None)
            process.poll.return_value = None
            with (
                mock.patch.object(
                    recorder.subprocess, "Popen", return_value=process
                ) as popen,
                mock.patch.object(recorder, "_wait_for_recorder"),
            ):
                recorder.start_recording(paths, source="", ready=lambda: True)

        command = popen.call_args.args[0]
        self.assertEqual(command[0], "pw-record")
        self.assertNotIn("--target", command)

    def test_recorder_keeps_explicit_source_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            process = mock.Mock(pid=os.getpid(), returncode=None)
            process.poll.return_value = None
            with (
                mock.patch.object(
                    recorder.subprocess, "Popen", return_value=process
                ) as popen,
                mock.patch.object(recorder, "_wait_for_recorder"),
            ):
                recorder.start_recording(
                    paths, source="legacy-microphone", ready=lambda: True
                )

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--target") + 1], "legacy-microphone")

    def test_playback_uses_system_sink_unless_configured(self) -> None:
        default = default_user_config().audio
        explicit = replace(default, sink="hdmi-speakers")
        with mock.patch.object(tts_queue.subprocess, "Popen") as popen:
            tts_queue._open_playback(16000, default)
            tts_queue._open_playback(16000, explicit)

        default_command = popen.call_args_list[0].args[0]
        explicit_command = popen.call_args_list[1].args[0]
        self.assertNotIn("--target", default_command)
        self.assertEqual(
            explicit_command[explicit_command.index("--target") + 1], "hdmi-speakers"
        )

    def test_streaming_client_keeps_explicit_sink(self) -> None:
        client = tts_client.StreamingPlayback(
            "hello", replace(default_user_config().audio, sink="usb-dac")
        )
        with mock.patch.object(tts_client.subprocess, "Popen") as popen:
            client._open_playback(22050)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--target") + 1], "usb-dac")


class ProfileAwareDiagnosticTests(unittest.TestCase):
    def test_showcase_reports_venice_pipewire_and_cpu_dictation(self) -> None:
        config = default_user_config()
        snapshot = checks.DiagnosticSnapshot(
            config=replace(
                config,
                compute=replace(config.compute, dictation_device=ComputeDevice.CPU),
            ),
            registry=checks.build_integration_registry(config),
        )
        with (
            mock.patch.object(checks, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(
                checks,
                "_run",
                return_value=mock.Mock(returncode=0, stdout=_WPCTL_STATUS),
            ),
            mock.patch.object(checks, "get_venice_api_key", return_value="token"),
        ):
            backend = checks.check_backend_configuration(snapshot)
            pipewire = checks.check_pipewire_devices(snapshot)
            compute = checks.check_compute_modes(snapshot)
            cuda = checks.check_cuda(snapshot)
            models = checks.check_model_file(snapshot)

        self.assertEqual(backend[0].severity, Severity.OK)
        self.assertIn("venice", backend[0].detail)
        self.assertEqual(pipewire[0].severity, Severity.OK)
        self.assertTrue(
            any(
                "dictation configured compute=cpu" in result.detail
                for result in compute
            )
        )
        self.assertIn("CUDA tools were not invoked", cuda[0].detail)
        self.assertIn("not required by the Venice LLM", models[0].detail)

    def test_missing_venice_credentials_are_actionable(self) -> None:
        from local_voice_harness.credentials import CredentialError

        with mock.patch.object(
            checks,
            "get_venice_api_key",
            side_effect=CredentialError("Venice API key is not stored"),
        ):
            results = checks.check_backend_configuration(_snapshot())
        self.assertEqual(results[0].severity, Severity.FATAL)
        self.assertIn("not stored", results[0].detail)
