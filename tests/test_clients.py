from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.stt import client as stt_client
from local_voice_harness.stt import server as stt_server
from local_voice_harness.tts import client as tts_client
from local_voice_harness.user_config import DictationDevice
from tests.support import join_threads


class SpeechToTextClientTests(unittest.TestCase):
    def test_normalize_replaces_whole_terms_case_insensitively(self) -> None:
        with mock.patch.dict(
            stt_server.REPLACEMENTS, {"cursa": "Cursor", "herder": "Herdr"}, clear=True
        ):
            result = stt_server.normalize("Cursa and herder, not cursative")

        self.assertEqual(result, "Cursor and Herdr, not cursative")

    def test_resolve_language_maps_aliases_and_auto_detect(self) -> None:
        self.assertEqual(stt_server.resolve_language("en"), "en")
        self.assertEqual(stt_server.resolve_language("ZH"), "zh")
        self.assertEqual(stt_server.resolve_language("English"), "en")
        self.assertEqual(stt_server.resolve_language(" chinese "), "zh")
        self.assertIsNone(stt_server.resolve_language(""))
        self.assertIsNone(stt_server.resolve_language("auto"))

    def test_resolve_backend_accepts_supported_values(self) -> None:
        self.assertEqual(stt_server.resolve_backend("parakeet"), "parakeet")
        self.assertEqual(stt_server.resolve_backend(" whisper "), "whisper")

    def test_resolve_backend_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported DICTATION_BACKEND"):
            stt_server.resolve_backend("invalid")

    def test_resolve_model_name_maps_legacy_whisper_default_to_parakeet(self) -> None:
        self.assertEqual(
            stt_server.resolve_model_name("parakeet", "large-v3-turbo"),
            stt_server.PARAKEET_DEFAULT_MODEL,
        )

    def test_resolve_model_name_honors_explicit_parakeet_model(self) -> None:
        self.assertEqual(
            stt_server.resolve_model_name("parakeet", "nemo-parakeet-tdt-0.6b-v3"),
            "nemo-parakeet-tdt-0.6b-v3",
        )

    def test_resolve_quantization_defaults_to_int8_for_parakeet(self) -> None:
        self.assertEqual(stt_server.resolve_quantization("parakeet"), "int8")

    def test_resolve_quantization_can_be_disabled(self) -> None:
        self.assertIsNone(stt_server.resolve_quantization("parakeet", "none"))

    def test_load_transcriber_builds_parakeet_backend(self) -> None:
        instance = mock.Mock()
        instance.transcribe.return_value = " hello "
        settings = stt_server.STTRuntimeSettings(
            backend="parakeet",
            device=DictationDevice.AUTO,
            model_name="nemo-parakeet-tdt-0.6b-v2",
            quantization="int8",
            compute_type="float16",
            language=None,
            prompt="prompt",
            replacements={},
        )
        with mock.patch.object(
            stt_server, "ParakeetTranscriber", return_value=instance
        ) as parakeet_cls:
            transcriber = stt_server.load_transcriber(settings)

        parakeet_cls.assert_called_once_with(
            "nemo-parakeet-tdt-0.6b-v2",
            device=DictationDevice.AUTO,
            quantization="int8",
        )
        self.assertEqual(transcriber.transcribe("/tmp/audio.wav"), " hello ")

    def test_parakeet_cpu_uses_only_cpu_provider_without_cuda_discovery(self) -> None:
        onnx_asr = mock.Mock()
        onnxruntime = mock.Mock()
        with mock.patch.dict(
            sys.modules,
            {"onnx_asr": onnx_asr, "onnxruntime": onnxruntime},
        ):
            stt_server.ParakeetTranscriber(
                "model",
                device=DictationDevice.CPU,
                quantization="int8",
            )

        onnxruntime.get_available_providers.assert_not_called()
        self.assertEqual(
            onnx_asr.load_model.call_args.kwargs["providers"],
            ["CPUExecutionProvider"],
        )

    def test_parakeet_explicit_cuda_fails_when_provider_is_unavailable(self) -> None:
        onnx_asr = mock.Mock()
        onnxruntime = mock.Mock()
        onnxruntime.get_available_providers.return_value = ["CPUExecutionProvider"]
        with (
            mock.patch.dict(
                sys.modules,
                {"onnx_asr": onnx_asr, "onnxruntime": onnxruntime},
            ),
            self.assertRaisesRegex(RuntimeError, "CUDA dictation was requested"),
        ):
            stt_server.ParakeetTranscriber(
                "model",
                device=DictationDevice.CUDA,
                quantization="int8",
            )
        onnx_asr.load_model.assert_not_called()

    def test_parakeet_auto_falls_back_to_cpu_provider(self) -> None:
        onnxruntime = mock.Mock()
        onnxruntime.get_available_providers.return_value = ["CPUExecutionProvider"]
        with mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime}):
            device, providers = stt_server._parakeet_device(DictationDevice.AUTO)

        self.assertEqual(device, "cpu")
        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_parakeet_explicit_cuda_disables_cpu_fallback(self) -> None:
        onnxruntime = mock.Mock()
        onnxruntime.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        with mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime}):
            device, providers = stt_server._parakeet_device(DictationDevice.CUDA)

        self.assertEqual(device, "cuda")
        self.assertEqual(providers, ["CUDAExecutionProvider"])

    def test_parakeet_auto_keeps_cpu_fallback_after_cuda(self) -> None:
        onnxruntime = mock.Mock()
        onnxruntime.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        with mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime}):
            device, providers = stt_server._parakeet_device(DictationDevice.AUTO)

        self.assertEqual(device, "cuda")
        self.assertEqual(
            providers,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_whisper_cpu_avoids_cuda_probe_and_uses_cpu_compute(self) -> None:
        faster_whisper = mock.Mock()
        ctranslate2 = mock.Mock()
        with mock.patch.dict(
            sys.modules,
            {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2},
        ):
            stt_server.WhisperTranscriber(
                "model",
                device=DictationDevice.CPU,
                compute_type="float16",
                language=None,
                prompt="prompt",
            )

        ctranslate2.get_cuda_device_count.assert_not_called()
        faster_whisper.WhisperModel.assert_called_once_with(
            "model", device="cpu", compute_type="int8"
        )

    def test_whisper_explicit_cuda_fails_clearly_when_unavailable(self) -> None:
        ctranslate2 = mock.Mock()
        ctranslate2.get_cuda_device_count.return_value = 0
        with (
            mock.patch.dict(sys.modules, {"ctranslate2": ctranslate2}),
            self.assertRaisesRegex(RuntimeError, "CUDA dictation was requested"),
        ):
            stt_server._whisper_device(DictationDevice.CUDA)

    def test_whisper_auto_falls_back_to_cpu(self) -> None:
        ctranslate2 = mock.Mock()
        ctranslate2.get_cuda_device_count.return_value = 0
        with mock.patch.dict(sys.modules, {"ctranslate2": ctranslate2}):
            device = stt_server._whisper_device(DictationDevice.AUTO)

        self.assertEqual(device, "cpu")

    def test_transcribe_sends_path_and_returns_trimmed_text(self) -> None:
        audio = Path("/tmp/request.wav")
        output = io.StringIO()
        with (
            mock.patch.object(
                stt_client,
                "_v2_request",
                return_value={
                    "ok": True,
                    "version": stt_client.PROTOCOL_VERSION,
                    "type": "transcript",
                    "delivery_id": "delivery-id",
                    "text": "  hello world \n",
                },
            ) as request,
            mock.patch.object(
                stt_client.time, "perf_counter", side_effect=[10.0, 10.125]
            ),
            redirect_stdout(output),
        ):
            result = stt_client.transcribe(audio)

        self.assertEqual(result, "hello world")
        request.assert_called_once_with(
            audio,
            timeout=120,
        )
        self.assertEqual(json.loads(output.getvalue())["stage"], "stt")

    def test_transcribe_reports_backend_empty_and_transport_errors(self) -> None:
        cases = [
            (
                {
                    "ok": False,
                    "version": stt_client.PROTOCOL_VERSION,
                    "error": {
                        "code": "invalid_audio_path",
                        "message": "not allowed",
                    },
                },
                "invalid_audio_path: not allowed",
            ),
            (
                {
                    "ok": True,
                    "version": stt_client.PROTOCOL_VERSION,
                    "type": "transcript",
                    "delivery_id": "empty",
                    "text": "",
                },
                "did not recognize any speech",
            ),
        ]
        for response, message in cases:
            with (
                self.subTest(response=response),
                mock.patch.object(stt_client, "_v2_request", return_value=response),
                self.assertRaisesRegex(HarnessError, message),
            ):
                stt_client.transcribe(Path("/runtime/recordings/request-test.wav"))

        with (
            mock.patch.object(
                stt_client, "_v2_request", side_effect=ConnectionRefusedError()
            ),
            self.assertRaisesRegex(HarnessError, "STT request failed"),
        ):
            stt_client.transcribe(Path("/runtime/recordings/request-test.wav"))

    def test_transcribe_retries_busy_generation_then_succeeds(self) -> None:
        audio = Path(
            "/runtime/voice-harness/recordings/"
            "request-0123456789abcdef0123456789abcdef.wav"
        )
        busy = {
            "ok": False,
            "version": stt_client.PROTOCOL_VERSION,
            "error": {
                "code": "server_busy",
                "message": "active transcription",
            },
        }
        success = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery",
            "text": "hello",
        }
        with (
            mock.patch.object(
                stt_client, "_v2_request", side_effect=[busy, success]
            ) as request,
            mock.patch.object(stt_client.time, "sleep") as sleep,
        ):
            result = stt_client.transcribe(audio)

        self.assertEqual(result, "hello")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in request.call_args_list],
            [audio, audio],
        )
        sleep.assert_called_once_with(stt_client.BUSY_BACKOFF_SECONDS)

    def test_retained_transcribe_propagates_uncertain_ack_state(self) -> None:
        response = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery",
            "text": "cancel that job",
            "_retained_state": "uncertain",
        }
        with mock.patch.object(stt_client, "_v2_request", return_value=response):
            delivery = stt_client.transcribe_retained(
                Path("/runtime/request.wav"),
                woke=False,
            )

        self.assertEqual(delivery.state, "uncertain")

    def test_transcribe_exhausted_busy_preserves_retry_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "request-generation.wav"
            audio.write_bytes(b"RIFF" + b"\0" * 64)
            busy = {
                "ok": False,
                "version": stt_client.PROTOCOL_VERSION,
                "error": {
                    "code": "server_busy",
                    "message": "active transcription",
                },
            }
            with (
                mock.patch.object(stt_client, "REQUEST_DEADLINE_SECONDS", 0.0),
                mock.patch.object(stt_client, "_v2_request", return_value=busy),
                self.assertRaisesRegex(
                    HarnessError,
                    rf"voice-harness transcribe --generation {audio}",
                ),
            ):
                stt_client.transcribe(audio)

            self.assertTrue(audio.exists())

    def test_v2_request_frames_multiline_text_and_sends_matching_ack(self) -> None:
        response = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery-id",
            "text": "first line\n第二行",
        }
        connection = mock.Mock()
        connection.recv.side_effect = [
            json.dumps(response, separators=(",", ":")).encode() + b"\n",
            b"",
        ]
        with mock.patch.object(stt_client.socket, "socket", return_value=connection):
            result = stt_client._v2_request(Path("/runtime/request.wav"), timeout=3)

        self.assertEqual(result, response)
        request = json.loads(connection.sendall.call_args_list[0].args[0])
        acknowledgment = json.loads(connection.sendall.call_args_list[1].args[0])
        self.assertEqual(request["type"], "transcribe")
        self.assertEqual(request["audio_path"], "/runtime/request.wav")
        self.assertEqual(
            acknowledgment,
            {
                "version": stt_client.PROTOCOL_VERSION,
                "type": "ack",
                "delivery_id": "delivery-id",
            },
        )

    def test_retained_ack_records_wake_context_before_returning(self) -> None:
        response = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery-id",
            "text": "do the thing",
        }
        connection = mock.Mock()
        connection.recv.side_effect = [
            json.dumps(response, separators=(",", ":")).encode() + b"\n",
            b"",
        ]
        with (
            mock.patch.object(stt_client.socket, "socket", return_value=connection),
            mock.patch.object(stt_client, "_delivery_request") as delivery_request,
        ):
            result = stt_client._v2_request(
                Path("/runtime/request.wav"),
                timeout=3,
                retain=True,
                woke=True,
            )

        self.assertEqual(result["delivery_id"], response["delivery_id"])
        self.assertEqual(result["text"], response["text"])
        acknowledgment = json.loads(connection.sendall.call_args_list[1].args[0])
        self.assertEqual(acknowledgment["disposition"], "retain")
        self.assertIs(acknowledgment["woke"], True)
        delivery_request.assert_called_once_with("pending", "delivery-id")
        self.assertEqual(result["_retained_state"], "pending")

    def test_pending_authorization_failure_remains_uncertain(self) -> None:
        response = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery-id",
            "text": "do the thing",
        }
        connection = mock.Mock()
        connection.recv.side_effect = [
            json.dumps(response, separators=(",", ":")).encode() + b"\n",
            b"",
        ]
        with (
            mock.patch.object(stt_client.socket, "socket", return_value=connection),
            mock.patch.object(
                stt_client,
                "_delivery_request",
                side_effect=HarnessError("STT unavailable"),
            ) as delivery_request,
        ):
            result = stt_client._v2_request(
                Path("/runtime/request.wav"),
                timeout=3,
                retain=True,
                woke=True,
            )

        delivery_request.assert_called_once_with("pending", "delivery-id")
        self.assertEqual(result["_retained_state"], "uncertain")

    def test_retained_ack_transport_failure_is_fenced_in_process(self) -> None:
        response = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery-id",
            "text": "do the thing",
        }
        connection = mock.Mock()
        connection.recv.side_effect = [
            json.dumps(response, separators=(",", ":")).encode() + b"\n",
            ConnectionResetError("reset after ack"),
        ]
        with (
            mock.patch.object(stt_client.socket, "socket", return_value=connection),
            mock.patch.object(stt_client, "_delivery_request") as delivery_request,
        ):
            result = stt_client._v2_request(
                Path("/runtime/request.wav"),
                timeout=3,
                retain=True,
            )

        delivery_request.assert_called_once_with("ambiguous", "delivery-id")
        self.assertEqual(result["_retained_state"], "ambiguous")

    def test_retained_ack_transport_failure_surfaces_uncertain_recovery(self) -> None:
        response = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery-id",
            "text": "do the thing",
        }
        connection = mock.Mock()
        connection.recv.side_effect = [
            json.dumps(response, separators=(",", ":")).encode() + b"\n",
            ConnectionResetError("reset after ack"),
        ]
        with (
            mock.patch.object(stt_client.socket, "socket", return_value=connection),
            mock.patch.object(
                stt_client,
                "_delivery_request",
                side_effect=HarnessError("STT unavailable"),
            ),
        ):
            result = stt_client._v2_request(
                Path("/runtime/request.wav"),
                timeout=3,
                retain=True,
            )

        self.assertEqual(result["_retained_state"], "uncertain")

    def test_transcribe_falls_back_to_legacy_server_once(self) -> None:
        audio = Path("/runtime/recordings/request.wav")
        with (
            mock.patch.object(
                stt_client, "_v2_request", side_effect=stt_client._LegacyServer
            ),
            mock.patch.object(
                stt_client, "unix_request", return_value=b"legacy transcript"
            ) as legacy,
        ):
            result = stt_client.transcribe(audio)

        self.assertEqual(result, "legacy transcript")
        legacy.assert_called_once_with(
            stt_client.STT_SOCKET,
            f"{audio}\n".encode(),
            timeout=mock.ANY,
        )
        timeout = legacy.call_args.kwargs["timeout"]
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, stt_client.REQUEST_DEADLINE_SECONDS)

    def test_post_ack_protocol_failure_does_not_trigger_legacy_retry(self) -> None:
        response = {
            "ok": True,
            "version": stt_client.PROTOCOL_VERSION,
            "type": "transcript",
            "delivery_id": "delivery-id",
            "text": "hello",
        }
        connection = mock.Mock()
        connection.recv.side_effect = [
            json.dumps(response, separators=(",", ":")).encode() + b"\n",
            b'{"version":1}\n',
            b"",
        ]
        with (
            mock.patch.object(stt_client.socket, "socket", return_value=connection),
            mock.patch.object(stt_client, "unix_request") as legacy,
            self.assertRaisesRegex(HarnessError, "invalid acknowledgment response"),
        ):
            stt_client.transcribe(Path("/runtime/request.wav"))

        legacy.assert_not_called()

    def test_legacy_fallback_respects_expired_overall_deadline(self) -> None:
        audio = Path("/runtime/recordings/request.wav")
        with (
            mock.patch.object(stt_client, "REQUEST_DEADLINE_SECONDS", 1.0),
            mock.patch.object(
                stt_client.time,
                "monotonic",
                side_effect=[100.0, 101.1],
            ),
            mock.patch.object(
                stt_client, "_v2_request", side_effect=stt_client._LegacyServer
            ),
            mock.patch.object(stt_client, "unix_request") as legacy,
            self.assertRaisesRegex(HarnessError, "deadline expired"),
        ):
            stt_client.transcribe(audio)

        legacy.assert_not_called()

    def test_protocol_error_only_suggests_an_existing_retry_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "request.wav"
            audio.write_bytes(b"RIFF" + b"\0" * 64)
            response = {
                "ok": False,
                "version": stt_client.PROTOCOL_VERSION,
                "error": {
                    "code": "transcription_failed",
                    "message": "backend failed",
                    "retry_path": str(audio),
                },
            }
            with (
                mock.patch.object(stt_client, "_v2_request", return_value=response),
                self.assertRaisesRegex(
                    HarnessError,
                    rf"voice-harness transcribe --generation {audio}",
                ),
            ):
                stt_client.transcribe(audio)

            audio.unlink()
            with (
                mock.patch.object(stt_client, "_v2_request", return_value=response),
                self.assertRaises(HarnessError) as caught,
            ):
                stt_client.transcribe(audio)
            self.assertNotIn("voice-harness transcribe", str(caught.exception))


class TextToSpeechClientTests(unittest.TestCase):
    def test_synthesize_sends_json_plays_output_and_returns_timings(self) -> None:
        response = {
            "ok": True,
            "audio_seconds": 1.5,
            "generation_seconds": 0.25,
        }
        output = io.StringIO()
        with (
            mock.patch.object(tts_client, "STATE_DIR", Path("/runtime")),
            mock.patch.object(
                tts_client, "unix_request", return_value=json.dumps(response).encode()
            ) as request,
            mock.patch.object(tts_client.subprocess, "run") as run,
            mock.patch.object(
                tts_client, "playback_slot", return_value=contextlib.nullcontext()
            ),
            mock.patch.object(
                tts_client.time, "perf_counter", side_effect=[20.0, 20.125]
            ),
            mock.patch.object(
                tts_client.uuid, "uuid4", return_value=mock.Mock(hex="request-id")
            ),
            redirect_stdout(output),
        ):
            settings = replace(
                tts_client.default_user_config().audio,
                voice="/tmp/voice.wav",
                sink="usb-dac",
            )
            result = tts_client.synthesize_and_play("hello", settings)

        payload = request.call_args.args[1]
        self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(
            json.loads(payload),
            {
                "text": "hello",
                "output": "/runtime/reply-request-id.wav",
                "voice": "/tmp/voice.wav",
            },
        )
        request.assert_called_once_with(tts_client.TTS_SOCKET, payload, timeout=120)
        run.assert_called_once_with(
            [
                "pw-play",
                "--target",
                "usb-dac",
                "/runtime/reply-request-id.wav",
            ],
            check=True,
        )
        self.assertEqual(result["stage"], "tts")
        self.assertEqual(result["request_seconds"], 0.125)
        self.assertEqual(json.loads(output.getvalue())["stage"], "tts")

    def test_synthesize_reports_backend_failure_without_playing(self) -> None:
        with (
            mock.patch.object(
                tts_client,
                "unix_request",
                return_value=b'{"ok": false, "error": "GPU unavailable"}',
            ),
            mock.patch.object(tts_client.subprocess, "run") as run,
            self.assertRaisesRegex(HarnessError, "GPU unavailable"),
        ):
            tts_client.synthesize_and_play("hello")

        run.assert_not_called()

    def test_streaming_playback_is_serialized_across_callers(self) -> None:
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        def run_playback(*, should_interrupt: object = None) -> dict[str, object]:
            del should_interrupt
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return {"ok": True}

        playback = mock.Mock()
        playback.run.side_effect = run_playback
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(tts_client, "STATE_DIR", Path(temporary)),
            mock.patch.object(tts_client, "StreamingPlayback", return_value=playback),
        ):
            threads = [
                threading.Thread(target=tts_client.stream_and_play, args=("hello",))
                for _ in range(2)
            ]
            for thread in threads:
                thread.start()
            join_threads(threads)

        self.assertEqual(maximum_active, 1)

    def test_synthesize_rejects_malformed_responses(self) -> None:
        for response in (b"not json", b"[]"):
            with (
                self.subTest(response=response),
                mock.patch.object(tts_client, "unix_request", return_value=response),
                self.assertRaisesRegex(HarnessError, "invalid response"),
            ):
                tts_client.synthesize_and_play("hello")

    def test_synthesize_wraps_transport_errors(self) -> None:
        with (
            mock.patch.object(
                tts_client, "unix_request", side_effect=ConnectionRefusedError()
            ),
            self.assertRaisesRegex(HarnessError, "TTS request failed"),
        ):
            tts_client.synthesize_and_play("hello")


if __name__ == "__main__":
    unittest.main()
