from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.stt import client as stt_client
from local_voice_harness.stt import server as stt_server
from local_voice_harness.tts import client as tts_client


class SpeechToTextClientTests(unittest.TestCase):
    def test_normalize_replaces_whole_terms_case_insensitively(self) -> None:
        with mock.patch.dict(
            stt_server.REPLACEMENTS, {"cursa": "Cursor", "herder": "Herdr"}, clear=True
        ):
            result = stt_server.normalize("Cursa and herder, not cursative")

        self.assertEqual(result, "Cursor and Herdr, not cursative")

    def test_transcribe_sends_path_and_returns_trimmed_text(self) -> None:
        audio = Path("/tmp/request.wav")
        output = io.StringIO()
        with (
            mock.patch.object(
                stt_client, "unix_request", return_value=b"  hello world \n"
            ) as request,
            mock.patch.object(
                stt_client.time, "perf_counter", side_effect=[10.0, 10.125]
            ),
            redirect_stdout(output),
        ):
            result = stt_client.transcribe(audio)

        self.assertEqual(result, "hello world")
        request.assert_called_once_with(
            stt_client.STT_SOCKET, b"/tmp/request.wav\n", timeout=120
        )
        self.assertEqual(json.loads(output.getvalue())["stage"], "stt")

    def test_transcribe_reports_backend_empty_and_transport_errors(self) -> None:
        cases = [
            (b"__DICTATION_ERROR__:RuntimeError: failed", "RuntimeError: failed"),
            (b" \n", "did not recognize any speech"),
        ]
        for response, message in cases:
            with (
                self.subTest(response=response),
                mock.patch.object(stt_client, "unix_request", return_value=response),
                self.assertRaisesRegex(HarnessError, message),
            ):
                stt_client.transcribe()

        with (
            mock.patch.object(
                stt_client, "unix_request", side_effect=ConnectionRefusedError()
            ),
            self.assertRaisesRegex(HarnessError, "Whisper request failed"),
        ):
            stt_client.transcribe()


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
                tts_client.time, "perf_counter", side_effect=[20.0, 20.125]
            ),
            mock.patch.dict(
                os.environ, {"VOICE_HARNESS_VOICE": "/tmp/voice.wav"}, clear=True
            ),
            redirect_stdout(output),
        ):
            result = tts_client.synthesize_and_play("hello")

        payload = request.call_args.args[1]
        self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(
            json.loads(payload),
            {
                "text": "hello",
                "output": "/runtime/reply.wav",
                "voice": "/tmp/voice.wav",
            },
        )
        request.assert_called_once_with(tts_client.TTS_SOCKET, payload, timeout=120)
        run.assert_called_once_with(["pw-play", "/runtime/reply.wav"], check=True)
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
            self.assertRaisesRegex(HarnessError, "Chatterbox request failed"),
        ):
            tts_client.synthesize_and_play("hello")


if __name__ == "__main__":
    unittest.main()
