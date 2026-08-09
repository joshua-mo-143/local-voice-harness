from __future__ import annotations

import io
import json
import queue
import sys
import tempfile
import types
import unittest
import wave
from dataclasses import replace
from email.message import Message
from pathlib import Path
from unittest import mock

from local_voice_harness.config import load_backend_settings
from local_voice_harness.tts import client, server


class TextSplittingTests(unittest.TestCase):
    def test_prefers_sentences_and_bounds_long_text(self) -> None:
        text = (
            "The first sentence is short. "
            "This second sentence has several clauses, which are useful split points, "
            "and enough additional words to exceed a deliberately small limit."
        )

        chunks = server.split_text(text, max_chars=55)

        self.assertEqual(chunks[0], "The first sentence is short.")
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 55 for chunk in chunks))

    def test_empty_whitespace_has_no_chunks(self) -> None:
        self.assertEqual(server.split_text(" \n\t "), [])


class ServerStreamingTests(unittest.TestCase):
    def test_cancellation_stops_before_the_next_model_call(self) -> None:
        handler = mock.Mock()
        handler.wfile = io.BytesIO()
        fake_soundfile = types.SimpleNamespace(
            write=lambda path, audio, rate, subtype: Path(path).write_bytes(b"wav")
        )
        events: list[dict[str, object]] = []

        def emit(_handler: object, event: dict[str, object]) -> None:
            events.append(event)
            if event.get("event") == "chunk":
                server.ACTIVE_STREAMS["request-1"].set()

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(server, "OUTPUT_ROOT", Path(temporary)),
            mock.patch.object(server, "MODEL", types.SimpleNamespace(sr=24_000)),
            mock.patch.object(
                server, "_generate", return_value=([0.0] * 2_400, 0.1)
            ) as generate,
            mock.patch.object(server, "_write_json", side_effect=emit),
            mock.patch.dict(sys.modules, {"soundfile": fake_soundfile}),
        ):
            server._stream_response(
                handler,
                {
                    "text": "First sentence. Second sentence.",
                    "request_id": "request-1",
                },
            )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            [event["event"] for event in events], ["start", "chunk", "done"]
        )
        self.assertTrue(events[-1]["cancelled"])
        self.assertNotIn("request-1", server.ACTIVE_STREAMS)

    def test_venice_tts_requests_pcm_wav_with_configured_model_and_voice(self) -> None:
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(24_000)
            target.writeframes(b"\x00\x00" * 2_400)
        headers = Message()
        headers["Content-Type"] = "audio/wav"
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = headers
        response.read.return_value = output.getvalue()
        settings = replace(
            load_backend_settings({}),
            tts_provider="venice",
            tts_model="tts-kokoro",
            tts_voice="af_sky",
            tts_speed=1.25,
            tts_endpoint="https://api.venice.ai/api/v1/audio/speech",
            tts_timeout=19,
        )

        with (
            mock.patch.object(server, "SETTINGS", settings),
            mock.patch.object(
                server, "get_venice_api_key", return_value="venice-secret"
            ),
            mock.patch.object(
                server.urllib.request, "urlopen", return_value=response
            ) as urlopen,
        ):
            audio, rate, duration, _elapsed = server._venice_audio("Hello.")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(audio, output.getvalue())
        self.assertEqual((rate, duration), (24_000, 0.1))
        self.assertEqual(
            payload,
            {
                "model": "tts-kokoro",
                "voice": "af_sky",
                "input": "Hello.",
                "response_format": "wav",
                "speed": 1.25,
            },
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer venice-secret")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 19)


class _CapturingStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, value: bytes) -> int:
        self.data.extend(value)
        return len(value)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _CapturingStdin()
        self.stderr = io.BytesIO()
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _FakeSocket:
    def __init__(self, payload: bytes) -> None:
        self.responses = [payload]
        self.sent = b""
        self.connected = ""
        self.closed = False

    def connect(self, path: str) -> None:
        self.connected = path

    def sendall(self, value: bytes) -> None:
        self.sent += value

    def shutdown(self, _direction: int) -> None:
        pass

    def recv(self, _size: int) -> bytes:
        return self.responses.pop(0) if self.responses else b""

    def close(self) -> None:
        self.closed = True


class ClientPlaybackTests(unittest.TestCase):
    def test_worker_uses_one_raw_pipewire_process_for_all_chunks(self) -> None:
        session = client.StreamingPlayback("hello")
        process = _FakeProcess()
        chunks: queue.Queue[dict[str, object] | None] = queue.Queue()
        expected = bytearray()

        with tempfile.TemporaryDirectory() as temporary:
            for index, text in enumerate(("first", "second")):
                output = Path(temporary) / f"{index}.wav"
                frames = bytes([index + 1, 0]) * 240
                expected.extend(frames)
                with wave.open(str(output), "wb") as target:
                    target.setnchannels(1)
                    target.setsampwidth(2)
                    target.setframerate(24_000)
                    target.writeframes(frames)
                chunks.put({"output": str(output), "text": text})
            chunks.put(None)

            with mock.patch.object(
                client.subprocess, "Popen", return_value=process
            ) as popen:
                session._play_chunks(chunks, 24_000)

        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertIn("--raw", command)
        self.assertIn("--rate=24000", command)
        self.assertEqual(process.stdin.data, expected)
        self.assertEqual(session.played_text, "first second")

    def test_stream_events_are_queued_and_reported(self) -> None:
        session = client.StreamingPlayback("One. Two.")
        events = [
            {"ok": True, "event": "start", "sample_rate": 24_000},
            {
                "ok": True,
                "event": "chunk",
                "output": "/tmp/nonexistent-voice-chunk.wav",
                "text": "One.",
            },
            {"ok": True, "event": "done", "cancelled": False, "chunks": 1},
        ]
        payload = b"".join(json.dumps(event).encode() + b"\n" for event in events)
        fake_socket = _FakeSocket(payload)

        def consume(chunks: queue.Queue[dict[str, object] | None], _rate: int) -> None:
            while (item := chunks.get()) is not None:
                with session._state_lock:
                    session._played.append(str(item["text"]))

        with (
            mock.patch.object(client.socket, "socket", return_value=fake_socket),
            mock.patch.object(
                client.select, "select", return_value=([fake_socket], [], [])
            ),
            mock.patch.object(session, "_play_chunks", side_effect=consume),
        ):
            result = session.run()

        request = json.loads(fake_socket.sent)
        self.assertTrue(request["stream"])
        self.assertEqual(request["request_id"], session.request_id)
        self.assertEqual(result["played_text"], "One.")
        self.assertFalse(result["interrupted"])

    def test_interrupt_monitor_continues_after_server_finishes(self) -> None:
        session = client.StreamingPlayback("One.")
        events = [
            {"ok": True, "event": "start", "sample_rate": 24_000},
            {
                "ok": True,
                "event": "chunk",
                "output": "/tmp/nonexistent-late-voice-chunk.wav",
                "text": "One.",
            },
            {"ok": True, "event": "done", "cancelled": False, "chunks": 1},
        ]
        payload = b"".join(json.dumps(event).encode() + b"\n" for event in events)
        fake_socket = _FakeSocket(payload)
        checks = 0

        def wait_for_cancel(
            chunks: queue.Queue[dict[str, object] | None], _rate: int
        ) -> None:
            chunks.get()
            while not session.cancelled.wait(0.01):
                pass

        def should_interrupt() -> bool:
            nonlocal checks
            checks += 1
            return checks == 2

        with (
            mock.patch.object(client.socket, "socket", return_value=fake_socket),
            mock.patch.object(
                client.select, "select", return_value=([fake_socket], [], [])
            ),
            mock.patch.object(session, "_play_chunks", side_effect=wait_for_cancel),
            mock.patch.object(client, "unix_request", return_value=b'{"ok":true}\n'),
        ):
            result = session.run(should_interrupt=should_interrupt)

        self.assertTrue(result["interrupted"])
        self.assertEqual(checks, 2)

    def test_explicit_cancel_notifies_server_and_stops_process(self) -> None:
        session = client.StreamingPlayback("hello")
        process = _FakeProcess()
        session._process = process  # type: ignore[reportAttributeAccessIssue]

        with mock.patch.object(
            client, "unix_request", return_value=b'{"ok":true}\n'
        ) as send:
            session.cancel()
            session.cancel()

        self.assertTrue(process.terminated)
        send.assert_called_once()
        request = json.loads(send.call_args.args[1])
        self.assertEqual(request, {"op": "cancel", "request_id": session.request_id})

    def test_external_cancel_ends_run_as_an_interruption(self) -> None:
        session = client.StreamingPlayback("hello")
        fake_socket = _FakeSocket(b"")

        def cancel_while_waiting(
            _read: object, _write: object, _error: object, _timeout: float
        ) -> tuple[list[_FakeSocket], list[object], list[object]]:
            session.cancel()
            return [fake_socket], [], []

        with (
            mock.patch.object(client.socket, "socket", return_value=fake_socket),
            mock.patch.object(
                client.select, "select", side_effect=cancel_while_waiting
            ),
            mock.patch.object(client, "unix_request", return_value=b'{"ok":true}\n'),
        ):
            result = session.run()

        self.assertTrue(result["interrupted"])

    def test_legacy_api_still_uses_complete_waveform_request(self) -> None:
        response = json.dumps({"ok": True, "output": "/tmp/reply.wav"}).encode()
        with (
            mock.patch.object(client, "unix_request", return_value=response) as send,
            mock.patch.object(client.subprocess, "run") as play,
        ):
            result = client.synthesize_and_play("hello")

        request = json.loads(send.call_args.args[1])
        self.assertNotIn("stream", request)
        self.assertEqual(request["text"], "hello")
        self.assertEqual(result["stage"], "tts")
        play.assert_called_once()


if __name__ == "__main__":
    unittest.main()
