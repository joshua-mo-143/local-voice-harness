from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from local_voice_harness.stt import server


class _Transcriber(server.Transcriber):
    def __init__(self) -> None:
        self.paths: list[str] = []

    def transcribe(self, audio_path: str) -> str:
        self.paths.append(audio_path)
        return "hello"


class _BlockingTranscriber(server.Transcriber):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.contents: list[bytes] = []
        self.paths: list[Path] = []

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        self.paths.append(path)
        self.contents.append(path.read_bytes())
        self.entered.set()
        if not self.release.wait(2):
            raise RuntimeError("test transcription was not released")
        return "hello"


def _request(socket_path: Path, payload: bytes) -> bytes:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    try:
        client.connect(str(socket_path))
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := client.recv(4096):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        client.close()


def _error(response: bytes) -> dict[str, object]:
    value = json.loads(response)
    assert isinstance(value, dict)
    error = value["error"]
    assert isinstance(error, dict)
    return error


def _generation(writable: Path, contents: bytes) -> Path:
    generations = writable.parent / "recordings"
    generations.mkdir(mode=0o700, exist_ok=True)
    generations.chmod(0o700)
    path = generations / f"{writable.stem}-{uuid.uuid4().hex}.wav"
    path.write_bytes(contents)
    return path


@contextmanager
def _running_server(
    root: Path,
    transcriber: server.Transcriber,
) -> Iterator[tuple[Path, Path]]:
    socket_path = root / "stt.sock"
    audio_path = root / "voice-harness" / "request.wav"
    dictation_path = root / "dictation" / "recording.wav"
    audio_path.parent.mkdir()
    dictation_path.parent.mkdir()
    stopped = threading.Event()
    with (
        mock.patch.object(server.config, "WAV_PATH", audio_path),
        mock.patch.object(server.config, "DICTATION_WAV_PATH", dictation_path),
        mock.patch.object(server.config, "RECORDING_LOCK", root / "recording.lock"),
    ):
        thread = threading.Thread(
            target=server.serve,
            args=(transcriber,),
            kwargs={"socket_path": socket_path, "stop_event": stopped},
        )
        thread.start()
        deadline = time.monotonic() + 2
        while not socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not socket_path.exists():
            raise RuntimeError("test STT server did not start")
        try:
            yield socket_path, audio_path
        finally:
            stopped.set()
            thread.join(2)
            if thread.is_alive():
                raise RuntimeError("test STT server did not stop")


class SpeechToTextProtocolTests(unittest.TestCase):
    def test_valid_owned_regular_wav_is_transcribed_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transcriber = _Transcriber()
            with _running_server(Path(temporary), transcriber) as (
                socket_path,
                audio_path,
            ):
                generation = _generation(audio_path, b"RIFF" + b"\0" * 64)
                response = _request(socket_path, f"{generation}\n".encode())

                self.assertEqual(response, b"hello")
                self.assertEqual(len(transcriber.paths), 1)
                self.assertIn("stt-processing", transcriber.paths[0])
                self.assertFalse(generation.exists())

    def test_malformed_and_oversized_requests_return_structured_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with _running_server(Path(temporary), _Transcriber()) as (
                socket_path,
                _audio_path,
            ):
                malformed = _request(socket_path, b"\xff\n")
                oversized = _request(
                    socket_path, b"x" * (server.MAX_REQUEST_BYTES + 1) + b"\n"
                )

            self.assertEqual(_error(malformed)["code"], "invalid_encoding")
            self.assertEqual(_error(oversized)["code"], "request_too_large")

    def test_slow_client_does_not_block_a_later_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transcriber = _Transcriber()
            with (
                mock.patch.object(server, "READ_TIMEOUT_SECONDS", 0.1),
                _running_server(Path(temporary), transcriber) as (
                    socket_path,
                    audio_path,
                ),
            ):
                slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                slow.settimeout(2)
                slow.connect(str(socket_path))
                slow.sendall(b"/unfinished")
                generation = _generation(audio_path, b"RIFF" + b"\0" * 64)

                started = time.monotonic()
                response = _request(socket_path, f"{generation}\n".encode())
                elapsed = time.monotonic() - started
                timeout_response = slow.recv(4096)
                slow.close()

            self.assertEqual(response, b"hello")
            self.assertLess(elapsed, 0.5)
            self.assertEqual(_error(timeout_response)["code"], "request_timeout")

    def test_disconnected_client_does_not_block_a_later_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with _running_server(Path(temporary), _Transcriber()) as (
                socket_path,
                audio_path,
            ):
                abandoned = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                abandoned.connect(str(socket_path))
                abandoned.sendall(b"/unfinished")
                abandoned.close()
                generation = _generation(audio_path, b"RIFF" + b"\0" * 64)

                response = _request(socket_path, f"{generation}\n".encode())

            self.assertEqual(response, b"hello")

    def test_paths_outside_allowlist_and_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.wav"
            outside.write_bytes(b"RIFF" + b"\0" * 64)
            with _running_server(root, _Transcriber()) as (
                socket_path,
                audio_path,
            ):
                outside_response = _request(socket_path, f"{outside}\n".encode())
                symlink = _generation(audio_path, b"placeholder")
                symlink.unlink()
                symlink.symlink_to(outside)
                symlink_response = _request(socket_path, f"{symlink}\n".encode())

            self.assertEqual(_error(outside_response)["code"], "invalid_audio_path")
            self.assertEqual(_error(symlink_response)["code"], "invalid_audio_path")

    def test_new_recording_is_not_transcribed_or_deleted_by_earlier_request(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transcriber = _BlockingTranscriber()
            with _running_server(Path(temporary), transcriber) as (
                socket_path,
                audio_path,
            ):
                old_audio = b"RIFF-old" + b"\0" * 64
                new_audio = b"RIFF-new" + b"\0" * 64
                old_generation = _generation(audio_path, old_audio)
                responses: list[bytes] = []
                request = threading.Thread(
                    target=lambda: responses.append(
                        _request(socket_path, f"{old_generation}\n".encode())
                    )
                )
                request.start()
                self.assertTrue(transcriber.entered.wait(1))

                new_generation = _generation(audio_path, new_audio)
                transcriber.release.set()
                request.join(2)

                self.assertFalse(request.is_alive())
                self.assertEqual(responses, [b"hello"])
                self.assertEqual(transcriber.contents, [old_audio])
                self.assertEqual(new_generation.read_bytes(), new_audio)
                self.assertFalse(transcriber.paths[0].exists())

    def test_valid_request_gets_server_busy_while_model_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transcriber = _BlockingTranscriber()
            with _running_server(Path(temporary), transcriber) as (
                socket_path,
                audio_path,
            ):
                first_generation = _generation(audio_path, b"RIFF-first" + b"\0" * 64)
                first_response: list[bytes] = []
                first = threading.Thread(
                    target=lambda: first_response.append(
                        _request(socket_path, f"{first_generation}\n".encode())
                    )
                )
                first.start()
                self.assertTrue(transcriber.entered.wait(1))
                second_generation = _generation(audio_path, b"RIFF-second" + b"\0" * 64)

                busy = _request(socket_path, f"{second_generation}\n".encode())
                self.assertTrue(second_generation.exists())
                transcriber.release.set()
                first.join(2)
                retry = _request(socket_path, f"{second_generation}\n".encode())

            self.assertEqual(_error(busy)["code"], "server_busy")
            self.assertEqual(first_response, [b"hello"])
            self.assertEqual(retry, b"hello")
            self.assertFalse(second_generation.exists())
            self.assertEqual(len(transcriber.contents), 2)

    def test_writable_recording_path_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with _running_server(Path(temporary), _Transcriber()) as (
                socket_path,
                audio_path,
            ):
                contents = b"RIFF-active" + b"\0" * 64
                audio_path.write_bytes(contents)

                response = _request(socket_path, f"{audio_path}\n".encode())

                self.assertEqual(_error(response)["code"], "invalid_audio_path")
                self.assertEqual(audio_path.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
