from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from local_voice_harness.stt import client as stt_client
from local_voice_harness.stt import server
from local_voice_harness.user_config import DictationDevice


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


class _FailingTranscriber(server.Transcriber):
    def transcribe(self, audio_path: str) -> str:
        raise RuntimeError(f"backend failed for {audio_path}")


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


def _open_v2_request(socket_path: Path, generation: Path) -> socket.socket:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    client.connect(str(socket_path))
    client.sendall(
        json.dumps(
            {
                "version": server.PROTOCOL_VERSION,
                "type": "transcribe",
                "audio_path": str(generation),
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    return client


def _receive_frame(client: socket.socket) -> dict[str, object]:
    response = bytearray()
    while b"\n" not in response:
        chunk = client.recv(4096)
        if not chunk:
            raise RuntimeError("server closed before completing a response frame")
        response.extend(chunk)
    frame, newline, trailing = response.partition(b"\n")
    if not newline or trailing:
        raise RuntimeError("server returned invalid response framing")
    value = json.loads(frame)
    assert isinstance(value, dict)
    return value


def _acknowledge(
    client: socket.socket,
    response: dict[str, object],
    *,
    delivery_id: str | None = None,
) -> bytes:
    client.sendall(
        json.dumps(
            {
                "version": server.PROTOCOL_VERSION,
                "type": "ack",
                "delivery_id": delivery_id or response["delivery_id"],
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    client.shutdown(socket.SHUT_WR)
    trailing = bytearray()
    while chunk := client.recv(4096):
        trailing.extend(chunk)
    client.close()
    return bytes(trailing)


def _acknowledged_request(
    socket_path: Path,
    generation: Path,
) -> dict[str, object]:
    client = _open_v2_request(socket_path, generation)
    try:
        response = _receive_frame(client)
        if response.get("ok") is True:
            trailing = _acknowledge(client, response)
            if trailing:
                raise RuntimeError(f"unexpected acknowledgment response: {trailing!r}")
        else:
            client.close()
        return response
    finally:
        client.close()


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise RuntimeError("condition was not met before the deadline")
        time.sleep(0.01)


def _error(response: bytes) -> dict[str, object]:
    value = json.loads(response)
    assert isinstance(value, dict)
    error = value["error"]
    assert isinstance(error, dict)
    return error


def _protocol_error(response: dict[str, object]) -> dict[str, object]:
    error = response["error"]
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
    ready = threading.Event()
    with (
        mock.patch.object(server.config, "WAV_PATH", audio_path),
        mock.patch.object(server.config, "DICTATION_WAV_PATH", dictation_path),
        mock.patch.object(server.config, "RECORDING_LOCK", root / "recording.lock"),
    ):
        thread = threading.Thread(
            target=server.serve,
            args=(transcriber,),
            kwargs={
                "socket_path": socket_path,
                "stop_event": stopped,
                "ready_event": ready,
            },
        )
        thread.start()
        if not ready.wait(2):
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
                client = _open_v2_request(socket_path, generation)
                response = _receive_frame(client)

                self.assertEqual(response["text"], "hello")
                self.assertEqual(len(transcriber.paths), 1)
                self.assertIn("stt-processing", transcriber.paths[0])
                self.assertTrue(Path(transcriber.paths[0]).exists())
                self.assertFalse(generation.exists())

                self.assertEqual(_acknowledge(client, response), b"")

                self.assertFalse(generation.exists())
                self.assertFalse(Path(transcriber.paths[0]).exists())

    def test_wake_crash_retains_then_restart_recovers_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transcriber = _Transcriber()
            with _running_server(Path(temporary), transcriber) as (
                socket_path,
                audio_path,
            ):
                generation = _generation(
                    audio_path,
                    b"RIFF-process-crash" + b"\0" * 64,
                )
                process = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os, sys; from pathlib import Path; "
                            "from local_voice_harness.stt import client; "
                            "client.STT_SOCKET = Path(sys.argv[1]); "
                            "client.transcribe_retained(Path(sys.argv[2]), woke=True); "
                            "os._exit(73)"
                        ),
                        str(socket_path),
                        str(generation),
                    ],
                    check=False,
                    timeout=2,
                )

                self.assertEqual(process.returncode, 73)
                with mock.patch.object(stt_client, "STT_SOCKET", socket_path):
                    recovered = stt_client.recover_retained_transcripts()
                    self.assertEqual(len(recovered), 1)
                    self.assertEqual(recovered[0].text, "hello")
                    self.assertTrue(recovered[0].woke)
                    self.assertEqual(recovered[0].state, "pending")

                    routed: list[str] = []
                    routed.append(recovered[0].text)
                    recovered[0].release()

                    self.assertEqual(stt_client.recover_retained_transcripts(), ())

                self.assertEqual(routed, ["hello"])
                self.assertEqual(len(transcriber.paths), 1)
                self.assertFalse(generation.exists())

    def test_ambiguous_delivery_survives_restart_until_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with _running_server(Path(temporary), _Transcriber()) as (
                socket_path,
                audio_path,
            ):
                generation = _generation(
                    audio_path,
                    b"RIFF-ambiguous" + b"\0" * 64,
                )
                with mock.patch.object(stt_client, "STT_SOCKET", socket_path):
                    delivery = stt_client.transcribe_retained(generation, woke=False)
                    delivery.mark_ambiguous()

                    recovered = stt_client.recover_retained_transcripts()
                    self.assertEqual(len(recovered), 1)
                    self.assertEqual(recovered[0].state, "ambiguous")

                    recovered[0].mark_ambiguous()
                    recovered[0].release()
                    self.assertEqual(stt_client.recover_retained_transcripts(), ())

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
                response = _acknowledged_request(socket_path, generation)
                elapsed = time.monotonic() - started
                timeout_response = slow.recv(4096)
                slow.close()

            self.assertEqual(response["text"], "hello")
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

                response = _acknowledged_request(socket_path, generation)

            self.assertEqual(response["text"], "hello")

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
                responses: list[dict[str, object]] = []
                request = threading.Thread(
                    target=lambda: responses.append(
                        _acknowledged_request(socket_path, old_generation)
                    )
                )
                request.start()
                self.assertTrue(transcriber.entered.wait(1))

                new_generation = _generation(audio_path, new_audio)
                transcriber.release.set()
                request.join(2)

                self.assertFalse(request.is_alive())
                self.assertEqual(
                    [response["text"] for response in responses], ["hello"]
                )
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
                first_response: list[dict[str, object]] = []
                first = threading.Thread(
                    target=lambda: first_response.append(
                        _acknowledged_request(socket_path, first_generation)
                    )
                )
                first.start()
                self.assertTrue(transcriber.entered.wait(1))
                second_generation = _generation(audio_path, b"RIFF-second" + b"\0" * 64)

                busy_client = _open_v2_request(socket_path, second_generation)
                busy = _receive_frame(busy_client)
                busy_client.close()
                self.assertTrue(second_generation.exists())
                transcriber.release.set()
                first.join(2)
                retry = _acknowledged_request(socket_path, second_generation)

            self.assertEqual(_protocol_error(busy)["code"], "server_busy")
            self.assertEqual(
                [response["text"] for response in first_response], ["hello"]
            )
            self.assertEqual(retry["text"], "hello")
            self.assertFalse(second_generation.exists())
            self.assertEqual(len(transcriber.contents), 2)

    def test_legacy_success_restores_unacknowledged_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with _running_server(Path(temporary), _Transcriber()) as (
                socket_path,
                audio_path,
            ):
                contents = b"RIFF-legacy" + b"\0" * 64
                generation = _generation(audio_path, contents)

                response = _request(socket_path, f"{generation}\n".encode())

                self.assertEqual(response, b"hello")
                self.assertEqual(generation.read_bytes(), contents)

    def test_backend_and_normalization_failures_restore_before_error(self) -> None:
        cases: list[tuple[server.Transcriber, object]] = [
            (_FailingTranscriber(), mock.DEFAULT),
            (_Transcriber(), RuntimeError("normalization failed")),
        ]
        for transcriber, normalization_error in cases:
            with (
                self.subTest(transcriber=type(transcriber).__name__),
                tempfile.TemporaryDirectory() as temporary,
            ):
                normalize = (
                    mock.patch.object(
                        server, "normalize", side_effect=normalization_error
                    )
                    if normalization_error is not mock.DEFAULT
                    else mock.patch.object(server, "normalize", wraps=server.normalize)
                )
                with (
                    normalize,
                    _running_server(Path(temporary), transcriber) as (
                        socket_path,
                        audio_path,
                    ),
                ):
                    contents = b"RIFF-failure" + b"\0" * 64
                    generation = _generation(audio_path, contents)

                    response = _acknowledged_request(socket_path, generation)

                    error = _protocol_error(response)
                    self.assertEqual(error["code"], "transcription_failed")
                    self.assertEqual(error["retry_path"], str(generation))
                    self.assertEqual(generation.read_bytes(), contents)

    def test_disconnect_during_transcription_restores_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transcriber = _BlockingTranscriber()
            with _running_server(Path(temporary), transcriber) as (
                socket_path,
                audio_path,
            ):
                contents = b"RIFF-disconnect" + b"\0" * 64
                generation = _generation(audio_path, contents)
                client = _open_v2_request(socket_path, generation)
                self.assertTrue(transcriber.entered.wait(1))

                client.close()
                transcriber.release.set()
                _wait_for(generation.exists)

                self.assertEqual(generation.read_bytes(), contents)

    def test_ack_timeout_and_mismatch_restore_retry_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(server, "READ_TIMEOUT_SECONDS", 0.1),
                _running_server(Path(temporary), _Transcriber()) as (
                    socket_path,
                    audio_path,
                ),
            ):
                timed_out = _generation(audio_path, b"RIFF-ack-timeout" + b"\0" * 64)
                timeout_client = _open_v2_request(socket_path, timed_out)
                self.assertEqual(_receive_frame(timeout_client)["text"], "hello")
                timeout_error = _receive_frame(timeout_client)
                timeout_client.close()

                self.assertEqual(
                    _protocol_error(timeout_error)["retry_path"], str(timed_out)
                )
                self.assertTrue(timed_out.exists())

                mismatched = _generation(audio_path, b"RIFF-ack-mismatch" + b"\0" * 64)
                mismatch_client = _open_v2_request(socket_path, mismatched)
                transcript = _receive_frame(mismatch_client)
                trailing = _acknowledge(
                    mismatch_client,
                    transcript,
                    delivery_id="0" * 32,
                )

                mismatch_error = _error(trailing)
                self.assertEqual(mismatch_error["code"], "invalid_ack")
                self.assertEqual(mismatch_error["retry_path"], str(mismatched))
                self.assertTrue(mismatched.exists())

    def test_duplicate_request_is_busy_until_ack_and_transcribes_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transcriber = _Transcriber()
            with _running_server(Path(temporary), transcriber) as (
                socket_path,
                audio_path,
            ):
                generation = _generation(audio_path, b"RIFF-duplicate" + b"\0" * 64)
                first = _open_v2_request(socket_path, generation)
                first_response = _receive_frame(first)

                duplicate = _open_v2_request(socket_path, generation)
                duplicate_response = _receive_frame(duplicate)
                duplicate.close()

                self.assertEqual(
                    _protocol_error(duplicate_response)["code"], "server_busy"
                )
                self.assertEqual(len(transcriber.paths), 1)
                self.assertEqual(_acknowledge(first, first_response), b"")
                self.assertFalse(generation.exists())

    def test_startup_recovers_processing_and_finishes_delivered_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "voice-harness" / "request.wav"
            dictation_path = root / "dictation" / "recording.wav"
            audio_path.parent.mkdir()
            dictation_path.parent.mkdir()
            generations = audio_path.parent / "recordings"
            generations.mkdir(mode=0o700)
            original = generations / f"request-{uuid.uuid4().hex}.wav"
            processing_dir = audio_path.parent / server.PROCESSING_DIRECTORY
            delivered_dir = audio_path.parent / server.DELIVERED_DIRECTORY
            processing_dir.mkdir(mode=0o700)
            delivered_dir.mkdir(mode=0o700)
            processing = processing_dir / f"{original.stem}-{uuid.uuid4().hex}.wav"
            delivered = (
                delivered_dir / f"request-{uuid.uuid4().hex}-{uuid.uuid4().hex}.wav"
            )
            contents = b"RIFF-recover" + b"\0" * 64
            processing.write_bytes(contents)
            delivered.write_bytes(b"RIFF-delivered" + b"\0" * 64)

            with (
                mock.patch.object(server.config, "WAV_PATH", audio_path),
                mock.patch.object(server.config, "DICTATION_WAV_PATH", dictation_path),
                mock.patch.object(
                    server.config, "RECORDING_LOCK", root / "recording.lock"
                ),
            ):
                server.recover_stranded_audio()

            self.assertEqual(original.read_bytes(), contents)
            self.assertFalse(processing.exists())
            self.assertFalse(delivered.exists())

    def test_startup_recovery_is_idempotent_and_quarantines_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "voice-harness" / "request.wav"
            dictation_path = root / "dictation" / "recording.wav"
            audio_path.parent.mkdir()
            dictation_path.parent.mkdir()
            generations = audio_path.parent / "recordings"
            generations.mkdir(mode=0o700)
            processing_dir = audio_path.parent / server.PROCESSING_DIRECTORY
            processing_dir.mkdir(mode=0o700)

            partial_original = generations / f"request-{uuid.uuid4().hex}.wav"
            partial_processing = (
                processing_dir / f"{partial_original.stem}-{uuid.uuid4().hex}.wav"
            )
            partial_original.write_bytes(b"RIFF-partial" + b"\0" * 64)
            partial_processing.hardlink_to(partial_original)

            conflict_original = generations / f"request-{uuid.uuid4().hex}.wav"
            conflict_processing = (
                processing_dir / f"{conflict_original.stem}-{uuid.uuid4().hex}.wav"
            )
            conflict_original.write_bytes(b"RIFF-original" + b"\0" * 64)
            conflict_processing.write_bytes(b"RIFF-conflict" + b"\0" * 64)
            malformed = processing_dir / "not-a-claim.wav"
            malformed.write_bytes(b"RIFF-malformed" + b"\0" * 64)

            with (
                mock.patch.object(server.config, "WAV_PATH", audio_path),
                mock.patch.object(server.config, "DICTATION_WAV_PATH", dictation_path),
                mock.patch.object(
                    server.config, "RECORDING_LOCK", root / "recording.lock"
                ),
                mock.patch.object(server, "log") as log,
            ):
                server.recover_stranded_audio()
                server.recover_stranded_audio()

            quarantine = audio_path.parent / server.QUARANTINE_DIRECTORY
            self.assertTrue(partial_original.exists())
            self.assertFalse(partial_processing.exists())
            self.assertEqual(
                conflict_original.read_bytes(), b"RIFF-original" + b"\0" * 64
            )
            self.assertEqual(len(tuple(quarantine.iterdir())), 2)
            self.assertTrue(
                any(
                    "quarantined STT audio at" in str(call)
                    for call in log.call_args_list
                )
            )

    def test_restoration_cleanup_failure_does_not_expose_retry_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "request.wav"
            paths = server.recorder.RecorderPaths(
                root,
                audio_path,
                root / "recording.pid",
                root / "recording.log",
            )
            paths.generations.mkdir(mode=0o700)
            processing_dir = root / server.PROCESSING_DIRECTORY
            processing_dir.mkdir(mode=0o700)
            original = paths.generations / f"request-{uuid.uuid4().hex}.wav"
            processing = processing_dir / f"{original.stem}-{uuid.uuid4().hex}.wav"
            processing.write_bytes(b"RIFF-cleanup" + b"\0" * 64)
            real_unlink = Path.unlink

            def fail_processing_unlink(path: Path, *, missing_ok: bool = False) -> None:
                if path == processing:
                    raise PermissionError("cleanup denied")
                real_unlink(path, missing_ok=missing_ok)

            with mock.patch.object(Path, "unlink", fail_processing_unlink):
                result = server._restore_claim(
                    server.AudioClaim(original, processing, paths)
                )

            self.assertIsNone(result.retry_path)
            self.assertEqual(result.preserved_path, processing)
            self.assertFalse(original.exists())
            self.assertTrue(processing.exists())

    def test_recovery_quarantines_small_claim_and_symlink_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "request.wav"
            paths = server.recorder.RecorderPaths(
                root,
                audio_path,
                root / "recording.pid",
                root / "recording.log",
            )
            paths.generations.mkdir(mode=0o700)
            processing_dir = root / server.PROCESSING_DIRECTORY
            processing_dir.mkdir(mode=0o700)

            small_original = paths.generations / f"request-{uuid.uuid4().hex}.wav"
            small_processing = (
                processing_dir / f"{small_original.stem}-{uuid.uuid4().hex}.wav"
            )
            small_processing.write_bytes(b"short")

            symlink_original = paths.generations / f"request-{uuid.uuid4().hex}.wav"
            symlink_processing = (
                processing_dir / f"{symlink_original.stem}-{uuid.uuid4().hex}.wav"
            )
            contents = b"RIFF-symlink" + b"\0" * 64
            symlink_processing.write_bytes(contents)
            symlink_original.symlink_to(symlink_processing)

            with server.recorder.recording_lock(paths.state_dir, paths.lock):
                small = server._restore_claim_locked(
                    server.AudioClaim(small_original, small_processing, paths)
                )
                collision = server._restore_claim_locked(
                    server.AudioClaim(
                        symlink_original,
                        symlink_processing,
                        paths,
                    )
                )

            self.assertIsNone(small.retry_path)
            self.assertIsNotNone(small.quarantine_path)
            assert small.quarantine_path is not None
            self.assertEqual(small.quarantine_path.read_bytes(), b"short")
            self.assertIsNone(collision.retry_path)
            self.assertIsNotNone(collision.quarantine_path)
            assert collision.quarantine_path is not None
            self.assertEqual(collision.quarantine_path.read_bytes(), contents)
            self.assertTrue(symlink_original.is_symlink())
            self.assertFalse(symlink_original.exists())

    def test_commit_remains_successful_after_post_rename_lock_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "request.wav"
            paths = server.recorder.RecorderPaths(
                root,
                audio_path,
                root / "recording.pid",
                root / "recording.log",
            )
            processing_dir = root / server.PROCESSING_DIRECTORY
            processing_dir.mkdir(mode=0o700)
            original = root / "recordings" / f"request-{uuid.uuid4().hex}.wav"
            processing = processing_dir / f"{original.stem}-{uuid.uuid4().hex}.wav"
            processing.write_bytes(b"RIFF-commit" + b"\0" * 64)

            @contextmanager
            def lock_with_failed_cleanup(
                _state_dir: Path, _lock: Path
            ) -> Iterator[None]:
                yield
                raise OSError("lock cleanup failed")

            with (
                mock.patch.object(
                    server.recorder,
                    "recording_lock",
                    side_effect=lock_with_failed_cleanup,
                ),
                mock.patch.object(server, "log") as log,
            ):
                server._commit_claim(server.AudioClaim(original, processing, paths))

            self.assertFalse(processing.exists())
            self.assertFalse(
                (root / server.DELIVERED_DIRECTORY / processing.name).exists()
            )
            self.assertTrue(
                any(
                    "delivery committed despite" in str(call)
                    for call in log.call_args_list
                )
            )

    def test_retention_remains_successful_after_post_rename_lock_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = server.recorder.RecorderPaths(
                root,
                root / "request.wav",
                root / "recording.pid",
                root / "recording.log",
            )
            processing_dir = root / server.PROCESSING_DIRECTORY
            processing_dir.mkdir(mode=0o700)
            original = root / "recordings" / f"request-{uuid.uuid4().hex}.wav"
            processing = processing_dir / f"{original.stem}-{uuid.uuid4().hex}.wav"
            processing.write_bytes(b"RIFF-retain" + b"\0" * 64)
            processing.chmod(0o600)
            delivery_id = uuid.uuid4().hex

            @contextmanager
            def lock_with_failed_cleanup(
                _state_dir: Path, _lock: Path
            ) -> Iterator[None]:
                yield
                raise OSError("lock cleanup failed")

            with (
                mock.patch.object(
                    server.recorder,
                    "recording_lock",
                    side_effect=lock_with_failed_cleanup,
                ),
                mock.patch.object(server, "log") as log,
            ):
                server._write_retained_claim(
                    server.AudioClaim(original, processing, paths),
                    delivery_id=delivery_id,
                    text="hello",
                    woke=True,
                )

            retained = root / server.RETAINED_DIRECTORY / delivery_id
            self.assertTrue((retained / server.RETAINED_AUDIO).exists())
            self.assertEqual(
                server._load_retained_delivery(retained)["state"],
                "ambiguous",
            )
            self.assertTrue(
                any(
                    "retention committed despite" in str(call)
                    for call in log.call_args_list
                )
            )

    def test_retained_deliveries_recover_in_creation_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = server.recorder.RecorderPaths(
                root,
                root / "request.wav",
                root / "recording.pid",
                root / "recording.log",
            )
            retained_root = root / server.RETAINED_DIRECTORY
            retained_root.mkdir(mode=0o700)
            deliveries = (
                ("0" * 32, "later", 2.0),
                ("f" * 32, "earlier", 1.0),
            )
            for delivery_id, text, created_at in deliveries:
                delivery = retained_root / delivery_id
                delivery.mkdir(mode=0o700)
                metadata = delivery / server.RETAINED_METADATA
                metadata.write_text(
                    json.dumps(
                        {
                            "version": server.PROTOCOL_VERSION,
                            "delivery_id": delivery_id,
                            "text": text,
                            "woke": True,
                            "state": "pending",
                            "created_at": created_at,
                        }
                    )
                )
                metadata.chmod(0o600)
                audio = delivery / server.RETAINED_AUDIO
                audio.write_bytes(b"RIFF-retained" + b"\0" * 64)
                audio.chmod(0o600)

            with mock.patch.object(
                server, "_recorder_path_sets", return_value=(paths,)
            ):
                recovered = server._recover_retained_deliveries()

        self.assertEqual(
            [delivery["text"] for delivery in recovered],
            ["earlier", "later"],
        )

    def test_main_recovers_before_loading_model(self) -> None:
        order: list[str] = []
        settings = server.STTRuntimeSettings(
            backend="parakeet",
            device=DictationDevice.AUTO,
            model_name="model",
            quantization="int8",
            compute_type="float16",
            language=None,
            prompt="prompt",
            replacements={},
        )
        with (
            mock.patch.object(server, "load_user_config"),
            mock.patch.object(server, "runtime_settings", return_value=settings),
            mock.patch.object(
                server,
                "recover_stranded_audio",
                side_effect=lambda: order.append("recover"),
            ),
            mock.patch.object(
                server,
                "load_transcriber",
                side_effect=lambda _settings: order.append("load") or _Transcriber(),
            ),
            mock.patch.object(
                server,
                "serve",
                side_effect=lambda _model, **_kwargs: order.append("serve"),
            ),
        ):
            server.main()

        self.assertEqual(order, ["recover", "load", "serve"])

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
