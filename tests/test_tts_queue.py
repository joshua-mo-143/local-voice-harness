from __future__ import annotations

import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.tts import queue as tts_queue
from local_voice_harness.tts.queue import (
    PlaybackQueue,
    PlaybackRequest,
    PrefetchedUtterance,
    PrefetchHandle,
)


def _write_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 8)
    return path


class PlaybackQueueTests(unittest.TestCase):
    def test_enqueue_defers_prefetch_until_explicitly_started(self) -> None:
        queue = PlaybackQueue()
        with mock.patch("local_voice_harness.tts.queue.PrefetchHandle") as prefetch:
            queue.enqueue(PlaybackRequest(text="first"))
            queue.enqueue(PlaybackRequest(text="second"))
            prefetch.assert_not_called()
            queue.start_prefetch()

        self.assertEqual(
            prefetch.call_args_list,
            [
                mock.call(
                    "first",
                    skip_first_speed=False,
                    preflight_speed=False,
                )
            ],
        )

    def test_prefetch_wait_progress_can_be_interrupted(self) -> None:
        handle = PrefetchHandle.__new__(PrefetchHandle)
        handle._progress = threading.Condition()
        handle._chunks = []
        handle._chunk_texts = []
        handle._sample_rate = 0
        handle._stream_done = False
        handle._stream_error = None
        should_interrupt = mock.Mock(return_value=True)

        with (
            mock.patch("local_voice_harness.tts.queue.STREAM_POLL_SECONDS", 0.001),
            mock.patch.object(handle, "cancel") as cancel,
        ):
            result = handle.wait_progress(
                0, timeout=1, should_interrupt=should_interrupt
            )

        self.assertIsNone(result)
        should_interrupt.assert_called()
        cancel.assert_called_once()

    def test_prefetch_wait_can_be_interrupted(self) -> None:
        handle = PrefetchHandle.__new__(PrefetchHandle)
        handle.text = "slow response"
        handle._event = threading.Event()
        handle._result = PrefetchedUtterance(
            sample_rate=0,
            chunks=[],
            chunk_texts=[],
            error=HarnessError("cancelled"),
        )
        should_interrupt = mock.Mock(return_value=True)

        with (
            mock.patch("local_voice_harness.tts.queue.STREAM_POLL_SECONDS", 0.001),
            mock.patch.object(handle, "cancel") as cancel,
        ):
            result = handle.wait(timeout=1, should_interrupt=should_interrupt)

        self.assertIsNone(result)
        should_interrupt.assert_called_once()
        cancel.assert_called_once()

    def test_drain_reports_interruption_while_prefetching(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(text="slow response")
        handle = mock.create_autospec(PrefetchHandle, instance=True)
        handle.wait_progress.return_value = None
        queue._items.append((request, handle))
        on_played = mock.Mock()
        should_interrupt = mock.Mock(return_value=True)

        batch = queue.drain(
            should_interrupt=should_interrupt,
            on_played=on_played,
        )

        handle.wait_progress.assert_called_once_with(
            0,
            should_interrupt=should_interrupt,
            on_poll=None,
        )
        handle.discard.assert_called_once()
        self.assertEqual(len(queue), 0)
        self.assertEqual(batch[0][1:], (True, request))
        self.assertEqual(batch[0][0]["played_text"], "")
        on_played.assert_called_once_with(batch[0][0], True, request)

    def test_enqueue_during_drain_starts_prefetch_immediately(self) -> None:
        queue = PlaybackQueue()
        queue._draining = True

        with mock.patch("local_voice_harness.tts.queue.PrefetchHandle") as prefetch:
            queue.enqueue(PlaybackRequest(text="arrived during playback"))

        prefetch.assert_called_once_with(
            "arrived during playback",
            skip_first_speed=False,
            preflight_speed=False,
        )
        self.assertIs(queue._items[0][1], prefetch.return_value)

    def test_drain_removes_and_discards_terminal_prefetch_failure(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(text="failed response")
        handle = mock.create_autospec(PrefetchHandle, instance=True)
        handle.wait_progress.side_effect = HarnessError("socket refused")
        queue._items.append((request, handle))

        with self.assertRaisesRegex(HarnessError, "socket refused"):
            queue.drain()

        handle.discard.assert_called_once()
        self.assertEqual(len(queue), 0)

    def test_late_pw_play_failure_does_not_run_delivery_callback(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(
            text="completed",
            job_id="aaaaaaaaaaaa",
            delivery_token="claim",
            job_status="completed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            chunk = Path(temporary) / "chunk.wav"
            with wave.open(str(chunk), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\x00\x00" * 8)
            handle = mock.create_autospec(PrefetchHandle, instance=True)
            handle.wait_progress.return_value = (
                16_000,
                [chunk],
                ["completed"],
                True,
            )
            queue._items.append((request, handle))
            process = mock.Mock()
            process.stdin = mock.Mock()
            process.stderr = mock.Mock()
            process.stderr.read.return_value = b"sink disconnected"
            process.wait.return_value = 1
            process.poll.return_value = 1
            on_played = mock.Mock()

            with (
                mock.patch(
                    "local_voice_harness.tts.queue._open_playback",
                    return_value=process,
                ),
                self.assertRaisesRegex(HarnessError, "sink disconnected"),
            ):
                queue.drain(on_played=on_played)

        on_played.assert_not_called()

    def test_later_prefetch_failure_acknowledges_successful_audio_prefix(self) -> None:
        queue = PlaybackQueue()
        first = PlaybackRequest(
            text="first",
            job_id="aaaaaaaaaaaa",
            delivery_token="claim1",
            job_status="completed",
        )
        second = PlaybackRequest(
            text="second",
            job_id="bbbbbbbbbbbb",
            delivery_token="claim2",
            job_status="completed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            chunk = Path(temporary) / "first.wav"
            with wave.open(str(chunk), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\x00\x00" * 8)
            first_handle = mock.create_autospec(PrefetchHandle, instance=True)
            first_handle.wait_progress.return_value = (
                16_000,
                [chunk],
                ["first"],
                True,
            )
            second_handle = mock.create_autospec(PrefetchHandle, instance=True)
            second_handle.wait_progress.side_effect = HarnessError(
                "second prefetch failed"
            )
            queue._items.extend([(first, first_handle), (second, second_handle)])
            process = mock.Mock()
            process.stdin = mock.Mock()
            process.wait.return_value = 0
            process.poll.return_value = 0
            on_played = mock.Mock()

            with (
                mock.patch(
                    "local_voice_harness.tts.queue._open_playback",
                    return_value=process,
                ),
                self.assertRaisesRegex(HarnessError, "second prefetch failed"),
            ):
                queue.drain(on_played=on_played)

        on_played.assert_called_once()
        result, interrupted, request = on_played.call_args.args
        self.assertEqual(result["played_text"], "first")
        self.assertFalse(interrupted)
        self.assertIs(request, first)
        second_handle.discard.assert_called_once()

    def test_partial_synthesis_failure_releases_delivery_claim(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(
            text="partially delivered",
            job_id="aaaaaaaaaaaa",
            delivery_token="claim",
            job_status="completed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            chunk = _write_wav(Path(temporary) / "first.wav")
            handle = mock.create_autospec(PrefetchHandle, instance=True)
            handle.done_meta = {}
            handle.wait_progress.side_effect = [
                (16_000, [chunk], ["partial"], False),
                HarnessError("later synthesis failed"),
            ]
            queue._items.append((request, handle))
            process = mock.Mock()
            process.stdin = mock.Mock()
            process.wait.return_value = 0
            process.poll.return_value = 0
            on_played = mock.Mock()

            with (
                mock.patch(
                    "local_voice_harness.tts.queue._open_playback",
                    return_value=process,
                ),
                self.assertRaisesRegex(HarnessError, "later synthesis failed"),
            ):
                queue.drain(on_played=on_played)

        on_played.assert_called_once()
        result, interrupted, played_request = on_played.call_args.args
        self.assertTrue(interrupted)
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["played_text"], "partial")
        self.assertIs(played_request, request)

    def test_drain_prefetches_next_item_before_playing_current(self) -> None:
        queue = PlaybackQueue()
        queue.enqueue(PlaybackRequest(text="first"))
        queue.enqueue(PlaybackRequest(text="second"))
        created: list[str] = []

        def fake_handle(
            text: str,
            settings: object | None = None,
            **_options: object,
        ) -> PrefetchHandle:
            created.append(text)
            handle = mock.create_autospec(PrefetchHandle, instance=True)
            handle.done_meta = {}
            chunk = first_chunk if text == "first" else second_chunk

            def wait_progress(
                have: int,
                **_kwargs: object,
            ) -> tuple[int, list[Path], list[str], bool]:
                if text == "first":
                    self.assertIn("second", created)
                return 16_000, [chunk], [text], True

            handle.wait_progress.side_effect = wait_progress
            return handle

        with tempfile.TemporaryDirectory() as temporary:
            first_chunk = _write_wav(Path(temporary) / "first.wav")
            second_chunk = _write_wav(Path(temporary) / "second.wav")
            process = mock.Mock()
            process.stdin = mock.Mock()
            process.wait.return_value = 0
            process.poll.return_value = 0
            with (
                mock.patch(
                    "local_voice_harness.tts.queue.PrefetchHandle",
                    side_effect=fake_handle,
                ),
                mock.patch(
                    "local_voice_harness.tts.queue._open_playback",
                    return_value=process,
                ),
            ):
                queue.drain()

        self.assertEqual(created, ["first", "second"])

    def test_drain_starts_playback_on_first_chunk_before_later_chunks(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(text="two clauses")
        playback_opened = mock.Mock()

        with tempfile.TemporaryDirectory() as temporary:
            first = _write_wav(Path(temporary) / "first.wav")
            second = _write_wav(Path(temporary) / "second.wav")
            handle = mock.create_autospec(PrefetchHandle, instance=True)
            handle.done_meta = {}

            def wait_progress(
                have: int,
                **_kwargs: object,
            ) -> tuple[int, list[Path], list[str], bool]:
                if have == 0:
                    playback_opened.assert_not_called()
                    return 16_000, [first], ["one"], False
                playback_opened.assert_called_once()
                return 16_000, [first, second], ["one", "two"], True

            handle.wait_progress.side_effect = wait_progress
            queue._items.append((request, handle))
            process = mock.Mock()
            process.stdin = mock.Mock()
            process.wait.return_value = 0
            process.poll.return_value = 0

            def open_playback(*_args: object, **_kwargs: object) -> mock.Mock:
                playback_opened()
                return process

            with mock.patch(
                "local_voice_harness.tts.queue._open_playback",
                side_effect=open_playback,
            ):
                batch = queue.drain()

        self.assertEqual(batch[0][0]["played_text"], "one two")
        self.assertFalse(batch[0][1])

    def test_drain_interrupt_before_later_chunks_discards_prefetch(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(text="two clauses")

        with tempfile.TemporaryDirectory() as temporary:
            first = _write_wav(Path(temporary) / "first.wav")
            handle = mock.create_autospec(PrefetchHandle, instance=True)
            handle.done_meta = {}
            handle.wait_progress.side_effect = [
                (16_000, [first], ["one"], False),
                None,
            ]
            queue._items.append((request, handle))
            process = mock.Mock()
            process.stdin = mock.Mock()
            process.wait.return_value = 0
            process.poll.return_value = 0

            with mock.patch(
                "local_voice_harness.tts.queue._open_playback",
                return_value=process,
            ):
                batch = queue.drain()

        self.assertTrue(batch[0][1])
        self.assertEqual(batch[0][0]["played_text"], "one")
        handle.discard.assert_called()
        self.assertEqual(len(queue), 0)

    def test_failed_handle_discard_cleans_partial_chunks_safely(self) -> None:
        handle = PrefetchHandle.__new__(PrefetchHandle)
        handle.text = "failed response"
        handle._event = threading.Event()
        handle._discard_lock = threading.Lock()
        handle._discarded = False
        handle._progress = threading.Condition()
        handle._chunks = []
        handle._thread = mock.Mock()
        handle._thread.is_alive.return_value = False
        with tempfile.TemporaryDirectory() as temporary:
            stream_dir = Path(temporary) / "stream-request"
            stream_dir.mkdir()
            chunk = stream_dir / "chunk.wav"
            chunk.write_bytes(b"partial")
            handle._result = PrefetchedUtterance(
                sample_rate=0,
                chunks=[chunk],
                chunk_texts=[],
                error=HarnessError("stream failed"),
            )
            handle._event.set()
            handle.discard()
            handle.discard()

            self.assertFalse(stream_dir.exists())
            handle._thread.join.assert_called_once_with(
                timeout=tts_queue.PREFETCH_JOIN_SECONDS
            )

    def test_discard_cancels_submitted_request_and_joins_worker(self) -> None:
        submitted = threading.Event()
        stream_socket = mock.Mock()
        stream_socket.sendall.side_effect = lambda _request: submitted.set()
        with (
            mock.patch.object(tts_queue.socket, "socket", return_value=stream_socket),
            mock.patch.object(
                tts_queue.select,
                "select",
                return_value=([], [], []),
            ),
            mock.patch.object(tts_queue, "STREAM_TIMEOUT_SECONDS", 60),
            mock.patch.object(
                tts_queue,
                "unix_request",
                return_value=b'{"ok":true,"cancelled":true}\n',
            ) as send,
        ):
            handle = PrefetchHandle("slow response")
            self.assertTrue(submitted.wait(timeout=1))
            handle.discard()
            handle.discard()

        self.assertFalse(handle._thread.is_alive())
        stream_socket.close.assert_called_once()
        send.assert_called_once()
        request = json.loads(send.call_args.args[1])
        self.assertEqual(
            request,
            {"op": "cancel", "request_id": handle.request_id},
        )

    def test_cancel_before_submission_never_starts_or_cancels_server_work(self) -> None:
        started = threading.Event()

        def wait_for_cancel(
            _text: str,
            *,
            cancelled: threading.Event,
            **_kwargs: object,
        ) -> PrefetchedUtterance:
            started.set()
            cancelled.wait(timeout=2)
            raise HarnessError("cancelled before submission")

        with (
            mock.patch.object(
                tts_queue,
                "_prefetch_utterance",
                side_effect=wait_for_cancel,
            ),
            mock.patch.object(tts_queue, "unix_request") as send,
        ):
            handle = PrefetchHandle("cancel immediately")
            self.assertTrue(started.wait(timeout=1))
            handle.discard()

        self.assertFalse(handle._thread.is_alive())
        send.assert_not_called()

    def test_truncated_prefetch_stream_fails_and_closes_socket(self) -> None:
        request_id = "request-1"
        events = [
            {
                "ok": True,
                "event": "start",
                "request_id": request_id,
                "sample_rate": 24_000,
                "chunks": 1,
            },
            {
                "ok": True,
                "event": "chunk",
                "request_id": request_id,
                "index": 0,
                "output": "/tmp/nonexistent-prefetch.wav",
                "text": "partial",
            },
        ]
        stream_socket = mock.Mock()
        stream_socket.recv.side_effect = [
            b"".join(json.dumps(event).encode() + b"\n" for event in events),
            b"",
        ]

        with (
            mock.patch.object(tts_queue.socket, "socket", return_value=stream_socket),
            mock.patch.object(
                tts_queue.select,
                "select",
                return_value=([stream_socket], [], []),
            ),
            self.assertRaisesRegex(HarnessError, "before completion"),
        ):
            tts_queue._prefetch_utterance(
                "partial",
                request_id=request_id,
                cancelled=threading.Event(),
                register_socket=mock.Mock(),
                submit_request=mock.Mock(),
                clear_socket=mock.Mock(),
            )

        stream_socket.close.assert_called_once()
        stream_socket.settimeout.assert_called_once_with(
            tts_queue.PREFETCH_CONNECT_SECONDS
        )

    def test_silent_prefetch_stream_times_out_and_closes_socket(self) -> None:
        stream_socket = mock.Mock()

        with (
            mock.patch.object(tts_queue.socket, "socket", return_value=stream_socket),
            mock.patch.object(tts_queue.select, "select", return_value=([], [], [])),
            mock.patch.object(tts_queue, "STREAM_TIMEOUT_SECONDS", 1),
            mock.patch.object(tts_queue.time, "monotonic", side_effect=[0, 2]),
            self.assertRaisesRegex(HarnessError, "timed out"),
        ):
            tts_queue._prefetch_utterance(
                "silent",
                request_id="request-1",
                cancelled=threading.Event(),
                register_socket=mock.Mock(),
                submit_request=mock.Mock(),
                clear_socket=mock.Mock(),
            )

        stream_socket.close.assert_called_once()
        stream_socket.settimeout.assert_called_once_with(
            tts_queue.PREFETCH_CONNECT_SECONDS
        )
