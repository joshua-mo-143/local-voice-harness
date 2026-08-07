from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from local_voice_harness.errors import HarnessError
from local_voice_harness.tts.queue import (
    PlaybackQueue,
    PlaybackRequest,
    PrefetchedUtterance,
    PrefetchHandle,
)


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
            [mock.call("first"), mock.call("second")],
        )

    def test_prefetch_wait_can_be_interrupted(self) -> None:
        handle = PrefetchHandle.__new__(PrefetchHandle)
        handle.text = "slow response"
        handle._event = threading.Event()
        handle._result = None
        should_interrupt = mock.Mock(return_value=True)

        with mock.patch("local_voice_harness.tts.queue.STREAM_POLL_SECONDS", 0.001):
            result = handle.wait(timeout=1, should_interrupt=should_interrupt)

        self.assertIsNone(result)
        should_interrupt.assert_called_once()

    def test_drain_reports_interruption_while_prefetching(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(text="slow response")
        handle = mock.create_autospec(PrefetchHandle, instance=True)
        handle.wait.return_value = None
        queue._items.append((request, handle))
        on_played = mock.Mock()
        should_interrupt = mock.Mock(return_value=True)

        batch = queue.drain(
            should_interrupt=should_interrupt,
            on_played=on_played,
        )

        handle.wait.assert_called_once_with(should_interrupt=should_interrupt)
        handle.discard.assert_called_once()
        self.assertEqual(len(queue), 0)
        self.assertEqual(batch[0][1:], (True, request))
        self.assertEqual(batch[0][0]["played_text"], "")
        on_played.assert_called_once_with(batch[0][0], True, request)

    def test_drain_removes_and_discards_terminal_prefetch_failure(self) -> None:
        queue = PlaybackQueue()
        request = PlaybackRequest(text="failed response")
        handle = mock.create_autospec(PrefetchHandle, instance=True)
        handle.wait.side_effect = HarnessError("socket refused")
        queue._items.append((request, handle))

        with self.assertRaisesRegex(HarnessError, "socket refused"):
            queue.drain()

        handle.discard.assert_called_once()
        self.assertEqual(len(queue), 0)

    def test_failed_handle_discard_cleans_partial_chunks_safely(self) -> None:
        handle = PrefetchHandle.__new__(PrefetchHandle)
        handle.text = "failed response"
        handle._event = threading.Event()
        handle._discard_lock = threading.Lock()
        handle._discarded = False
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
            with mock.patch("local_voice_harness.tts.queue.threading.Thread") as thread:
                handle.discard()
                cleanup = thread.call_args.kwargs["target"]
                cleanup()
                handle.discard()

            self.assertFalse(stream_dir.exists())
            thread.assert_called_once()
