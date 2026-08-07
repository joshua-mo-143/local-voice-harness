from __future__ import annotations

import threading
import unittest
from unittest import mock

from local_voice_harness.tts.queue import PlaybackQueue, PlaybackRequest, PrefetchHandle


class PlaybackQueueTests(unittest.TestCase):
    def test_enqueue_starts_prefetch_before_playback(self) -> None:
        queue = PlaybackQueue()
        started = threading.Event()

        class SlowPrefetch(PrefetchHandle):
            def _run(self) -> None:
                started.set()
                super()._run()

        with mock.patch("local_voice_harness.tts.queue.PrefetchHandle", SlowPrefetch):
            queue.enqueue(PlaybackRequest(text="first"))
            queue.enqueue(PlaybackRequest(text="second"))

        self.assertTrue(started.wait(timeout=1))

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
