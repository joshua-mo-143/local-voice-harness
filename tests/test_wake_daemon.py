from __future__ import annotations

import collections
import time
import unittest
from unittest import mock

from local_voice_harness.wake import daemon as wake_daemon
from local_voice_harness.wake.daemon import WakeConversationDaemon


def _bare_daemon() -> WakeConversationDaemon:
    """Build a daemon without running __init__ (which loads the wake model/VAD)."""
    instance = WakeConversationDaemon.__new__(WakeConversationDaemon)
    instance.history = []
    instance.cursor_session = None
    instance.conversation_deadline = 0.0
    instance.awaiting_followup = False
    instance.last_wake = 0.0
    instance.running = True
    instance.microphone = None
    instance.activation_thread = None
    instance.activation_error = None
    instance.pre_roll = collections.deque(maxlen=wake_daemon.PRE_ROLL_FRAMES)
    instance.wake_model = mock.Mock()
    return instance


class ProcessUtteranceTests(unittest.TestCase):
    def test_completed_turn_enables_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="what time is it"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", return_value=("it is noon", None)
            ) as qwen_turn,
            mock.patch.object(wake_daemon, "synthesize_and_play", return_value={}),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        qwen_turn.assert_called_once()
        self.assertTrue(
            daemon.awaiting_followup,
            "a completed turn must re-arm follow-up listening",
        )
        self.assertGreater(daemon.conversation_deadline, 0.0)
        self.assertEqual(
            daemon.history,
            [
                {"role": "user", "content": "what time is it"},
                {"role": "assistant", "content": "it is noon"},
            ],
        )

    def test_empty_wake_phrase_waits_for_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="hey jarvis"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(wake_daemon, "synthesize_and_play") as play,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=True)

        qwen_turn.assert_not_called()
        play.assert_not_called()
        self.assertTrue(daemon.awaiting_followup)
        self.assertGreater(daemon.conversation_deadline, 0.0)


class AnnounceJobTests(unittest.TestCase):
    def test_awaiting_user_job_enables_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(wake_daemon, "mark_delivered"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(wake_daemon, "synthesize_and_play", return_value={}),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.announce_job(
                {"id": "job1", "status": "awaiting_user", "question": "which repo?"}
            )

        self.assertTrue(daemon.awaiting_followup)
        self.assertEqual(daemon.cursor_session, "job1")
        self.assertGreater(daemon.conversation_deadline, 0.0)
        stop_components.assert_not_called()

    def test_completed_job_does_not_enable_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(wake_daemon, "mark_delivered"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(wake_daemon, "synthesize_and_play", return_value={}),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.announce_job({"id": "job2", "status": "completed", "result": "done"})

        self.assertFalse(daemon.awaiting_followup)


class RunLoopFollowupTests(unittest.TestCase):
    def test_followup_speech_is_recorded_without_wake_word(self) -> None:
        daemon = _bare_daemon()
        daemon.awaiting_followup = True
        daemon.conversation_deadline = time.monotonic() + 100

        recorded: list[list[bytes]] = []
        processed: list[bool] = []

        def fake_record(initial: list[bytes]) -> None:
            recorded.append(list(initial))

        def fake_process(*, woke: bool) -> None:
            processed.append(woke)
            daemon.running = False

        daemon.start_microphone = lambda: None  # type: ignore[method-assign]
        daemon.read_frame = lambda: b"\x00\x00"  # type: ignore[method-assign]
        daemon.is_speech = lambda frame: True  # type: ignore[method-assign]
        daemon.record_utterance = fake_record  # type: ignore[method-assign]
        daemon.process_utterance = fake_process  # type: ignore[method-assign]

        with mock.patch.object(wake_daemon, "pending_results", return_value=[]):
            daemon.run()

        self.assertEqual(processed, [False], "follow-up must be handled as woke=False")
        self.assertEqual(len(recorded), 1)


if __name__ == "__main__":
    unittest.main()
