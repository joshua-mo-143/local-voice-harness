from __future__ import annotations

import collections
import io
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


class MicrophoneStartupTests(unittest.TestCase):
    def test_retries_until_pipewire_target_is_available(self) -> None:
        daemon = _bare_daemon()
        unavailable = mock.Mock()
        unavailable.poll.return_value = 1
        unavailable.stderr = io.BytesIO(b"stream error: no target node available")
        available = mock.Mock()
        available.poll.return_value = None

        with (
            mock.patch.object(
                wake_daemon.subprocess,
                "Popen",
                side_effect=[unavailable, available],
            ) as popen,
            mock.patch.object(wake_daemon.time, "sleep"),
            mock.patch.object(wake_daemon, "log"),
        ):
            daemon.start_microphone()

        self.assertEqual(popen.call_count, 2)
        self.assertIs(daemon.microphone, available)

    def test_other_pipewire_errors_fail_without_retrying(self) -> None:
        daemon = _bare_daemon()
        failed = mock.Mock()
        failed.poll.return_value = 1
        failed.returncode = 1
        failed.stderr = io.BytesIO(b"permission denied")

        with (
            mock.patch.object(
                wake_daemon.subprocess, "Popen", return_value=failed
            ) as popen,
            mock.patch.object(wake_daemon.time, "sleep"),
            self.assertRaisesRegex(wake_daemon.HarnessError, "permission denied"),
        ):
            daemon.start_microphone()

        popen.assert_called_once()


class ProcessUtteranceTests(unittest.TestCase):
    def test_completed_turn_enables_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="what time is it"
            ),
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

    def test_active_job_followup_is_classified_by_qwen(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="work on APP-43 instead"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", return_value=("started", None)
            ) as qwen_turn,
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(wake_daemon, "synthesize_and_play", return_value={}),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        qwen_turn.assert_called_once_with(
            "work on APP-43 instead", mock.ANY, "oldjob123456"
        )
        cursor_turn.assert_not_called()

    def test_explicit_cursor_request_starts_fresh_job(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="ask Cursor to work on APP-43"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "cursor_turn", return_value=("started", None)
            ) as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(wake_daemon, "synthesize_and_play", return_value={}),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        cursor_turn.assert_called_once_with("ask Cursor to work on APP-43")
        qwen_turn.assert_not_called()


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
