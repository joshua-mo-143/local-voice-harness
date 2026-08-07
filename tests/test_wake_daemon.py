from __future__ import annotations

import collections
import io
import threading
import time
import unittest
from unittest import mock

from local_voice_harness.browser_context import RequestContext
from local_voice_harness.intent import Intent, IntentRoute
from local_voice_harness.tts.queue import PlaybackQueue, PlaybackRequest
from local_voice_harness.wake import daemon as wake_daemon
from local_voice_harness.wake.daemon import WakeConversationDaemon


def _playback_batch(
    text: str = "test",
    *,
    played_text: str = "",
    interrupted: bool = False,
    job_id: str | None = None,
    delivery_token: str | None = None,
    job_status: str | None = None,
) -> list[tuple[dict[str, object], bool, PlaybackRequest]]:
    request = PlaybackRequest(
        text=text,
        job_id=job_id,
        delivery_token=delivery_token,
        job_status=job_status,
    )
    return [
        (
            {
                "ok": True,
                "played_text": played_text or text,
                "interrupted": interrupted,
            },
            interrupted,
            request,
        )
    ]


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
    instance.microphone_paused = False
    instance.activation_thread = None
    instance.activation_error = None
    instance.component_lock = threading.Lock()
    instance.playback_queue = PlaybackQueue()
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


class StripWakePrefixTests(unittest.TestCase):
    def test_strips_leading_wake_phrase(self) -> None:
        text, found = wake_daemon.strip_wake_prefix("Hey Jarvis, what time is it?")
        self.assertTrue(found)
        self.assertEqual(text, "what time is it?")

    def test_strips_common_parakeet_mishearing(self) -> None:
        text, found = wake_daemon.strip_wake_prefix("hey service what time is it")
        self.assertTrue(found)
        self.assertEqual(text, "what time is it")

    def test_strips_wake_phrase_not_at_start(self) -> None:
        text, found = wake_daemon.strip_wake_prefix("um hey jarvis what time is it")
        self.assertTrue(found)
        self.assertEqual(text, "um what time is it")


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
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("what time is it\n\nGitHub context"),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        qwen_turn.assert_called_once_with(
            "what time is it\n\nGitHub context",
            mock.ANY,
            None,
            trusted_utterance="what time is it",
            delivery_claims=mock.ANY,
        )
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

    def test_turn_preserves_awaiting_job_session_played_before_response(
        self,
    ) -> None:
        daemon = _bare_daemon()
        batch = _playback_batch(
            "Cursor needs clarification. which repo?",
            job_id="job1",
            delivery_token="claim",
            job_status="awaiting_user",
        ) + _playback_batch("it is noon")
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="what time is it"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", return_value=("it is noon", None)
            ),
            mock.patch.object(
                daemon, "_drain_playback_queue", return_value=(batch, None)
            ),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("what time is it"),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "acknowledge_delivery"),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        self.assertEqual(daemon.cursor_session, "job1")

    def test_empty_wake_phrase_waits_for_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="hey jarvis"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(daemon, "_drain_playback_queue") as play,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=True)

        qwen_turn.assert_not_called()
        play.assert_not_called()
        self.assertTrue(daemon.awaiting_followup)
        self.assertGreater(daemon.conversation_deadline, 0.0)

    def test_missing_wake_prefix_still_accepts_command(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="what time is it"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", return_value=("it is noon", None)
            ) as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("what time is it"),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=True)

        qwen_turn.assert_called_once()

    def test_fuzzy_new_task_bypasses_main_qwen(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="work on APP-43 instead"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                wake_daemon, "cursor_turn", return_value=("started", None)
            ) as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        cursor_turn.assert_called_once_with(
            "work on APP-43 instead",
            utterance="work on APP-43 instead",
            context_repository=None,
            delivery_claims=mock.ANY,
        )
        qwen_turn.assert_not_called()

    def test_router_sends_clarification_answer_to_awaiting_job(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="use the api repository"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("use the api repository"),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_REPLY, "high"),
            ),
            mock.patch.object(
                wake_daemon, "cursor_turn", return_value=("continued", None)
            ) as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        cursor_turn.assert_called_once_with(
            "use the api repository",
            "oldjob123456",
            utterance="use the api repository",
            action="reply",
            job_id="oldjob123456",
            delivery_claims=mock.ANY,
        )
        qwen_turn.assert_not_called()

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
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext(
                    "ask Cursor to work on APP-43\n\nGitHub context"
                ),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        cursor_turn.assert_called_once_with(
            "ask Cursor to work on APP-43\n\nGitHub context",
            utterance="ask Cursor to work on APP-43",
            context_repository=None,
            delivery_claims=mock.ANY,
        )
        qwen_turn.assert_not_called()

    def test_failed_fresh_turn_stops_components(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="what time is it"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", side_effect=RuntimeError("LLM failed")
            ),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        stop_components.assert_called_once()
        self.assertEqual(daemon.conversation_deadline, 0.0)
        self.assertEqual(daemon.history, [])
        self.assertIsNone(daemon.cursor_session)

    def test_failed_active_followup_keeps_components_until_timeout(self) -> None:
        daemon = _bare_daemon()
        daemon.conversation_deadline = time.monotonic() + 30
        daemon.history = [{"role": "user", "content": "earlier"}]
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="try again"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", side_effect=RuntimeError("LLM failed")
            ),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        stop_components.assert_not_called()
        self.assertGreater(daemon.conversation_deadline, time.monotonic())
        self.assertEqual(daemon.history, [{"role": "user", "content": "earlier"}])

    def test_failed_playback_releases_foreground_delivery(self) -> None:
        daemon = _bare_daemon()

        def qwen_with_delivery(
            _text: str,
            _history: object,
            _session: object,
            *,
            trusted_utterance: str,
            delivery_claims: list[tuple[str, str]],
        ) -> tuple[str, None]:
            self.assertEqual(trusted_utterance, "question")
            delivery_claims.append(("123456789abc", "claim"))
            return "answer", None

        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="question"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "qwen_turn", side_effect=qwen_with_delivery),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                side_effect=RuntimeError("playback failed"),
            ),
            mock.patch.object(wake_daemon, "release_deliveries") as release,
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(woke=False)

        release.assert_called_once_with([("123456789abc", "claim")])
        self.assertEqual(daemon.history, [])


class AnnounceJobTests(unittest.TestCase):
    def test_play_response_finalizes_queued_job_announcements(self) -> None:
        daemon = _bare_daemon()
        job_batch = _playback_batch(
            "Cursor needs clarification. which repo?",
            job_id="job1",
            delivery_token="claim",
            job_status="awaiting_user",
        )
        response_batch = _playback_batch("Here is the answer.")
        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(job_batch + response_batch, None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.play_response("Here is the answer.")

        acknowledge.assert_called_once_with("job1", "claim")
        self.assertEqual(daemon.cursor_session, "job1")

    def test_awaiting_user_job_enables_followup(self) -> None:
        daemon = _bare_daemon()
        job: dict[str, object] = {
            "id": "job1",
            "status": "awaiting_user",
            "question": "which repo?",
            "_delivery_token": "claim",
        }
        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(
                    _playback_batch(
                        "Cursor needs clarification. which repo?",
                        job_id="job1",
                        delivery_token="claim",
                        job_status="awaiting_user",
                    ),
                    None,
                ),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(job)
            daemon._play_pending_announcements()

        acknowledge.assert_called_once_with("job1", "claim")
        self.assertTrue(daemon.awaiting_followup)
        self.assertEqual(daemon.cursor_session, "job1")
        self.assertGreater(daemon.conversation_deadline, 0.0)
        stop_components.assert_not_called()

    def test_completed_job_enables_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery"),
            mock.patch.object(wake_daemon, "release_delivery"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(
                    _playback_batch(
                        "Cursor finished. done",
                        job_id="job2",
                        job_status="completed",
                    ),
                    None,
                ),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                {"id": "job2", "status": "completed", "result": "done"}
            )
            daemon._play_pending_announcements()

        self.assertTrue(daemon.awaiting_followup)
        self.assertIsNone(daemon.cursor_session)
        self.assertEqual(
            daemon.history,
            [{"role": "assistant", "content": "Cursor finished. done"}],
        )

    def test_playback_failure_releases_delivery_without_acknowledging(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                side_effect=RuntimeError("speaker unavailable"),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                {
                    "id": "job2",
                    "status": "completed",
                    "result": "done",
                    "_delivery_token": "claim",
                }
            )
            daemon._play_pending_announcements()

        acknowledge.assert_not_called()
        release.assert_called_once_with("job2", "claim")
        self.assertEqual(len(daemon.playback_queue), 0)

    def test_announcement_prefetch_starts_only_after_components_are_ready(
        self,
    ) -> None:
        daemon = _bare_daemon()
        events: list[str] = []
        handle = mock.Mock()
        handle.wait.side_effect = wake_daemon.HarnessError("socket refused")

        def create_handle(_text: str) -> object:
            events.append("prefetch")
            return handle

        with (
            mock.patch.object(
                wake_daemon,
                "start_components",
                side_effect=lambda: events.append("components-ready"),
            ),
            mock.patch(
                "local_voice_harness.tts.queue.PrefetchHandle",
                side_effect=create_handle,
            ),
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                {
                    "id": "job2",
                    "status": "completed",
                    "result": "done",
                    "_delivery_token": "claim",
                }
            )
            self.assertEqual(events, [])
            daemon._play_pending_announcements()

        self.assertEqual(events, ["components-ready", "prefetch"])
        handle.discard.assert_called_once()
        acknowledge.assert_not_called()
        release.assert_called_once_with("job2", "claim")
        self.assertEqual(len(daemon.playback_queue), 0)

    def test_acknowledgement_happens_after_playback(self) -> None:
        daemon = _bare_daemon()
        events: list[str] = []

        def fake_drain(_response: str, *, on_played: object = None):
            events.append("played")
            return (
                _playback_batch(
                    "Cursor finished. done",
                    job_id="job2",
                    delivery_token="claim",
                    job_status="completed",
                ),
                None,
            )

        with (
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(daemon, "_drain_playback_queue", side_effect=fake_drain),
            mock.patch.object(
                wake_daemon,
                "acknowledge_delivery",
                side_effect=lambda _job, _token: events.append("acknowledged"),
            ),
            mock.patch.object(wake_daemon, "release_delivery"),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                {
                    "id": "job2",
                    "status": "completed",
                    "result": "done",
                    "_delivery_token": "claim",
                }
            )
            daemon._play_pending_announcements()

        self.assertEqual(events, ["played", "acknowledged"])

    def test_later_completed_job_does_not_clear_awaiting_session(self) -> None:
        daemon = _bare_daemon()
        daemon._enqueue_job_announcement(
            {"id": "job1", "status": "awaiting_user", "question": "which repo?"}
        )
        daemon._enqueue_job_announcement(
            {"id": "job2", "status": "completed", "result": "done"}
        )
        batch = _playback_batch(
            "Cursor needs clarification. which repo?",
            job_id="job1",
            job_status="awaiting_user",
        ) + _playback_batch(
            "Cursor finished. done",
            job_id="job2",
            job_status="completed",
        )
        with (
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                daemon, "_drain_playback_queue", return_value=(batch, None)
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._play_pending_announcements()

        self.assertEqual(daemon.cursor_session, "job1")

    def test_failure_releases_only_unplayed_deliveries(self) -> None:
        daemon = _bare_daemon()
        daemon._enqueue_job_announcement(
            {
                "id": "job1",
                "status": "completed",
                "result": "first",
                "_delivery_token": "claim1",
            }
        )
        daemon._enqueue_job_announcement(
            {
                "id": "job2",
                "status": "completed",
                "result": "second",
                "_delivery_token": "claim2",
            }
        )
        with daemon.playback_queue._lock:
            first_request = daemon.playback_queue._items[0][0]

        def fail_after_first(
            _response: str,
            *,
            on_played: object = None,
        ) -> None:
            callback = on_played
            callback(  # type: ignore[operator]
                {"played_text": first_request.text},
                False,
                first_request,
            )
            raise RuntimeError("second item failed")

        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                daemon, "_drain_playback_queue", side_effect=fail_after_first
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._play_pending_announcements()

        acknowledge.assert_called_once_with("job1", "claim1")
        release.assert_called_once_with("job2", "claim2")


class ComponentSynchronizationTests(unittest.TestCase):
    def test_cleanup_waits_for_activation_before_stopping(self) -> None:
        daemon = _bare_daemon()
        activation_started = threading.Event()
        finish_activation = threading.Event()
        events: list[str] = []

        def warm_qwen(*_args: object, **_kwargs: object) -> tuple[str, None]:
            activation_started.set()
            finish_activation.wait(timeout=2)
            return "OK", None

        with (
            mock.patch.object(wake_daemon.subprocess, "run"),
            mock.patch.object(wake_daemon, "llm_ready", return_value=True),
            mock.patch.object(wake_daemon, "qwen_turn", side_effect=warm_qwen),
            mock.patch.object(
                wake_daemon,
                "start_components",
                side_effect=lambda: events.append("started"),
            ),
            mock.patch.object(
                wake_daemon,
                "stop_components",
                side_effect=lambda: events.append("stopped"),
            ),
        ):
            daemon.begin_activation()
            self.assertTrue(activation_started.wait(timeout=1))
            cleanup = threading.Thread(target=daemon.stop_components_when_idle)
            cleanup.start()
            self.assertNotIn("stopped", events)
            finish_activation.set()
            assert daemon.activation_thread is not None
            daemon.activation_thread.join(timeout=2)
            cleanup.join(timeout=2)

        self.assertEqual(events, ["started", "stopped"])


class PlaybackBargeInTests(unittest.TestCase):
    def test_wake_word_cancels_playback_and_preserves_preroll(self) -> None:
        daemon = _bare_daemon()
        daemon.np = mock.Mock()  # type: ignore[reportAttributeAccessIssue]
        daemon.np.frombuffer.return_value = object()
        daemon.wake_key = "hey_jarvis"
        daemon.read_frame = mock.Mock(side_effect=[b"quiet", b"wake"])  # type: ignore[method-assign]
        daemon.wake_model.predict.side_effect = [
            {"hey_jarvis": 0.1},
            {"hey_jarvis": 0.9},
        ]

        def fake_drain(
            *, should_interrupt: object = None, on_played: object = None
        ) -> list[tuple[dict[str, object], bool, PlaybackRequest]]:
            check = should_interrupt
            self.assertFalse(check())  # type: ignore[operator]
            interrupted = check()  # type: ignore[operator]
            return [
                (
                    {"interrupted": interrupted, "played_text": "first sentence"},
                    interrupted,
                    PlaybackRequest(text="A harmless response."),
                )
            ]

        with (
            mock.patch.object(wake_daemon, "BARGE_IN_MODE", "wake"),
            mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain),
        ):
            result, interruption = daemon.play_response("A harmless response.")

        self.assertTrue(result["interrupted"])
        self.assertIsNotNone(interruption)
        assert interruption is not None
        self.assertTrue(interruption.woke)
        self.assertEqual(interruption.initial, [b"quiet", b"wake"])

    def test_vad_requires_configured_sustained_speech(self) -> None:
        daemon = _bare_daemon()
        daemon.read_frame = mock.Mock(return_value=b"speech")  # type: ignore[method-assign]
        daemon.is_speech = mock.Mock(return_value=True)  # type: ignore[method-assign]

        def fake_drain(
            *, should_interrupt: object = None, on_played: object = None
        ) -> list[tuple[dict[str, object], bool, PlaybackRequest]]:
            check = should_interrupt
            decisions = [check() for _ in range(3)]  # type: ignore[operator]
            return [
                (
                    {"interrupted": decisions[-1], "played_text": ""},
                    decisions[-1],
                    PlaybackRequest(text="response"),
                )
            ]

        with (
            mock.patch.object(wake_daemon, "BARGE_IN_MODE", "vad"),
            mock.patch.object(wake_daemon, "BARGE_IN_SPEECH_FRAMES", 3),
            mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain),
        ):
            _, interruption = daemon.play_response("response")

        self.assertIsNotNone(interruption)
        assert interruption is not None
        self.assertFalse(interruption.woke)
        self.assertEqual(len(interruption.initial), 3)

    def test_off_mode_drains_frames_without_acoustic_cancellation(self) -> None:
        daemon = _bare_daemon()
        daemon.read_frame = mock.Mock(return_value=b"speaker echo")  # type: ignore[method-assign]

        def fake_drain(
            *, should_interrupt: object = None, on_played: object = None
        ) -> list[tuple[dict[str, object], bool, PlaybackRequest]]:
            check = should_interrupt
            self.assertFalse(check())  # type: ignore[operator]
            self.assertFalse(check())  # type: ignore[operator]
            return [
                (
                    {"interrupted": False, "played_text": "response"},
                    False,
                    PlaybackRequest(text="response"),
                )
            ]

        with (
            mock.patch.object(wake_daemon, "BARGE_IN_MODE", "off"),
            mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain),
            mock.patch.object(daemon, "wait_for_playback_quiet") as quiet,
        ):
            _, interruption = daemon.play_response("response")

        self.assertIsNone(interruption)
        quiet.assert_called_once()
        daemon.wake_model.predict.assert_not_called()

    def test_response_wake_phrase_cannot_trigger_itself(self) -> None:
        daemon = _bare_daemon()
        daemon.np = mock.Mock()  # type: ignore[reportAttributeAccessIssue]
        daemon.np.frombuffer.return_value = object()
        daemon.wake_key = "hey_jarvis"
        daemon.read_frame = mock.Mock(return_value=b"echo")  # type: ignore[method-assign]
        daemon.wake_model.predict.return_value = {"hey_jarvis": 1.0}

        def fake_drain(
            *, should_interrupt: object = None, on_played: object = None
        ) -> list[tuple[dict[str, object], bool, PlaybackRequest]]:
            self.assertFalse(should_interrupt())  # type: ignore[operator]
            return [
                (
                    {"interrupted": False, "played_text": "Hey Jarvis."},
                    False,
                    PlaybackRequest(text="Say Hey Jarvis."),
                )
            ]

        with (
            mock.patch.object(wake_daemon, "BARGE_IN_MODE", "wake"),
            mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain),
            mock.patch.object(daemon, "wait_for_playback_quiet"),
            mock.patch.object(wake_daemon, "log"),
        ):
            _, interruption = daemon.play_response("Say Hey Jarvis.")

        self.assertIsNone(interruption)

    def test_quiet_gate_clears_echo_before_followup_rearms(self) -> None:
        daemon = _bare_daemon()
        daemon.microphone = mock.Mock()
        daemon.microphone.poll.return_value = None
        daemon.pre_roll.extend([b"old"])
        daemon.read_frame = mock.Mock(  # type: ignore[method-assign]
            side_effect=[b"speech", b"quiet", b"quiet"]
        )
        daemon.is_speech = lambda frame: frame == b"speech"  # type: ignore[method-assign]

        with (
            mock.patch.object(wake_daemon, "PLAYBACK_QUIET_FRAMES", 2),
            mock.patch.object(wake_daemon, "PLAYBACK_QUIET_TIMEOUT_SECONDS", 1),
        ):
            daemon.wait_for_playback_quiet()

        self.assertEqual(daemon.read_frame.call_count, 3)
        self.assertEqual(list(daemon.pre_roll), [])


class InterruptedTurnTests(unittest.TestCase):
    def test_only_played_assistant_prefix_is_added_to_history(self) -> None:
        daemon = _bare_daemon()
        interruption = wake_daemon.BargeIn(initial=[b"user"], woke=False)
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="tell me more"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", return_value=("first. second.", None)
            ),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                daemon,
                "play_response",
                return_value=(
                    {"interrupted": True, "played_text": "first."},
                    interruption,
                ),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            result = daemon.process_utterance(woke=False)

        self.assertIs(result, interruption)
        self.assertEqual(
            daemon.history,
            [
                {"role": "user", "content": "tell me more"},
                {"role": "assistant", "content": "first."},
            ],
        )

    def test_job_announcement_keeps_components_for_barge_in(self) -> None:
        daemon = _bare_daemon()
        interruption = wake_daemon.BargeIn(initial=[b"user"], woke=True)
        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(
                    _playback_batch(
                        "Cursor finished. done",
                        interrupted=True,
                        job_id="job3",
                        delivery_token="claim",
                        job_status="completed",
                    ),
                    interruption,
                ),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                {
                    "id": "job3",
                    "status": "completed",
                    "result": "done",
                    "_delivery_token": "claim",
                }
            )
            result = daemon._play_pending_announcements()

        self.assertIs(result, interruption)
        acknowledge.assert_not_called()
        release.assert_called_once_with("job3", "claim")
        stop_components.assert_not_called()


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

        with (
            mock.patch.object(wake_daemon, "recover_jobs") as recover,
            mock.patch.object(wake_daemon, "pending_results", return_value=[]),
        ):
            daemon.run()

        recover.assert_called_once()
        self.assertEqual(processed, [False], "follow-up must be handled as woke=False")
        self.assertEqual(len(recorded), 1)


if __name__ == "__main__":
    unittest.main()
