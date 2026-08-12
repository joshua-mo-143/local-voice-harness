from __future__ import annotations

import collections
import contextlib
import io
import json
import os
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_voice_harness import llm, llm_transport, recorder
from local_voice_harness.browser_context import RequestContext
from local_voice_harness.config import load_backend_settings
from local_voice_harness.cursor import service as cursor_service
from local_voice_harness.cursor.delivery import DeliveryClaim
from local_voice_harness.cursor.model import CursorJob, JobStatus
from local_voice_harness.cursor.service import CursorTurnRequest
from local_voice_harness.cursor.store import JobStore
from local_voice_harness.errors import HarnessError, NoSpeechError
from local_voice_harness.integrations.registry import build_integration_registry
from local_voice_harness.intent import Intent, IntentRoute
from local_voice_harness.questions import (
    AnswerProvenance,
    Choice,
    Question,
    QuestionKind,
    QuestionOrigin,
    QuestionSensitivity,
)
from local_voice_harness.responses import AssistantResponse
from local_voice_harness.tts.queue import PlaybackQueue, PlaybackRequest
from local_voice_harness.user_config import default_user_config, load_user_config
from local_voice_harness.wake import daemon as wake_daemon
from local_voice_harness.wake.daemon import WakeConversationDaemon

AUDIO_GENERATION = Path(
    "/runtime/voice-harness/recordings/request-0123456789abcdef0123456789abcdef.wav"
)
_DEFAULT_CONFIG = default_user_config()
USER_CONFIG = replace(
    _DEFAULT_CONFIG,
    providers=load_backend_settings(
        {
            "VOICE_HARNESS_LLM_PROVIDER": "local",
            "VOICE_HARNESS_TTS_PROVIDER": "local",
        },
        path=Path(os.devnull),
    ),
)


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


def _delivery_claim(
    job_id: str,
    status: str,
    *,
    result: str | None = None,
    question: str | None = None,
    voice_question: dict[str, object] | None = None,
    token: str = "claim",
) -> DeliveryClaim:
    job_id = {
        "job1": "aaaaaaaaaaaa",
        "job2": "bbbbbbbbbbbb",
        "job3": "cccccccccccc",
    }.get(job_id, job_id)
    values: dict[str, object] = {
        "id": job_id,
        "status": status,
        "request": "test",
        "created_at": 1,
        "delivered": False,
    }
    if result is not None:
        values["result"] = result
    if question is not None:
        values["question"] = question
    if voice_question is not None:
        values["voice_question"] = voice_question
    if status in {"completed", "failed", "cancelled", "blocked"}:
        values["completed_at"] = 1
    if status == "failed":
        values["error"] = result or "failed"
    return DeliveryClaim(CursorJob.from_dict(values), token)


def _bare_daemon() -> WakeConversationDaemon:
    """Build a daemon without running __init__ (which loads the wake model/VAD)."""
    instance = WakeConversationDaemon.__new__(WakeConversationDaemon)
    instance.user_config = USER_CONFIG
    instance.audio = USER_CONFIG.audio
    instance.platform = USER_CONFIG.platform
    instance.providers = USER_CONFIG.providers
    instance.integrations = build_integration_registry(USER_CONFIG)
    instance.history = []
    instance.cursor_session = None
    instance.completed_followup = None
    instance.recent_playback = collections.deque(
        maxlen=wake_daemon.RECENT_PLAYBACK_LIMIT
    )
    instance.conversation_deadline = 0.0
    instance.awaiting_followup = False
    instance.last_wake = 0.0
    instance.force_listen = threading.Event()
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


def _pending_choice_snapshot() -> wake_daemon.PendingQuestionSnapshot:
    question = Question(
        id="question-1",
        text="Which database should I use?",
        kind=QuestionKind.MULTIPLE_CHOICE,
        choices=(
            Choice("sqlite", "Use SQLite"),
            Choice("postgres", "Use PostgreSQL"),
        ),
        sensitivity=QuestionSensitivity.ROUTINE,
        origin=QuestionOrigin("cursor", "oldjob123456", "oldjob123456-routing-0"),
        owner="agent",
        asked_at=1,
    )
    return wake_daemon.PendingQuestionSnapshot(
        "oldjob123456",
        question.text,
        question.owner,
        question.id,
        question.origin.turn_token,
        question,
    )


def _pending_free_text_snapshot() -> wake_daemon.PendingQuestionSnapshot:
    question = Question(
        id="question-1",
        text="Which repository?",
        kind=QuestionKind.FREE_TEXT,
        sensitivity=QuestionSensitivity.ROUTINE,
        origin=QuestionOrigin("cursor", "oldjob123456", "oldjob123456-routing-0"),
        owner="repository",
        asked_at=1,
    )
    return wake_daemon.PendingQuestionSnapshot(
        "oldjob123456",
        question.text,
        question.owner,
        question.id,
        question.origin.turn_token,
        question,
    )


class StartupConfigSnapshotTests(unittest.TestCase):
    def test_restart_observes_file_change_without_mutating_running_daemon(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text('[audio]\nsource = "startup-microphone"\n')
            startup = load_user_config(
                {}, path=path, backends_path=Path(os.devnull), home=Path(temporary)
            )

            openwakeword = types.ModuleType("openwakeword")
            openwakeword.__file__ = "/tmp/openwakeword/__init__.py"
            model_module = types.ModuleType("openwakeword.model")
            wake_model = mock.Mock(models={"hey_jarvis": object()})
            model_module.Model = mock.Mock(return_value=wake_model)  # type: ignore[attr-defined]
            numpy = types.ModuleType("numpy")

            with (
                mock.patch.dict(
                    "sys.modules",
                    {
                        "numpy": numpy,
                        "openwakeword": openwakeword,
                        "openwakeword.model": model_module,
                    },
                ),
                mock.patch.object(wake_daemon, "SpeechDetector"),
            ):
                running = WakeConversationDaemon(startup)
                path.write_text('[audio]\nsource = "restarted-microphone"\n')
                restarted = WakeConversationDaemon(
                    load_user_config(
                        {},
                        path=path,
                        backends_path=Path(os.devnull),
                        home=Path(temporary),
                    )
                )

        self.assertEqual(running.audio.source, "startup-microphone")
        self.assertEqual(restarted.audio.source, "restarted-microphone")


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
    def test_response_channels_are_selected_at_wake_boundary(self) -> None:
        daemon = _bare_daemon()
        output = io.StringIO()
        response = AssistantResponse(
            spoken_text="The job started.",
            display_text="Started job 123456789abc in /tmp/example.",
        )
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="start it"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("start it"),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                wake_daemon, "qwen_turn", return_value=(response, None)
            ) as qwen,
            mock.patch.object(
                daemon,
                "play_response",
                return_value=({"played_text": response.spoken_text}, None),
            ) as play,
            mock.patch.object(wake_daemon, "notify"),
            contextlib.redirect_stdout(output),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        self.assertIn(f"Assistant: {response.display_text}", output.getvalue())
        self.assertNotIn(response.spoken_text, output.getvalue())
        play.assert_called_once_with(response)
        qwen.assert_called_once()

    def test_completed_turn_enables_followup(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="what time is it"
            ) as transcribe,
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
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        qwen_turn.assert_called_once_with(
            "what time is it\n\nGitHub context",
            mock.ANY,
            None,
            trusted_utterance="what time is it",
            delivery_claims=mock.ANY,
            allow_tools=False,
            settings=daemon.providers,
        )
        transcribe.assert_called_once_with(AUDIO_GENERATION)
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

    def test_end_conversation_route_closes_and_skips_tools(self) -> None:
        daemon = _bare_daemon()
        daemon.awaiting_followup = True
        daemon.conversation_deadline = time.monotonic() + 30
        daemon.history = [{"role": "user", "content": "earlier"}]
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="thanks, that's all"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("bye"), None),
            ),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.END_CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            result = daemon.process_utterance(AUDIO_GENERATION, woke=False)

        self.assertIsNone(result)
        qwen_turn.assert_not_called()
        cursor_turn.assert_not_called()
        self.assertFalse(daemon.awaiting_followup)
        self.assertEqual(daemon.conversation_deadline, 0.0)
        self.assertEqual(daemon.history, [])

    def test_end_conversation_barge_in_keeps_conversation_open(self) -> None:
        daemon = _bare_daemon()
        daemon.awaiting_followup = True
        daemon.conversation_deadline = time.monotonic() + 30
        interruption = wake_daemon.BargeIn(initial=[b"user"], woke=False)
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="that's everything then"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.END_CONVERSATION, "high"),
            ),
            mock.patch.object(
                daemon,
                "play_response",
                return_value=({"interrupted": True}, interruption),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            result = daemon.process_utterance(AUDIO_GENERATION, woke=False)

        self.assertIs(result, interruption)
        stop_components.assert_not_called()

    def test_venice_turn_streams_sentence_chunks_to_playback(self) -> None:
        daemon = _bare_daemon()
        daemon.providers = replace(daemon.providers, llm_provider="venice")
        played_requests: list[str] = []

        def streamed_turn(
            _text: str,
            _history: object,
            _session: object,
            **kwargs: object,
        ) -> tuple[str, None]:
            callback = kwargs["on_text_chunk"]
            assert callable(callback)
            callback("First sentence.")
            callback("Second sentence.")
            return "First sentence. Second sentence.", None

        def drain(
            _response: str,
            **_kwargs: object,
        ) -> tuple[
            list[tuple[dict[str, object], bool, PlaybackRequest]],
            None,
        ]:
            with daemon.playback_queue._lock:
                requests = [
                    request for request, _handle in daemon.playback_queue._items
                ]
                daemon.playback_queue._items.clear()
            played_requests.extend(request.text for request in requests)
            return [
                (
                    {
                        "ok": True,
                        "played_text": request.text,
                        "interrupted": False,
                    },
                    False,
                    request,
                )
                for request in requests
            ], None

        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="tell me something"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon, "qwen_turn", side_effect=streamed_turn
            ) as qwen_turn,
            mock.patch.object(daemon, "_drain_playback_queue", side_effect=drain),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("tell me something"),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        self.assertEqual(played_requests, ["First sentence.", "Second sentence."])
        self.assertEqual(
            daemon.history,
            [
                {"role": "user", "content": "tell me something"},
                {
                    "role": "assistant",
                    "content": "First sentence. Second sentence.",
                },
            ],
        )
        self.assertIn("on_text_chunk", qwen_turn.call_args.kwargs)

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
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

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
            daemon.process_utterance(AUDIO_GENERATION, woke=True)

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
            daemon.process_utterance(AUDIO_GENERATION, woke=True)

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
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_choice_snapshot(),
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
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "work on APP-43 instead",
                utterance="work on APP-43 instead",
                context_repository=None,
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
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
                daemon,
                "_pending_cursor_question",
                return_value=_pending_free_text_snapshot(),
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
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "use the api repository",
                "oldjob123456",
                utterance="use the api repository",
                action="reply",
                job_id="oldjob123456",
                expected_question_id="question-1",
                expected_question_turn="oldjob123456-routing-0",
                answer_provenance=AnswerProvenance.USER_VOICE,
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )
        qwen_turn.assert_not_called()

    def test_pending_question_closes_silently_on_filler_speech(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        daemon.awaiting_followup = True
        daemon.conversation_deadline = time.monotonic() + 30
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="um yeah so"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("um yeah so"),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_choice_snapshot(),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_REPLY, "high"),
            ),
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(daemon, "play_response") as play_response,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_not_called()
        qwen_turn.assert_not_called()
        play_response.assert_not_called()
        stop_components.assert_called_once()
        self.assertEqual(daemon.cursor_session, "oldjob123456")
        self.assertFalse(daemon.awaiting_followup)
        self.assertEqual(daemon.conversation_deadline, 0.0)

    def test_pending_free_text_question_does_not_accept_filler_as_answer(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        daemon.awaiting_followup = True
        daemon.conversation_deadline = time.monotonic() + 30
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="and uh"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("and uh"),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_free_text_snapshot(),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_REPLY, "high"),
            ),
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(daemon, "play_response") as play_response,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_not_called()
        qwen_turn.assert_not_called()
        play_response.assert_not_called()
        self.assertEqual(daemon.cursor_session, "oldjob123456")
        self.assertFalse(daemon.awaiting_followup)

    def test_pending_question_rejects_unrelated_multi_speaker_speech(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon,
                "transcribe",
                return_value="did you feed the dog no I thought you did",
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("did you feed the dog"),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_choice_snapshot(),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(daemon, "play_response") as play_response,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_not_called()
        qwen_turn.assert_not_called()
        play_response.assert_not_called()
        self.assertEqual(daemon.cursor_session, "oldjob123456")

    def test_pending_question_rejects_implicit_submit_from_contaminated_stt(
        self,
    ) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        transcript = (
            "Option Turser started local voice harness issue two zero eight fixture"
        )
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value=transcript),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext(transcript),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_choice_snapshot(),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(daemon, "play_response") as play_response,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_not_called()
        qwen_turn.assert_not_called()
        play_response.assert_not_called()
        self.assertEqual(daemon.cursor_session, "oldjob123456")

    def test_pending_question_accepts_explicit_option_despite_conversation_route(
        self,
    ) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon,
                "transcribe",
                return_value="Actually, I meant option two",
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("Actually, I meant option two"),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_choice_snapshot(),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                wake_daemon, "cursor_turn", return_value=("continued", None)
            ) as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("continued"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        request = cursor_turn.call_args.args[0]
        self.assertEqual(request.action, "reply")
        self.assertEqual(request.utterance, "Actually, I meant option two")
        qwen_turn.assert_not_called()

    def test_pending_question_allows_explicit_request_to_continue_talking(
        self,
    ) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon,
                "transcribe",
                return_value="keep talking with me about databases",
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("keep talking with me about databases"),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_choice_snapshot(),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION_CONTINUE, "high"),
            ),
            mock.patch.object(
                wake_daemon,
                "qwen_turn",
                return_value=("We can compare their tradeoffs.", "oldjob123456"),
            ) as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("tradeoffs"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        qwen_turn.assert_called_once()
        self.assertEqual(daemon.cursor_session, "oldjob123456")
        self.assertTrue(daemon.awaiting_followup)

    def test_pending_question_allows_related_consultation(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon,
                "transcribe",
                return_value="which option would you recommend",
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("which option would you recommend"),
            ),
            mock.patch.object(
                daemon,
                "_pending_cursor_question",
                return_value=_pending_choice_snapshot(),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.QUESTION_CONSULTATION, "high"),
            ),
            mock.patch.object(
                wake_daemon,
                "qwen_turn",
                return_value=("SQLite is simpler for a local tool.", "oldjob123456"),
            ) as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("SQLite is simpler."), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        qwen_turn.assert_called_once()
        self.assertEqual(daemon.cursor_session, "oldjob123456")
        self.assertTrue(daemon.awaiting_followup)

    def test_pending_question_snapshot_uses_one_store_read(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "aaaaaaaaaaaa"
        awaiting = CursorJob.from_dict(
            {
                "id": "aaaaaaaaaaaa",
                "status": "awaiting_user",
                "question": "Which repository?",
                "clarification_kind": "repository",
                "created_at": 1,
            }
        )
        with mock.patch.object(
            wake_daemon.CURSOR_STORE, "get", return_value=awaiting
        ) as get:
            snapshot = daemon._pending_cursor_question()

        get.assert_called_once_with("aaaaaaaaaaaa")
        assert snapshot is not None
        self.assertEqual(snapshot.text, "Which repository?")
        self.assertEqual(snapshot.owner, "repository")

    def test_answer_later_is_not_a_control_without_pending_snapshot(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="answer later"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("answer later"),
            ),
            mock.patch.object(daemon, "_pending_cursor_question", return_value=None),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(
                wake_daemon, "qwen_turn", return_value=("What should wait?", None)
            ) as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_not_called()
        qwen_turn.assert_called_once()

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
                    "ask Cursor to work on APP-43\n\nLinear context",
                    focused_issue="APP-43",
                    external_issue_reference="APP-43",
                    external_issue_source="linear",
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
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "ask Cursor to work on APP-43\n\nLinear context",
                utterance="ask Cursor to work on APP-43",
                context_repository=None,
                issue_key="APP-43",
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )
        qwen_turn.assert_not_called()

    def test_issue_list_scope_reaches_voice_submission(self) -> None:
        daemon = _bare_daemon()
        context = RequestContext(
            "work on issues 4 and 9\n\nLinear team issue list",
            issue_scope="ENG",
            issue_scope_source="linear",
        )
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="work on issues 4 and 9"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "request_context", return_value=context),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                wake_daemon, "cursor_turn", return_value=("started", None)
            ) as cursor_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        request = cursor_turn.call_args.args[0]
        self.assertEqual(request.issue_scope, "ENG")
        self.assertEqual(request.issue_scope_source, "linear")

    def test_uncertain_bare_ticket_batch_requests_repository_scope(self) -> None:
        daemon = _bare_daemon()
        text = "Can you work on issues 92, 93 and 95?"
        with (
            mock.patch.object(wake_daemon, "transcribe", return_value=text),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.UNCERTAIN, "low"),
            ),
            mock.patch.object(wake_daemon, "cursor_turn") as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "play_response",
                return_value=({"played_text": ""}, None),
            ) as play,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_not_called()
        qwen_turn.assert_not_called()
        play.assert_called_once_with(
            AssistantResponse.from_text(wake_daemon.MISSING_ISSUE_SCOPE_RESPONSE)
        )

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
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

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
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(wake_daemon, "stop_components") as stop_components,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

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
            delivery_claims: list[DeliveryClaim],
            allow_tools: bool = True,
            settings: object | None = None,
        ) -> tuple[str, None]:
            self.assertEqual(trusted_utterance, "question")
            delivery_claims.append(
                _delivery_claim("123456789abc", "completed", result="answer")
            )
            return "answer", None

        with (
            mock.patch.object(wake_daemon, "transcribe", return_value="question"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "qwen_turn", side_effect=qwen_with_delivery),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
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
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        release.assert_called_once_with(
            [_delivery_claim("123456789abc", "completed", result="answer")]
        )
        self.assertEqual(daemon.history, [])


class InboxIntentRoutingTests(unittest.TestCase):
    def test_list_intent_routes_to_inbox(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="what jobs are running"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_LIST, "high"),
            ),
            mock.patch.object(
                wake_daemon, "cursor_turn", return_value=("you have 2 jobs", None)
            ) as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_called_once_with(
            CursorTurnRequest("", None, action="list"),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )
        qwen_turn.assert_not_called()

    def test_dismiss_intent_passes_reference_and_session(self) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "oldjob123456"
        with (
            mock.patch.object(
                wake_daemon, "transcribe", return_value="dismiss the bug fix"
            ),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_DISMISS, "high"),
            ),
            mock.patch.object(
                wake_daemon, "cursor_turn", return_value=("Dismissed the update.", None)
            ) as cursor_turn,
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        cursor_turn.assert_called_once_with(
            CursorTurnRequest(
                "dismiss the bug fix",
                "oldjob123456",
                action="dismiss",
                job_id="oldjob123456",
                reference="dismiss the bug fix",
            ),
            delivery_claims=mock.ANY,
            integrations=mock.ANY,
        )
        qwen_turn.assert_not_called()


class AnnounceJobTests(unittest.TestCase):
    def test_multiple_choice_announcement_numbers_stored_options(self) -> None:
        daemon = _bare_daemon()
        claim = _delivery_claim(
            "job1",
            "awaiting_user",
            question="What should authorize irreversible audio deletion?",
            voice_question={
                "version": 1,
                "id": "question-1",
                "text": "What should authorize irreversible audio deletion?",
                "kind": "multiple_choice",
                "sensitivity": "architecture",
                "origin": {
                    "provider": "cursor",
                    "job_id": "aaaaaaaaaaaa",
                    "turn_token": "aaaaaaaaaaaa-1",
                },
                "choices": [
                    {
                        "id": "send-complete",
                        "label": "Successful server send completion",
                    },
                    {
                        "id": "client-ack",
                        "label": "Explicit client acknowledgment",
                    },
                ],
                "owner": "workflow",
                "state": "pending",
                "asked_at": 1,
            },
        )

        response = daemon._job_response(claim.job)

        self.assertEqual(
            response.spoken_text,
            "Cursor needs clarification for test. What should authorize irreversible "
            "audio deletion? Option 1 is Successful server send completion. "
            "Option 2 is Explicit client acknowledgment. Please choose one.",
        )
        self.assertIn(
            "Cursor job aaaaaaaaaaaa (test) needs clarification", response.display_text
        )
        self.assertIn("Option 2", response.display_text)

    def test_job_announcement_queues_only_spoken_and_prints_display(self) -> None:
        daemon = _bare_daemon()
        claim = _delivery_claim(
            "job2",
            "completed",
            result="Changed /srv/example/config.toml at commit abc123.",
        )
        response = daemon._job_response(claim.job)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            daemon._enqueue_job_announcement(claim)

        self.assertEqual(
            output.getvalue(),
            f"Assistant: {response.display_text}\n",
        )
        self.assertEqual(daemon.playback_queue.queued_text(), response.spoken_text)
        self.assertNotIn("/srv/example", daemon.playback_queue.queued_text())
        self.assertIn("/srv/example", response.display_text)
        with daemon.playback_queue._lock:
            queued_request = daemon.playback_queue._items[0][0]
        self.assertEqual(queued_request.job_completed_at, claim.job.completed_at)
        self.assertEqual(
            queued_request.display_fingerprint,
            wake_daemon._display_fingerprint(response.display_text),
        )

    def test_display_failure_does_not_queue_or_acknowledge_delivery(self) -> None:
        daemon = _bare_daemon()
        claim = _delivery_claim("job2", "completed", result="done")

        class BrokenDisplay(io.StringIO):
            def write(self, _value: str) -> int:
                raise OSError("display unavailable")

        with (
            contextlib.redirect_stdout(BrokenDisplay()),
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            self.assertRaisesRegex(OSError, "display unavailable"),
        ):
            daemon._enqueue_job_announcement(claim)

        acknowledge.assert_not_called()
        self.assertEqual(len(daemon.playback_queue), 0)

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

    def test_play_response_queues_only_the_spoken_channel(self) -> None:
        daemon = _bare_daemon()
        response = AssistantResponse(
            spoken_text="The job failed.",
            display_text="Job 123 failed in /tmp/example; inspect the logs.",
        )
        with mock.patch.object(
            daemon,
            "_drain_playback_queue",
            return_value=([], None),
        ) as drain:
            daemon.play_response(response)

        drain.assert_called_once_with(response.spoken_text, on_played=mock.ANY)
        self.assertEqual(daemon.playback_queue.queued_text(), response.spoken_text)
        self.assertNotIn(response.display_text, daemon.playback_queue.queued_text())

    def test_awaiting_user_job_enables_followup(self) -> None:
        daemon = _bare_daemon()
        claim = _delivery_claim("job1", "awaiting_user", question="which repo?")
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
            daemon._enqueue_job_announcement(claim)
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
                _delivery_claim("job2", "completed", result="done")
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
        errors = io.StringIO()
        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                side_effect=RuntimeError("speaker unavailable token=playback-secret"),
            ),
            mock.patch.object(wake_daemon, "notify") as notify,
            contextlib.redirect_stderr(errors),
        ):
            daemon._enqueue_job_announcement(
                _delivery_claim("job2", "completed", result="done")
            )
            daemon._play_pending_announcements()

        acknowledge.assert_not_called()
        release.assert_called_once_with("bbbbbbbbbbbb", "claim")
        self.assertEqual(len(daemon.playback_queue), 0)
        notify.assert_called_once_with(wake_daemon.PLAYBACK_FAILURE, error=True)
        self.assertNotIn("playback-secret", errors.getvalue())
        self.assertIn("[REDACTED]", errors.getvalue())

    def test_truncated_queue_stream_releases_without_acknowledging(self) -> None:
        daemon = _bare_daemon()
        stream_socket = mock.Mock()
        reads = 0

        def recv(_size: int) -> bytes:
            nonlocal reads
            reads += 1
            if reads > 1:
                return b""
            request = json.loads(stream_socket.sendall.call_args.args[0])
            events = [
                {
                    "ok": True,
                    "event": "start",
                    "request_id": request["request_id"],
                    "sample_rate": 24_000,
                    "chunks": 1,
                },
                {
                    "ok": True,
                    "event": "chunk",
                    "request_id": request["request_id"],
                    "index": 0,
                    "output": "/tmp/nonexistent-prefetch.wav",
                    "text": "partial",
                },
            ]
            return b"".join(json.dumps(event).encode() + b"\n" for event in events)

        stream_socket.recv.side_effect = recv
        with (
            mock.patch(
                "local_voice_harness.tts.queue.socket.socket",
                return_value=stream_socket,
            ),
            mock.patch(
                "local_voice_harness.tts.queue.select.select",
                return_value=([stream_socket], [], []),
            ),
            mock.patch(
                "local_voice_harness.tts.queue.unix_request",
                return_value=b'{"ok":true,"cancelled":true}\n',
            ),
            mock.patch(
                "local_voice_harness.tts.queue.playback_slot",
                return_value=contextlib.nullcontext(),
            ),
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                _delivery_claim("job2", "completed", result="done")
            )
            daemon._play_pending_announcements()

        acknowledge.assert_not_called()
        release.assert_called_once_with("bbbbbbbbbbbb", "claim")
        self.assertEqual(len(daemon.playback_queue), 0)
        stream_socket.close.assert_called_once()

    def test_expired_acknowledgement_releases_without_enabling_followup(self) -> None:
        daemon = _bare_daemon()
        request = PlaybackRequest(
            text="Cursor finished. done",
            job_id="bbbbbbbbbbbb",
            delivery_token="claim",
            job_status="completed",
        )

        with (
            mock.patch.object(
                wake_daemon,
                "acknowledge_delivery",
                return_value=False,
            ) as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
        ):
            daemon._finish_job_playback(
                request,
                {"played_text": request.text},
                interrupted=False,
            )

        acknowledge.assert_called_once_with("bbbbbbbbbbbb", "claim")
        release.assert_called_once_with("bbbbbbbbbbbb", "claim")
        self.assertFalse(daemon.awaiting_followup)
        self.assertIsNone(daemon.completed_followup)

    def test_interrupted_announcement_does_not_install_recent_details(self) -> None:
        daemon = _bare_daemon()
        request = PlaybackRequest(
            text="Cursor finished issue 42.",
            job_id="bbbbbbbbbbbb",
            delivery_token="claim",
            job_status="completed",
            job_completed_at=1,
            display_fingerprint="fingerprint",
        )
        with (
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon.CURSOR_STORE, "get") as get,
        ):
            daemon._finish_job_playback(
                request,
                {"played_text": ""},
                interrupted=True,
            )

        release.assert_called_once_with("bbbbbbbbbbbb", "claim")
        get.assert_not_called()
        self.assertIsNone(daemon.completed_followup)

    def test_active_announcement_renews_its_delivery_lease(self) -> None:
        daemon = _bare_daemon()
        renewed = threading.Event()

        def drain_with_renewal(
            _response: str,
            *,
            on_poll: object,
            on_played: object = None,
        ) -> tuple[
            list[tuple[dict[str, object], bool, PlaybackRequest]],
            None,
        ]:
            self.assertTrue(renewed.wait(timeout=1))
            on_poll()  # type: ignore[operator]
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
            mock.patch.object(wake_daemon, "DELIVERY_RENEW_SECONDS", 0.001),
            mock.patch.object(
                wake_daemon,
                "renew_delivery",
                side_effect=lambda _job, _token: renewed.set() or True,
            ) as renew,
            mock.patch.object(
                wake_daemon,
                "acknowledge_delivery",
                return_value=True,
            ),
            mock.patch.object(wake_daemon, "release_delivery"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(daemon.playback_queue, "start_prefetch"),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                side_effect=drain_with_renewal,
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                _delivery_claim("job2", "completed", result="done")
            )
            daemon._play_pending_announcements()

        self.assertGreaterEqual(renew.call_count, 1)
        renew.assert_called_with("bbbbbbbbbbbb", "claim")

    def test_lease_loss_cancels_and_releases_without_acknowledging(self) -> None:
        daemon = _bare_daemon()
        renewal_failed = threading.Event()

        def fail_after_lease_loss(
            _response: str,
            *,
            on_poll: object,
            on_played: object = None,
        ) -> None:
            self.assertTrue(renewal_failed.wait(timeout=1))
            on_poll()  # type: ignore[operator]

        with (
            mock.patch.object(wake_daemon, "DELIVERY_RENEW_SECONDS", 0.001),
            mock.patch.object(
                wake_daemon,
                "renew_delivery",
                side_effect=lambda _job, _token: renewal_failed.set() and False,
            ),
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(daemon.playback_queue, "start_prefetch"),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                side_effect=fail_after_lease_loss,
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                _delivery_claim("job2", "completed", result="done")
            )
            daemon._play_pending_announcements()

        acknowledge.assert_not_called()
        release.assert_called_once_with("bbbbbbbbbbbb", "claim")
        self.assertEqual(len(daemon.playback_queue), 0)

    def test_lease_renewal_exception_fails_the_guard_closed(self) -> None:
        attempted = threading.Event()
        request = PlaybackRequest(
            text="Cursor finished. done",
            job_id="bbbbbbbbbbbb",
            delivery_token="claim",
            job_status="completed",
        )

        def fail_renewal(_job_id: str, _token: str) -> bool:
            attempted.set()
            raise OSError("store unavailable")

        with (
            mock.patch.object(wake_daemon, "DELIVERY_RENEW_SECONDS", 0.001),
            mock.patch.object(
                wake_daemon,
                "renew_delivery",
                side_effect=fail_renewal,
            ),
        ):
            guard = wake_daemon._DeliveryLeaseGuard([request])
            guard.start()
            self.assertTrue(attempted.wait(timeout=1))
            with self.assertRaisesRegex(
                HarnessError,
                "lease renewal failed.*store unavailable",
            ):
                guard.maintain()
            guard.stop()

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
                side_effect=lambda _settings: events.append("components-ready"),
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
                _delivery_claim("job2", "completed", result="done")
            )
            self.assertEqual(events, [])
            daemon._play_pending_announcements()

        self.assertEqual(events, ["components-ready", "prefetch"])
        handle.discard.assert_called_once()
        acknowledge.assert_not_called()
        release.assert_called_once_with("bbbbbbbbbbbb", "claim")
        self.assertEqual(len(daemon.playback_queue), 0)

    def test_acknowledgement_happens_after_playback(self) -> None:
        daemon = _bare_daemon()
        events: list[str] = []

        def fake_drain(
            _response: str,
            *,
            on_poll: object = None,
            on_played: object = None,
        ):
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
                _delivery_claim("job2", "completed", result="done")
            )
            daemon._play_pending_announcements()

        self.assertEqual(events, ["played", "acknowledged"])

    def test_later_completed_job_does_not_clear_awaiting_session(self) -> None:
        daemon = _bare_daemon()
        daemon._enqueue_job_announcement(
            _delivery_claim("job1", "awaiting_user", question="which repo?")
        )
        daemon._enqueue_job_announcement(
            _delivery_claim("job2", "completed", result="done")
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
            _delivery_claim("job1", "completed", result="first", token="claim1")
        )
        daemon._enqueue_job_announcement(
            _delivery_claim("job2", "completed", result="second", token="claim2")
        )
        with daemon.playback_queue._lock:
            first_request = daemon.playback_queue._items[0][0]

        def fail_after_first(
            _response: str,
            *,
            on_poll: object = None,
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

        acknowledge.assert_called_once_with("aaaaaaaaaaaa", "claim1")
        release.assert_called_once_with("bbbbbbbbbbbb", "claim2")


class ComponentSynchronizationTests(unittest.TestCase):
    def test_warmup_rejects_tool_call_without_invoking_agent(self) -> None:
        daemon = _bare_daemon()
        settings = mock.Mock(
            llm_provider="local",
            llm_model="test-model",
            llm_endpoint="http://127.0.0.1:8090/v1/chat/completions",
            llm_timeout=1,
        )
        daemon.providers = settings
        response = io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "warmup-call",
                                        "type": "function",
                                        "function": {
                                            "name": "cursor",
                                            "arguments": json.dumps(
                                                {
                                                    "task": "mutate the workspace",
                                                    "action": "submit",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            ).encode()
        )

        with (
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                llm_transport.urllib.request,
                "urlopen",
                return_value=response,
            ) as urlopen,
            mock.patch.object(llm, "cursor_turn") as cursor_turn,
            mock.patch.object(wake_daemon, "log"),
        ):
            daemon.begin_activation()
            assert daemon.activation_thread is not None
            daemon.activation_thread.join(timeout=2)

        self.assertFalse(daemon.activation_thread.is_alive())
        self.assertIsInstance(daemon.activation_error, HarnessError)
        self.assertIn("tools are disabled", str(daemon.activation_error))
        cursor_turn.assert_not_called()
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

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
            mock.patch.object(
                wake_daemon, "qwen_turn", side_effect=warm_qwen
            ) as qwen_turn,
            mock.patch.object(
                wake_daemon,
                "start_components",
                side_effect=lambda _settings: events.append("started"),
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
        qwen_turn.assert_called_once_with(
            "Reply with only OK. Do not call a tool.",
            allow_tools=False,
            settings=daemon.providers,
        )


class PlaybackBargeInTests(unittest.TestCase):
    def test_wake_word_cancels_playback_and_preserves_preroll(self) -> None:
        daemon = _bare_daemon()
        daemon.audio = replace(daemon.audio, barge_in_mode="wake")
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

        with mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain):
            result, interruption = daemon.play_response("A harmless response.")

        self.assertTrue(result["interrupted"])
        self.assertIsNotNone(interruption)
        assert interruption is not None
        self.assertTrue(interruption.woke)
        self.assertEqual(interruption.initial, [b"quiet", b"wake"])

    def test_vad_requires_configured_sustained_speech(self) -> None:
        daemon = _bare_daemon()
        daemon.audio = replace(
            daemon.audio, barge_in_mode="vad", barge_in_speech_frames=3
        )
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

        with mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain):
            _, interruption = daemon.play_response("response")

        self.assertIsNotNone(interruption)
        assert interruption is not None
        self.assertFalse(interruption.woke)
        self.assertEqual(len(interruption.initial), 3)

    def test_off_mode_drains_frames_without_acoustic_cancellation(self) -> None:
        daemon = _bare_daemon()
        daemon.audio = replace(daemon.audio, barge_in_mode="off")
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
            mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain),
            mock.patch.object(daemon, "wait_for_playback_quiet") as quiet,
        ):
            _, interruption = daemon.play_response("response")

        self.assertIsNone(interruption)
        quiet.assert_called_once()
        daemon.wake_model.predict.assert_not_called()
        self.assertEqual(
            tuple(playback.text for playback in daemon.recent_playback),
            ("response",),
        )

    def test_response_wake_phrase_cannot_trigger_itself(self) -> None:
        daemon = _bare_daemon()
        daemon.audio = replace(daemon.audio, barge_in_mode="wake")
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
            mock.patch.object(daemon.playback_queue, "drain", side_effect=fake_drain),
            mock.patch.object(daemon, "wait_for_playback_quiet"),
            mock.patch.object(wake_daemon, "log"),
        ):
            _, interruption = daemon.play_response("Say Hey Jarvis.")

        self.assertIsNone(interruption)

    def test_quiet_gate_clears_echo_before_followup_rearms(self) -> None:
        daemon = _bare_daemon()
        daemon.audio = replace(
            daemon.audio,
            playback_quiet_frames=2,
            playback_quiet_timeout_seconds=1,
        )
        daemon.microphone = mock.Mock()
        daemon.microphone.poll.return_value = None
        daemon.pre_roll.extend([b"old"])
        daemon.read_frame = mock.Mock(  # type: ignore[method-assign]
            side_effect=[b"speech", b"quiet", b"quiet"]
        )
        daemon.is_speech = lambda frame: frame == b"speech"  # type: ignore[method-assign]

        daemon.wait_for_playback_quiet()

        self.assertEqual(daemon.read_frame.call_count, 3)
        self.assertEqual(list(daemon.pre_roll), [])


class PlaybackEchoSuppressionTests(unittest.TestCase):
    def test_recent_playback_prefix_is_not_admitted_as_followup(self) -> None:
        daemon = _bare_daemon()
        daemon.awaiting_followup = True
        daemon.conversation_deadline = time.monotonic() + 100
        daemon.recent_playback.append(
            wake_daemon.RecentPlayback(
                "I will report back when it finishes.",
                time.monotonic() + 5,
            )
        )

        with (
            mock.patch.object(
                wake_daemon,
                "transcribe",
                return_value="I will report back when",
            ),
            mock.patch.object(daemon, "ensure_components") as ensure_components,
            mock.patch.object(wake_daemon, "request_context") as request_context,
            mock.patch.object(wake_daemon, "log") as log,
        ):
            result = daemon.process_utterance(AUDIO_GENERATION, woke=False)

        self.assertIsNone(result)
        self.assertTrue(daemon.awaiting_followup)
        ensure_components.assert_not_called()
        request_context.assert_not_called()
        self.assertIn("matching recent local playback", log.call_args.args[0])

    def test_no_speech_from_leakage_keeps_followup_armed(self) -> None:
        daemon = _bare_daemon()
        daemon.awaiting_followup = True
        deadline = time.monotonic() + 100
        daemon.conversation_deadline = deadline

        with (
            mock.patch.object(
                wake_daemon,
                "transcribe",
                side_effect=NoSpeechError("STT did not recognize any speech"),
            ),
            mock.patch.object(wake_daemon, "release_deliveries") as release,
            mock.patch.object(daemon, "stop_components_when_idle") as stop_components,
            mock.patch.object(wake_daemon, "notify") as notify,
            mock.patch.object(wake_daemon, "log") as log,
        ):
            result = daemon.process_utterance(AUDIO_GENERATION, woke=False)

        self.assertIsNone(result)
        self.assertTrue(daemon.awaiting_followup)
        self.assertEqual(daemon.conversation_deadline, deadline)
        release.assert_called_once_with([])
        stop_components.assert_not_called()
        notify.assert_not_called()
        self.assertIn("listening remains armed", log.call_args.args[0])

    def test_genuine_vad_barge_in_over_playback_is_admitted(self) -> None:
        daemon = _bare_daemon()
        daemon.recent_playback.append(
            wake_daemon.RecentPlayback(
                "I will report back when it finishes.",
                time.monotonic() + 5,
            )
        )

        with (
            mock.patch.object(
                wake_daemon,
                "transcribe",
                return_value="No, cancel that job instead.",
            ),
            mock.patch.object(daemon, "ensure_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                return_value=RequestContext("No, cancel that job instead."),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CONVERSATION, "high"),
            ),
            mock.patch.object(
                wake_daemon,
                "qwen_turn",
                return_value=("Okay, I will cancel it.", None),
            ) as qwen_turn,
            mock.patch.object(
                daemon,
                "play_response",
                return_value=(
                    {
                        "interrupted": False,
                        "played_text": "Okay, I will cancel it.",
                    },
                    None,
                ),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            result = daemon.process_utterance(AUDIO_GENERATION, woke=False)

        self.assertIsNone(result)
        qwen_turn.assert_called_once()
        self.assertEqual(
            qwen_turn.call_args.kwargs["trusted_utterance"],
            "No, cancel that job instead.",
        )
        self.assertEqual(
            daemon.history[0],
            {"role": "user", "content": "No, cancel that job instead."},
        )

    def test_recent_playback_window_expires(self) -> None:
        daemon = _bare_daemon()
        with mock.patch.object(wake_daemon.time, "monotonic", return_value=10.0):
            daemon._remember_recent_playback("A bounded response.")

        with mock.patch.object(
            wake_daemon.time,
            "monotonic",
            return_value=10.0 + wake_daemon.PLAYBACK_ECHO_WINDOW_SECONDS,
        ):
            recent = daemon._active_recent_playback()

        self.assertEqual(recent, ())


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
                side_effect=lambda text, **_settings: RequestContext(text),
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
            result = daemon.process_utterance(AUDIO_GENERATION, woke=False)

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
                        job_id="cccccccccccc",
                        delivery_token="claim",
                        job_status="completed",
                    ),
                    interruption,
                ),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enqueue_job_announcement(
                _delivery_claim("job3", "completed", result="done")
            )
            result = daemon._play_pending_announcements()

        self.assertIs(result, interruption)
        acknowledge.assert_not_called()
        release.assert_called_once_with("cccccccccccc", "claim")
        stop_components.assert_not_called()

    def test_barge_in_releases_later_announcements_before_user_response(self) -> None:
        daemon = _bare_daemon()
        interruption = wake_daemon.BargeIn(initial=[b"user"], woke=True)
        daemon._enqueue_job_announcement(
            _delivery_claim("job1", "completed", result="first", token="claim1")
        )
        daemon._enqueue_job_announcement(
            _delivery_claim(
                "job2",
                "completed",
                result="Hey Jarvis, second",
                token="claim2",
            )
        )
        with daemon.playback_queue._lock:
            first_request = daemon.playback_queue._items[0][0]
        batch = [
            (
                {
                    "played_text": "",
                    "interrupted": True,
                },
                True,
                first_request,
            )
        ]

        with (
            mock.patch.object(wake_daemon, "acknowledge_delivery") as acknowledge,
            mock.patch.object(wake_daemon, "release_delivery") as release,
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(wake_daemon, "stop_components"),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(batch, interruption),
            ) as drain,
            mock.patch.object(wake_daemon, "notify"),
        ):
            result = daemon._play_pending_announcements()

        self.assertIs(result, interruption)
        acknowledge.assert_not_called()
        self.assertEqual(
            release.call_args_list,
            [
                mock.call("aaaaaaaaaaaa", "claim1"),
                mock.call("bbbbbbbbbbbb", "claim2"),
            ],
        )
        self.assertEqual(len(daemon.playback_queue), 0)
        self.assertEqual(
            drain.call_args.args[0],
            "Cursor finished test. Cursor finished test.",
        )
        self.assertNotIn("Hey Jarvis, second", drain.call_args.args[0])


class WakeRecordingHandoffTests(unittest.TestCase):
    def test_recorded_utterance_returns_explicit_generation(self) -> None:
        daemon = _bare_daemon()
        daemon.running = False
        daemon.is_speech = lambda _frame: True  # type: ignore[method-assign]

        with mock.patch.object(
            wake_daemon.recorder,
            "write_audio_generation",
            return_value=AUDIO_GENERATION,
        ) as handoff:
            result = daemon.record_utterance([b"\x01\x00"])

        self.assertEqual(result, AUDIO_GENERATION)
        handoff.assert_called_once_with(
            wake_daemon.RECORDING_PATHS,
            mock.ANY,
            conflicts=(wake_daemon.DICTATION_RECORDING_PATHS,),
        )

    def test_wake_capture_waits_for_fresh_speech_before_end_silence(self) -> None:
        daemon = _bare_daemon()
        quiet = [b"quiet"] * 9
        frames = [*quiet, b"request", *quiet]
        daemon.read_frame = mock.Mock(side_effect=frames)  # type: ignore[method-assign]
        daemon.is_speech = lambda frame: frame in {b"wake", b"request"}  # type: ignore[method-assign]

        with mock.patch.object(
            wake_daemon.recorder,
            "write_audio_generation",
            return_value=AUDIO_GENERATION,
        ):
            result = daemon.record_utterance(
                [b"wake"],
                wait_for_fresh_speech=True,
            )

        self.assertEqual(result, AUDIO_GENERATION)
        self.assertEqual(daemon.read_frame.call_count, len(frames))

    def test_capture_duration_excludes_pre_roll(self) -> None:
        daemon = _bare_daemon()
        daemon.is_speech = lambda _frame: False  # type: ignore[method-assign]
        daemon.read_frame = mock.Mock(return_value=b"quiet")  # type: ignore[method-assign]
        expected_frames = (
            int(wake_daemon.MAX_UTTERANCE_SECONDS * 1000 / wake_daemon.FRAME_MS) + 1
        )

        with mock.patch.object(
            wake_daemon.recorder,
            "write_audio_generation",
            return_value=AUDIO_GENERATION,
        ):
            daemon.record_utterance([b"pre-roll"] * wake_daemon.PRE_ROLL_FRAMES)

        self.assertEqual(daemon.read_frame.call_count, expected_frames)

    def test_run_requires_fresh_speech_after_wake_detection(self) -> None:
        daemon = _bare_daemon()
        daemon.np = mock.Mock()  # type: ignore[assignment]
        daemon.wake_key = "wake"
        daemon.wake_model.predict.return_value = {
            "wake": daemon.audio.wake_threshold + 0.1
        }
        daemon.start_microphone = lambda: None  # type: ignore[method-assign]
        daemon.read_frame = lambda: b"wake"  # type: ignore[method-assign]

        def process(_audio_path: Path, *, woke: bool) -> None:
            self.assertTrue(woke)
            daemon.running = False

        daemon.process_utterance = process  # type: ignore[method-assign]
        with (
            mock.patch.object(
                daemon,
                "record_utterance_safely",
                return_value=AUDIO_GENERATION,
            ) as record,
            mock.patch.object(daemon, "begin_activation"),
            mock.patch.object(wake_daemon, "recover_jobs"),
            mock.patch.object(wake_daemon, "pending_results", return_value=[]),
            mock.patch.object(wake_daemon, "log"),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.run()

        record.assert_called_once_with(
            [b"wake"],
            wait_for_fresh_speech=True,
        )

    def test_handoff_conflict_is_suppressed_without_terminating_daemon(self) -> None:
        daemon = _bare_daemon()
        with (
            mock.patch.object(
                wake_daemon.recorder, "any_recording_active", return_value=False
            ),
            mock.patch.object(
                daemon,
                "record_utterance",
                side_effect=HarnessError("another recording is active"),
            ),
            mock.patch.object(wake_daemon, "log") as log,
            mock.patch.object(wake_daemon, "notify"),
        ):
            result = daemon.record_utterance_safely([b"\x01\x00"])

        self.assertIsNone(result)
        self.assertTrue(daemon.running)
        self.assertIn("suppressed", log.call_args.args[0])

    def test_run_suppresses_wake_during_manual_capture_without_touching_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared_lock = root / "recording.lock"
            manual = recorder.RecorderPaths(
                root / "manual",
                root / "manual" / "request.wav",
                root / "manual" / "recording.pid",
                root / "manual" / "pw-record.log",
                shared_lock,
            )
            dictation = recorder.RecorderPaths(
                root / "dictation",
                root / "dictation" / "recording.wav",
                root / "dictation" / "recording.pid",
                root / "dictation" / "pw-record.log",
                shared_lock,
            )
            manual.state_dir.mkdir()
            identity = recorder.process_identity(os.getpid())
            assert identity is not None
            state = json.dumps({"pid": os.getpid(), "process_start": identity})
            manual.process.write_text(state)
            audio = b"manual capture in progress"
            manual.audio.write_bytes(audio)

            daemon = _bare_daemon()
            daemon.np = mock.Mock()  # type: ignore[assignment]
            daemon.wake_key = "wake"
            daemon.wake_model.predict.side_effect = [
                {"wake": daemon.audio.wake_threshold + 0.1},
                {"wake": 0.0},
            ]
            reads = 0

            def read_frame() -> bytes:
                nonlocal reads
                reads += 1
                if reads == 2:
                    daemon.running = False
                return b"\x00\x00"

            daemon.start_microphone = lambda: None  # type: ignore[method-assign]
            daemon.read_frame = read_frame  # type: ignore[method-assign]
            with (
                mock.patch.object(wake_daemon, "CAPTURE_PATHS", (manual, dictation)),
                mock.patch.object(daemon, "record_utterance") as record,
                mock.patch.object(daemon, "begin_activation") as activate,
                mock.patch.object(wake_daemon, "recover_jobs"),
                mock.patch.object(wake_daemon, "pending_results", return_value=[]),
                mock.patch.object(wake_daemon, "log") as log,
                mock.patch.object(wake_daemon, "notify"),
            ):
                daemon.run()

            self.assertEqual(reads, 2, "daemon must continue after suppression")
            record.assert_not_called()
            activate.assert_not_called()
            self.assertEqual(manual.audio.read_bytes(), audio)
            self.assertEqual(manual.process.read_text(), state)
            self.assertTrue(
                any("deferred" in call.args[0] for call in log.call_args_list)
            )

    def test_wake_busy_exhaustion_logs_preserved_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generation = Path(temporary) / "request-generation.wav"
            generation.write_bytes(b"RIFF" + b"\0" * 64)
            daemon = _bare_daemon()
            daemon.pause_microphone = lambda: None  # type: ignore[method-assign]
            daemon.resume_microphone = lambda: None  # type: ignore[method-assign]
            daemon.stop_components_when_idle = lambda: None  # type: ignore[method-assign]
            error = HarnessError(
                "STT remained busy; audio was preserved. Retry with "
                f"`voice-harness transcribe --generation {generation}`"
            )
            with (
                mock.patch.object(wake_daemon, "transcribe", side_effect=error),
                mock.patch.object(wake_daemon, "release_deliveries"),
                mock.patch.object(wake_daemon, "log") as log,
                mock.patch.object(wake_daemon, "notify"),
            ):
                daemon.process_utterance(generation, woke=True)

            self.assertTrue(generation.exists())
            self.assertIn(str(generation), log.call_args.args[0])


class PeriodicCursorRecoveryTests(unittest.TestCase):
    def test_next_delivery_poll_recovers_worker_that_died_after_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs_dir = root / "jobs"
            legacy_dir = root / "legacy"
            store = JobStore(jobs_dir, legacy_dir)
            store.create(
                CursorJob.from_dict(
                    {
                        "id": "123456789abc",
                        "status": "running",
                        "request": "test",
                        "created_at": 1,
                        "delivered": False,
                        "herdr_target": "agent",
                        "worker_token": "claim",
                        "worker_pid": 42,
                        "worker_boot_id": "boot",
                        "worker_process_start": "start",
                    }
                )
            )
            alive = True
            launch = mock.Mock()

            with (
                mock.patch.object(cursor_service, "JOBS_DIR", jobs_dir),
                mock.patch.object(cursor_service, "LEGACY_JOBS_DIR", legacy_dir),
                mock.patch.object(
                    cursor_service, "_worker_is_alive", side_effect=lambda _job: alive
                ),
                mock.patch.object(cursor_service, "launch_worker", launch),
            ):
                cursor_service.recover_jobs()
                launch.assert_not_called()
                alive = False
                self.assertEqual(wake_daemon.pending_results(), [])

            launch.assert_called_once_with("123456789abc")
            self.assertEqual(
                store.get("123456789abc").status,
                JobStatus.QUEUED,
            )


class RunLoopFollowupTests(unittest.TestCase):
    def test_followup_speech_is_recorded_without_wake_word(self) -> None:
        daemon = _bare_daemon()
        daemon.awaiting_followup = True
        daemon.conversation_deadline = time.monotonic() + 100

        recorded: list[list[bytes]] = []
        processed: list[tuple[Path, bool]] = []

        def fake_record(
            initial: list[bytes],
            *,
            wait_for_fresh_speech: bool = False,
        ) -> Path:
            self.assertFalse(wait_for_fresh_speech)
            recorded.append(list(initial))
            return AUDIO_GENERATION

        def fake_process(audio_path: Path, *, woke: bool) -> None:
            processed.append((audio_path, woke))
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
        self.assertEqual(
            processed,
            [(AUDIO_GENERATION, False)],
            "follow-up must pass its generation and be handled as woke=False",
        )
        self.assertEqual(len(recorded), 1)


class ForceListenTests(unittest.TestCase):
    def test_force_listen_starts_a_turn_without_the_wake_word(self) -> None:
        daemon = _bare_daemon()
        daemon.force_listen.set()

        processed: list[tuple[Path, bool]] = []

        def fake_process(audio_path: Path, *, woke: bool) -> None:
            processed.append((audio_path, woke))
            daemon.running = False

        daemon.start_microphone = lambda: None  # type: ignore[method-assign]
        daemon.read_frame = lambda: b"\x00\x00"  # type: ignore[method-assign]
        daemon.record_utterance = (  # type: ignore[method-assign]
            lambda _initial, *, wait_for_fresh_speech=False: AUDIO_GENERATION
        )
        daemon.process_utterance = fake_process  # type: ignore[method-assign]
        daemon.begin_activation = lambda: None  # type: ignore[method-assign]

        with (
            mock.patch.object(wake_daemon, "recover_jobs"),
            mock.patch.object(wake_daemon, "pending_results", return_value=[]),
            mock.patch.object(
                wake_daemon.recorder, "any_recording_active", return_value=False
            ),
            mock.patch.object(wake_daemon, "log"),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.run()

        self.assertEqual(processed, [(AUDIO_GENERATION, False)])
        self.assertFalse(
            daemon.force_listen.is_set(),
            "the request must be consumed exactly once",
        )

    def test_request_listen_signals_the_recorded_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "wake.pid"
            pid_path.write_text(
                json.dumps({"pid": 4321, "process_start": "daemon-start"})
            )
            handle = mock.Mock()
            with (
                mock.patch.object(wake_daemon, "WAKE_PID_PATH", pid_path),
                mock.patch.object(
                    wake_daemon.ProcessHandle,
                    "open",
                    return_value=handle,
                ) as open_handle,
            ):
                wake_daemon.request_listen()

        open_handle.assert_called_once_with(4321, expected_start="daemon-start")
        handle.send_signal.assert_called_once_with(wake_daemon.signal.SIGUSR1)
        handle.close.assert_called_once()

    def test_request_listen_reports_a_stopped_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "missing.pid"
            with (
                mock.patch.object(wake_daemon, "WAKE_PID_PATH", pid_path),
                self.assertRaisesRegex(HarnessError, "not running"),
            ):
                wake_daemon.request_listen()

    def test_request_listen_reports_a_dead_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "wake.pid"
            pid_path.write_text(
                json.dumps({"pid": 4321, "process_start": "daemon-start"})
            )
            with (
                mock.patch.object(wake_daemon, "WAKE_PID_PATH", pid_path),
                mock.patch.object(wake_daemon.ProcessHandle, "open", return_value=None),
                self.assertRaisesRegex(HarnessError, "not running"),
            ):
                wake_daemon.request_listen()

    def test_request_listen_rejects_pid_reuse_without_signalling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "wake.pid"
            pid_path.write_text(
                json.dumps({"pid": 4321, "process_start": "original-start"})
            )
            with (
                mock.patch.object(wake_daemon, "WAKE_PID_PATH", pid_path),
                mock.patch.object(
                    wake_daemon.ProcessHandle, "open", return_value=None
                ) as open_handle,
                self.assertRaisesRegex(HarnessError, "not running"),
            ):
                wake_daemon.request_listen()

        open_handle.assert_called_once_with(4321, expected_start="original-start")

    def test_main_rejects_duplicate_daemon_startup(self) -> None:
        with (
            mock.patch.object(
                wake_daemon,
                "_acquire_wake_singleton",
                side_effect=HarnessError("wake daemon is already running"),
            ),
            mock.patch.object(wake_daemon, "WakeConversationDaemon"),
            self.assertRaisesRegex(HarnessError, "already running"),
        ):
            wake_daemon.main()


class CompletedFollowupContextTests(unittest.TestCase):
    def _completed_context(
        self,
        job: CursorJob,
    ) -> wake_daemon.CompletedFollowup:
        response = wake_daemon.cursor_service.render_job_announcement(job)
        return wake_daemon.CompletedFollowup(
            job_id=job.id,
            completed_at=job.completed_at,
            expires_at=time.monotonic() + 60,
            display_fingerprint=wake_daemon._display_fingerprint(response.display_text),
        )

    def test_completed_announcement_installs_context(self) -> None:
        daemon = _bare_daemon()
        daemon.platform = replace(daemon.platform, cursor_followup_enabled=True)
        with (
            mock.patch.object(
                wake_daemon.CURSOR_STORE,
                "get",
                return_value=mock.Mock(completed_at=123.0),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enable_post_job_conversation(
                job_id="bbbbbbbbbbbb", job_status="completed", played_text="done"
            )

        assert daemon.completed_followup is not None
        self.assertEqual(daemon.completed_followup.job_id, "bbbbbbbbbbbb")
        self.assertEqual(daemon.completed_followup.completed_at, 123.0)
        self.assertGreater(daemon.completed_followup.expires_at, time.monotonic())

    def test_foreground_completion_installs_context_after_playback(self) -> None:
        daemon = _bare_daemon()
        daemon.platform = replace(daemon.platform, cursor_followup_enabled=True)
        claim = _delivery_claim("job2", "completed", result="done")

        def finish_in_foreground(
            _request: CursorTurnRequest,
            *,
            delivery_claims: list[DeliveryClaim],
            integrations: object,
        ) -> tuple[str, None]:
            delivery_claims.append(claim)
            return "done", None

        def acknowledge(
            claims: list[DeliveryClaim],
        ) -> list[DeliveryClaim]:
            acknowledged = list(claims)
            claims.clear()
            return acknowledged

        with (
            mock.patch.object(wake_daemon.CURSOR_STORE, "get", return_value=claim.job),
            mock.patch.object(wake_daemon, "transcribe", return_value="do the task"),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(
                wake_daemon,
                "route_intent",
                return_value=IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            ),
            mock.patch.object(
                wake_daemon, "cursor_turn", side_effect=finish_in_foreground
            ),
            mock.patch.object(
                wake_daemon, "acknowledge_deliveries", side_effect=acknowledge
            ),
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("done"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)

        assert daemon.completed_followup is not None
        self.assertEqual(daemon.completed_followup.job_id, "bbbbbbbbbbbb")
        self.assertEqual(daemon.completed_followup.completed_at, 1)

    def test_kill_switch_disables_context(self) -> None:
        daemon = _bare_daemon()
        daemon.platform = replace(daemon.platform, cursor_followup_enabled=False)
        with (
            mock.patch.object(wake_daemon.CURSOR_STORE, "get") as get,
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enable_post_job_conversation(
                job_id="bbbbbbbbbbbb", job_status="completed", played_text="done"
            )

        self.assertIsNone(daemon.completed_followup)
        get.assert_not_called()

    def test_awaiting_job_completion_swaps_slots(self) -> None:
        daemon = _bare_daemon()
        daemon.platform = replace(daemon.platform, cursor_followup_enabled=True)
        daemon.cursor_session = "bbbbbbbbbbbb"
        with (
            mock.patch.object(
                wake_daemon.CURSOR_STORE,
                "get",
                return_value=mock.Mock(completed_at=5.0),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon._enable_post_job_conversation(
                job_id="bbbbbbbbbbbb", job_status="completed", played_text="done"
            )

        self.assertIsNone(daemon.cursor_session)
        assert daemon.completed_followup is not None
        self.assertEqual(daemon.completed_followup.job_id, "bbbbbbbbbbbb")

    def test_active_context_expires(self) -> None:
        daemon = _bare_daemon()
        daemon.completed_followup = wake_daemon.CompletedFollowup(
            job_id="bbbbbbbbbbbb",
            completed_at=1.0,
            expires_at=time.monotonic() - 1,
        )
        self.assertIsNone(daemon._active_completed_followup())
        self.assertIsNone(daemon.completed_followup)

    def test_recent_details_return_exact_display_rendering_and_consume_context(
        self,
    ) -> None:
        daemon = _bare_daemon()
        job = _delivery_claim(
            "job2",
            "completed",
            result="Changed /srv/example/config.toml.",
        ).job
        context = self._completed_context(job)
        daemon.completed_followup = context
        expected = wake_daemon.cursor_service.render_job_announcement(job)
        with mock.patch.object(wake_daemon.CURSOR_STORE, "get", return_value=job):
            response = daemon._recent_completion_details(context)

        self.assertEqual(response.display_text, expected.display_text)
        self.assertNotEqual(response.spoken_text, expected.display_text)
        self.assertIsNone(daemon.completed_followup)

    def test_recent_details_fail_closed_when_job_identity_no_longer_matches(
        self,
    ) -> None:
        daemon = _bare_daemon()
        job = _delivery_claim("job2", "completed", result="done").job
        context = self._completed_context(job)
        context.completed_at = 999
        daemon.completed_followup = context
        with mock.patch.object(wake_daemon.CURSOR_STORE, "get", return_value=job):
            response = daemon._recent_completion_details(context)

        self.assertEqual(response.display_text, wake_daemon.RECENT_DETAILS_UNAVAILABLE)
        self.assertIsNone(daemon.completed_followup)

    def test_recent_details_fail_closed_when_rendering_changed(self) -> None:
        daemon = _bare_daemon()
        announced = _delivery_claim("job2", "completed", result="original").job
        changed = announced.evolve(result="different")
        context = self._completed_context(announced)
        daemon.completed_followup = context
        with mock.patch.object(wake_daemon.CURSOR_STORE, "get", return_value=changed):
            response = daemon._recent_completion_details(context)

        self.assertEqual(response.display_text, wake_daemon.RECENT_DETAILS_UNAVAILABLE)
        self.assertIsNone(daemon.completed_followup)

    def test_newer_completed_announcement_supersedes_retained_context(self) -> None:
        daemon = _bare_daemon()
        daemon.platform = replace(daemon.platform, cursor_followup_enabled=True)
        first = _delivery_claim("job1", "completed", result="first").job
        second = _delivery_claim("job2", "completed", result="second").job
        with mock.patch.object(
            wake_daemon.CURSOR_STORE,
            "get",
            side_effect=[first, second],
        ):
            for job in (first, second):
                display = wake_daemon.cursor_service.render_job_announcement(
                    job
                ).display_text
                daemon._remember_completed_job(
                    job.id,
                    expected_completed_at=job.completed_at,
                    display_fingerprint=wake_daemon._display_fingerprint(display),
                )

        assert daemon.completed_followup is not None
        self.assertEqual(daemon.completed_followup.job_id, second.id)

    def _run_route(
        self,
        daemon: WakeConversationDaemon,
        route: IntentRoute,
        *,
        transcript: str,
        follow_up_started: bool = True,
        expire_during_routing: bool = False,
    ) -> mock.Mock:
        def run_cursor(request: CursorTurnRequest, **_kwargs: object):
            if follow_up_started and request.on_follow_up_started is not None:
                request.on_follow_up_started()
            return "ok", None

        cursor_turn = mock.Mock(side_effect=run_cursor)

        def routed(*_args: object, **_kwargs: object) -> IntentRoute:
            if expire_during_routing and daemon.completed_followup is not None:
                daemon.completed_followup.expires_at = time.monotonic() - 1
            return route

        with (
            mock.patch.object(wake_daemon, "transcribe", return_value=transcript),
            mock.patch.object(wake_daemon, "start_components"),
            mock.patch.object(
                wake_daemon,
                "request_context",
                side_effect=lambda text, **_settings: RequestContext(text),
            ),
            mock.patch.object(wake_daemon, "route_intent", side_effect=routed),
            mock.patch.object(wake_daemon, "cursor_turn", cursor_turn),
            mock.patch.object(wake_daemon, "qwen_turn") as qwen_turn,
            mock.patch.object(
                daemon,
                "_drain_playback_queue",
                return_value=(_playback_batch("ok"), None),
            ),
            mock.patch.object(wake_daemon, "notify"),
        ):
            daemon.process_utterance(AUDIO_GENERATION, woke=False)
        self._last_qwen = qwen_turn
        return cursor_turn

    def test_followup_route_dispatches_and_consumes_slot(self) -> None:
        daemon = _bare_daemon()
        daemon.completed_followup = wake_daemon.CompletedFollowup(
            job_id="bbbbbbbbbbbb",
            completed_at=9.0,
            expires_at=time.monotonic() + 60,
        )
        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_FOLLOWUP, "high"),
            transcript="review the changes",
        )

        cursor_turn.assert_called_once()
        request = cursor_turn.call_args.args[0]
        self.assertEqual(request.action, "follow_up")
        self.assertEqual(request.job_id, "bbbbbbbbbbbb")
        self.assertEqual(request.expected_completed_at, 9.0)
        self.assertIsNone(daemon.completed_followup)

    def test_details_route_reads_snapshot_without_starting_tools(self) -> None:
        daemon = _bare_daemon()
        job = _delivery_claim("job2", "completed", result="full details").job
        daemon.completed_followup = self._completed_context(job)
        with mock.patch.object(wake_daemon.CURSOR_STORE, "get", return_value=job):
            cursor_turn = self._run_route(
                daemon,
                IntentRoute(Intent.CURSOR_DETAILS, "high"),
                transcript="tell me more",
            )

        cursor_turn.assert_not_called()
        self._last_qwen.assert_not_called()
        self.assertIsNone(daemon.completed_followup)

    def test_followup_route_without_context_declines_without_tools(self) -> None:
        daemon = _bare_daemon()
        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_FOLLOWUP, "high"),
            transcript="review the changes",
        )
        cursor_turn.assert_not_called()
        self._last_qwen.assert_not_called()

    def test_busy_followup_preserves_context_until_expiry(self) -> None:
        daemon = _bare_daemon()
        retained = wake_daemon.CompletedFollowup(
            job_id="bbbbbbbbbbbb",
            completed_at=9.0,
            expires_at=time.monotonic() + 60,
        )
        daemon.completed_followup = retained

        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_FOLLOWUP, "high"),
            transcript="review the changes",
            follow_up_started=False,
        )

        cursor_turn.assert_called_once()
        self.assertIs(daemon.completed_followup, retained)
        retained.expires_at = time.monotonic() - 1
        self.assertIsNone(daemon._active_completed_followup())

    def test_followup_expiring_during_routing_is_not_dispatched(self) -> None:
        daemon = _bare_daemon()
        daemon.completed_followup = wake_daemon.CompletedFollowup(
            job_id="bbbbbbbbbbbb",
            completed_at=9.0,
            expires_at=time.monotonic() + 60,
        )

        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_FOLLOWUP, "high"),
            transcript="review the changes",
            expire_during_routing=True,
        )

        cursor_turn.assert_not_called()
        self.assertIsNone(daemon.completed_followup)
        self._last_qwen.assert_not_called()

    def test_pending_clarification_takes_precedence_over_completed_context(
        self,
    ) -> None:
        daemon = _bare_daemon()
        daemon.cursor_session = "aaaaaaaaaaaa"
        retained = wake_daemon.CompletedFollowup(
            job_id="bbbbbbbbbbbb",
            completed_at=9.0,
            expires_at=time.monotonic() + 60,
        )
        daemon.completed_followup = retained
        daemon._pending_cursor_question = mock.Mock(
            return_value=wake_daemon.PendingQuestionSnapshot(
                "aaaaaaaaaaaa",
                "Which repository?",
                "repository",
                "question-1",
                "aaaaaaaaaaaa-routing-0",
            )
        )

        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_REPLY, "high"),
            transcript="use the api repository",
        )

        request = cursor_turn.call_args.args[0]
        self.assertEqual(request.action, "reply")
        self.assertEqual(request.job_id, "aaaaaaaaaaaa")
        self.assertIs(daemon.completed_followup, retained)

    def test_non_actionable_submit_uses_deterministic_safe_response(self) -> None:
        daemon = _bare_daemon()
        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_SUBMIT, "medium"),
            transcript="maybe change the code",
        )

        cursor_turn.assert_not_called()
        self._last_qwen.assert_not_called()

    def test_explicit_submit_clears_context(self) -> None:
        daemon = _bare_daemon()
        daemon.completed_followup = wake_daemon.CompletedFollowup(
            job_id="bbbbbbbbbbbb",
            completed_at=9.0,
            expires_at=time.monotonic() + 60,
        )
        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_SUBMIT, "high"),
            transcript="work on a new ticket",
        )
        cursor_turn.assert_called_once()
        self.assertEqual(cursor_turn.call_args.args[0].action, "submit")
        self.assertIsNone(daemon.completed_followup)

    def test_pr_unsupported_declines_without_starting_a_job(self) -> None:
        daemon = _bare_daemon()
        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_PR_UNSUPPORTED, "high"),
            transcript="open a pull request",
        )
        cursor_turn.assert_not_called()
        self._last_qwen.assert_not_called()
        self.assertTrue(daemon.awaiting_followup)

    def test_medium_confidence_pr_is_still_declined_without_tools(self) -> None:
        daemon = _bare_daemon()
        cursor_turn = self._run_route(
            daemon,
            IntentRoute(Intent.CURSOR_PR_UNSUPPORTED, "medium"),
            transcript="open a pull request",
        )
        cursor_turn.assert_not_called()
        self._last_qwen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
