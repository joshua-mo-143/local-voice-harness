from __future__ import annotations

import collections
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import recorder
from ..agents.model import AgentJob as CursorJob
from ..agents.model import JobStatus
from ..agents.service import AgentTurnRequest as CursorTurnRequest
from ..agents.service import agent_turn as cursor_turn
from ..agents.service import recover_jobs
from ..browser_context import RequestContext, request_context
from ..components import start_components, stop_components
from ..config import (
    DICTATION_PID_PATH,
    DICTATION_RECORDER_LOG,
    DICTATION_STATE_DIR,
    DICTATION_WAV_PATH,
    JOBS_DIR,
    LEGACY_JOBS_DIR,
    PID_PATH,
    PROJECT_ROOT,
    RECORDER_LOG,
    RECORDING_LOCK,
    STATE_DIR,
    WAKE_LOCK,
    WAKE_PID_PATH,
    WAV_PATH,
)
from ..critical_targets import (
    CriticalTarget,
    ReadbackCandidate,
    ReadbackReply,
    TargetSelection,
    new_candidate,
    readback_response,
    resolve_readback,
    select_submit_target,
)
from ..cursor import consultation as cursor_consultation
from ..cursor import questions as cursor_questions
from ..cursor import service as cursor_service
from ..cursor.delivery import (
    DELIVERY_RENEW_SECONDS,
    DELIVERY_WINDOW,
    DeliveryClaim,
    DeliveryClaims,
)
from ..cursor.delivery import (
    acknowledge_deliveries as acknowledge_claims,
)
from ..cursor.delivery import (
    acknowledge_delivery as acknowledge_claim,
)
from ..cursor.delivery import (
    release_deliveries as release_claims,
)
from ..cursor.delivery import (
    release_delivery as release_claim,
)
from ..cursor.delivery import (
    renew_delivery as renew_claim,
)
from ..cursor.store import JobStore
from ..diagnostic_safety import (
    DAEMON_FAILURE,
    PLAYBACK_FAILURE,
    RECORDING_FAILURE,
    VOICE_REQUEST_FAILURE,
    redact_diagnostic,
)
from ..errors import HarnessError, NoSpeechError
from ..integrations.registry import IntegrationRegistry, build_integration_registry
from ..intent import (
    NON_ACTIONABLE_SUBMIT_RESPONSE,
    ForkIntent,
    Intent,
    IntentRoute,
    decide_fork_intent,
    is_grouped_repository_mapping,
    route_intent,
)
from ..llm import qwen_turn
from ..notifications import notify
from ..process import ProcessHandle, process_identity
from ..questions import (
    AnswerOutcome,
    AnswerProvenance,
    Question,
    question_control,
    resolve_answer,
)
from ..responses import AssistantResponse, ResponseLike, as_assistant_response
from ..speech import SpeechRenderer, StreamingSpeechRenderer
from ..stt.client import transcribe
from ..ticket_targets import MISSING_ISSUE_SCOPE_RESPONSE, extract_ticket_targets
from ..tts.queue import PlaybackQueue, PlaybackRequest
from ..user_config import UserConfig, load_user_config
from ..vad import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, SpeechDetector

RECORDING_PATHS = recorder.RecorderPaths(
    STATE_DIR, WAV_PATH, PID_PATH, RECORDER_LOG, RECORDING_LOCK
)
DICTATION_RECORDING_PATHS = recorder.RecorderPaths(
    DICTATION_STATE_DIR,
    DICTATION_WAV_PATH,
    DICTATION_PID_PATH,
    DICTATION_RECORDER_LOG,
    RECORDING_LOCK,
)
CAPTURE_PATHS = (RECORDING_PATHS, DICTATION_RECORDING_PATHS)
CURSOR_STORE = JobStore(JOBS_DIR, LEGACY_JOBS_DIR)
END_SILENCE_MS = 720
MAX_UTTERANCE_SECONDS = 15
CONVERSATION_TIMEOUT_SECONDS = 60
PRE_ROLL_FRAMES = 25
MICROPHONE_START_ATTEMPTS = 30
MICROPHONE_RETRY_SECONDS = 1
PLAYBACK_ECHO_WINDOW_SECONDS = 8.0
RECENT_PLAYBACK_LIMIT = 8


@dataclass(frozen=True, slots=True)
class PendingTargetReadback:
    candidate: ReadbackCandidate
    request: CursorTurnRequest


def _critical_target_request(target: CriticalTarget) -> CursorTurnRequest:
    """Build a dispatch request containing only the confirmed canonical identity."""

    text = f"Work on {target.canonical}."
    if target.provider == "github":
        return CursorTurnRequest(
            text,
            utterance=text,
            context_repository=target.repository,
            github_repository=target.repository,
            github_issue=int(target.ticket),
            issue_scope=target.repository,
            issue_scope_source="github",
        )
    return CursorTurnRequest(
        text,
        utterance=text,
        issue_key=target.canonical,
        issue_scope=target.repository,
        issue_scope_source="linear",
    )


def acknowledge_delivery(job_id: str, token: str) -> bool:
    return acknowledge_claim(CURSOR_STORE, job_id, token)


def release_delivery(job_id: str, token: str) -> bool:
    return release_claim(CURSOR_STORE, job_id, token)


def renew_delivery(job_id: str, token: str) -> bool:
    return renew_claim(CURSOR_STORE, job_id, token)


def acknowledge_deliveries(claims: DeliveryClaims) -> list[DeliveryClaim]:
    return acknowledge_claims(CURSOR_STORE, claims)


def release_deliveries(claims: DeliveryClaims) -> None:
    release_claims(CURSOR_STORE, claims)


def pending_results(
    integrations: IntegrationRegistry | None = None,
) -> list[DeliveryClaim]:
    return cursor_service.pending_results(
        limit=DELIVERY_WINDOW,
        integrations=integrations,
    )


class _DeliveryLeaseGuard:
    def __init__(self, requests: list[PlaybackRequest]) -> None:
        self._claims = {
            (request.job_id, request.delivery_token)
            for request in requests
            if request.job_id and request.delivery_token
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: HarnessError | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._claims or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="voice-delivery-lease",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(DELIVERY_RENEW_SECONDS):
            with self._lock:
                for job_id, token in self._claims:
                    try:
                        renewed = renew_delivery(job_id, token)
                    except Exception as exc:
                        self._error = HarnessError(
                            "delivery lease renewal failed for "
                            f"{job_id}: {type(exc).__name__}: {exc}"
                        )
                        self._stop.set()
                        return
                    if not renewed:
                        self._error = HarnessError(
                            f"delivery lease lost for job {job_id}"
                        )
                        self._stop.set()
                        return
                if not self._claims:
                    return

    def maintain(self) -> None:
        with self._lock:
            if self._error is not None:
                raise self._error

    def complete(self, request: PlaybackRequest) -> None:
        if request.job_id and request.delivery_token:
            with self._lock:
                self._claims.discard((request.job_id, request.delivery_token))
                if not self._claims:
                    self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise HarnessError("delivery lease worker did not stop")


WAKE_NAME = r"(?:jarvis|travis|service|jarvus|jervis)"
WAKE_PREFIX = re.compile(rf"^\s*hey[,\s]+{WAKE_NAME}\b[\s,;:!?.-]*", re.IGNORECASE)
WAKE_ANYWHERE = re.compile(rf"\bhey[,\s]+{WAKE_NAME}\b[\s,;:!?.-]*", re.IGNORECASE)
SPOKEN_WAKE_PATTERN = re.compile(rf"\bhey[,\s]+{WAKE_NAME}\b", re.IGNORECASE)


def strip_wake_prefix(text: str) -> tuple[str, bool]:
    """Remove a leading wake phrase, tolerating Parakeet mis-transcriptions."""

    match = WAKE_PREFIX.match(text)
    if match is not None:
        return text[match.end() :].strip(), True
    match = WAKE_ANYWHERE.search(text)
    if match is not None:
        return (text[: match.start()] + text[match.end() :]).strip(), True
    return text.strip(), False


CLOSE_PATTERN = re.compile(
    r"\b(?:goodbye|stop listening|go to sleep|end conversation)\b", re.IGNORECASE
)
PENDING_SUBMIT_PATTERN = re.compile(
    r"\b(?:work\s+on|fix|change|update|implement|add|remove|run|review|inspect|"
    r"start|create|build|refactor|test)\b",
    re.IGNORECASE,
)
FILLER_WORDS = frozenset(
    {
        "ah",
        "and",
        "but",
        "erm",
        "hmm",
        "i",
        "like",
        "mean",
        "mm",
        "okay",
        "ok",
        "so",
        "uh",
        "um",
        "well",
        "yeah",
        "you",
        "know",
    }
)
END_CONVERSATION_RESPONSE = "Okay, I'll be here if you need me."
RECENT_DETAILS_UNAVAILABLE = (
    "I no longer have details for that recent announcement. "
    "Ask for the job by name or ID."
)


@dataclass
class BargeIn:
    initial: list[bytes]
    woke: bool


@dataclass
class CompletedFollowup:
    """A bounded, one-shot reference to the last announced completed job.

    ``expires_at`` is a ``time.monotonic()`` deadline, so it is intentionally
    volatile and cannot survive a restart.
    """

    job_id: str
    completed_at: float | None
    expires_at: float
    display_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class RecentPlayback:
    text: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class PendingQuestionSnapshot:
    job_id: str
    text: str
    owner: str
    question_id: str
    turn_token: str
    question: Question | None = None


def log(message: str) -> None:
    print(
        f"[voice-harness-wake] {redact_diagnostic(message)}",
        file=sys.stderr,
        flush=True,
    )


def _display_fingerprint(display_text: str) -> str:
    return hashlib.sha256(display_text.encode("utf-8")).hexdigest()


def _transcript_words(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold(), flags=re.UNICODE)
    )


def _matches_playback_prefix(transcript: str, playback: str) -> bool:
    captured_words = _transcript_words(transcript)
    played_words = _transcript_words(playback)
    if not captured_words or len(captured_words) > len(played_words):
        return False
    if captured_words == played_words:
        return True
    return (
        len(captured_words) >= 3
        and captured_words == played_words[: len(captured_words)]
    )


def _is_filler_speech(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    return bool(words) and all(word in FILLER_WORDS for word in words)


class WakeConversationDaemon:
    def __init__(self, user_config: UserConfig) -> None:
        import numpy as np
        import openwakeword
        from openwakeword.model import Model

        self.user_config = user_config
        self.audio = user_config.audio
        self.platform = user_config.platform
        self.providers = user_config.providers
        self.speech_renderer = SpeechRenderer.from_local_config(
            local_checkout=PROJECT_ROOT
        )
        self.integrations = build_integration_registry(user_config)
        self.np = np
        module_path = openwakeword.__file__
        if module_path is None:
            raise HarnessError("Could not locate OpenWakeWord package resources")
        model_path = (
            Path(module_path).parent / "resources" / "models" / "hey_jarvis_v0.1.onnx"
        )
        self.wake_model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            vad_threshold=0.0,
        )
        self.wake_key = next(iter(self.wake_model.models))
        self.speech_detector = SpeechDetector(minimum_rms=self.audio.min_speech_rms)
        self.pre_roll: collections.deque[bytes] = collections.deque(
            maxlen=PRE_ROLL_FRAMES
        )
        self.history: list[dict[str, str]] = []
        self.cursor_session: str | None = None
        self.completed_followup: CompletedFollowup | None = None
        self.recent_playback: collections.deque[RecentPlayback] = collections.deque(
            maxlen=RECENT_PLAYBACK_LIMIT
        )
        self.pending_target_readback: PendingTargetReadback | None = None
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.last_wake = 0.0
        self.force_listen = threading.Event()
        self.running = True
        self.microphone: subprocess.Popen[bytes] | None = None
        self.microphone_paused = False
        self.activation_thread: threading.Thread | None = None
        self.activation_error: Exception | None = None
        self.component_lock = threading.Lock()
        self.playback_queue = PlaybackQueue(self.audio)

    def is_speech(self, frame: bytes) -> bool:
        return self.speech_detector.is_speech(frame)

    def read_frame(self) -> bytes:
        if self.microphone is None or self.microphone.stdout is None:
            raise HarnessError("microphone process is unavailable")
        data = bytearray()
        while len(data) < FRAME_BYTES:
            chunk = self.microphone.stdout.read(FRAME_BYTES - len(data))
            if not chunk:
                detail = (
                    self.microphone.stderr.read().decode(errors="replace").strip()
                    if self.microphone.stderr is not None
                    else ""
                )
                raise HarnessError(f"microphone stream ended: {detail}")
            data.extend(chunk)
        return bytes(data)

    def start_microphone(self) -> None:
        command = ["pw-record", "--raw"]
        if self.audio.source:
            command.extend(("--target", self.audio.source))
        command.extend(("--channels=1", "--rate=16000", "--format=s16", "-"))
        for attempt in range(1, MICROPHONE_START_ATTEMPTS + 1):
            self.microphone = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            time.sleep(0.2)
            if self.microphone.poll() is None:
                log(
                    "listening for Hey Jarvis on "
                    f"{self.audio.source or 'PipeWire default source'}"
                )
                return

            detail = (
                self.microphone.stderr.read().decode(errors="replace").strip()
                if self.microphone.stderr
                else ""
            )
            if (
                "no target node available" not in detail.lower()
                or attempt == MICROPHONE_START_ATTEMPTS
            ):
                raise HarnessError(
                    f"pw-record failed: {detail or self.microphone.returncode}"
                )
            log(
                f"microphone is not ready; retrying in "
                f"{MICROPHONE_RETRY_SECONDS}s ({attempt}/{MICROPHONE_START_ATTEMPTS})"
            )
            time.sleep(MICROPHONE_RETRY_SECONDS)

    def pause_microphone(self) -> None:
        if (
            not self.microphone_paused
            and self.microphone is not None
            and self.microphone.poll() is None
        ):
            os.kill(self.microphone.pid, signal.SIGSTOP)
            self.microphone_paused = True

    def resume_microphone(self) -> None:
        if not self.microphone_paused:
            return
        if self.microphone is None or self.microphone.poll() is not None:
            self.microphone_paused = False
            return
        os.kill(self.microphone.pid, signal.SIGCONT)
        self.microphone_paused = False
        for _ in range(4):
            self.read_frame()
        self.pre_roll.clear()

    def record_utterance(
        self,
        initial: list[bytes],
        *,
        wait_for_fresh_speech: bool = False,
    ) -> Path:
        frames = list(initial)
        has_speech = (
            False
            if wait_for_fresh_speech
            else any(self.is_speech(frame) for frame in frames)
        )
        silence_ms = 0
        captured_frames = 0
        while self.running:
            frame = self.read_frame()
            frames.append(frame)
            captured_frames += 1
            if self.is_speech(frame):
                has_speech = True
                silence_ms = 0
            elif has_speech:
                silence_ms += FRAME_MS
            duration = captured_frames * FRAME_MS / 1000
            if has_speech and silence_ms >= END_SILENCE_MS:
                break
            if duration >= MAX_UTTERANCE_SECONDS:
                break

        def write_audio(path: Path) -> None:
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(SAMPLE_RATE)
                output.writeframes(b"".join(frames))

        return recorder.write_audio_generation(
            RECORDING_PATHS,
            write_audio,
            conflicts=(DICTATION_RECORDING_PATHS,),
        )

    def record_utterance_safely(
        self,
        initial: list[bytes],
        *,
        wait_for_fresh_speech: bool = False,
    ) -> Path | None:
        try:
            if recorder.any_recording_active(CAPTURE_PATHS):
                message = "wake activation deferred while another recording is active"
                log(message)
                notify(message)
                return None
            return self.record_utterance(
                initial,
                wait_for_fresh_speech=wait_for_fresh_speech,
            )
        except HarnessError as exc:
            message = f"wake recording suppressed: {exc}"
            log(message)
            notify(RECORDING_FAILURE, error=True)
            return None

    def begin_activation(self) -> None:
        if self.activation_thread is not None and self.activation_thread.is_alive():
            return
        self.activation_error = None

        def activate() -> None:
            try:
                with self.component_lock:
                    start_components(self.providers)
                    if self.providers.llm_provider == "local":
                        qwen_turn(
                            "Reply with only OK. Do not call a tool.",
                            allow_tools=False,
                            settings=self.providers,
                        )
                    log("LLM tool graph and TTS backend are warm")
            except Exception as exc:
                self.activation_error = exc

        self.activation_thread = threading.Thread(
            target=activate, name="voice-harness-activation", daemon=True
        )
        self.activation_thread.start()

    def ensure_components(self) -> None:
        if self.activation_thread is None:
            with self.component_lock:
                start_components(self.providers)
            return
        self.activation_thread.join(timeout=60)
        if self.activation_thread.is_alive():
            raise HarnessError("model activation did not finish within 60 seconds")
        if self.activation_error is not None:
            raise HarnessError(f"model activation failed: {self.activation_error}")

    def stop_components_when_idle(self) -> None:
        with self.component_lock:
            stop_components()

    def close_conversation(self, reason: str) -> None:
        log(f"conversation closed: {reason}")
        self.history.clear()
        self.cursor_session = None
        self.completed_followup = None
        self.recent_playback.clear()
        self.pending_target_readback = None
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.pause_microphone()
        try:
            self.stop_components_when_idle()
        finally:
            self.resume_microphone()
        notify("Conversation closed")

    def close_pending_capture(self, reason: str) -> None:
        """Stop ambient follow-up capture without discarding the durable question."""
        log(f"pending-question capture closed: {reason}")
        self.history.clear()
        self.completed_followup = None
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.pause_microphone()
        try:
            self.stop_components_when_idle()
        finally:
            self.resume_microphone()

    def end_conversation(self) -> BargeIn | None:
        """Speak a brief farewell, then close unless the user barges in."""
        response = as_assistant_response(END_CONVERSATION_RESPONSE)
        print(f"Assistant: {response.display_text}", flush=True)
        _playback, interruption = self.play_response(response)
        if interruption is not None:
            return interruption
        self.close_conversation("assistant ended the conversation")
        return None

    def wait_for_playback_quiet(self) -> None:
        if self.microphone is None or self.microphone.poll() is not None:
            self.pre_roll.clear()
            return
        quiet = 0
        max_frames = max(
            self.audio.playback_quiet_frames,
            int(self.audio.playback_quiet_timeout_seconds * 1000 / FRAME_MS),
        )
        for _ in range(max_frames):
            frame = self.read_frame()
            quiet = quiet + 1 if not self.is_speech(frame) else 0
            if quiet >= self.audio.playback_quiet_frames:
                break
        self.pre_roll.clear()

    def _remember_recent_playback(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.recent_playback.append(
            RecentPlayback(
                text=text,
                expires_at=time.monotonic() + PLAYBACK_ECHO_WINDOW_SECONDS,
            )
        )

    def _active_recent_playback(self) -> tuple[RecentPlayback, ...]:
        now = time.monotonic()
        while self.recent_playback and self.recent_playback[0].expires_at <= now:
            self.recent_playback.popleft()
        return tuple(self.recent_playback)

    @staticmethod
    def _is_playback_echo(
        transcript: str, recent_playback: tuple[RecentPlayback, ...]
    ) -> bool:
        return any(
            _matches_playback_prefix(transcript, playback.text)
            for playback in recent_playback
        )

    def _build_interrupt_checker(
        self, response: str
    ) -> tuple[Callable[[], bool], Callable[[], BargeIn | None]]:
        self.resume_microphone()
        self.pre_roll.clear()
        self.wake_model.reset()
        speech_streak = 0
        interruption: BargeIn | None = None
        wake_barge_enabled = SPOKEN_WAKE_PATTERN.search(response) is None
        if self.audio.barge_in_mode == "wake" and not wake_barge_enabled:
            log(
                "wake barge-in suppressed because the response contains the wake phrase"
            )

        def should_interrupt() -> bool:
            nonlocal speech_streak, interruption
            frame = self.read_frame()
            self.pre_roll.append(frame)
            if self.audio.barge_in_mode == "off":
                return False
            if self.audio.barge_in_mode == "vad":
                speech_streak = speech_streak + 1 if self.is_speech(frame) else 0
                detected = speech_streak >= self.audio.barge_in_speech_frames
                woke = False
            else:
                samples = self.np.frombuffer(frame, dtype="<i2")
                score = float(self.wake_model.predict(samples).get(self.wake_key, 0.0))
                detected = wake_barge_enabled and score >= self.audio.wake_threshold
                woke = True
            if detected:
                interruption = BargeIn(initial=list(self.pre_roll), woke=woke)
                self.pre_roll.clear()
                log(f"barge-in detected ({self.audio.barge_in_mode})")
                return True
            return False

        def interruption_result() -> BargeIn | None:
            return interruption

        return should_interrupt, interruption_result

    def _drain_playback_queue(
        self,
        response: str,
        *,
        on_poll: Callable[[], None] | None = None,
        on_played: Callable[[dict[str, object], bool, PlaybackRequest], None]
        | None = None,
    ) -> tuple[list[tuple[dict[str, object], bool, PlaybackRequest]], BargeIn | None]:
        should_interrupt, interruption_result = self._build_interrupt_checker(response)
        try:
            if on_poll is None:
                batch = self.playback_queue.drain(
                    should_interrupt=should_interrupt,
                    on_played=on_played,
                )
            else:
                batch = self.playback_queue.drain(
                    should_interrupt=should_interrupt,
                    on_poll=on_poll,
                    on_played=on_played,
                )
        finally:
            self.wake_model.reset()
        interruption = interruption_result()
        for playback, _interrupted, _request in batch:
            self._remember_recent_playback(str(playback.get("played_text") or ""))
        if interruption is None and batch and not batch[-1][1]:
            self.wait_for_playback_quiet()
        return batch, interruption

    def _render_speech(self, text: str) -> str:
        renderer = getattr(self, "speech_renderer", None)
        return (
            renderer.render(text)
            if renderer is not None
            else SpeechRenderer().render(text)
        )

    def play_response(
        self, response: ResponseLike
    ) -> tuple[dict[str, object], BargeIn | None]:
        spoken_text = self._render_speech(as_assistant_response(response).spoken_text)
        self.playback_queue.enqueue(PlaybackRequest(text=spoken_text))
        finished: set[int] = set()

        def finish_job(
            playback: dict[str, object],
            interrupted: bool,
            request: PlaybackRequest,
        ) -> None:
            if request.job_id:
                self._finish_job_playback(request, playback, interrupted=interrupted)
            finished.add(id(request))

        batch, interruption = self._drain_playback_queue(
            spoken_text,
            on_played=finish_job,
        )
        for playback, interrupted, request in batch:
            if id(request) not in finished:
                finish_job(playback, interrupted, request)
        if not batch:
            return {}, interruption
        return batch[-1][0], interruption

    def play_streamed_response(
        self,
        generate: Callable[
            [Callable[[str], None]],
            tuple[str, str | None],
        ],
    ) -> tuple[str, str | None, dict[str, object], BargeIn | None]:
        condition = threading.Condition()
        finished = False
        stopped = False
        chunks: list[str] = []
        played: list[dict[str, object]] = []
        playback_errors: list[BaseException] = []
        interruption: BargeIn | None = None
        stream_renderer = StreamingSpeechRenderer(
            getattr(self, "speech_renderer", SpeechRenderer())
        )

        def on_text_chunk(text: str) -> None:
            nonlocal stopped
            if not text.strip():
                return
            with condition:
                if stopped:
                    return
                chunks.append(text.strip())
                for spoken_text in stream_renderer.feed(text):
                    self.playback_queue.enqueue(PlaybackRequest(text=spoken_text))
                condition.notify()

        def flush_text_chunks() -> None:
            with condition:
                if stopped:
                    return
                for spoken_text in stream_renderer.flush():
                    self.playback_queue.enqueue(PlaybackRequest(text=spoken_text))
                condition.notify()

        def player() -> None:
            nonlocal stopped, interruption
            try:
                while True:
                    with condition:
                        condition.wait_for(
                            lambda: len(self.playback_queue) > 0 or finished
                        )
                        if len(self.playback_queue) == 0 and finished:
                            return
                        response_so_far = " ".join(chunks)
                    batch, current_interruption = self._drain_playback_queue(
                        response_so_far
                    )
                    played.extend(result for result, _interrupted, _request in batch)
                    if current_interruption is not None:
                        interruption = current_interruption
                        with condition:
                            stopped = True
                            self.playback_queue.clear()
                        return
            except BaseException as exc:
                playback_errors.append(exc)
                with condition:
                    stopped = True
                    self.playback_queue.clear()

        playback_thread = threading.Thread(
            target=player,
            name="voice-streamed-playback",
            daemon=True,
        )
        playback_thread.start()
        try:
            response, next_cursor_session = generate(on_text_chunk)
            flush_text_chunks()
        finally:
            with condition:
                finished = True
                condition.notify_all()
            playback_thread.join()
        if playback_errors:
            raise playback_errors[0]
        if not chunks and interruption is None:
            playback, interruption = self.play_response(response)
            return (
                response,
                next_cursor_session,
                playback,
                interruption,
            )
        played_text = " ".join(
            str(result.get("played_text") or "").strip() for result in played
        ).strip()
        playback = {
            "ok": True,
            "stage": "tts",
            "interrupted": interruption is not None,
            "played_text": played_text,
        }
        return response, next_cursor_session, playback, interruption

    def _job_response(self, job: CursorJob) -> AssistantResponse:
        return cursor_service.render_job_announcement(job)

    def _enqueue_job_announcement(self, claim: DeliveryClaim) -> None:
        job = claim.job
        job_id = job.id
        response = self._job_response(job)
        log(f"job {job_id} completion queued: {response.display_text}")
        print(f"Assistant: {response.display_text}", flush=True)
        self.playback_queue.enqueue(
            PlaybackRequest(
                text=self._render_speech(response.spoken_text),
                job_id=job_id,
                delivery_token=claim.token,
                job_status=job.status.value,
                job_completed_at=job.completed_at,
                display_fingerprint=_display_fingerprint(response.display_text),
            )
        )

    def _enable_post_job_conversation(
        self,
        *,
        job_id: str,
        job_status: str,
        played_text: str,
        job_completed_at: float | None = None,
        display_fingerprint: str | None = None,
    ) -> None:
        if played_text:
            self.history.append({"role": "assistant", "content": played_text})
            self.history = self.history[-8:]
        if job_status == "awaiting_user":
            self.cursor_session = job_id
        elif job_status == "completed":
            # A job that was awaiting clarification and then completes must give
            # up the clarification slot and take the completed slot atomically.
            if self.cursor_session == job_id:
                self.cursor_session = None
            self._remember_completed_job(
                job_id,
                expected_completed_at=job_completed_at,
                display_fingerprint=display_fingerprint,
            )
        self.conversation_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
        self.awaiting_followup = True
        notify("Listening for a follow-up…")

    def _remember_completed_job(
        self,
        job_id: str,
        *,
        expected_completed_at: float | None = None,
        display_fingerprint: str | None = None,
    ) -> None:
        """Install the most recently played completed job as follow-up context.

        Called only after a successful, uninterrupted announcement was
        acknowledged, so the reference reflects work the user just heard about.
        The last successfully announced completion wins.
        """
        if not self.platform.cursor_followup_enabled:
            return
        try:
            job = CURSOR_STORE.get(job_id)
        except HarnessError:
            return
        except Exception as exc:  # noqa: BLE001 - never let context tracking crash a turn
            log(f"follow-up context skipped for {job_id}: {type(exc).__name__}: {exc}")
            return
        if (
            expected_completed_at is not None
            and job.completed_at != expected_completed_at
        ):
            return
        self.completed_followup = CompletedFollowup(
            job_id=job_id,
            completed_at=job.completed_at,
            expires_at=(
                time.monotonic() + self.platform.cursor_followup_window_seconds
            ),
            display_fingerprint=display_fingerprint,
        )
        log(f"follow-up context retained for completed job {job_id}")

    def _active_completed_followup(self) -> CompletedFollowup | None:
        """Return the live completed reference, clearing it once it has expired."""
        followup = self.completed_followup
        if followup is None:
            return None
        if time.monotonic() >= followup.expires_at:
            self.completed_followup = None
            return None
        return followup

    def _recent_completion_details(
        self,
        followup: CompletedFollowup,
    ) -> AssistantResponse:
        if self.completed_followup is not followup:
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        try:
            job = CURSOR_STORE.get(followup.job_id)
        except Exception as exc:  # noqa: BLE001 - retrieval must fail closed
            log(
                "recent completion details unavailable for "
                f"{followup.job_id}: {type(exc).__name__}: {exc}"
            )
            self.completed_followup = None
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        if (
            job.status != JobStatus.COMPLETED
            or job.completed_at != followup.completed_at
            or followup.display_fingerprint is None
        ):
            self.completed_followup = None
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        rendered = cursor_service.render_job_announcement(job)
        if _display_fingerprint(rendered.display_text) != followup.display_fingerprint:
            self.completed_followup = None
            return AssistantResponse.from_text(RECENT_DETAILS_UNAVAILABLE)
        self.completed_followup = None
        return AssistantResponse(
            spoken_text="I've displayed the details from that completed Cursor job.",
            display_text=rendered.display_text,
        )

    def _pending_cursor_question(self) -> PendingQuestionSnapshot | None:
        """Load one immutable question snapshot for routing and answer fencing."""
        if self.cursor_session is None:
            return None
        try:
            job = CURSOR_STORE.get(self.cursor_session)
        except Exception as exc:  # noqa: BLE001 - routing must fail closed
            log(
                "clarification context unavailable for "
                f"{self.cursor_session}: {type(exc).__name__}: {exc}"
            )
            return None
        if job.status != JobStatus.AWAITING_USER:
            return None
        question = cursor_questions.current(job)
        if question is None:
            return None
        return PendingQuestionSnapshot(
            job_id=job.id,
            text=question.text,
            owner=question.owner,
            question_id=question.id,
            turn_token=question.origin.turn_token,
            question=question,
        )

    def _finish_job_playback(
        self,
        request: PlaybackRequest,
        playback: dict[str, object],
        *,
        interrupted: bool,
    ) -> None:
        job_id = request.job_id or ""
        delivery_token = request.delivery_token or ""
        if interrupted:
            if delivery_token:
                release_delivery(job_id, delivery_token)
            return
        acknowledged = (
            acknowledge_delivery(job_id, delivery_token) if delivery_token else True
        )
        if not acknowledged:
            if delivery_token:
                release_delivery(job_id, delivery_token)
            return
        played_text = str(playback.get("played_text") or "").strip() or request.text
        self._enable_post_job_conversation(
            job_id=job_id,
            job_status=str(request.job_status or ""),
            played_text=played_text,
            job_completed_at=request.job_completed_at,
            display_fingerprint=request.display_fingerprint,
        )

    def _play_pending_announcements(self) -> BargeIn | None:
        if len(self.playback_queue) == 0:
            return None
        self.pause_microphone()
        batch: list[tuple[dict[str, object], bool, PlaybackRequest]] = []
        interruption: BargeIn | None = None
        finished: set[tuple[str, str] | int] = set()
        with self.playback_queue._lock:
            pending_requests = [request for request, _ in self.playback_queue._items]
        lease_guard = _DeliveryLeaseGuard(pending_requests)

        def request_key(request: PlaybackRequest) -> tuple[str, str] | int:
            if request.job_id and request.delivery_token:
                return request.job_id, request.delivery_token
            return id(request)

        def finish_job(
            playback: dict[str, object],
            interrupted: bool,
            request: PlaybackRequest,
        ) -> None:
            lease_guard.complete(request)
            if request.job_id:
                self._finish_job_playback(request, playback, interrupted=interrupted)
            finished.add(request_key(request))

        try:
            lease_guard.start()
            with self.component_lock:
                start_components(self.providers)
                lease_guard.maintain()
                self.playback_queue.start_prefetch(limit=DELIVERY_WINDOW)
            batch, interruption = self._drain_playback_queue(
                self.playback_queue.queued_text(),
                on_poll=lease_guard.maintain,
                on_played=finish_job,
            )
            for playback, interrupted, request in batch:
                if request_key(request) not in finished:
                    finish_job(playback, interrupted, request)
            if interruption is not None:
                # Barge-in must not leave later announcements ahead of the
                # user's response. Release their durable claims so they can be
                # announced again only after the interrupted turn completes.
                self.playback_queue.clear()
                for request in pending_requests:
                    if (
                        request_key(request) not in finished
                        and request.delivery_token
                        and request.job_id
                    ):
                        release_delivery(request.job_id, request.delivery_token)
            return interruption
        except Exception as exc:
            self.playback_queue.clear()
            for request in pending_requests:
                if (
                    request_key(request) not in finished
                    and request.delivery_token
                    and request.job_id
                ):
                    release_delivery(request.job_id, request.delivery_token)
            log(f"queued playback failed: {type(exc).__name__}: {exc}")
            notify(PLAYBACK_FAILURE, error=True)
            return None
        finally:
            try:
                lease_guard.stop()
            finally:
                if not self.conversation_deadline and interruption is None:
                    self.stop_components_when_idle()
                self.resume_microphone()
                self.wake_model.reset()

    def continue_after_barge_in(self, interruption: BargeIn | None) -> None:
        while interruption is not None and self.running:
            audio_path = self.record_utterance_safely(
                interruption.initial,
                wait_for_fresh_speech=interruption.woke,
            )
            if audio_path is None:
                return
            interruption = self.process_utterance(audio_path, woke=interruption.woke)

    def process_utterance(self, audio_path: Path, *, woke: bool) -> BargeIn | None:
        had_active_conversation = bool(self.conversation_deadline)
        delivery_claims: DeliveryClaims = []
        recent_playback = self._active_recent_playback() if not woke else ()
        self.pause_microphone()
        try:
            try:
                text = transcribe(audio_path)
            except NoSpeechError:
                if getattr(self, "pending_target_readback", None) is None:
                    raise
                log("critical-target reply contained no recognizable speech")
                self.awaiting_followup = True
                self.conversation_deadline = (
                    time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
                )
                notify("I didn't catch that. Please repeat yes, no, or the correction.")
                return None
            if woke:
                text, found_wake = strip_wake_prefix(text)
                if not found_wake:
                    if text:
                        log(
                            "wake prefix absent in STT; trusting OpenWakeWord: "
                            f"{text!r}"
                        )
                    else:
                        log(f"rejected wake candidate: {text!r}")
                        self.stop_components_when_idle()
                        return None
            if self._is_playback_echo(text, recent_playback):
                log("rejected follow-up matching recent local playback")
                self.awaiting_followup = True
                return None
            if not text:
                log("wake phrase contained no request; waiting for follow-up")
                notify("Listening for a follow-up…")
                self.awaiting_followup = True
                self.conversation_deadline = (
                    time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
                )
                return None
            self.awaiting_followup = False
            log(f"user: {text}")
            if CLOSE_PATTERN.search(text):
                self.close_conversation("spoken command")
                return None
            self.ensure_components()
            print(f"You: {text}", flush=True)
            next_cursor_session = self.cursor_session
            next_history = list(self.history)
            remember_response = False
            streamed_playback = False
            playback: dict[str, object] = {}
            interruption: BargeIn | None = None
            readback_result: AssistantResponse | None = None
            confirmed_request: CursorTurnRequest | None = None
            pending_readback = getattr(self, "pending_target_readback", None)
            routing_context = RequestContext(text)
            context = (
                request_context(
                    text,
                    platform=self.platform,
                    integrations=self.integrations,
                )
                if pending_readback is not None
                else routing_context
            )
            if pending_readback is not None:
                resolution = resolve_readback(
                    pending_readback.candidate,
                    text,
                    context,
                )
                self.pending_target_readback = None
                if resolution.reply == ReadbackReply.AFFIRMATIVE:
                    confirmed_request = pending_readback.request
                elif resolution.reply == ReadbackReply.CORRECTION:
                    assert resolution.replacement is not None
                    replacement = new_candidate(
                        TargetSelection(
                            resolution.replacement,
                            pending_readback.candidate.context_binding,
                        ),
                        origin_turn=uuid.uuid4().hex,
                    )
                    self.pending_target_readback = PendingTargetReadback(
                        replacement,
                        _critical_target_request(resolution.replacement),
                    )
                    readback_result = readback_response(replacement)
                elif resolution.reply == ReadbackReply.NEGATIVE:
                    readback_result = AssistantResponse.from_text(
                        "Okay, I didn't start that work."
                    )
                elif resolution.reply == ReadbackReply.EXPIRED:
                    readback_result = AssistantResponse.from_text(
                        "That confirmation expired because the target context changed. "
                        "Please repeat the request."
                    )
            active_completed = self._active_completed_followup()
            pending = self._pending_cursor_question()
            if readback_result is not None or confirmed_request is not None:
                route = IntentRoute(Intent.UNCERTAIN, "low")
            else:
                route = route_intent(
                    text,
                    routing_context,
                    cursor_session=pending.job_id if pending is not None else None,
                    pending_question=pending.text if pending is not None else None,
                    clarification_kind=pending.owner if pending is not None else None,
                    recent_completion=active_completed is not None,
                    settings=self.providers,
                )
                if pending is not None:
                    deterministic_answer = (
                        resolve_answer(
                            pending.question,
                            text,
                            provenance=AnswerProvenance.USER_VOICE,
                        ).outcome
                        if pending.question is not None
                        else None
                    )
                    resolved_as_answer = deterministic_answer in {
                        AnswerOutcome.REPEAT,
                        AnswerOutcome.DEFERRED,
                    } or (
                        deterministic_answer == AnswerOutcome.ACCEPTED
                        and pending.question is not None
                        and bool(pending.question.choices)
                    )
                    grouped_repository_answer = (
                        pending.owner == "grouped_repository"
                        and is_grouped_repository_mapping(text)
                    )
                    if (
                        resolved_as_answer
                        or grouped_repository_answer
                        or question_control(text) is not None
                    ):
                        route = IntentRoute(Intent.AGENT_REPLY, "high")
                    invalid_pending_reply = (
                        not resolved_as_answer
                        and route.intent == Intent.AGENT_REPLY
                        and (
                            (
                                pending.question is not None
                                and bool(pending.question.choices)
                            )
                            or _is_filler_speech(text)
                        )
                    )
                    implicit_submit = (
                        route.intent == Intent.AGENT_SUBMIT
                        and PENDING_SUBMIT_PATTERN.search(text) is None
                    )
                    if not woke and (
                        not route.actionable or invalid_pending_reply or implicit_submit
                    ):
                        # Follow-up VAD can capture nearby conversation. A pending
                        # structured question makes conversational fallback unsafe:
                        # close silently and leave the durable question untouched.
                        self.close_pending_capture("non-actionable speech")
                        return None
            if pending_readback is None and (
                route.intent == Intent.CONVERSATION
                or (
                    route.actionable
                    and route.intent
                    in {
                        Intent.AGENT_SUBMIT,
                        Intent.WORKSPACE_CONSULTATION,
                    }
                )
            ):
                context = request_context(
                    text,
                    platform=self.platform,
                    integrations=self.integrations,
                )
            if route.actionable and route.intent == Intent.END_CONVERSATION:
                return self.end_conversation()
            fork_requested = decide_fork_intent(text) == ForkIntent.AFFIRMATIVE
            github_arguments = (
                {
                    "github_repository": context.github_repository,
                    "github_issue": context.github_issue,
                    "github_issue_context": context.github_issue_context,
                    "fork_requested": fork_requested,
                    "github_pull_request": context.github_pull_request,
                }
                if context.github_repository
                or context.github_issue
                or fork_requested
                or context.github_pull_request
                else {}
            )
            extraction = extract_ticket_targets(
                text,
                scope_source=context.issue_scope_source,
                scope=context.issue_scope,
            )
            missing_ticket_scope = extraction.has_unresolved_scope and route.intent in {
                Intent.AGENT_SUBMIT,
                Intent.UNCERTAIN,
            }
            if readback_result is not None:
                response = readback_result
            elif confirmed_request is not None:
                self.completed_followup = None
                response, next_cursor_session = cursor_turn(
                    confirmed_request,
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif missing_ticket_scope:
                response = MISSING_ISSUE_SCOPE_RESPONSE
            elif route.actionable and route.intent == Intent.QUESTION_CONSULTATION:
                snapshot = cursor_consultation.pending_question_snapshot(
                    CURSOR_STORE,
                    pending.job_id if pending is not None else None,
                )
                if snapshot is None:
                    response = cursor_consultation.NO_PENDING_QUESTION
                else:
                    try:
                        response = cursor_consultation.consult_pending_question(
                            self.integrations.herdr_client(),
                            CURSOR_STORE,
                            snapshot,
                            context.text,
                        )
                    except Exception as exc:  # noqa: BLE001 - consultation fails closed
                        response = (
                            str(exc)
                            if isinstance(exc, HarnessError)
                            and str(exc) == cursor_consultation.STALE_PENDING_QUESTION
                            else cursor_consultation.CONSULTATION_FAILED
                        )
            elif route.actionable and route.intent == Intent.WORKSPACE_CONSULTATION:
                completed_job = None
                if active_completed is not None:
                    try:
                        completed_job = CURSOR_STORE.get(active_completed.job_id)
                    except Exception:  # noqa: BLE001 - selection must fail closed
                        completed_job = None
                try:
                    client = self.integrations.herdr_client()
                    target = cursor_consultation.workspace_target(
                        client,
                        focused_repository=context.focused_repository,
                        completed_job=completed_job,
                    )
                    response = (
                        cursor_consultation.consult(client, target, context.text)
                        if target is not None
                        else cursor_consultation.NO_WORKSPACE
                    )
                except Exception:  # noqa: BLE001 - consultation fails closed
                    response = cursor_consultation.CONSULTATION_FAILED
            elif route.actionable and route.intent == Intent.AGENT_LIST:
                response, next_cursor_session = cursor_turn(
                    CursorTurnRequest(
                        "",
                        self.cursor_session,
                        action="list",
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.AGENT_CANCEL:
                response, next_cursor_session = cursor_turn(
                    CursorTurnRequest(
                        text,
                        self.cursor_session,
                        action="cancel",
                        job_id=self.cursor_session,
                        reference=text,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.AGENT_STATUS:
                response, next_cursor_session = cursor_turn(
                    CursorTurnRequest(
                        text,
                        self.cursor_session,
                        action="status",
                        job_id=self.cursor_session,
                        reference=text,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent in {
                Intent.AGENT_DISMISS,
                Intent.AGENT_REPEAT,
            }:
                action = (
                    "reply"
                    if route.intent == Intent.AGENT_REPEAT and pending is not None
                    else (
                        "dismiss" if route.intent == Intent.AGENT_DISMISS else "repeat"
                    )
                )
                response, next_cursor_session = cursor_turn(
                    CursorTurnRequest(
                        text,
                        self.cursor_session,
                        utterance=text if action == "reply" else None,
                        action=action,
                        job_id=self.cursor_session,
                        reference=text,
                        expected_question_id=(
                            pending.question_id
                            if action == "reply" and pending is not None
                            else None
                        ),
                        expected_question_turn=(
                            pending.turn_token
                            if action == "reply" and pending is not None
                            else None
                        ),
                        answer_provenance=(
                            AnswerProvenance.USER_VOICE
                            if action == "reply"
                            else AnswerProvenance.USER_TEXT
                        ),
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif (
                route.actionable
                and route.intent == Intent.AGENT_REPLY
                and pending is not None
            ):
                response, next_cursor_session = cursor_turn(
                    CursorTurnRequest(
                        context.text,
                        pending.job_id,
                        utterance=text,
                        action="reply",
                        job_id=pending.job_id,
                        expected_question_id=(
                            pending.question_id if pending is not None else None
                        ),
                        expected_question_turn=(
                            pending.turn_token if pending is not None else None
                        ),
                        answer_provenance=AnswerProvenance.USER_VOICE,
                    ),
                    delivery_claims=delivery_claims,
                    integrations=self.integrations,
                )
            elif route.actionable and route.intent == Intent.AGENT_SUBMIT:
                # Explicit new work invalidates any retained completed-job slot.
                self.completed_followup = None
                selection = select_submit_target(extraction, context)
                if selection is not None and not fork_requested:
                    candidate = new_candidate(
                        selection,
                        origin_turn=uuid.uuid4().hex,
                    )
                    self.pending_target_readback = PendingTargetReadback(
                        candidate,
                        _critical_target_request(candidate.target),
                    )
                    response = readback_response(candidate)
                else:
                    response, next_cursor_session = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            context_repository=context.focused_repository,
                            issue_key=context.external_issue_reference,
                            issue_scope=context.issue_scope,
                            issue_scope_source=context.issue_scope_source,
                            **github_arguments,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
                    )
            elif route.intent == Intent.AGENT_SUBMIT:
                response = NON_ACTIONABLE_SUBMIT_RESPONSE
            elif route.intent == Intent.AGENT_PR_UNSUPPORTED:
                response = (
                    "I can't open pull requests. I can review the changes or run "
                    "the tests in that checkout instead."
                )
            elif route.intent == Intent.AGENT_DETAILS:
                current_completed = self._active_completed_followup()
                if (
                    self.cursor_session is not None
                    or current_completed is None
                    or current_completed is not active_completed
                ):
                    response = RECENT_DETAILS_UNAVAILABLE
                else:
                    response = self._recent_completion_details(current_completed)
            elif route.actionable and route.intent == Intent.AGENT_FOLLOWUP:
                current_completed = self._active_completed_followup()
                if (
                    self.cursor_session is not None
                    or current_completed is None
                    or current_completed is not active_completed
                ):
                    response = (
                        "I don't have a recent completed Cursor job to follow up on."
                    )
                else:
                    log(
                        "follow-up dispatched for completed job "
                        f"{current_completed.job_id}"
                    )

                    def consume_completed_followup() -> None:
                        # Consume only after the child is durably created. A busy
                        # checkout is retryable while this context remains live.
                        if self.completed_followup is current_completed:
                            self.completed_followup = None

                    response, next_cursor_session = cursor_turn(
                        CursorTurnRequest(
                            context.text,
                            utterance=text,
                            action="follow_up",
                            job_id=current_completed.job_id,
                            expected_completed_at=current_completed.completed_at,
                            on_follow_up_started=consume_completed_followup,
                        ),
                        delivery_claims=delivery_claims,
                        integrations=self.integrations,
                    )
            else:
                # The authoritative router handles every mutating action above.
                # Conversation fallback is always tool-free.
                if self.providers.llm_provider == "venice":
                    (
                        response,
                        next_cursor_session,
                        playback,
                        interruption,
                    ) = self.play_streamed_response(
                        lambda on_text_chunk: qwen_turn(
                            context.text,
                            self.history,
                            self.cursor_session,
                            **github_arguments,
                            trusted_utterance=text,
                            delivery_claims=delivery_claims,
                            on_text_chunk=on_text_chunk,
                            allow_tools=False,
                            settings=self.providers,
                        )
                    )
                    streamed_playback = True
                else:
                    response, next_cursor_session = qwen_turn(
                        context.text,
                        self.history,
                        self.cursor_session,
                        **github_arguments,
                        trusted_utterance=text,
                        delivery_claims=delivery_claims,
                        allow_tools=False,
                        settings=self.providers,
                    )
                remember_response = True
            rendered_response = as_assistant_response(response)
            print(f"Assistant: {rendered_response.display_text}", flush=True)
            cursor_session_before_playback = self.cursor_session
            if not streamed_playback:
                playback, interruption = self.play_response(rendered_response)
            if remember_response:
                played_text = (
                    str(playback.get("played_text") or "").strip()
                    if playback.get("interrupted")
                    else rendered_response.spoken_text
                )
                next_history.append({"role": "user", "content": text})
                if played_text:
                    next_history.append({"role": "assistant", "content": played_text})
                next_history = next_history[-8:]
            if interruption is not None:
                release_deliveries(delivery_claims)
                self.history = next_history
                return interruption
            acknowledged = acknowledge_deliveries(delivery_claims)
            completed_claims = [
                claim
                for claim in acknowledged
                if claim.job.status == JobStatus.COMPLETED
            ]
            if len(completed_claims) == 1:
                completed = completed_claims[0].job
                self._remember_completed_job(
                    completed.id,
                    expected_completed_at=completed.completed_at,
                    display_fingerprint=_display_fingerprint(
                        rendered_response.display_text
                    ),
                )
            if self.cursor_session == cursor_session_before_playback:
                self.cursor_session = next_cursor_session
            self.history = next_history
            self.conversation_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
            self.awaiting_followup = True
            notify("Listening for a follow-up…")
        except NoSpeechError as exc:
            release_deliveries(delivery_claims)
            if had_active_conversation:
                log(
                    "follow-up contained no recognizable speech; listening remains armed"
                )
                self.awaiting_followup = True
            else:
                log(f"turn failed: NoSpeechError: {exc}")
                notify(VOICE_REQUEST_FAILURE, error=True)
                self.awaiting_followup = False
                self.history.clear()
                self.cursor_session = None
                self.conversation_deadline = 0.0
                self.stop_components_when_idle()
        except Exception as exc:
            release_deliveries(delivery_claims)
            log(f"turn failed: {type(exc).__name__}: {exc}")
            notify(VOICE_REQUEST_FAILURE, error=True)
            self.awaiting_followup = False
            if not had_active_conversation:
                self.history.clear()
                self.cursor_session = None
                self.conversation_deadline = 0.0
                self.stop_components_when_idle()
        finally:
            self.resume_microphone()
            self.wake_model.reset()
        return None

    def run(self) -> None:
        recover_jobs(integrations=self.integrations)
        self.start_microphone()
        speech_streak = 0
        while self.running:
            for job in pending_results(self.integrations):
                self._enqueue_job_announcement(job)
            if len(self.playback_queue) > 0:
                self.continue_after_barge_in(self._play_pending_announcements())
                speech_streak = 0
                continue
            frame = self.read_frame()
            now = time.monotonic()
            self.pre_roll.append(frame)
            if self.conversation_deadline and now >= self.conversation_deadline:
                self.close_conversation("inactivity")
                speech_streak = 0
                continue
            if self.force_listen.is_set():
                self.force_listen.clear()
                log("force-listen requested")
                initial = list(self.pre_roll)
                self.pre_roll.clear()
                speech_streak = 0
                audio_path = self.record_utterance_safely(initial)
                if audio_path is None:
                    continue
                notify("Listening…")
                self.begin_activation()
                self.continue_after_barge_in(
                    self.process_utterance(audio_path, woke=False)
                )
                continue
            if self.awaiting_followup:
                speech_streak = speech_streak + 1 if self.is_speech(frame) else 0
                if speech_streak >= 5:
                    log("follow-up speech detected")
                    initial = list(self.pre_roll)
                    self.pre_roll.clear()
                    speech_streak = 0
                    audio_path = self.record_utterance_safely(initial)
                    if audio_path is None:
                        speech_streak = 0
                        continue
                    self.continue_after_barge_in(
                        self.process_utterance(audio_path, woke=False)
                    )
                continue
            samples = self.np.frombuffer(frame, dtype="<i2")
            score = float(self.wake_model.predict(samples).get(self.wake_key, 0.0))
            if score >= self.audio.wake_threshold and now - self.last_wake >= 2.0:
                self.last_wake = now
                log(f"wake detected: score={score:.3f}")
                initial = list(self.pre_roll)
                self.pre_roll.clear()
                audio_path = self.record_utterance_safely(
                    initial,
                    wait_for_fresh_speech=True,
                )
                if audio_path is None:
                    speech_streak = 0
                    continue
                notify("Wake detected — listening…")
                self.begin_activation()
                self.continue_after_barge_in(
                    self.process_utterance(audio_path, woke=True)
                )

    def stop(self) -> None:
        self.running = False
        if self.microphone is not None and self.microphone.poll() is None:
            if self.microphone_paused:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(self.microphone.pid, signal.SIGCONT)
                self.microphone_paused = False
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.microphone.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.microphone.wait(timeout=2)


def _read_wake_state() -> tuple[int, str] | None:
    try:
        raw = WAKE_PID_PATH.read_text().strip()
    except OSError:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    pid = value.get("pid")
    start = value.get("process_start")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if not isinstance(start, str) or not start:
        return None
    return pid, start


def request_listen() -> None:
    """Ask a running wake daemon to start a conversation without the wake word."""
    state = _read_wake_state()
    if state is None:
        raise HarnessError("wake daemon is not running")
    pid, start = state
    handle = ProcessHandle.open(pid, expected_start=start)
    if handle is None:
        raise HarnessError("wake daemon is not running")
    try:
        handle.send_signal(signal.SIGUSR1)
    except ProcessLookupError as exc:
        raise HarnessError("wake daemon is not running") from exc
    finally:
        handle.close()


def _acquire_wake_singleton() -> int:
    WAKE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(WAKE_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise HarnessError("wake daemon is already running") from exc
    return descriptor


def _write_pidfile() -> None:
    identity = process_identity(os.getpid())
    if identity is None:
        raise HarnessError("could not establish wake daemon process identity")
    WAKE_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = WAKE_PID_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"pid": os.getpid(), "process_start": identity}, separators=(",", ":")
        )
        + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, WAKE_PID_PATH)


def _remove_pidfile() -> None:
    with contextlib.suppress(OSError):
        WAKE_PID_PATH.unlink()


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-wake: ok")
        return
    singleton = _acquire_wake_singleton()
    daemon = WakeConversationDaemon(load_user_config())

    def handle_signal(_signum: int, _frame: object) -> None:
        daemon.stop()

    def handle_listen(_signum: int, _frame: object) -> None:
        daemon.force_listen.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGUSR1, handle_listen)
    _write_pidfile()
    try:
        daemon.run()
    except Exception as exc:
        if daemon.running:
            log(f"fatal: {type(exc).__name__}: {exc}")
            notify(DAEMON_FAILURE, error=True)
            raise
    finally:
        daemon.stop()
        _remove_pidfile()
        fcntl.flock(singleton, fcntl.LOCK_UN)
        os.close(singleton)


if __name__ == "__main__":
    main()
