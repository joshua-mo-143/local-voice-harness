from __future__ import annotations

import collections
import contextlib
import os
import re
import signal
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..browser_context import request_context
from ..components import llm_ready, start_components, stop_components
from ..config import DEFAULT_SOURCE, STATE_DIR, WAV_PATH
from ..cursor.jobs import (
    DeliveryClaims,
    acknowledge_deliveries,
    acknowledge_delivery,
    cursor_turn,
    pending_results,
    recover_jobs,
    release_deliveries,
    release_delivery,
)
from ..errors import HarnessError
from ..intent import ForkIntent, Intent, decide_fork_intent, route_intent
from ..llm import qwen_turn
from ..notifications import notify
from ..stt.client import transcribe
from ..tts.queue import PlaybackQueue, PlaybackRequest

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
VAD_CHUNK_BYTES = SAMPLE_RATE * 20 // 1000 * 2
END_SILENCE_MS = 720
MAX_UTTERANCE_SECONDS = 15
CONVERSATION_TIMEOUT_SECONDS = 60
WAKE_THRESHOLD = float(os.environ.get("VOICE_HARNESS_WAKE_THRESHOLD", "0.55"))
MIN_SPEECH_RMS = float(os.environ.get("VOICE_HARNESS_MIN_SPEECH_RMS", "1100"))
SOURCE = os.environ.get("VOICE_HARNESS_SOURCE", DEFAULT_SOURCE)
PRE_ROLL_FRAMES = 25
MICROPHONE_START_ATTEMPTS = 30
MICROPHONE_RETRY_SECONDS = 1
BARGE_IN_MODE = os.environ.get("VOICE_HARNESS_BARGE_IN_MODE", "wake").strip().lower()
BARGE_IN_SPEECH_FRAMES = max(
    1, int(os.environ.get("VOICE_HARNESS_BARGE_IN_SPEECH_FRAMES", "5"))
)
PLAYBACK_QUIET_FRAMES = max(
    1, int(os.environ.get("VOICE_HARNESS_PLAYBACK_QUIET_FRAMES", "4"))
)
PLAYBACK_QUIET_TIMEOUT_SECONDS = max(
    0.0, float(os.environ.get("VOICE_HARNESS_PLAYBACK_QUIET_TIMEOUT_SECONDS", "2"))
)
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


@dataclass
class BargeIn:
    initial: list[bytes]
    woke: bool


def log(message: str) -> None:
    print(f"[voice-harness-wake] {message}", file=sys.stderr, flush=True)


class WakeConversationDaemon:
    def __init__(self) -> None:
        import numpy as np
        import openwakeword
        import webrtcvad
        from openwakeword.model import Model

        if BARGE_IN_MODE not in {"wake", "vad", "off"}:
            raise HarnessError(
                "VOICE_HARNESS_BARGE_IN_MODE must be 'wake', 'vad', or 'off'"
            )
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
        self.vad = webrtcvad.Vad(2)
        self.pre_roll: collections.deque[bytes] = collections.deque(
            maxlen=PRE_ROLL_FRAMES
        )
        self.history: list[dict[str, str]] = []
        self.cursor_session: str | None = None
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.last_wake = 0.0
        self.running = True
        self.microphone: subprocess.Popen[bytes] | None = None
        self.microphone_paused = False
        self.activation_thread: threading.Thread | None = None
        self.activation_error: Exception | None = None
        self.component_lock = threading.Lock()
        self.playback_queue = PlaybackQueue()

    def is_speech(self, frame: bytes) -> bool:
        samples = self.np.frombuffer(frame, dtype="<i2").astype(self.np.float64)
        rms = float(self.np.sqrt(self.np.mean(samples * samples)))
        if rms < MIN_SPEECH_RMS:
            return False
        chunks = [
            frame[offset : offset + VAD_CHUNK_BYTES]
            for offset in range(0, len(frame), VAD_CHUNK_BYTES)
        ]
        valid = [chunk for chunk in chunks if len(chunk) == VAD_CHUNK_BYTES]
        return sum(self.vad.is_speech(chunk, SAMPLE_RATE) for chunk in valid) >= 2

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
        if SOURCE:
            command.extend(("--target", SOURCE))
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
                    f"listening for Hey Jarvis on {SOURCE or 'PipeWire default source'}"
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

    def record_utterance(self, initial: list[bytes]) -> None:
        frames = list(initial)
        has_speech = any(self.is_speech(frame) for frame in frames)
        silence_ms = 0
        while self.running:
            frame = self.read_frame()
            frames.append(frame)
            if self.is_speech(frame):
                has_speech = True
                silence_ms = 0
            elif has_speech:
                silence_ms += FRAME_MS
            duration = len(frames) * FRAME_MS / 1000
            if has_speech and silence_ms >= END_SILENCE_MS:
                break
            if duration >= MAX_UTTERANCE_SECONDS:
                break
        STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        with wave.open(str(WAV_PATH), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(b"".join(frames))

    def begin_activation(self) -> None:
        if self.activation_thread is not None and self.activation_thread.is_alive():
            return
        self.activation_error = None

        def activate() -> None:
            try:
                with self.component_lock:
                    subprocess.run(
                        [
                            "systemctl",
                            "--user",
                            "start",
                            "voice-harness-llm.service",
                            "voice-harness-tts.service",
                        ],
                        check=True,
                    )
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline and not llm_ready():
                        time.sleep(0.1)
                    if not llm_ready():
                        raise HarnessError(
                            "Qwen did not become ready within 30 seconds"
                        )
                    qwen_turn("Reply with only OK. Do not call a tool.")
                    start_components()
                    log("Qwen tool graph and Chatterbox are warm")
            except Exception as exc:
                self.activation_error = exc

        self.activation_thread = threading.Thread(
            target=activate, name="voice-harness-activation", daemon=True
        )
        self.activation_thread.start()

    def ensure_components(self) -> None:
        if self.activation_thread is None:
            with self.component_lock:
                start_components()
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
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.pause_microphone()
        try:
            self.stop_components_when_idle()
        finally:
            self.resume_microphone()
        notify("Conversation closed")

    def wait_for_playback_quiet(self) -> None:
        if self.microphone is None or self.microphone.poll() is not None:
            self.pre_roll.clear()
            return
        quiet = 0
        max_frames = max(
            PLAYBACK_QUIET_FRAMES,
            int(PLAYBACK_QUIET_TIMEOUT_SECONDS * 1000 / FRAME_MS),
        )
        for _ in range(max_frames):
            frame = self.read_frame()
            quiet = quiet + 1 if not self.is_speech(frame) else 0
            if quiet >= PLAYBACK_QUIET_FRAMES:
                break
        self.pre_roll.clear()

    def _build_interrupt_checker(
        self, response: str
    ) -> tuple[Callable[[], bool], Callable[[], BargeIn | None]]:
        self.resume_microphone()
        self.pre_roll.clear()
        self.wake_model.reset()
        speech_streak = 0
        interruption: BargeIn | None = None
        wake_barge_enabled = SPOKEN_WAKE_PATTERN.search(response) is None
        if BARGE_IN_MODE == "wake" and not wake_barge_enabled:
            log(
                "wake barge-in suppressed because the response contains the wake phrase"
            )

        def should_interrupt() -> bool:
            nonlocal speech_streak, interruption
            frame = self.read_frame()
            self.pre_roll.append(frame)
            if BARGE_IN_MODE == "off":
                return False
            if BARGE_IN_MODE == "vad":
                speech_streak = speech_streak + 1 if self.is_speech(frame) else 0
                detected = speech_streak >= BARGE_IN_SPEECH_FRAMES
                woke = False
            else:
                samples = self.np.frombuffer(frame, dtype="<i2")
                score = float(self.wake_model.predict(samples).get(self.wake_key, 0.0))
                detected = wake_barge_enabled and score >= WAKE_THRESHOLD
                woke = True
            if detected:
                interruption = BargeIn(initial=list(self.pre_roll), woke=woke)
                self.pre_roll.clear()
                log(f"barge-in detected ({BARGE_IN_MODE})")
                return True
            return False

        def interruption_result() -> BargeIn | None:
            return interruption

        return should_interrupt, interruption_result

    def _drain_playback_queue(
        self,
        response: str,
        *,
        on_played: Callable[[dict[str, object], bool, PlaybackRequest], None]
        | None = None,
    ) -> tuple[list[tuple[dict[str, object], bool, PlaybackRequest]], BargeIn | None]:
        should_interrupt, interruption_result = self._build_interrupt_checker(response)
        try:
            batch = self.playback_queue.drain(
                should_interrupt=should_interrupt,
                on_played=on_played,
            )
        finally:
            self.wake_model.reset()
        interruption = interruption_result()
        if interruption is None and batch and not batch[-1][1]:
            self.wait_for_playback_quiet()
        return batch, interruption

    def play_response(self, response: str) -> tuple[dict[str, object], BargeIn | None]:
        self.playback_queue.enqueue(PlaybackRequest(text=response))
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
            response,
            on_played=finish_job,
        )
        for playback, interrupted, request in batch:
            if id(request) not in finished:
                finish_job(playback, interrupted, request)
        if not batch:
            return {}, interruption
        return batch[-1][0], interruption

    def _job_response_text(self, job: dict[str, object]) -> str:
        if job.get("status") == "completed":
            return f"Cursor finished. {str(job.get('result') or '').strip()}"
        if job.get("status") == "awaiting_user":
            return (
                "Cursor needs clarification. "
                + str(job.get("question") or job.get("result") or "").strip()
            )
        if job.get("status") == "blocked":
            return str(
                job.get("result")
                or f"Cursor agent {job.get('herdr_target') or ''} needs attention."
            ).strip()
        if job.get("status") == "cancelled":
            return str(job.get("result") or "Cursor job was cancelled.").strip()
        return f"Cursor job failed. {str(job.get('error') or 'Unknown error').strip()}"

    def _enqueue_job_announcement(self, job: dict[str, object]) -> None:
        job_id = str(job.get("id") or "")
        response = self._job_response_text(job)
        log(f"job {job_id} completion queued: {response}")
        print(f"Assistant: {response}", flush=True)
        self.playback_queue.enqueue(
            PlaybackRequest(
                text=response,
                job_id=job_id,
                delivery_token=str(job.get("_delivery_token") or "") or None,
                job_status=str(job.get("status") or ""),
            )
        )

    def _enable_post_job_conversation(
        self,
        *,
        job_id: str,
        job_status: str,
        played_text: str,
    ) -> None:
        if played_text:
            self.history.append({"role": "assistant", "content": played_text})
            self.history = self.history[-8:]
        if job_status == "awaiting_user":
            self.cursor_session = job_id
        elif job_status == "completed" and self.cursor_session == job_id:
            self.cursor_session = None
        self.conversation_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
        self.awaiting_followup = True
        notify("Listening for a follow-up…")

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
            return
        played_text = str(playback.get("played_text") or "").strip() or request.text
        self._enable_post_job_conversation(
            job_id=job_id,
            job_status=str(request.job_status or ""),
            played_text=played_text,
        )

    def _play_pending_announcements(self) -> BargeIn | None:
        if len(self.playback_queue) == 0:
            return None
        self.pause_microphone()
        batch: list[tuple[dict[str, object], bool, PlaybackRequest]] = []
        interruption: BargeIn | None = None
        finished: set[int] = set()
        with self.playback_queue._lock:
            pending_requests = [request for request, _ in self.playback_queue._items]

        def finish_job(
            playback: dict[str, object],
            interrupted: bool,
            request: PlaybackRequest,
        ) -> None:
            if request.job_id:
                self._finish_job_playback(request, playback, interrupted=interrupted)
            finished.add(id(request))

        try:
            with self.component_lock:
                start_components()
                self.playback_queue.start_prefetch()
            batch, interruption = self._drain_playback_queue(
                self.playback_queue.peek_text(),
                on_played=finish_job,
            )
            for playback, interrupted, request in batch:
                if id(request) not in finished:
                    finish_job(playback, interrupted, request)
            return interruption
        except Exception as exc:
            self.playback_queue.clear()
            for request in pending_requests:
                if (
                    id(request) not in finished
                    and request.delivery_token
                    and request.job_id
                ):
                    release_delivery(request.job_id, request.delivery_token)
            log(f"queued playback failed: {type(exc).__name__}: {exc}")
            notify(str(exc) or type(exc).__name__, error=True)
            return None
        finally:
            if not self.conversation_deadline and interruption is None:
                self.stop_components_when_idle()
            self.resume_microphone()
            self.wake_model.reset()

    def continue_after_barge_in(self, interruption: BargeIn | None) -> None:
        while interruption is not None and self.running:
            self.record_utterance(interruption.initial)
            interruption = self.process_utterance(woke=interruption.woke)

    def process_utterance(self, *, woke: bool) -> BargeIn | None:
        had_active_conversation = bool(self.conversation_deadline)
        delivery_claims: DeliveryClaims = []
        self.pause_microphone()
        try:
            text = transcribe()
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
            context = request_context(text)
            route = route_intent(text, context, cursor_session=self.cursor_session)
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
            if (
                route.actionable
                and route.intent == Intent.CURSOR_CANCEL
                and self.cursor_session is not None
            ):
                response, next_cursor_session = cursor_turn(
                    "",
                    action="cancel",
                    job_id=self.cursor_session,
                    delivery_claims=delivery_claims,
                )
            elif (
                route.actionable
                and route.intent == Intent.CURSOR_STATUS
                and self.cursor_session is not None
            ):
                response, next_cursor_session = cursor_turn(
                    "",
                    self.cursor_session,
                    action="status",
                    job_id=self.cursor_session,
                    delivery_claims=delivery_claims,
                )
            elif route.actionable and route.intent == Intent.CURSOR_SUBMIT:
                response, next_cursor_session = cursor_turn(
                    context.text,
                    utterance=text,
                    context_repository=context.focused_repository,
                    **github_arguments,
                    delivery_claims=delivery_claims,
                )
            elif (
                route.actionable
                and route.intent == Intent.CURSOR_REPLY
                and self.cursor_session is not None
            ):
                response, next_cursor_session = cursor_turn(
                    context.text,
                    self.cursor_session,
                    utterance=text,
                    action="reply",
                    job_id=self.cursor_session,
                    delivery_claims=delivery_claims,
                )
            else:
                response, next_cursor_session = qwen_turn(
                    context.text,
                    self.history,
                    self.cursor_session,
                    **github_arguments,
                    trusted_utterance=text,
                    delivery_claims=delivery_claims,
                )
                remember_response = True
            print(f"Assistant: {response}", flush=True)
            cursor_session_before_playback = self.cursor_session
            playback, interruption = self.play_response(response)
            if remember_response:
                played_text = (
                    str(playback.get("played_text") or "").strip()
                    if playback.get("interrupted")
                    else response
                )
                next_history.append({"role": "user", "content": text})
                if played_text:
                    next_history.append({"role": "assistant", "content": played_text})
                next_history = next_history[-8:]
            if interruption is not None:
                release_deliveries(delivery_claims)
                self.history = next_history
                return interruption
            acknowledge_deliveries(delivery_claims)
            if self.cursor_session == cursor_session_before_playback:
                self.cursor_session = next_cursor_session
            self.history = next_history
            self.conversation_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
            self.awaiting_followup = True
            notify("Listening for a follow-up…")
        except Exception as exc:
            release_deliveries(delivery_claims)
            log(f"turn failed: {type(exc).__name__}: {exc}")
            notify(str(exc) or type(exc).__name__, error=True)
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
        recover_jobs()
        self.start_microphone()
        speech_streak = 0
        while self.running:
            for job in pending_results():
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
            if self.awaiting_followup:
                speech_streak = speech_streak + 1 if self.is_speech(frame) else 0
                if speech_streak >= 5:
                    log("follow-up speech detected")
                    initial = list(self.pre_roll)
                    self.pre_roll.clear()
                    speech_streak = 0
                    self.record_utterance(initial)
                    self.continue_after_barge_in(self.process_utterance(woke=False))
                continue
            samples = self.np.frombuffer(frame, dtype="<i2")
            score = float(self.wake_model.predict(samples).get(self.wake_key, 0.0))
            if score >= WAKE_THRESHOLD and now - self.last_wake >= 2.0:
                self.last_wake = now
                log(f"wake detected: score={score:.3f}")
                notify("Wake detected — listening…")
                self.begin_activation()
                initial = list(self.pre_roll)
                self.pre_roll.clear()
                self.record_utterance(initial)
                self.continue_after_barge_in(self.process_utterance(woke=True))

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


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("voice-harness-wake: ok")
        return
    daemon = WakeConversationDaemon()

    def handle_signal(_signum: int, _frame: object) -> None:
        daemon.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        daemon.run()
    except Exception as exc:
        if daemon.running:
            log(f"fatal: {type(exc).__name__}: {exc}")
            notify(str(exc) or type(exc).__name__, error=True)
            raise
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
