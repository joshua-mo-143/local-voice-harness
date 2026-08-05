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
from dataclasses import dataclass
from pathlib import Path

from ..browser_context import enrich_request
from ..components import llm_ready, start_components, stop_components
from ..config import CURSOR_PATTERN, DEFAULT_SOURCE, STATE_DIR, WAV_PATH
from ..cursor.jobs import cursor_turn, mark_delivered, pending_results
from ..errors import HarnessError
from ..llm import qwen_turn
from ..notifications import notify
from ..stt.client import transcribe
from ..tts.client import stream_and_play

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
WAKE_PREFIX = re.compile(r"^\s*hey[,\s]+(?:jarvis|travis)\b[\s,;:!?.-]*", re.IGNORECASE)
SPOKEN_WAKE_PATTERN = re.compile(r"\bhey[,\s]+(?:jarvis|travis)\b", re.IGNORECASE)
CLOSE_PATTERN = re.compile(
    r"\b(?:goodbye|stop listening|go to sleep|end conversation)\b", re.IGNORECASE
)
JOB_CANCEL_PATTERN = re.compile(
    r"\b(?:cancel|stop)\s+(?:that\s+)?(?:cursor\s+)?job\b", re.IGNORECASE
)
JOB_STATUS_PATTERN = re.compile(
    r"\b(?:status|progress|what(?:'s| is)\s+happening)\b", re.IGNORECASE
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
                    raise HarnessError("Qwen did not become ready within 30 seconds")
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
            start_components()
            return
        self.activation_thread.join(timeout=60)
        if self.activation_thread.is_alive():
            raise HarnessError("model activation did not finish within 60 seconds")
        if self.activation_error is not None:
            raise HarnessError(f"model activation failed: {self.activation_error}")

    def close_conversation(self, reason: str) -> None:
        if (
            not self.conversation_deadline
            and not self.history
            and not self.cursor_session
        ):
            return
        log(f"conversation closed: {reason}")
        self.history.clear()
        self.cursor_session = None
        self.conversation_deadline = 0.0
        self.awaiting_followup = False
        self.pause_microphone()
        try:
            stop_components()
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

    def play_response(self, response: str) -> tuple[dict[str, object], BargeIn | None]:
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

        try:
            result = stream_and_play(response, should_interrupt=should_interrupt)
        finally:
            self.wake_model.reset()
        if interruption is None:
            self.wait_for_playback_quiet()
        return result, interruption

    def continue_after_barge_in(self, interruption: BargeIn | None) -> None:
        while interruption is not None and self.running:
            self.record_utterance(interruption.initial)
            interruption = self.process_utterance(woke=interruption.woke)

    def process_utterance(self, *, woke: bool) -> BargeIn | None:
        self.pause_microphone()
        try:
            text = transcribe()
            if woke:
                match = WAKE_PREFIX.match(text)
                if match is None:
                    log(f"rejected wake candidate: {text!r}")
                    if self.activation_thread is not None:
                        self.activation_thread.join(timeout=60)
                    stop_components()
                    return None
                text = text[match.end() :].strip()
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
            remember_response = False
            if self.cursor_session is not None and JOB_CANCEL_PATTERN.search(text):
                response, self.cursor_session = cursor_turn(
                    "", action="cancel", job_id=self.cursor_session
                )
            elif self.cursor_session is not None and JOB_STATUS_PATTERN.search(text):
                response, self.cursor_session = cursor_turn(
                    "", self.cursor_session, action="status", job_id=self.cursor_session
                )
            elif CURSOR_PATTERN.match(text):
                response, self.cursor_session = cursor_turn(enrich_request(text))
            else:
                response, self.cursor_session = qwen_turn(
                    enrich_request(text), self.history, self.cursor_session
                )
                remember_response = True
            print(f"Assistant: {response}", flush=True)
            playback, interruption = self.play_response(response)
            if remember_response:
                played_text = (
                    str(playback.get("played_text") or "").strip()
                    if playback.get("interrupted")
                    else response
                )
                self.history.append({"role": "user", "content": text})
                if played_text:
                    self.history.append({"role": "assistant", "content": played_text})
                self.history = self.history[-8:]
            if interruption is not None:
                return interruption
            self.conversation_deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
            self.awaiting_followup = True
            notify("Listening for a follow-up…")
        except Exception as exc:
            log(f"turn failed: {type(exc).__name__}: {exc}")
            notify(str(exc) or type(exc).__name__, error=True)
            self.awaiting_followup = False
        finally:
            self.resume_microphone()
            self.wake_model.reset()
        return None

    def announce_job(self, job: dict[str, object]) -> BargeIn | None:
        job_id = str(job.get("id") or "")
        mark_delivered(job_id)
        self.pause_microphone()
        conversation_active = bool(self.conversation_deadline)
        try:
            start_components()
            if job.get("status") == "completed":
                response = f"Cursor finished. {str(job.get('result') or '').strip()}"
            elif job.get("status") == "awaiting_user":
                response = (
                    "Cursor needs clarification. "
                    + str(job.get("question") or job.get("result") or "").strip()
                )
                self.cursor_session = job_id
                self.conversation_deadline = (
                    time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
                )
                self.awaiting_followup = True
                conversation_active = True
            elif job.get("status") == "blocked":
                response = str(
                    job.get("result")
                    or f"Cursor agent {job.get('herdr_target') or ''} needs attention."
                ).strip()
            elif job.get("status") == "cancelled":
                response = str(job.get("result") or "Cursor job was cancelled.").strip()
            else:
                response = f"Cursor job failed. {str(job.get('error') or 'Unknown error').strip()}"
            log(f"job {job_id} completion: {response}")
            print(f"Assistant: {response}", flush=True)
            _, interruption = self.play_response(response)
            if interruption is not None:
                conversation_active = True
                return interruption
        except Exception as exc:
            log(f"job {job_id} announcement failed: {type(exc).__name__}: {exc}")
            notify(str(exc) or type(exc).__name__, error=True)
        finally:
            if not conversation_active:
                stop_components()
            self.resume_microphone()
            self.wake_model.reset()
        return None

    def run(self) -> None:
        self.start_microphone()
        speech_streak = 0
        while self.running:
            jobs = pending_results()
            if jobs:
                self.continue_after_barge_in(self.announce_job(jobs[0]))
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
