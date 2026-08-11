from __future__ import annotations

import select
import signal
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import HarnessError
from .user_config import DictationSettings

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
VAD_CHUNK_MS = 20
VAD_CHUNK_BYTES = SAMPLE_RATE * VAD_CHUNK_MS // 1000 * 2


@dataclass(frozen=True)
class VadCaptureSettings:
    end_silence_ms: float
    max_seconds: float
    minimum_rms: float
    start_speech_frames: int

    @classmethod
    def from_dictation(cls, settings: DictationSettings) -> VadCaptureSettings:
        return cls(
            end_silence_ms=settings.vad_end_silence_ms,
            max_seconds=settings.vad_max_seconds,
            minimum_rms=settings.vad_min_speech_rms,
            start_speech_frames=settings.vad_start_speech_frames,
        )


class SpeechDetector:
    """RMS-gated WebRTC voice activity detector for 16-bit mono PCM frames."""

    def __init__(self, *, minimum_rms: float, aggressiveness: int = 2) -> None:
        import numpy as np
        import webrtcvad

        self._np: Any = np
        self._vad: Any = webrtcvad.Vad(aggressiveness)
        self.minimum_rms = minimum_rms

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != FRAME_BYTES:
            return False
        samples = self._np.frombuffer(frame, dtype="<i2").astype(self._np.float64)
        rms = float(self._np.sqrt(self._np.mean(samples * samples)))
        if rms < self.minimum_rms:
            return False
        chunks = [
            frame[offset : offset + VAD_CHUNK_BYTES]
            for offset in range(0, len(frame), VAD_CHUNK_BYTES)
        ]
        return (
            sum(
                self._vad.is_speech(chunk, SAMPLE_RATE)
                for chunk in chunks
                if len(chunk) == VAD_CHUNK_BYTES
            )
            >= 2
        )


def _read_frame(
    process: subprocess.Popen[bytes],
    stop_requested: threading.Event,
    *,
    poll_timeout: float = 0.5,
) -> bytes:
    if process.stdout is None:
        raise HarnessError("microphone stream is unavailable")
    data = bytearray()
    while len(data) < FRAME_BYTES:
        if stop_requested.is_set():
            raise HarnessError("VAD dictation was cancelled")
        stdout = process.stdout
        pollable = False
        try:
            pollable = select.select([stdout], [], [], poll_timeout) == (
                [stdout],
                [],
                [],
            )
        except (OSError, ValueError):
            pollable = True
        if not pollable:
            continue
        chunk = stdout.read(FRAME_BYTES - len(data))
        if not chunk:
            detail = (
                process.stderr.read().decode(errors="replace").strip()
                if process.stderr is not None
                else ""
            )
            raise HarnessError(
                f"microphone stream ended: {detail or process.returncode}"
            )
        data.extend(chunk)
    return bytes(data)


def _stop_microphone(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def capture_vad_audio(
    path: Path,
    *,
    source: str,
    settings: VadCaptureSettings,
    detector: SpeechDetector,
    stop_requested: threading.Event,
) -> str:
    """Wait for speech and capture a WAV until silence or the duration limit."""

    command = ["pw-record", "--raw"]
    if source:
        command.extend(("--target", source))
    command.extend(("--channels=1", "--rate=16000", "--format=s16", "-"))
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    has_speech = False
    candidate_frames: list[bytes] = []
    silence_ms = 0.0
    captured_frame_count = 0
    outcome = "maximum duration"
    try:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            while True:
                if stop_requested.is_set():
                    raise HarnessError("VAD dictation was cancelled")
                frame = _read_frame(process, stop_requested)
                speech_detected = detector.is_speech(frame)
                if not has_speech:
                    if speech_detected:
                        candidate_frames.append(frame)
                    else:
                        candidate_frames.clear()
                    if len(candidate_frames) < settings.start_speech_frames:
                        continue
                    has_speech = True
                    for candidate in candidate_frames:
                        output.writeframesraw(candidate)
                    captured_frame_count += len(candidate_frames)
                    candidate_frames.clear()
                    silence_ms = 0
                elif speech_detected:
                    output.writeframesraw(frame)
                    captured_frame_count += 1
                    silence_ms = 0
                else:
                    output.writeframesraw(frame)
                    captured_frame_count += 1
                    silence_ms += FRAME_MS

                duration = captured_frame_count * FRAME_MS / 1000
                if has_speech and silence_ms >= settings.end_silence_ms:
                    outcome = "silence"
                    break
                if has_speech and duration >= settings.max_seconds:
                    break
    finally:
        _stop_microphone(process)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return outcome
