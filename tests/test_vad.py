from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

from local_voice_harness import vad
from local_voice_harness.errors import HarnessError


def _settings(
    *,
    silence_ms: float = 160,
    max_seconds: float = 2,
    start_speech_frames: int = 3,
) -> vad.VadCaptureSettings:
    return vad.VadCaptureSettings(
        end_silence_ms=silence_ms,
        max_seconds=max_seconds,
        minimum_rms=0,
        start_speech_frames=start_speech_frames,
    )


def _process(frames: list[bytes]) -> mock.Mock:
    process = mock.Mock()
    process.stdout = io.BytesIO(b"".join(frames))
    process.stderr = io.BytesIO()
    process.poll.return_value = None
    process.wait.return_value = 0
    return process


class VadSettingsTests(unittest.TestCase):
    def test_defaults_are_suitable_for_dictation(self) -> None:
        settings = vad.VadCaptureSettings.from_environment({})

        self.assertEqual(settings.end_silence_ms, 900)
        self.assertEqual(settings.max_seconds, 120)
        self.assertEqual(settings.minimum_rms, 1100)
        self.assertEqual(settings.start_speech_frames, 3)

    def test_invalid_values_are_rejected(self) -> None:
        for name, value in (
            ("DICTATION_VAD_END_SILENCE_MS", "0"),
            ("DICTATION_VAD_MAX_SECONDS", "forever"),
            ("DICTATION_VAD_MIN_SPEECH_RMS", "-1"),
            ("DICTATION_VAD_START_SPEECH_FRAMES", "1.5"),
        ):
            with self.subTest(name=name), self.assertRaises(HarnessError):
                vad.VadCaptureSettings.from_environment({name: value})


class _FakeSamples:
    def astype(self, _dtype: object) -> _FakeSamples:
        return self

    def __mul__(self, _other: object) -> _FakeSamples:
        return self


class _FakeNumpy:
    float64 = object()

    def __init__(self, rms: float) -> None:
        self.rms = rms

    def frombuffer(self, _frame: bytes, *, dtype: str) -> _FakeSamples:
        assert dtype == "<i2"
        return _FakeSamples()

    def mean(self, _samples: _FakeSamples) -> float:
        return self.rms * self.rms

    def sqrt(self, value: float) -> float:
        return value**0.5


class SpeechDetectorTests(unittest.TestCase):
    def _detector(self, rms: float, decisions: list[bool]) -> vad.SpeechDetector:
        detector = vad.SpeechDetector.__new__(vad.SpeechDetector)
        detector._np = _FakeNumpy(rms)
        detector._vad = mock.Mock()
        detector._vad.is_speech.side_effect = decisions
        detector.minimum_rms = 100
        return detector

    def test_rms_gate_skips_webrtc(self) -> None:
        detector = self._detector(99, [])

        self.assertFalse(detector.is_speech(b"\0" * vad.FRAME_BYTES))
        detector._vad.is_speech.assert_not_called()

    def test_two_speech_chunks_mark_frame_as_speech(self) -> None:
        detector = self._detector(101, [True, False, True, False])

        self.assertTrue(detector.is_speech(b"\0" * vad.FRAME_BYTES))
        self.assertEqual(detector._vad.is_speech.call_count, 4)

    def test_incomplete_frame_is_not_speech(self) -> None:
        detector = self._detector(101, [True] * 4)

        self.assertFalse(detector.is_speech(b"\0"))
        detector._vad.is_speech.assert_not_called()


class VadCaptureTests(unittest.TestCase):
    def _capture(
        self,
        frames: list[bytes],
        speech: list[bool],
        *,
        settings: vad.VadCaptureSettings | None = None,
    ) -> tuple[str, bytes, mock.Mock]:
        process = _process(frames)
        detector = mock.Mock()
        detector.is_speech.side_effect = speech
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.wav"
            with mock.patch.object(vad.subprocess, "Popen", return_value=process):
                outcome = vad.capture_vad_audio(
                    path,
                    source="microphone",
                    settings=settings or _settings(),
                    detector=detector,
                    stop_requested=threading.Event(),
                )
            audio = path.read_bytes()
        return outcome, audio, process

    def test_sustained_silence_finishes_valid_wav(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, audio, process = self._capture(
            [frame] * 5, [True, True, True, False, False]
        )

        self.assertEqual(outcome, "silence")
        process.send_signal.assert_called_once()
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * 5)
            self.assertEqual(recorded.getframerate(), vad.SAMPLE_RATE)

    def test_short_pause_does_not_finish(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, _audio, _process = self._capture(
            [frame] * 8,
            [True, True, True, False, True, True, False, False],
        )

        self.assertEqual(outcome, "silence")

    def test_pre_speech_silence_is_not_written_to_wav(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, audio, _process = self._capture(
            [frame] * 7,
            [False, False, True, True, True, False, False],
        )

        self.assertEqual(outcome, "silence")
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * 5)

    def test_isolated_speech_frame_does_not_activate_capture(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, audio, _process = self._capture(
            [frame] * 8,
            [True, False, False, True, True, True, False, False],
        )

        self.assertEqual(outcome, "silence")
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * 5)

    def test_stop_while_waiting_for_speech_cleans_up_microphone(self) -> None:
        frame = b"\0" * vad.FRAME_BYTES
        process = _process([frame])
        stop_requested = threading.Event()
        stop_requested.set()
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(vad.subprocess, "Popen", return_value=process),
            self.assertRaisesRegex(HarnessError, "cancelled"),
        ):
            vad.capture_vad_audio(
                Path(temporary) / "capture.wav",
                source="",
                settings=_settings(),
                detector=mock.Mock(is_speech=mock.Mock(return_value=False)),
                stop_requested=stop_requested,
            )

        process.send_signal.assert_called_once()

    def test_stop_during_stalled_pipe_read_cleans_up_microphone(self) -> None:
        read_fd, write_fd = os.pipe()
        process = mock.Mock()
        process.stdout = os.fdopen(read_fd, "rb", buffering=0)
        process.stderr = io.BytesIO()
        process.poll.return_value = None
        stop_requested = threading.Event()

        def request_stop_after_delay() -> None:
            time.sleep(0.05)
            stop_requested.set()

        try:
            with (
                tempfile.TemporaryDirectory() as temporary,
                mock.patch.object(vad.subprocess, "Popen", return_value=process),
                self.assertRaisesRegex(HarnessError, "cancelled"),
            ):
                threading.Thread(target=request_stop_after_delay, daemon=True).start()
                vad.capture_vad_audio(
                    Path(temporary) / "capture.wav",
                    source="",
                    settings=_settings(),
                    detector=mock.Mock(is_speech=mock.Mock(return_value=False)),
                    stop_requested=stop_requested,
                )
        finally:
            os.close(write_fd)

        process.send_signal.assert_called_once()

    def test_maximum_duration_finishes_active_speech(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, _audio, _process = self._capture(
            [frame] * 3,
            [True, True, True],
            settings=_settings(max_seconds=0.24),
        )

        self.assertEqual(outcome, "maximum duration")


if __name__ == "__main__":
    unittest.main()
