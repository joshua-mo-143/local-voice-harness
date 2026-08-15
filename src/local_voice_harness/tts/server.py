from __future__ import annotations

import io
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

from ..config import (
    STATE_DIR,
    TTS_SOCKET,
    BackendSettings,
)
from ..credentials import get_venice_api_key
from ..diagnostic_safety import redact_diagnostic
from ..http_pool import urlopen as pooled_urlopen
from ..user_config import default_user_config, load_user_config

SOCKET_PATH = TTS_SOCKET
OUTPUT_ROOT = STATE_DIR
OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
LOCK = threading.Lock()
ACTIVE_STREAMS: dict[str, threading.Event] = {}
PENDING_CANCELLATIONS: dict[str, float] = {}
ACTIVE_STREAMS_LOCK = threading.Lock()
PENDING_CANCELLATION_SECONDS = 120.0
MAX_PENDING_CANCELLATIONS = 1024
MAX_CHUNK_CHARS = 160
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MODEL: Any = None
SETTINGS: BackendSettings | None = None
VENICE_API_KEY: str | None = None
MAX_AUDIO_BYTES = 32 * 1024 * 1024
VENICE_AUDIO_ATTEMPTS = 3
VENICE_RETRY_BACKOFF_SECONDS = 0.2
VENICE_RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


def log(message: str) -> None:
    print(
        f"[voice-harness-tts] {redact_diagnostic(message)}",
        file=sys.stderr,
        flush=True,
    )


def split_text(text: str, *, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split prose into short TTS requests without cutting normal words."""
    text = " ".join(text.split())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue
        clauses = re.split(r"(?<=[,;:])\s+", sentence)
        current = ""
        for clause in clauses:
            if len(clause) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                words = clause.split()
                word_chunk = ""
                for word in words:
                    if len(word) > max_chars:
                        if word_chunk:
                            chunks.append(word_chunk)
                            word_chunk = ""
                        chunks.extend(
                            word[offset : offset + max_chars]
                            for offset in range(0, len(word), max_chars)
                        )
                    elif not word_chunk:
                        word_chunk = word
                    elif len(word_chunk) + len(word) + 1 <= max_chars:
                        word_chunk += f" {word}"
                    else:
                        chunks.append(word_chunk)
                        word_chunk = word
                if word_chunk:
                    chunks.append(word_chunk)
            elif not current:
                current = clause
            elif len(current) + len(clause) + 1 <= max_chars:
                current += f" {clause}"
            else:
                chunks.append(current)
                current = clause
        if current:
            chunks.append(current)
    return chunks


def _write_json(handler: socketserver.StreamRequestHandler, value: object) -> None:
    handler.wfile.write((json.dumps(value) + "\n").encode())
    handler.wfile.flush()


def _write_response(
    handler: socketserver.StreamRequestHandler,
    value: object,
) -> None:
    try:
        _write_json(handler, value)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _request_text(request: dict[str, object]) -> str:
    text = str(request.get("text", "")).strip()
    if not text:
        raise ValueError("text must not be empty")
    if len(text) > 2_000:
        raise ValueError("text exceeds 2000 characters")
    return text


def _request_voice(request: dict[str, object]) -> Path | str | None:
    voice_value = str(request.get("voice", "")).strip()
    if _settings().tts_provider == "venice":
        return voice_value or None
    voice = Path(voice_value).expanduser().resolve() if voice_value else None
    if voice is not None and not voice.is_file():
        raise FileNotFoundError(f"reference voice not found: {voice}")
    return voice


def _request_bool(
    request: dict[str, object],
    name: str,
    *,
    default: bool,
) -> bool:
    value = request.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _generate(text: str, voice: Path | None) -> tuple[Any, float]:
    import torch

    started = time.perf_counter()
    waveform = MODEL.generate(
        text,
        audio_prompt_path=str(voice) if voice else None,
        temperature=0.8,
        top_p=0.95,
    )
    torch.cuda.synchronize()
    return waveform.detach().cpu().float().squeeze().numpy(), (
        time.perf_counter() - started
    )


def _settings() -> BackendSettings:
    return SETTINGS or default_user_config().providers


def _venice_api_key() -> str:
    return VENICE_API_KEY or get_venice_api_key()


def _wav_metadata(audio: bytes) -> tuple[int, float]:
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            compression = source.getcomptype()
            pcm = source.readframes(source.getnframes())
    except (EOFError, wave.Error) as exc:
        raise RuntimeError("Venice TTS returned an invalid WAV response") from exc
    if channels != 1 or sample_width != 2 or compression != "NONE":
        raise RuntimeError(
            "Venice TTS WAV must be mono, 16-bit, uncompressed PCM audio"
        )
    frame_width = channels * sample_width
    if sample_rate <= 0 or not pcm or len(pcm) % frame_width:
        raise RuntimeError("Venice TTS returned an empty or truncated WAV response")
    return sample_rate, len(pcm) / frame_width / sample_rate


def _venice_http_failure(exc: urllib.error.HTTPError) -> RuntimeError:
    try:
        detail = exc.read(4096).decode("utf-8", errors="replace").strip()
    except OSError:
        detail = ""
    finally:
        exc.close()
    detail = redact_diagnostic(detail)
    suffix = f": {detail}" if detail else ""
    return RuntimeError(
        f"Venice TTS request failed: HTTP {exc.code} {exc.reason}{suffix}"
    )


def _venice_audio(
    text: str,
    voice: str | None = None,
) -> tuple[bytes, int, float, float]:
    settings = _settings()
    payload = json.dumps(
        {
            "model": settings.tts_model,
            "voice": voice or settings.tts_voice,
            "input": text,
            "response_format": "wav",
            # Venice models do not apply this consistently. Request the original
            # cadence and perform one predictable, pitch-preserving transform.
            "speed": 1,
        }
    ).encode()
    request = urllib.request.Request(
        settings.tts_endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {_venice_api_key()}",
            "Content-Type": "application/json",
        },
    )
    attempts_remaining = VENICE_AUDIO_ATTEMPTS
    backoff = VENICE_RETRY_BACKOFF_SECONDS
    deadline = time.monotonic() + settings.tts_timeout
    last_error: RuntimeError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Venice TTS request failed: timed out")
        started = time.perf_counter()
        try:
            with pooled_urlopen(request, timeout=remaining) as response:
                content_type = response.headers.get_content_type()
                audio = response.read(MAX_AUDIO_BYTES + 1)
        except urllib.error.HTTPError as exc:
            error = _venice_http_failure(exc)
            last_error = error
            attempts_remaining -= 1
            if exc.code not in VENICE_RETRYABLE_HTTP_CODES or attempts_remaining <= 0:
                raise error from exc
            if deadline - time.monotonic() <= backoff:
                raise error from exc
            log(f"Venice TTS HTTP {exc.code}; retrying ({attempts_remaining} left)")
            time.sleep(backoff)
            backoff *= 2
            continue
        except (OSError, urllib.error.URLError) as exc:
            error = RuntimeError(f"Venice TTS request failed: {exc}")
            last_error = error
            attempts_remaining -= 1
            if attempts_remaining <= 0:
                raise error from exc
            if deadline - time.monotonic() <= backoff:
                raise error from exc
            log(f"Venice TTS transport failed; retrying ({attempts_remaining} left)")
            time.sleep(backoff)
            backoff *= 2
            continue
        elapsed = time.perf_counter() - started
        if content_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
            raise RuntimeError(
                f"Venice TTS returned unexpected content type {content_type}"
            )
        if len(audio) > MAX_AUDIO_BYTES:
            raise RuntimeError("Venice TTS response exceeds the 32 MiB limit")
        sample_rate, duration = _wav_metadata(audio)
        return audio, sample_rate, duration, elapsed


def _atempo_filter(speed: float) -> str:
    """Build quality-preserving FFmpeg atempo stages within the 0.5-2 range."""
    factors: list[float] = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2:
        factors.append(2)
        remaining /= 2
    factors.append(remaining)
    return ",".join(f"atempo={factor:g}" for factor in factors)


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required for pitch-preserving Venice TTS speed adjustment"
        )
    return ffmpeg


def _apply_venice_speed(
    audio: bytes,
    output: Path,
    speed: float,
    *,
    timeout: float,
) -> tuple[int, float, float]:
    started = time.perf_counter()
    if speed == 1:
        output.write_bytes(audio)
    else:
        ffmpeg = _require_ffmpeg()
        source_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".voice-harness-tts-",
                suffix=".wav",
                dir=output.parent,
                delete=False,
            ) as source:
                source.write(audio)
                source_path = Path(source.name)
            process = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(source_path),
                    "-filter:a",
                    _atempo_filter(speed),
                    "-c:a",
                    "pcm_s16le",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            output.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg TTS speed adjustment failed: {exc}") from exc
        finally:
            if source_path is not None:
                source_path.unlink(missing_ok=True)
        if process.returncode:
            output.unlink(missing_ok=True)
            detail = process.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg TTS speed adjustment failed: {detail}")

    try:
        processed = output.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"could not read processed Venice TTS audio: {exc}") from exc
    sample_rate, duration = _wav_metadata(processed)
    return sample_rate, duration, time.perf_counter() - started


def _synthesize(
    text: str,
    voice: Path | str | None,
    output: Path,
    *,
    apply_speed: bool = True,
) -> tuple[int, float, float]:
    if _settings().tts_provider == "venice":
        settings = _settings()
        audio, _sample_rate, _duration, generation_elapsed = _venice_audio(
            text,
            voice if isinstance(voice, str) else None,
        )
        sample_rate, duration, processing_elapsed = _apply_venice_speed(
            audio,
            output,
            settings.tts_speed if apply_speed else 1,
            timeout=settings.tts_timeout,
        )
        return sample_rate, duration, generation_elapsed + processing_elapsed

    import soundfile as sf

    audio, elapsed = _generate(text, voice if isinstance(voice, Path) else None)
    sf.write(str(output), audio, MODEL.sr, subtype="PCM_16")
    return MODEL.sr, len(audio) / MODEL.sr, elapsed


def _cancel_stream(request_id: str) -> bool:
    now = time.monotonic()
    with ACTIVE_STREAMS_LOCK:
        expired = [
            pending_id
            for pending_id, expires_at in PENDING_CANCELLATIONS.items()
            if expires_at <= now
        ]
        for pending_id in expired:
            PENDING_CANCELLATIONS.pop(pending_id, None)
        cancelled = ACTIVE_STREAMS.get(request_id)
        if cancelled is None:
            if len(PENDING_CANCELLATIONS) >= MAX_PENDING_CANCELLATIONS:
                oldest = min(
                    PENDING_CANCELLATIONS,
                    key=PENDING_CANCELLATIONS.__getitem__,
                )
                PENDING_CANCELLATIONS.pop(oldest, None)
            PENDING_CANCELLATIONS[request_id] = now + PENDING_CANCELLATION_SECONDS
        else:
            cancelled.set()
    return True


def _stream_response(
    handler: socketserver.StreamRequestHandler,
    request: dict[str, object],
) -> None:
    text = _request_text(request)
    voice = _request_voice(request)
    request_id = str(request.get("request_id", ""))
    if not REQUEST_ID.fullmatch(request_id):
        raise ValueError("request_id must contain 1-64 letters, numbers, '_' or '-'")
    chunks = split_text(text)
    skip_first_speed = _request_bool(request, "skip_first_speed", default=False)
    preflight_speed = _request_bool(request, "preflight_speed", default=False)
    settings = _settings()
    if (
        settings.tts_provider == "venice"
        and settings.tts_speed != 1
        and preflight_speed
    ):
        _require_ffmpeg()
    output_dir = OUTPUT_ROOT / f"stream-{request_id}"
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    cancelled = threading.Event()
    with ACTIVE_STREAMS_LOCK:
        if request_id in ACTIVE_STREAMS:
            raise ValueError("request_id is already active")
        pending_cancel = PENDING_CANCELLATIONS.pop(request_id, None)
        if pending_cancel is not None and pending_cancel > time.monotonic():
            cancelled.set()
        ACTIVE_STREAMS[request_id] = cancelled

    generation_seconds = 0.0
    audio_seconds = 0.0
    emitted = 0
    sample_rate: int | None = None
    try:
        # Keep one stream contiguous on the single GPU. A cancellation handler does
        # not acquire this lock, so it can still stop us between model calls.
        with LOCK:
            for index, chunk in enumerate(chunks):
                if cancelled.is_set():
                    break
                output = output_dir / f"{index:04d}.wav"
                rate, duration, elapsed = _synthesize(
                    chunk,
                    voice,
                    output,
                    apply_speed=not (skip_first_speed and index == 0),
                )
                generation_seconds += elapsed
                if cancelled.is_set():
                    output.unlink(missing_ok=True)
                    break
                if sample_rate is None:
                    sample_rate = rate
                    _write_json(
                        handler,
                        {
                            "ok": True,
                            "event": "start",
                            "request_id": request_id,
                            "sample_rate": sample_rate,
                            "chunks": len(chunks),
                        },
                    )
                elif rate != sample_rate:
                    raise RuntimeError(
                        "TTS backend changed sample rate during one response"
                    )
                audio_seconds += duration
                _write_json(
                    handler,
                    {
                        "ok": True,
                        "event": "chunk",
                        "request_id": request_id,
                        "index": index,
                        "text": chunk,
                        "output": str(output),
                        "audio_seconds": round(duration, 3),
                        "generation_seconds": round(elapsed, 3),
                    },
                )
                emitted += 1
        _write_json(
            handler,
            {
                "ok": True,
                "event": "done",
                "request_id": request_id,
                "cancelled": cancelled.is_set(),
                "chunks": emitted,
                "audio_seconds": round(audio_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "realtime_factor": (
                    round(generation_seconds / audio_seconds, 3)
                    if audio_seconds
                    else None
                ),
            },
        )
        if cancelled.is_set() and not emitted:
            shutil.rmtree(output_dir, ignore_errors=True)
    except (BrokenPipeError, ConnectionResetError):
        cancelled.set()
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    finally:
        with ACTIVE_STREAMS_LOCK:
            ACTIVE_STREAMS.pop(request_id, None)


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline(64 * 1024)
            if not line:
                _write_response(self, {"ok": True, "ready": True})
                return
            request = json.loads(line)
            if request.get("op") == "cancel":
                request_id = str(request.get("request_id", ""))
                if not REQUEST_ID.fullmatch(request_id):
                    raise ValueError(
                        "request_id must contain 1-64 letters, numbers, '_' or '-'"
                    )
                response = {
                    "ok": True,
                    "cancelled": _cancel_stream(request_id),
                    "request_id": request_id,
                }
                _write_response(self, response)
                return
            if request.get("op") == "validate_voice":
                voice = _request_voice(request)
                output = OUTPUT_ROOT / (
                    f".voice-validation-{threading.get_ident()}-"
                    f"{time.monotonic_ns()}.wav"
                )
                try:
                    with LOCK:
                        _sample_rate, duration, _elapsed = _synthesize(
                            "Voice validation.",
                            voice,
                            output,
                            apply_speed=False,
                        )
                    if (
                        duration <= 0
                        or not output.is_file()
                        or output.stat().st_size == 0
                    ):
                        raise RuntimeError(
                            "TTS provider returned empty validation audio"
                        )
                    response = {"ok": True, "voice_usable": True}
                finally:
                    output.unlink(missing_ok=True)
                _write_response(self, response)
                return
            if request.get("stream"):
                _stream_response(self, request)
                return
            text = _request_text(request)
            output = Path(
                str(request.get("output", OUTPUT_ROOT / "reply.wav"))
            ).resolve()
            if OUTPUT_ROOT.resolve() not in output.parents:
                raise ValueError(
                    "output must be inside the runtime voice-harness directory"
                )
            voice = _request_voice(request)
            with LOCK:
                sample_rate, duration, elapsed = _synthesize(text, voice, output)
            response = {
                "ok": True,
                "output": str(output),
                "sample_rate": sample_rate,
                "audio_seconds": round(duration, 3),
                "generation_seconds": round(elapsed, 3),
                "realtime_factor": round(elapsed / duration, 3),
            }
        except Exception as exc:
            diagnostic = redact_diagnostic(f"{type(exc).__name__}: {exc}")
            log(f"request failed: {diagnostic}")
            response = {"ok": False, "error": diagnostic}
        _write_response(self, response)


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main() -> None:
    global MODEL, SETTINGS, VENICE_API_KEY
    if "--check" in sys.argv[1:]:
        print("voice-harness-tts: ok")
        return
    SETTINGS = load_user_config().providers
    if SETTINGS.tts_provider == "local":
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        log("loading Chatterbox Turbo on CUDA")
        MODEL = ChatterboxTurboTTS.from_pretrained(device="cuda")
        torch.cuda.synchronize()
        log("model ready")
    else:
        VENICE_API_KEY = get_venice_api_key()
        log(f"using Venice TTS model {SETTINGS.tts_model} ({SETTINGS.tts_voice})")
    SOCKET_PATH.unlink(missing_ok=True)
    try:
        with Server(str(SOCKET_PATH), RequestHandler) as server:
            os.chmod(SOCKET_PATH, 0o600)
            log(f"listening on {SOCKET_PATH}")
            server.serve_forever()
    finally:
        SOCKET_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
