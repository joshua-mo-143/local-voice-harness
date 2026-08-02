from __future__ import annotations

import json
import os
import re
import shutil
import socketserver
import sys
import threading
import time
from pathlib import Path

RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
SOCKET_PATH = RUNTIME / "voice-harness-tts.sock"
OUTPUT_ROOT = RUNTIME / "voice-harness"
OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
LOCK = threading.Lock()
ACTIVE_STREAMS: dict[str, threading.Event] = {}
ACTIVE_STREAMS_LOCK = threading.Lock()
MAX_CHUNK_CHARS = 160
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MODEL: object | None = None


def log(message: str) -> None:
    print(f"[voice-harness-tts] {message}", file=sys.stderr, flush=True)


def split_text(text: str, *, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split prose into short Chatterbox requests without cutting normal words."""
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


def _request_text(request: dict[str, object]) -> str:
    text = str(request.get("text", "")).strip()
    if not text:
        raise ValueError("text must not be empty")
    if len(text) > 2_000:
        raise ValueError("text exceeds 2000 characters")
    return text


def _request_voice(request: dict[str, object]) -> Path | None:
    voice_value = str(request.get("voice", "")).strip()
    voice = Path(voice_value).expanduser().resolve() if voice_value else None
    if voice is not None and not voice.is_file():
        raise FileNotFoundError(f"reference voice not found: {voice}")
    return voice


def _generate(text: str, voice: Path | None) -> tuple[object, float]:
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


def _cancel_stream(request_id: str) -> bool:
    with ACTIVE_STREAMS_LOCK:
        cancelled = ACTIVE_STREAMS.get(request_id)
    if cancelled is None:
        return False
    cancelled.set()
    return True


def _stream_response(
    handler: socketserver.StreamRequestHandler,
    request: dict[str, object],
) -> None:
    import soundfile as sf

    text = _request_text(request)
    voice = _request_voice(request)
    request_id = str(request.get("request_id", ""))
    if not REQUEST_ID.fullmatch(request_id):
        raise ValueError("request_id must contain 1-64 letters, numbers, '_' or '-'")
    chunks = split_text(text)
    output_dir = OUTPUT_ROOT / f"stream-{request_id}"
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    cancelled = threading.Event()
    with ACTIVE_STREAMS_LOCK:
        if request_id in ACTIVE_STREAMS:
            raise ValueError("request_id is already active")
        ACTIVE_STREAMS[request_id] = cancelled

    generation_seconds = 0.0
    audio_seconds = 0.0
    emitted = 0
    try:
        _write_json(
            handler,
            {
                "ok": True,
                "event": "start",
                "request_id": request_id,
                "sample_rate": MODEL.sr,
                "chunks": len(chunks),
            },
        )
        # Keep one stream contiguous on the single GPU. A cancellation handler does
        # not acquire this lock, so it can still stop us between model calls.
        with LOCK:
            for index, chunk in enumerate(chunks):
                if cancelled.is_set():
                    break
                audio, elapsed = _generate(chunk, voice)
                generation_seconds += elapsed
                if cancelled.is_set():
                    break
                output = output_dir / f"{index:04d}.wav"
                sf.write(str(output), audio, MODEL.sr, subtype="PCM_16")
                duration = len(audio) / MODEL.sr
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
            import soundfile as sf

            line = self.rfile.readline(64 * 1024)
            if not line:
                _write_json(self, {"ok": True, "ready": True})
                return
            request = json.loads(line)
            if request.get("op") == "cancel":
                request_id = str(request.get("request_id", ""))
                response = {
                    "ok": True,
                    "cancelled": _cancel_stream(request_id),
                    "request_id": request_id,
                }
                _write_json(self, response)
                return
            if request.get("stream"):
                _stream_response(self, request)
                return
            text = _request_text(request)
            output = Path(
                str(request.get("output", OUTPUT_ROOT / "reply.wav"))
            ).resolve()
            if OUTPUT_ROOT.resolve() not in output.parents:
                raise ValueError("output must be inside the runtime voice-harness directory")
            voice = _request_voice(request)
            with LOCK:
                audio, elapsed = _generate(text, voice)
            sf.write(str(output), audio, MODEL.sr)
            duration = len(audio) / MODEL.sr
            response = {
                "ok": True,
                "output": str(output),
                "sample_rate": MODEL.sr,
                "audio_seconds": round(duration, 3),
                "generation_seconds": round(elapsed, 3),
                "realtime_factor": round(elapsed / duration, 3),
            }
        except Exception as exc:
            log(f"request failed: {type(exc).__name__}: {exc}")
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            _write_json(self, response)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main() -> None:
    global MODEL
    if "--check" in sys.argv[1:]:
        print("voice-harness-tts: ok")
        return
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    log("loading Chatterbox Turbo on CUDA")
    MODEL = ChatterboxTurboTTS.from_pretrained(device="cuda")
    torch.cuda.synchronize()
    log("model ready")
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
