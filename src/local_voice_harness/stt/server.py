from __future__ import annotations

import os
import re
import socket
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
SOCKET_PATH = RUNTIME / "dictation.sock"
WHISPER_MODELS = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    }
)
PARAKEET_DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v2"
WHISPER_DEFAULT_MODEL = "large-v3-turbo"
PROMPT = os.environ.get(
    "DICTATION_PROMPT",
    "Technical software engineering dictation that may mention Cursor, Herdr, "
    "code, files, functions, terminals, and command-line tools.",
)
REPLACEMENTS = {
    source.strip(): target.strip()
    for entry in os.environ.get(
        "DICTATION_REPLACEMENTS", "herder:herdr;cursa:Cursor;curser:Cursor"
    ).split(";")
    if entry.strip() and ":" in entry
    for source, target in [entry.split(":", 1)]
    if source.strip()
}
LANGUAGE_ALIASES = {"english": "en", "chinese": "zh", "mandarin": "zh"}


def resolve_backend(value: str) -> Literal["parakeet", "whisper"]:
    backend = value.strip().lower()
    if backend == "parakeet":
        return "parakeet"
    if backend == "whisper":
        return "whisper"
    raise ValueError(f"unsupported DICTATION_BACKEND {value!r}")


def resolve_language(value: str) -> str | None:
    """Map a configured language to a Whisper code, or ``None`` to auto-detect."""

    normalized = value.strip().lower()
    if normalized in {"", "auto", "detect"}:
        return None
    return LANGUAGE_ALIASES.get(normalized, normalized)


def resolve_model_name(backend: Literal["parakeet", "whisper"]) -> str:
    configured = os.environ.get("DICTATION_MODEL", "").strip()
    if backend == "parakeet":
        if not configured or configured in WHISPER_MODELS:
            return PARAKEET_DEFAULT_MODEL
        return configured
    return configured or WHISPER_DEFAULT_MODEL


def resolve_quantization(backend: Literal["parakeet", "whisper"]) -> str | None:
    if backend != "parakeet":
        return None
    value = os.environ.get("DICTATION_QUANTIZATION", "int8").strip().lower()
    if value in {"", "none", "off", "fp32"}:
        return None
    return value


BACKEND = resolve_backend(os.environ.get("DICTATION_BACKEND", "parakeet"))
MODEL_NAME = resolve_model_name(BACKEND)
QUANTIZATION = resolve_quantization(BACKEND)
COMPUTE_TYPE = os.environ.get("DICTATION_COMPUTE", "float16")
LANGUAGE = resolve_language(os.environ.get("DICTATION_LANGUAGE", "auto"))
LOCK = threading.Lock()


def log(message: str) -> None:
    print(f"[dictation] {message}", file=sys.stderr, flush=True)


def normalize(text: str) -> str:
    for source, target in REPLACEMENTS.items():
        text = re.sub(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            lambda _match, replacement=target: replacement,
            text,
            flags=re.IGNORECASE,
        )
    return text


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class ParakeetTranscriber(Transcriber):
    def __init__(self, model_name: str, *, quantization: str | None) -> None:
        import onnx_asr

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        log(
            f"loading Parakeet {model_name} on CUDA"
            + (f" ({quantization})" if quantization else "")
        )
        self._model = onnx_asr.load_model(
            model_name,
            quantization=quantization,
            providers=providers,
        )

    def transcribe(self, audio_path: str) -> str:
        return str(self._model.recognize(audio_path) or "").strip()


class WhisperTranscriber(Transcriber):
    def __init__(self, model_name: str, *, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        log(f"loading faster-whisper {model_name} on CUDA ({compute_type})")
        self._model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        self._language = LANGUAGE

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(
            audio_path,
            task="transcribe",
            language=self._language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=PROMPT,
        )
        return normalize("".join(segment.text for segment in segments).strip())


def load_transcriber() -> Transcriber:
    if BACKEND == "parakeet":
        return ParakeetTranscriber(MODEL_NAME, quantization=QUANTIZATION)
    return WhisperTranscriber(MODEL_NAME, compute_type=COMPUTE_TYPE)


def main() -> None:
    transcriber = load_transcriber()
    log(
        f"model ready (backend={BACKEND}, model={MODEL_NAME}, "
        f"language={LANGUAGE or 'auto-detect'})"
    )
    SOCKET_PATH.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o600)
        server.listen(4)
        log(f"listening on {SOCKET_PATH}")
        while True:
            connection, _ = server.accept()
            try:
                request = bytearray()
                while not request.endswith(b"\n"):
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    request.extend(chunk)
                audio_path = request.decode(errors="replace").strip()
                if not audio_path:
                    continue
                try:
                    with LOCK:
                        text = normalize(transcriber.transcribe(audio_path))
                except Exception as exc:
                    log(f"transcription failed: {exc}")
                    text = f"__DICTATION_ERROR__:{type(exc).__name__}: {exc}"
                connection.sendall(text.encode())
            finally:
                connection.close()
    finally:
        server.close()
        SOCKET_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
