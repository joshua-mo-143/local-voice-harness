from __future__ import annotations

import json
import os
import re
import socket
import stat
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from .. import config, recorder, vocabulary

SOCKET_PATH = config.STT_SOCKET
MAX_REQUEST_BYTES = 4096
ACCEPT_TIMEOUT_SECONDS = 0.5
READ_TIMEOUT_SECONDS = 2.0
MAX_CONNECTIONS = 16
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
        "DICTATION_REPLACEMENTS",
        "herder:herdr;cursa:Cursor;curser:Cursor;"
        "service:Jarvis;jarvus:Jarvis;jervis:Jarvis",
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


class ProtocolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def log(message: str) -> None:
    print(f"[dictation] {message}", file=sys.stderr, flush=True)


def _user_replacements() -> tuple[vocabulary.Replacement, ...]:
    """Load user vocabulary corrections, tolerating a missing or broken store.

    The store is read on each transcription so that ``voice-harness vocabulary``
    edits take effect without restarting the dictation service.
    """

    try:
        return vocabulary.load(config.VOCABULARY_PATH).replacements
    except (vocabulary.VocabularyError, OSError):
        return ()


def normalize(text: str) -> str:
    """Apply text corrections with user vocabulary taking precedence.

    Precedence is user vocabulary first, then ``DICTATION_REPLACEMENTS`` and the
    built-in defaults; a user correction overrides any static entry with the same
    spoken source.
    """

    user = _user_replacements()
    overridden = {replacement.spoken.casefold() for replacement in user}
    for replacement in user:
        text = replacement.apply(text)
    for source, target in REPLACEMENTS.items():
        if source.casefold() in overridden:
            continue
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


def _read_frame(connection: socket.socket) -> bytes:
    request = bytearray()
    while True:
        try:
            chunk = connection.recv(min(4096, MAX_REQUEST_BYTES + 2 - len(request)))
        except TimeoutError as exc:
            raise ProtocolError(
                "request_timeout", "request was not completed before the deadline"
            ) from exc
        if not chunk:
            raise ProtocolError("incomplete_request", "request must end with a newline")
        request.extend(chunk)
        newline = request.find(b"\n")
        if newline >= 0:
            if newline > MAX_REQUEST_BYTES:
                raise ProtocolError("request_too_large", "request exceeds size limit")
            if newline != len(request) - 1:
                raise ProtocolError(
                    "unexpected_data", "only one newline-delimited request is allowed"
                )
            return bytes(request[:newline])
        if len(request) > MAX_REQUEST_BYTES:
            raise ProtocolError("request_too_large", "request exceeds size limit")


def _parse_audio_path(frame: bytes) -> tuple[Path, recorder.RecorderPaths]:
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("invalid_encoding", "request must be valid UTF-8") from exc
    if not text:
        raise ProtocolError("invalid_request", "audio path is required")
    requested = Path(text)
    path_sets = tuple(
        recorder.RecorderPaths(
            audio.parent,
            audio,
            audio.parent / "recording.pid",
            audio.parent / "pw-record.log",
            config.RECORDING_LOCK,
        )
        for audio in (config.WAV_PATH, config.DICTATION_WAV_PATH)
    )
    for paths in path_sets:
        if recorder.is_generation_path(paths, requested):
            return requested, paths
    raise ProtocolError(
        "invalid_audio_path", "audio path is not a harness recording generation"
    )


def _claim_audio_path(requested: Path, paths: recorder.RecorderPaths) -> Path:
    processing_dir = paths.state_dir / "stt-processing"
    with recorder.recording_lock(paths.state_dir, paths.lock):
        try:
            generation_directory_metadata = paths.generations.lstat()
            processing_dir.mkdir(mode=0o700, exist_ok=True)
            directory_metadata = processing_dir.lstat()
        except OSError as exc:
            raise ProtocolError(
                "invalid_audio_path", "audio processing directory is inaccessible"
            ) from exc
        if (
            not stat.S_ISDIR(generation_directory_metadata.st_mode)
            or generation_directory_metadata.st_uid != os.getuid()
            or generation_directory_metadata.st_mode & 0o077
        ):
            raise ProtocolError(
                "invalid_audio_path",
                "audio generation directory must be private and harness-owned",
            )
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.getuid()
        ):
            raise ProtocolError(
                "invalid_audio_path",
                "audio processing directory must be harness-owned",
            )
        processing_dir.chmod(0o700)
        claimed = processing_dir / f"{requested.stem}-{uuid.uuid4().hex}.wav"
        try:
            requested.rename(claimed)
        except FileNotFoundError as exc:
            raise ProtocolError("audio_not_found", "audio file does not exist") from exc
        except OSError as exc:
            raise ProtocolError(
                "invalid_audio_path", "audio file could not be claimed"
            ) from exc
        try:
            metadata = claimed.lstat()
        except OSError as exc:
            claimed.unlink(missing_ok=True)
            raise ProtocolError(
                "invalid_audio_path", "claimed audio file is inaccessible"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            claimed.unlink(missing_ok=True)
            raise ProtocolError(
                "invalid_audio_path", "audio path must be a harness-owned regular file"
            )
        return claimed


def _error_response(error: ProtocolError) -> bytes:
    return (
        json.dumps(
            {
                "ok": False,
                "error": {"code": error.code, "message": str(error)},
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _send(connection: socket.socket, payload: bytes) -> None:
    try:
        connection.sendall(payload)
    except (BrokenPipeError, ConnectionResetError, TimeoutError):
        pass


def handle_connection(connection: socket.socket, transcriber: Transcriber) -> None:
    connection.settimeout(READ_TIMEOUT_SECONDS)
    audio_path: Path | None = None
    try:
        try:
            requested, paths = _parse_audio_path(_read_frame(connection))
            if not LOCK.acquire(blocking=False):
                raise ProtocolError(
                    "server_busy", "another transcription is already active"
                )
            try:
                audio_path = _claim_audio_path(requested, paths)
                text = normalize(transcriber.transcribe(str(audio_path)))
            finally:
                LOCK.release()
        except ProtocolError as exc:
            _send(connection, _error_response(exc))
            return
        except Exception as exc:
            log(f"transcription failed: {exc}")
            _send(
                connection,
                _error_response(
                    ProtocolError(
                        "transcription_failed", f"{type(exc).__name__}: {exc}"
                    )
                ),
            )
            return
        _send(connection, text.encode())
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
        connection.close()


def serve(
    transcriber: Transcriber,
    *,
    socket_path: Path = SOCKET_PATH,
    stop_event: threading.Event | None = None,
    ready_event: threading.Event | None = None,
) -> None:
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
    threads: set[threading.Thread] = set()

    def run_connection(connection: socket.socket) -> None:
        try:
            handle_connection(connection, transcriber)
        finally:
            slots.release()

    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(4)
        if ready_event is not None:
            ready_event.set()
        server.settimeout(ACCEPT_TIMEOUT_SECONDS)
        log(f"listening on {socket_path}")
        while stop_event is None or not stop_event.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                threads = {thread for thread in threads if thread.is_alive()}
                continue
            if not slots.acquire(blocking=False):
                _send(
                    connection,
                    _error_response(
                        ProtocolError(
                            "server_busy", "too many incomplete requests are active"
                        )
                    ),
                )
                connection.close()
                continue
            thread = threading.Thread(
                target=run_connection, args=(connection,), daemon=True
            )
            threads.add(thread)
            thread.start()
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    transcriber = load_transcriber()
    log(
        f"model ready (backend={BACKEND}, model={MODEL_NAME}, "
        f"language={LANGUAGE or 'auto-detect'})"
    )
    serve(transcriber)


if __name__ == "__main__":
    main()
