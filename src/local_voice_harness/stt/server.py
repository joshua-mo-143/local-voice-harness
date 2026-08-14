from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import stat
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .. import config, recorder, vocabulary
from ..diagnostic_safety import redact_diagnostic
from ..transcript import (
    TranscriptReplacement,
    effective_replacements,
    normalize_transcript,
)
from ..user_config import (
    DEFAULT_DICTATION_REPLACEMENTS,
    DictationDevice,
    UserConfig,
    load_user_config,
)

SOCKET_PATH = config.STT_SOCKET
PROTOCOL_VERSION = 2
MAX_REQUEST_BYTES = 4096
ACCEPT_TIMEOUT_SECONDS = 0.5
READ_TIMEOUT_SECONDS = 2.0
MAX_CONNECTIONS = 16
PROCESSING_DIRECTORY = "stt-processing"
DELIVERED_DIRECTORY = "stt-delivered"
RETAINED_DIRECTORY = "stt-retained"
QUARANTINE_DIRECTORY = "stt-quarantine"
RETAINED_METADATA = "delivery.json"
RETAINED_AUDIO = "audio.wav"
MAX_RETAINED_DELIVERIES = 32
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
REPLACEMENTS = dict(DEFAULT_DICTATION_REPLACEMENTS)
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


def resolve_model_name(
    backend: Literal["parakeet", "whisper"], configured: str = ""
) -> str:
    configured = configured.strip()
    if backend == "parakeet":
        if not configured or configured in WHISPER_MODELS:
            return PARAKEET_DEFAULT_MODEL
        return configured
    return configured or WHISPER_DEFAULT_MODEL


def resolve_quantization(
    backend: Literal["parakeet", "whisper"], configured: str = "int8"
) -> str | None:
    if backend != "parakeet":
        return None
    value = configured.strip().lower()
    if value in {"", "none", "off", "fp32"}:
        return None
    return value


LOCK = threading.Lock()


@dataclass(frozen=True)
class STTRuntimeSettings:
    backend: Literal["parakeet", "whisper"]
    device: DictationDevice
    model_name: str
    quantization: str | None
    compute_type: str
    language: str | None
    prompt: str
    replacements: Mapping[str, str]


def runtime_settings(config: UserConfig) -> STTRuntimeSettings:
    """Derive the STT process snapshot from the unified typed configuration."""

    backend = resolve_backend(config.compute.dictation_backend)
    return STTRuntimeSettings(
        backend=backend,
        device=config.compute.dictation_device,
        model_name=resolve_model_name(backend, config.compute.dictation_model),
        quantization=resolve_quantization(
            backend, config.compute.dictation_quantization
        ),
        compute_type=config.compute.dictation_compute,
        language=resolve_language(config.compute.dictation_language),
        prompt=config.dictation.prompt,
        replacements=dict(config.dictation.replacements),
    )


class ProtocolError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retry_path: Path | None = None,
        quarantine_path: Path | None = None,
        preserved_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_path = retry_path
        self.quarantine_path = quarantine_path
        self.preserved_path = preserved_path


@dataclass(frozen=True)
class TranscriptionRequest:
    requested: Path
    paths: recorder.RecorderPaths
    version: int | None


@dataclass(frozen=True)
class DeliveryRequest:
    operation: Literal["recover", "release", "ambiguous"]
    delivery_id: str | None = None


@dataclass(frozen=True)
class AudioClaim:
    original: Path
    processing: Path
    paths: recorder.RecorderPaths


@dataclass(frozen=True)
class RecoveryResult:
    retry_path: Path | None = None
    quarantine_path: Path | None = None
    preserved_path: Path | None = None


def log(message: str) -> None:
    print(
        f"[dictation] {redact_diagnostic(message)}",
        file=sys.stderr,
        flush=True,
    )


def _user_replacements() -> tuple[vocabulary.Replacement, ...]:
    """Load user vocabulary corrections, tolerating a missing or broken store.

    The store is read on each transcription so that ``voice-harness vocabulary``
    edits take effect without restarting the dictation service.
    """

    try:
        return vocabulary.load(config.VOCABULARY_PATH).replacements
    except (vocabulary.VocabularyError, OSError):
        return ()


def transcript_replacements(
    configured: Mapping[str, str] = REPLACEMENTS,
) -> tuple[TranscriptReplacement, ...]:
    """Return the effective rules used by the current STT configuration."""

    return effective_replacements(_user_replacements(), configured)


def normalize(text: str, configured: Mapping[str, str] = REPLACEMENTS) -> str:
    """Apply text corrections with user vocabulary taking precedence.

    Precedence is user vocabulary first, then the injected process-start
    replacements; a user correction overrides any static entry with the same
    spoken source.
    """

    return normalize_transcript(text, transcript_replacements(configured))


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


def _parakeet_device(
    requested: DictationDevice,
) -> tuple[Literal["cpu", "cuda"], list[str]]:
    if requested is DictationDevice.CPU:
        return "cpu", ["CPUExecutionProvider"]

    try:
        import onnxruntime

        cuda_available = (
            "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        )
    except Exception as exc:
        if requested is DictationDevice.CUDA:
            raise RuntimeError(
                "CUDA dictation was requested, but ONNX Runtime could not inspect "
                "CUDA providers; install the dictation-cuda dependency profile "
                "and verify the NVIDIA driver"
            ) from exc
        raise
    if requested is DictationDevice.CUDA and not cuda_available:
        raise RuntimeError(
            "CUDA dictation was requested, but ONNX Runtime's "
            "CUDAExecutionProvider is unavailable; install the dictation-cuda "
            "dependency profile and verify the NVIDIA driver"
        )
    if requested is DictationDevice.CUDA:
        return "cuda", ["CUDAExecutionProvider"]
    if cuda_available:
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return "cpu", ["CPUExecutionProvider"]


class ParakeetTranscriber(Transcriber):
    def __init__(
        self,
        model_name: str,
        *,
        device: DictationDevice,
        quantization: str | None,
    ) -> None:
        resolved_device, providers = _parakeet_device(device)
        import onnx_asr

        log(
            f"loading Parakeet {model_name} on {resolved_device.upper()}"
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
    def __init__(
        self,
        model_name: str,
        *,
        device: DictationDevice,
        compute_type: str,
        language: str | None,
        prompt: str,
    ) -> None:
        resolved_device = _whisper_device(device)
        resolved_compute = _whisper_compute_type(resolved_device, compute_type)
        from faster_whisper import WhisperModel

        log(
            f"loading faster-whisper {model_name} on "
            f"{resolved_device.upper()} ({resolved_compute})"
        )
        self._model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=resolved_compute,
        )
        self._language = language
        self._prompt = prompt

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(
            audio_path,
            task="transcribe",
            language=self._language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self._prompt,
        )
        return "".join(segment.text for segment in segments).strip()


def _whisper_device(
    requested: DictationDevice,
) -> Literal["cpu", "cuda"]:
    if requested is DictationDevice.CPU:
        return "cpu"

    try:
        import ctranslate2

        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda_available = False
    if requested is DictationDevice.CUDA and not cuda_available:
        raise RuntimeError(
            "CUDA dictation was requested, but faster-whisper cannot access a "
            "CUDA device; verify the NVIDIA driver and CUDA libraries"
        )
    return "cuda" if cuda_available else "cpu"


def _whisper_compute_type(device: Literal["cpu", "cuda"], configured: str) -> str:
    if device == "cpu" and configured.strip().lower() in {"float16", "int8_float16"}:
        return "int8"
    return configured


def load_transcriber(settings: STTRuntimeSettings) -> Transcriber:
    if settings.backend == "parakeet":
        return ParakeetTranscriber(
            settings.model_name,
            device=settings.device,
            quantization=settings.quantization,
        )
    return WhisperTranscriber(
        settings.model_name,
        device=settings.device,
        compute_type=settings.compute_type,
        language=settings.language,
        prompt=settings.prompt,
    )


def _read_frame(connection: socket.socket) -> bytes:
    frame = bytearray()
    while True:
        try:
            chunk = connection.recv(min(4096, MAX_REQUEST_BYTES + 2 - len(frame)))
        except TimeoutError as exc:
            raise ProtocolError(
                "request_timeout", "request was not completed before the deadline"
            ) from exc
        if not chunk:
            raise ProtocolError("incomplete_request", "request must end with a newline")
        frame.extend(chunk)
        newline = frame.find(b"\n")
        if newline >= 0:
            if newline > MAX_REQUEST_BYTES:
                raise ProtocolError("request_too_large", "request exceeds size limit")
            if newline != len(frame) - 1:
                raise ProtocolError(
                    "unexpected_data", "only one newline-delimited frame is allowed"
                )
            return bytes(frame[:newline])
        if len(frame) > MAX_REQUEST_BYTES:
            raise ProtocolError("request_too_large", "request exceeds size limit")


def _recorder_path_sets() -> tuple[recorder.RecorderPaths, ...]:
    return tuple(
        recorder.RecorderPaths(
            audio.parent,
            audio,
            audio.parent / "recording.pid",
            audio.parent / "pw-record.log",
            config.RECORDING_LOCK,
        )
        for audio in (config.WAV_PATH, config.DICTATION_WAV_PATH)
    )


def _resolve_audio_path(
    value: str,
) -> tuple[Path, recorder.RecorderPaths]:
    if not value:
        raise ProtocolError("invalid_request", "audio path is required")
    requested = Path(value)
    for paths in _recorder_path_sets():
        if recorder.is_generation_path(paths, requested):
            return requested, paths
    raise ProtocolError(
        "invalid_audio_path", "audio path is not a harness recording generation"
    )


def _parse_request(frame: bytes) -> TranscriptionRequest | DeliveryRequest:
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("invalid_encoding", "request must be valid UTF-8") from exc

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        requested, paths = _resolve_audio_path(text)
        return TranscriptionRequest(requested, paths, None)

    if not isinstance(value, dict):
        raise ProtocolError("invalid_request", "request must be a JSON object")
    if value.get("version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported_protocol", "unsupported STT protocol version")
    request_type = value.get("type")
    if request_type in {"recover", "release", "ambiguous"}:
        delivery_id = value.get("delivery_id")
        if request_type != "recover" and (
            not isinstance(delivery_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", delivery_id) is None
        ):
            raise ProtocolError("invalid_request", "a valid delivery_id is required")
        return DeliveryRequest(request_type, delivery_id)
    if request_type != "transcribe":
        raise ProtocolError("invalid_request", "request type must be transcribe")
    audio_path = value.get("audio_path")
    if not isinstance(audio_path, str):
        raise ProtocolError("invalid_request", "audio_path must be a string")
    requested, paths = _resolve_audio_path(audio_path)
    return TranscriptionRequest(requested, paths, PROTOCOL_VERSION)


def _prepare_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise OSError(f"directory is not private and harness-owned: {path}")
    path.chmod(0o700)


def _quarantine_locked(
    path: Path,
    paths: recorder.RecorderPaths,
    *,
    reason: str,
) -> Path:
    quarantine_dir = paths.state_dir / QUARANTINE_DIRECTORY
    _prepare_private_directory(quarantine_dir)
    destination = quarantine_dir / path.name
    if destination.exists() or destination.is_symlink():
        destination = quarantine_dir / f"{path.stem}-{uuid.uuid4().hex}{path.suffix}"
    path.rename(destination)
    log(f"quarantined STT audio at {destination}: {reason}")
    return destination


def _claim_audio_path(requested: Path, paths: recorder.RecorderPaths) -> AudioClaim:
    processing_dir = paths.state_dir / PROCESSING_DIRECTORY
    with recorder.recording_lock(paths.state_dir, paths.lock):
        try:
            generation_directory_metadata = paths.generations.lstat()
            metadata = requested.lstat()
            _prepare_private_directory(processing_dir)
        except FileNotFoundError as exc:
            raise ProtocolError("audio_not_found", "audio file does not exist") from exc
        except OSError as exc:
            raise ProtocolError(
                "invalid_audio_path", "audio processing path is inaccessible"
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
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 44
        ):
            raise ProtocolError(
                "invalid_audio_path",
                "audio path must be a non-empty harness-owned regular file",
            )

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
            raise ProtocolError(
                "invalid_audio_path",
                "claimed audio file is inaccessible",
                preserved_path=claimed,
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            quarantine = _quarantine_locked(
                claimed, paths, reason="claimed audio failed ownership validation"
            )
            raise ProtocolError(
                "invalid_audio_path",
                "claimed audio failed ownership validation",
                quarantine_path=quarantine,
            )
        try:
            claimed.chmod(0o600)
        except OSError as exc:
            recovery = _restore_claim_locked(AudioClaim(requested, claimed, paths))
            raise _recovery_error(
                "invalid_audio_path",
                "claimed audio permissions could not be secured",
                recovery,
            ) from exc
        return AudioClaim(requested, claimed, paths)


def _original_path_for_claim(
    claim_path: Path, paths: recorder.RecorderPaths
) -> Path | None:
    stem, separator, claim_id = claim_path.stem.rpartition("-")
    if not separator or re.fullmatch(r"[0-9a-f]{32}", claim_id) is None or not stem:
        return None
    original = paths.generations / f"{stem}{claim_path.suffix}"
    if not recorder.is_generation_path(paths, original):
        return None
    return original


def _restore_claim_locked(claim: AudioClaim) -> RecoveryResult:
    try:
        metadata = claim.processing.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 44
        ):
            quarantine = _quarantine_locked(
                claim.processing,
                claim.paths,
                reason="processing claim failed ownership validation",
            )
            return RecoveryResult(quarantine_path=quarantine)
        _prepare_private_directory(claim.paths.generations)
        os.link(claim.processing, claim.original, follow_symlinks=False)
    except FileExistsError:
        try:
            processing_metadata = claim.processing.lstat()
            original_metadata = claim.original.lstat()
        except OSError as exc:
            quarantine = _quarantine_locked(
                claim.processing,
                claim.paths,
                reason=f"could not inspect restoration collision: {exc}",
            )
            return RecoveryResult(quarantine_path=quarantine)
        if (
            stat.S_ISREG(processing_metadata.st_mode)
            and stat.S_ISREG(original_metadata.st_mode)
            and processing_metadata.st_uid == os.getuid()
            and original_metadata.st_uid == os.getuid()
            and processing_metadata.st_size > 44
            and original_metadata.st_size > 44
            and processing_metadata.st_dev == original_metadata.st_dev
            and processing_metadata.st_ino == original_metadata.st_ino
        ):
            try:
                claim.processing.unlink()
            except OSError as exc:
                quarantine = _quarantine_locked(
                    claim.processing,
                    claim.paths,
                    reason=f"partial restoration cleanup failed: {exc}",
                )
                return RecoveryResult(
                    quarantine_path=quarantine,
                    preserved_path=claim.original,
                )
            else:
                return RecoveryResult(retry_path=claim.original)
        quarantine = _quarantine_locked(
            claim.processing,
            claim.paths,
            reason=f"restoration would overwrite {claim.original}",
        )
        return RecoveryResult(quarantine_path=quarantine)
    except OSError as exc:
        quarantine = _quarantine_locked(
            claim.processing,
            claim.paths,
            reason=f"could not restore generation: {exc}",
        )
        return RecoveryResult(quarantine_path=quarantine)

    try:
        claim.processing.unlink()
    except OSError as exc:
        try:
            claim.original.unlink()
        except OSError as rollback_exc:
            quarantine = _quarantine_locked(
                claim.processing,
                claim.paths,
                reason=(
                    f"processing cleanup failed ({exc}) and restoration rollback "
                    f"failed ({rollback_exc})"
                ),
            )
            return RecoveryResult(
                quarantine_path=quarantine,
                preserved_path=claim.original,
            )
        log(
            f"restoration cleanup failed; STT audio retained at {claim.processing}: {exc}"
        )
        return RecoveryResult(preserved_path=claim.processing)
    return RecoveryResult(retry_path=claim.original)


def _restore_claim(claim: AudioClaim) -> RecoveryResult:
    try:
        with recorder.recording_lock(claim.paths.state_dir, claim.paths.lock):
            result = _restore_claim_locked(claim)
    except Exception as exc:
        log(f"could not restore STT audio retained at {claim.processing}: {exc}")
        return RecoveryResult(preserved_path=claim.processing)
    if result.retry_path is not None:
        log(f"restored retryable STT audio at {result.retry_path}")
    return result


def _commit_claim(claim: AudioClaim) -> None:
    delivered_dir = claim.paths.state_dir / DELIVERED_DIRECTORY
    delivered: Path | None = None
    committed = False
    try:
        with recorder.recording_lock(claim.paths.state_dir, claim.paths.lock):
            _prepare_private_directory(delivered_dir)
            delivered = delivered_dir / claim.processing.name
            if delivered.exists() or delivered.is_symlink():
                raise FileExistsError(f"delivered claim already exists: {delivered}")
            claim.processing.rename(delivered)
            committed = True
    except Exception as exc:
        if not committed:
            raise
        log(f"STT delivery committed despite recording lock cleanup failure: {exc}")
    assert delivered is not None
    try:
        delivered.unlink()
    except OSError as exc:
        log(
            f"acknowledged STT audio retained for startup cleanup at {delivered}: {exc}"
        )


def _retained_root(paths: recorder.RecorderPaths) -> Path:
    return paths.state_dir / RETAINED_DIRECTORY


def _write_retained_claim(
    claim: AudioClaim,
    *,
    delivery_id: str,
    text: str,
    woke: bool,
) -> None:
    retained_root = _retained_root(claim.paths)
    temporary: Path | None = None
    committed = False
    try:
        with recorder.recording_lock(claim.paths.state_dir, claim.paths.lock):
            _prepare_private_directory(retained_root)
            deliveries = tuple(retained_root.iterdir())
            if len(deliveries) >= MAX_RETAINED_DELIVERIES:
                raise ProtocolError(
                    "retention_full",
                    "wake delivery retention is full and requires reconciliation",
                    preserved_path=claim.processing,
                )
            destination = retained_root / delivery_id
            if destination.exists() or destination.is_symlink():
                raise ProtocolError(
                    "delivery_conflict", "delivery identifier already exists"
                )
            temporary = retained_root / f".{delivery_id}-{uuid.uuid4().hex}"
            temporary.mkdir(mode=0o700)
            metadata = {
                "version": PROTOCOL_VERSION,
                "delivery_id": delivery_id,
                "text": text,
                "woke": woke,
                "state": "pending",
                "created_at": time.time(),
            }
            metadata_path = temporary / RETAINED_METADATA
            metadata_path.write_text(
                json.dumps(metadata, separators=(",", ":")),
                encoding="utf-8",
            )
            metadata_path.chmod(0o600)
            with metadata_path.open("rb") as metadata_file:
                os.fsync(metadata_file.fileno())
            claim.processing.rename(temporary / RETAINED_AUDIO)
            directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            temporary.rename(destination)
            committed = True
            temporary = None
            root_fd = os.open(retained_root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
    except Exception as exc:
        if committed:
            log(f"STT wake retention committed despite lock cleanup failure: {exc}")
            return
        if temporary is not None and claim.processing.exists():
            with contextlib.suppress(OSError):
                (temporary / RETAINED_METADATA).unlink()
            with contextlib.suppress(OSError):
                temporary.rmdir()
        raise


def _load_retained_delivery(path: Path) -> dict[str, object]:
    metadata_path = path / RETAINED_METADATA
    audio_path = path / RETAINED_AUDIO
    directory = path.lstat()
    metadata = metadata_path.lstat()
    audio = audio_path.lstat()
    if (
        not stat.S_ISDIR(directory.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or not stat.S_ISREG(audio.st_mode)
        or directory.st_uid != os.getuid()
        or metadata.st_uid != os.getuid()
        or audio.st_uid != os.getuid()
        or directory.st_mode & 0o077
        or metadata.st_mode & 0o077
        or audio.st_mode & 0o077
    ):
        raise OSError("retained delivery is not private and harness-owned")
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("version") != PROTOCOL_VERSION
        or re.fullmatch(r"[0-9a-f]{32}", str(value.get("delivery_id", ""))) is None
        or (not path.name.startswith(".") and value.get("delivery_id") != path.name)
        or not isinstance(value.get("text"), str)
        or not isinstance(value.get("woke"), bool)
        or value.get("state") not in {"pending", "ambiguous"}
    ):
        raise OSError("retained delivery metadata is invalid")
    return value


def _find_retained_delivery(delivery_id: str) -> tuple[Path, dict[str, object]]:
    for paths in _recorder_path_sets():
        delivery = _retained_root(paths) / delivery_id
        if delivery.exists():
            return delivery, _load_retained_delivery(delivery)
    raise ProtocolError("delivery_not_found", "retained delivery does not exist")


def _recover_retained_deliveries() -> list[dict[str, object]]:
    recovered: list[dict[str, object]] = []
    for paths in _recorder_path_sets():
        root = _retained_root(paths)
        if not root.exists():
            continue
        _prepare_private_directory(root)
        for delivery in sorted(root.iterdir()):
            if delivery.name.startswith("."):
                try:
                    metadata = _load_retained_delivery(delivery)
                    delivery_id = str(metadata["delivery_id"])
                    destination = root / delivery_id
                    if destination.exists() or destination.is_symlink():
                        raise OSError("retained delivery recovery destination exists")
                    delivery.rename(destination)
                    delivery = destination
                except (OSError, json.JSONDecodeError) as exc:
                    log(
                        "incomplete retained STT delivery requires manual recovery "
                        f"at {delivery}: {exc}"
                    )
                    continue
            try:
                recovered.append(_load_retained_delivery(delivery))
            except (OSError, json.JSONDecodeError) as exc:
                log(
                    f"retained STT delivery requires manual recovery at {delivery}: {exc}"
                )
    return recovered


def _update_retained_state(delivery_id: str, state: Literal["ambiguous"]) -> None:
    delivery, metadata = _find_retained_delivery(delivery_id)
    metadata["state"] = state
    replacement = delivery / f".{RETAINED_METADATA}-{uuid.uuid4().hex}"
    replacement.write_text(
        json.dumps(metadata, separators=(",", ":")),
        encoding="utf-8",
    )
    replacement.chmod(0o600)
    os.replace(replacement, delivery / RETAINED_METADATA)


def _release_retained_delivery(delivery_id: str) -> None:
    delivery, _metadata = _find_retained_delivery(delivery_id)
    (delivery / RETAINED_AUDIO).unlink()
    (delivery / RETAINED_METADATA).unlink()
    delivery.rmdir()


def _recovery_error(
    code: str,
    message: str,
    recovery: RecoveryResult,
) -> ProtocolError:
    return ProtocolError(
        code,
        message,
        retry_path=recovery.retry_path,
        quarantine_path=recovery.quarantine_path,
        preserved_path=recovery.preserved_path,
    )


def _recover_processing_directory(paths: recorder.RecorderPaths) -> None:
    processing_dir = paths.state_dir / PROCESSING_DIRECTORY
    if not processing_dir.exists():
        return
    _prepare_private_directory(processing_dir)
    for processing in tuple(processing_dir.iterdir()):
        original = _original_path_for_claim(processing, paths)
        if original is None:
            _quarantine_locked(
                processing, paths, reason="processing filename is not recoverable"
            )
            continue
        result = _restore_claim_locked(AudioClaim(original, processing, paths))
        if result.retry_path is not None:
            log(f"recovered retryable STT audio at {result.retry_path}")


def _recover_delivered_directory(paths: recorder.RecorderPaths) -> None:
    delivered_dir = paths.state_dir / DELIVERED_DIRECTORY
    if not delivered_dir.exists():
        return
    _prepare_private_directory(delivered_dir)
    for delivered in tuple(delivered_dir.iterdir()):
        original = _original_path_for_claim(delivered, paths)
        try:
            metadata = delivered.lstat()
        except OSError as exc:
            log(f"could not inspect delivered STT audio at {delivered}: {exc}")
            continue
        if (
            original is None
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            _quarantine_locked(
                delivered, paths, reason="delivered claim failed validation"
            )
            continue
        try:
            delivered.unlink()
        except OSError as exc:
            log(f"could not finish delivered STT cleanup at {delivered}: {exc}")


def recover_stranded_audio() -> None:
    for paths in _recorder_path_sets():
        with recorder.recording_lock(paths.state_dir, paths.lock):
            _recover_processing_directory(paths)
            _recover_delivered_directory(paths)


def _error_response(error: ProtocolError) -> bytes:
    details: dict[str, object] = {"code": error.code, "message": str(error)}
    if error.retry_path is not None:
        details["retry_path"] = str(error.retry_path)
    if error.quarantine_path is not None:
        details["quarantine_path"] = str(error.quarantine_path)
    if error.preserved_path is not None:
        details["preserved_path"] = str(error.preserved_path)
    return (
        json.dumps(
            {
                "ok": False,
                "version": PROTOCOL_VERSION,
                "error": details,
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _send(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(payload)


def _send_error(connection: socket.socket, error: ProtocolError) -> None:
    try:
        _send(connection, _error_response(error))
    except OSError as exc:
        log(f"could not deliver STT error response: {exc}")


def _success_response(text: str, delivery_id: str) -> bytes:
    return (
        json.dumps(
            {
                "ok": True,
                "version": PROTOCOL_VERSION,
                "type": "transcript",
                "delivery_id": delivery_id,
                "text": text,
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _validate_ack(frame: bytes, delivery_id: str) -> tuple[bool, bool]:
    try:
        value = json.loads(frame)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid_ack", "acknowledgment must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("invalid_ack", "acknowledgment must be a JSON object")
    if (
        value.get("version") != PROTOCOL_VERSION
        or value.get("type") != "ack"
        or value.get("delivery_id") != delivery_id
    ):
        raise ProtocolError(
            "invalid_ack", "acknowledgment does not match the transcription"
        )
    retain = value.get("disposition", "complete") == "retain"
    if value.get("disposition", "complete") not in {"complete", "retain"}:
        raise ProtocolError("invalid_ack", "unsupported acknowledgment disposition")
    woke = value.get("woke", False)
    if not isinstance(woke, bool):
        raise ProtocolError("invalid_ack", "woke must be a boolean")
    return retain, woke


def _delivery_response(request: DeliveryRequest) -> bytes:
    if request.operation == "recover":
        payload: dict[str, object] = {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "type": "deliveries",
            "deliveries": _recover_retained_deliveries(),
        }
    else:
        assert request.delivery_id is not None
        if request.operation == "release":
            _release_retained_delivery(request.delivery_id)
        else:
            _update_retained_state(request.delivery_id, "ambiguous")
        payload = {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "type": request.operation,
            "delivery_id": request.delivery_id,
        }
    return json.dumps(payload, separators=(",", ":")).encode() + b"\n"


def handle_connection(
    connection: socket.socket,
    transcriber: Transcriber,
    *,
    replacements: Mapping[str, str] = REPLACEMENTS,
) -> None:
    connection.settimeout(READ_TIMEOUT_SECONDS)
    claim: AudioClaim | None = None
    slot_acquired = False
    try:
        try:
            request = _parse_request(_read_frame(connection))
            if isinstance(request, DeliveryRequest):
                try:
                    _send(connection, _delivery_response(request))
                except ProtocolError as exc:
                    _send_error(connection, exc)
                except Exception as exc:
                    log(f"retained delivery operation failed: {exc}")
                    _send_error(
                        connection,
                        ProtocolError(
                            "delivery_operation_failed",
                            "retained delivery operation could not be completed",
                        ),
                    )
                return
            if not LOCK.acquire(blocking=False):
                raise ProtocolError(
                    "server_busy", "another transcription is already active"
                )
            slot_acquired = True
            try:
                claim = _claim_audio_path(request.requested, request.paths)
                text = normalize(
                    transcriber.transcribe(str(claim.processing)),
                    replacements,
                )
            except ProtocolError:
                raise
            except Exception as exc:
                log(f"transcription failed: {exc}")
                recovery = (
                    _restore_claim(claim) if claim is not None else RecoveryResult()
                )
                claim = None
                raise _recovery_error(
                    "transcription_failed",
                    redact_diagnostic(f"{type(exc).__name__}: {exc}"),
                    recovery,
                ) from exc
        except ProtocolError as exc:
            _send_error(connection, exc)
            return

        assert claim is not None
        if request.version is None:
            try:
                _send(connection, text.encode())
            except OSError as exc:
                log(f"legacy STT response delivery failed: {exc}")
            recovery = _restore_claim(claim)
            claim = None
            if recovery.retry_path is None:
                log("legacy STT audio could not be restored to its retry generation")
            return

        delivery_id = uuid.uuid4().hex
        try:
            _send(connection, _success_response(text, delivery_id))
        except OSError as exc:
            log(f"STT response delivery failed: {exc}")
            return

        try:
            retain, woke = _validate_ack(_read_frame(connection), delivery_id)
        except ProtocolError as exc:
            recovery = _restore_claim(claim)
            claim = None
            _send_error(
                connection,
                _recovery_error(exc.code, str(exc), recovery),
            )
            return

        try:
            if retain:
                _write_retained_claim(
                    claim,
                    delivery_id=delivery_id,
                    text=text,
                    woke=woke,
                )
            else:
                _commit_claim(claim)
        except Exception as exc:
            log(f"could not commit acknowledged STT audio deletion: {exc}")
            recovery = _restore_claim(claim)
            claim = None
            _send_error(
                connection,
                _recovery_error(
                    "delivery_commit_failed",
                    "transcription was received but audio cleanup could not be committed",
                    recovery,
                ),
            )
            return
        claim = None
    finally:
        if claim is not None:
            _restore_claim(claim)
        if slot_acquired:
            LOCK.release()
        connection.close()


def serve(
    transcriber: Transcriber,
    *,
    replacements: Mapping[str, str] = REPLACEMENTS,
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
            handle_connection(connection, transcriber, replacements=replacements)
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
                _send_error(
                    connection,
                    ProtocolError(
                        "server_busy", "too many incomplete requests are active"
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
    settings = runtime_settings(load_user_config())
    recover_stranded_audio()
    transcriber = load_transcriber(settings)
    log(
        f"model ready (backend={settings.backend}, model={settings.model_name}, "
        f"language={settings.language or 'auto-detect'})"
    )
    serve(transcriber, replacements=settings.replacements)


if __name__ == "__main__":
    main()
