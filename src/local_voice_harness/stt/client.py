from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import STT_SOCKET
from ..errors import HarnessError, NoSpeechError
from ..ipc import unix_request

PROTOCOL_VERSION = 2
REQUEST_DEADLINE_SECONDS = 120.0
BUSY_BACKOFF_SECONDS = 0.1
MAX_BUSY_BACKOFF_SECONDS = 1.0
MAX_RESPONSE_BYTES = 1024 * 1024


class _LegacyServer(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RetainedTranscript:
    delivery_id: str
    text: str
    woke: bool
    state: str = "pending"

    def mark_ambiguous(self) -> None:
        _delivery_request("ambiguous", self.delivery_id)

    def release(self) -> None:
        _delivery_request("release", self.delivery_id)


def _busy_timeout_error(audio_path: Path) -> HarnessError:
    if not audio_path.exists():
        return HarnessError(
            "STT remained busy; audio is still owned by the server and no "
            "retryable generation is currently available"
        )
    return HarnessError(
        "STT remained busy; audio was preserved. Retry with "
        f"`voice-harness transcribe --generation {audio_path}`"
    )


def _read_response_frame(client: socket.socket) -> bytes:
    response = bytearray()
    while True:
        chunk = client.recv(min(64 * 1024, MAX_RESPONSE_BYTES + 2 - len(response)))
        if not chunk:
            raise _LegacyServer
        response.extend(chunk)
        newline = response.find(b"\n")
        if newline >= 0:
            if newline > MAX_RESPONSE_BYTES:
                raise HarnessError("STT response exceeds the protocol size limit")
            if newline != len(response) - 1:
                raise HarnessError("STT response contained unexpected trailing data")
            return bytes(response[:newline])
        if len(response) > MAX_RESPONSE_BYTES:
            raise HarnessError("STT response exceeds the protocol size limit")


def _decode_protocol(frame: bytes) -> dict[str, Any]:
    try:
        value = json.loads(frame)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _LegacyServer from exc
    if not isinstance(value, dict) or value.get("version") != PROTOCOL_VERSION:
        raise _LegacyServer
    return value


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("STT request deadline expired")
    return remaining


def _v2_request(
    audio_path: Path,
    *,
    timeout: float,
    retain: bool = False,
    woke: bool = False,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    request = {
        "version": PROTOCOL_VERSION,
        "type": "transcribe",
        "audio_path": str(audio_path),
    }
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(STT_SOCKET))
        client.settimeout(_remaining_timeout(deadline))
        client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        response = _decode_protocol(_read_response_frame(client))
        if response.get("ok") is False:
            return response
        if (
            response.get("ok") is not True
            or response.get("type") != "transcript"
            or not isinstance(response.get("delivery_id"), str)
            or not isinstance(response.get("text"), str)
        ):
            raise HarnessError("STT server returned an invalid transcription response")

        acknowledgment = {
            "version": PROTOCOL_VERSION,
            "type": "ack",
            "delivery_id": response["delivery_id"],
        }
        if retain:
            acknowledgment["disposition"] = "retain"
            acknowledgment["woke"] = woke
        client.settimeout(_remaining_timeout(deadline))
        client.sendall(
            json.dumps(acknowledgment, separators=(",", ":")).encode() + b"\n"
        )
        try:
            client.shutdown(socket.SHUT_WR)
        except OSError:
            pass

        trailing = bytearray()
        try:
            while chunk := client.recv(64 * 1024):
                trailing.extend(chunk)
        except OSError:
            if retain:
                response["_retained_state"] = "uncertain"
                try:
                    _delivery_request("ambiguous", str(response["delivery_id"]))
                except (HarnessError, OSError):
                    pass
                else:
                    response["_retained_state"] = "ambiguous"
                return response
            return response
        if trailing:
            try:
                trailing_response = _decode_protocol(bytes(trailing).strip())
            except _LegacyServer as exc:
                raise HarnessError(
                    "STT server returned an invalid acknowledgment response"
                ) from exc
            if trailing_response.get("ok") is False:
                return trailing_response
            raise HarnessError("STT server returned unexpected acknowledgment data")
        if retain:
            try:
                _delivery_request("pending", str(response["delivery_id"]))
            except (HarnessError, OSError):
                response["_retained_state"] = "uncertain"
            else:
                response["_retained_state"] = "pending"
        return response
    finally:
        client.close()


def _delivery_request(operation: str, delivery_id: str | None = None) -> dict[str, Any]:
    request: dict[str, object] = {
        "version": PROTOCOL_VERSION,
        "type": operation,
    }
    if delivery_id is not None:
        request["delivery_id"] = delivery_id
    response = unix_request(
        STT_SOCKET,
        json.dumps(request, separators=(",", ":")).encode() + b"\n",
        timeout=REQUEST_DEADLINE_SECONDS,
    )
    try:
        protocol = _decode_protocol(response.strip())
    except _LegacyServer as exc:
        raise HarnessError(
            "STT server does not support durable wake deliveries"
        ) from exc
    if protocol.get("ok") is False:
        code, message, details = _error_details(protocol)
        raise HarnessError(_error_message(code, message, details))
    return protocol


def recover_retained_transcripts() -> tuple[RetainedTranscript, ...]:
    protocol = _delivery_request("recover")
    deliveries = protocol.get("deliveries")
    if protocol.get("type") != "deliveries" or not isinstance(deliveries, list):
        raise HarnessError("STT server returned invalid retained deliveries")
    recovered: list[RetainedTranscript] = []
    for value in deliveries:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("delivery_id"), str)
            or not isinstance(value.get("text"), str)
            or not isinstance(value.get("woke"), bool)
            or value.get("state") not in {"pending", "ambiguous"}
        ):
            raise HarnessError("STT server returned an invalid retained delivery")
        recovered.append(
            RetainedTranscript(
                value["delivery_id"],
                value["text"],
                value["woke"],
                str(value["state"]),
            )
        )
    return tuple(recovered)


def _legacy_request(audio_path: Path, *, timeout: float) -> str:
    response = unix_request(STT_SOCKET, f"{audio_path}\n".encode(), timeout=timeout)
    return response.decode(errors="replace").strip()


def _error_details(protocol: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    error = protocol.get("error")
    if not isinstance(error, dict):
        raise HarnessError("STT server returned an invalid error response")
    return (
        str(error.get("code", "protocol_error")),
        str(error.get("message", "STT request failed")),
        error,
    )


def _error_message(
    code: str,
    message: str,
    details: dict[str, Any],
) -> str:
    result = f"{code}: {message}"
    retry_path = details.get("retry_path")
    quarantine_path = details.get("quarantine_path")
    preserved_path = details.get("preserved_path")
    if isinstance(retry_path, str) and Path(retry_path).exists():
        result += f". Retry with `voice-harness transcribe --generation {retry_path}`"
    elif isinstance(quarantine_path, str):
        result += f". Audio was quarantined at {quarantine_path}"
    elif isinstance(preserved_path, str):
        result += f". Audio was preserved at {preserved_path}"
    return result


def transcribe(audio_path: Path) -> str:
    started = time.perf_counter()
    deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
    backoff = BUSY_BACKOFF_SECONDS
    first_attempt = True
    text = ""
    while True:
        if first_attempt:
            timeout = REQUEST_DEADLINE_SECONDS
        else:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise _busy_timeout_error(audio_path)
        first_attempt = False
        try:
            try:
                protocol = _v2_request(audio_path, timeout=timeout)
            except _LegacyServer:
                text = _legacy_request(
                    audio_path,
                    timeout=_remaining_timeout(deadline),
                )
                protocol = None
        except OSError as exc:
            message = f"STT request failed: {exc}"
            if audio_path.exists():
                message += (
                    f". Retry with `voice-harness transcribe --generation {audio_path}`"
                )
            raise HarnessError(message) from exc
        if protocol is None:
            try:
                legacy_protocol = json.loads(text)
            except json.JSONDecodeError:
                legacy_protocol = None
            if (
                isinstance(legacy_protocol, dict)
                and legacy_protocol.get("ok") is False
                and isinstance(legacy_protocol.get("error"), dict)
            ):
                code, message, details = _error_details(legacy_protocol)
            else:
                break
        elif protocol.get("ok") is False:
            code, message, details = _error_details(protocol)
        else:
            text = str(protocol["text"]).strip()
            break
        if code != "server_busy":
            raise HarnessError(_error_message(code, message, details))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _busy_timeout_error(audio_path)
        delay = min(backoff, remaining)
        time.sleep(delay)
        backoff = min(backoff * 2, MAX_BUSY_BACKOFF_SECONDS)
    if text.startswith("__DICTATION_ERROR__:"):
        raise HarnessError(text.removeprefix("__DICTATION_ERROR__:"))
    if not text:
        raise NoSpeechError("STT did not recognize any speech")
    print(
        json.dumps({"stage": "stt", "seconds": round(time.perf_counter() - started, 3)})
    )
    return text


def transcribe_retained(audio_path: Path, *, woke: bool) -> RetainedTranscript:
    started = time.perf_counter()
    deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
    backoff = BUSY_BACKOFF_SECONDS
    while True:
        try:
            protocol = _v2_request(
                audio_path,
                timeout=_remaining_timeout(deadline),
                retain=True,
                woke=woke,
            )
        except _LegacyServer as exc:
            raise HarnessError(
                "STT server does not support durable wake deliveries"
            ) from exc
        except OSError as exc:
            raise HarnessError(f"STT request failed: {exc}") from exc
        if protocol.get("ok") is not False:
            break
        code, message, details = _error_details(protocol)
        if code != "server_busy":
            raise HarnessError(_error_message(code, message, details))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _busy_timeout_error(audio_path)
        delay = min(backoff, remaining)
        time.sleep(delay)
        backoff = min(backoff * 2, MAX_BUSY_BACKOFF_SECONDS)
    text = str(protocol["text"]).strip()
    print(
        json.dumps({"stage": "stt", "seconds": round(time.perf_counter() - started, 3)})
    )
    return RetainedTranscript(
        str(protocol["delivery_id"]),
        text,
        woke,
        str(protocol.get("_retained_state", "pending")),
    )
