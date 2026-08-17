from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .errors import HarnessError, SpeechDeliveryError

DIAGNOSTIC_LIMIT = 2_000
NOTIFICATION_LIMIT = 240
COMMAND_FAILURE = "The command failed. Check the logs for details."
VOICE_REQUEST_FAILURE = "The voice request failed. Check the logs for details."
RECORDING_FAILURE = "Audio recording failed. Check the logs for details."
PLAYBACK_FAILURE = "Audio playback failed. Check the logs for details."
SPEECH_DELIVERY_FAILURE = "Speech delivery failed. Check the logs for details."
DAEMON_FAILURE = "The voice service failed. Check the logs for details."
CURSOR_TOOL_FAILURE = "The Cursor tool failed. Check the logs for details."

_AUTHORIZATION = re.compile(r"(?i)(\bauthorization\b\s*[:=]\s*)(?:[^\r\n]+)")
_SCHEME_CREDENTIAL = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
_NAMED_CREDENTIAL = re.compile(
    r"""(?ix)
    \b(
        api[_-]?key|access[_-]?token|refresh[_-]?token|
        auth[_-]?token|token|password|passwd|secret
    )\b
    (\s*[:=]\s*)
    (?:
        "[^"\r\n]*"|'[^'\r\n]*'|[^\s,;&\r\n]+
    )
    """
)
_KNOWN_TOKEN = re.compile(
    r"(?i)\b(?:github_pat_|gh[pousr]_|sk-|vce[_-])[A-Za-z0-9._-]{8,}"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")
_URL_USERINFO = re.compile(r"(?i)\b(https?://)([^/\s:@]+):([^@\s/]+)@")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def redact_diagnostic(
    value: object,
    *,
    limit: int | None = DIAGNOSTIC_LIMIT,
) -> str:
    """Return a single-line, bounded diagnostic with known credentials removed."""

    if limit is not None and limit < 1:
        raise ValueError("diagnostic limit must be positive")
    text = str(value)
    text = _AUTHORIZATION.sub(r"\1[REDACTED]", text)
    text = _SCHEME_CREDENTIAL.sub(r"\1 [REDACTED]", text)
    text = _NAMED_CREDENTIAL.sub(r"\1\2[REDACTED]", text)
    text = _KNOWN_TOKEN.sub("[REDACTED_TOKEN]", text)
    text = _JWT.sub("[REDACTED_TOKEN]", text)
    text = _URL_USERINFO.sub(r"\1[REDACTED]@", text)
    text = _CONTROL.sub(" ", text)
    normalized = _WHITESPACE.sub(" ", text).strip()
    return normalized if limit is None else normalized[:limit]


def redact_fields(
    value: object,
    *,
    limit: int | None = DIAGNOSTIC_LIMIT,
) -> object:
    """Recursively redact strings before structured diagnostic logging."""

    if isinstance(value, str):
        return redact_diagnostic(value, limit=limit)
    if isinstance(value, Mapping):
        return {
            str(key): redact_fields(item, limit=limit) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_fields(item, limit=limit) for item in value]
    return value


def diagnostic_log_path(
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Session-scoped JSONL sink for CLI diagnostics when stderr is discarded."""

    runtime = Path(
        (os.environ if environment is None else environment).get(
            "XDG_RUNTIME_DIR", "/tmp"
        )
    )
    return runtime / "voice-harness" / "diagnostics.jsonl"


def user_facing_failure_message(exc: BaseException) -> str:
    """Return a notification-safe failure string for a CLI exception."""

    if isinstance(exc, SpeechDeliveryError):
        return SPEECH_DELIVERY_FAILURE
    if isinstance(exc, HarnessError):
        message = redact_diagnostic(exc, limit=NOTIFICATION_LIMIT)
        if message:
            return message
    return COMMAND_FAILURE


def _append_diagnostic_log(payload: str) -> None:
    try:
        path = diagnostic_log_path()
        path.parent.mkdir(mode=0o700, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC
        fd = os.open(path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload.encode())
        finally:
            os.close(fd)
    except OSError:
        return


def log_diagnostic(component: str, event: str, value: object) -> None:
    """Write one redacted structured diagnostic event to stderr and the session log."""

    payload = json.dumps(
        {
            "component": component,
            "event": event,
            "diagnostic": redact_diagnostic(value),
        },
        ensure_ascii=False,
    )
    print(payload, file=sys.stderr, flush=True)
    _append_diagnostic_log(payload + "\n")
