from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest

from local_voice_harness.diagnostic_safety import (
    COMMAND_FAILURE,
    SPEECH_DELIVERY_FAILURE,
    diagnostic_log_path,
    log_diagnostic,
    redact_diagnostic,
    redact_fields,
    user_facing_failure_message,
)
from local_voice_harness.errors import HarnessError, NoSpeechError, SpeechDeliveryError


@pytest.mark.parametrize(
    ("diagnostic", "secret"),
    [
        ("Authorization: Bearer top-secret", "top-secret"),
        ("Authorization=Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("request failed with Bearer abc.def-123", "abc.def-123"),
        ("password='correct horse battery staple'", "correct horse battery staple"),
        ("api_key=venice-secret", "venice-secret"),
        ("token: arbitrary-token-value", "arbitrary-token-value"),
        ("github token ghp_1234567890abcdef", "ghp_1234567890abcdef"),
        ("key sk-1234567890abcdef", "sk-1234567890abcdef"),
        (
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123",
            "eyJhbGciOiJIUzI1NiJ9",
        ),
        ("clone https://user:password@example.com/repo", "user:password"),
    ],
)
def test_redact_diagnostic_removes_known_credentials(
    diagnostic: str,
    secret: str,
) -> None:
    redacted = redact_diagnostic(diagnostic)

    assert secret not in redacted
    assert "[REDACTED" in redacted


def test_redact_diagnostic_redacts_before_bounding_and_normalizes_lines() -> None:
    secret = "s" * 80
    diagnostic = f"first line\nAuthorization: Bearer {secret}\n" + "x" * 200

    redacted = redact_diagnostic(diagnostic, limit=64)

    assert len(redacted) == 64
    assert "\n" not in redacted
    assert secret not in redacted
    assert "[REDACTED]" in redacted


def test_redact_fields_recurses_through_structured_log_values() -> None:
    redacted = redact_fields(
        {
            "error": "password=hunter2",
            "events": ["Authorization: Basic abc123", {"token": "token=secret"}],
        }
    )

    assert redacted == {
        "error": "password=[REDACTED]",
        "events": [
            "Authorization: [REDACTED]",
            {"token": "token=[REDACTED]"},
        ],
    }


def test_log_diagnostic_persists_when_stderr_is_discarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    with Path(os.devnull).open("w", encoding="utf-8") as discarded:
        with contextlib.redirect_stderr(discarded):
            log_diagnostic(
                "cli",
                "command_failed",
                "NoSpeechError: STT did not recognize any speech",
            )

    path = diagnostic_log_path()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["component"] == "cli"
    assert persisted["event"] == "command_failed"
    assert persisted["diagnostic"] == "NoSpeechError: STT did not recognize any speech"
    assert path.stat().st_mode & 0o777 == 0o600


def test_log_diagnostic_write_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "runtime"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(blocker))
    log_diagnostic("cli", "command_failed", "boom")


def test_user_facing_failure_message_prefers_harness_errors() -> None:
    assert (
        user_facing_failure_message(NoSpeechError("STT did not recognize any speech"))
        == "STT did not recognize any speech"
    )
    assert user_facing_failure_message(RuntimeError("unexpected")) == COMMAND_FAILURE
    assert (
        user_facing_failure_message(SpeechDeliveryError("Venice failed"))
        == SPEECH_DELIVERY_FAILURE
    )
    redacted = user_facing_failure_message(HarnessError("token=super-secret-value"))
    assert "super-secret-value" not in redacted
    assert "[REDACTED]" in redacted
