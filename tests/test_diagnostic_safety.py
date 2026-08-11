from __future__ import annotations

import pytest

from local_voice_harness.diagnostic_safety import redact_diagnostic, redact_fields


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
