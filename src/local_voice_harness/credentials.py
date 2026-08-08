from __future__ import annotations

import shutil
import subprocess

from .errors import HarnessError

SECRET_TOOL = "secret-tool"
SECRET_ATTRIBUTES = ("application", "local-voice-harness", "provider", "venice")
SECRET_LABEL = "Local Voice Harness Venice API key"
MAX_API_KEY_CHARS = 4096


class CredentialError(HarnessError):
    """A desktop Secret Service operation failed."""


def _secret_tool() -> str:
    executable = shutil.which(SECRET_TOOL)
    if executable is None:
        raise CredentialError(
            "secret-tool is unavailable; install libsecret with "
            "`paru -S --needed libsecret`"
        )
    return executable


def _valid_api_key(value: str) -> str:
    key = value.strip()
    if (
        not key
        or len(key) > MAX_API_KEY_CHARS
        or any(character.isspace() for character in key)
    ):
        raise CredentialError("Venice API key must be one non-empty token")
    return key


def _operation_error(action: str, process: subprocess.CompletedProcess[str]) -> None:
    detail = process.stderr.strip() or f"secret {action} failed"
    if "not activatable" in detail.casefold():
        raise CredentialError(
            "no desktop Secret Service provider is available; install and start one "
            "such as oo7 (`paru -S --needed oo7`)"
        )
    raise CredentialError(f"could not {action} Venice credential: {detail}")


def get_venice_api_key() -> str:
    try:
        process = subprocess.run(
            [_secret_tool(), "lookup", *SECRET_ATTRIBUTES],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CredentialError(
            f"could not access desktop Secret Service: {exc}"
        ) from exc
    if process.returncode:
        _operation_error("read", process)
    if not process.stdout.strip():
        raise CredentialError(
            "Venice API key is not stored; run `voice-harness credentials set`"
        )
    return _valid_api_key(process.stdout)


def store_venice_api_key(value: str) -> None:
    key = _valid_api_key(value)
    try:
        process = subprocess.run(
            [
                _secret_tool(),
                "store",
                f"--label={SECRET_LABEL}",
                *SECRET_ATTRIBUTES,
            ],
            input=key,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CredentialError(
            f"could not access desktop Secret Service: {exc}"
        ) from exc
    if process.returncode:
        _operation_error("store", process)


def delete_venice_api_key() -> None:
    try:
        process = subprocess.run(
            [_secret_tool(), "clear", *SECRET_ATTRIBUTES],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CredentialError(
            f"could not access desktop Secret Service: {exc}"
        ) from exc
    if process.returncode:
        _operation_error("delete", process)
