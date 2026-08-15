from __future__ import annotations

import os
from collections.abc import Iterator
from unittest import mock

import pytest

_AMBIENT_PREFIXES = ("VOICE_HARNESS_", "DICTATION_")
_AMBIENT_KEYS = ("STATE_DIRECTORY", "VENICE_API_KEY")


@pytest.fixture(autouse=True)
def isolate_user_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep default tests off the invoking user's HOME, XDG, and backend env."""

    root = tmp_path_factory.mktemp("isolated-home")
    home = root / "home"
    config_home = root / "config"
    state_home = root / "state"
    cache_home = root / "cache"
    runtime_dir = root / "runtime"
    for path in (home, config_home, state_home, cache_home, runtime_dir):
        path.mkdir()
    secret_bin = root / "bin"
    secret_bin.mkdir()
    secret_tool = secret_bin / "secret-tool"
    secret_tool.write_text(
        "#!/bin/sh\n"
        "echo 'isolated test environment has no Secret Service' >&2\n"
        "exit 1\n"
    )
    secret_tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{secret_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    for key in list(os.environ):
        if key.startswith(_AMBIENT_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    for key in _AMBIENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def mock_browser_desktop() -> Iterator[None]:
    with mock.patch(
        "local_voice_harness.browser_context.get_desktop",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def mock_rofi_repository_prompts() -> Iterator[None]:
    with (
        mock.patch(
            "local_voice_harness.integrations.herdr.repository.choose_repository",
            return_value=None,
        ),
        mock.patch(
            "local_voice_harness.integrations.herdr.repository.confirm_clone",
            return_value=False,
        ),
    ):
        yield
