from __future__ import annotations

from collections.abc import Iterator
from unittest import mock

import pytest


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
