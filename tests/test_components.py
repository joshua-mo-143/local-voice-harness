from __future__ import annotations

import unittest
import urllib.error
from types import TracebackType
from unittest import mock

from local_voice_harness import components
from local_voice_harness.errors import HarnessError


class _HTTPResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _HTTPResponse:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None


class ComponentReadinessTests(unittest.TestCase):
    def test_llm_ready_requires_successful_health_response(self) -> None:
        with mock.patch.object(
            components.urllib.request, "urlopen", return_value=_HTTPResponse(200)
        ):
            self.assertTrue(components.llm_ready())
        with mock.patch.object(
            components.urllib.request, "urlopen", return_value=_HTTPResponse(503)
        ):
            self.assertFalse(components.llm_ready())
        with mock.patch.object(
            components.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            self.assertFalse(components.llm_ready())

    def test_start_waits_until_both_components_are_ready(self) -> None:
        with (
            mock.patch.object(components.subprocess, "run") as run,
            mock.patch.object(components, "llm_ready", side_effect=[False, True]),
            mock.patch.object(components, "socket_ready", return_value=True) as ready,
            mock.patch.object(
                components.time, "monotonic", side_effect=[10.0, 10.0, 10.25]
            ),
            mock.patch.object(components.time, "sleep") as sleep,
        ):
            components.start_components(timeout=1)

        run.assert_called_once_with(
            [
                "systemctl",
                "--user",
                "start",
                "voice-harness-llm.service",
                "voice-harness-tts.service",
            ],
            check=True,
        )
        ready.assert_called_once_with(components.TTS_SOCKET)
        sleep.assert_called_once_with(0.25)

    def test_start_reports_the_configured_timeout(self) -> None:
        with (
            mock.patch.object(components.subprocess, "run"),
            mock.patch.object(components.time, "monotonic", side_effect=[10.0, 10.5]),
            self.assertRaisesRegex(HarnessError, "within 0.25 seconds"),
        ):
            components.start_components(timeout=0.25)

    def test_stop_is_best_effort(self) -> None:
        with mock.patch.object(components.subprocess, "run") as run:
            components.stop_components()

        run.assert_called_once_with(
            [
                "systemctl",
                "--user",
                "stop",
                "voice-harness-llm.service",
                "voice-harness-tts.service",
            ],
            check=False,
            stdout=components.subprocess.DEVNULL,
            stderr=components.subprocess.DEVNULL,
        )


if __name__ == "__main__":
    unittest.main()
