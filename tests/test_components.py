from __future__ import annotations

import tempfile
import threading
import time
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from unittest import mock

from local_voice_harness import components
from local_voice_harness.config import load_backend_settings
from local_voice_harness.credentials import CredentialError
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
    def setUp(self) -> None:
        self.settings = load_backend_settings(
            {}, path=Path("/nonexistent/backends.toml")
        )
        patcher = mock.patch.object(
            components,
            "default_user_config",
            return_value=mock.Mock(providers=self.settings),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_start_reports_the_component_that_is_not_ready(self) -> None:
        with (
            mock.patch.object(components.subprocess, "run"),
            mock.patch.object(components, "llm_ready", return_value=True),
            mock.patch.object(components, "socket_ready", return_value=False),
            mock.patch.object(
                components.time, "monotonic", side_effect=[10.0, 10.0, 10.5]
            ),
            mock.patch.object(components.time, "sleep"),
            self.assertRaisesRegex(
                HarnessError, "TTS backend did not become ready within 0.25 seconds"
            ),
        ):
            components.start_components(timeout=0.25)

    def test_start_checks_venice_credential_before_starting_services(self) -> None:
        settings = replace(
            self.settings,
            tts_provider="venice",
        )
        with (
            mock.patch.object(
                components,
                "default_user_config",
                return_value=mock.Mock(providers=settings),
            ),
            mock.patch.object(
                components,
                "get_venice_api_key",
                side_effect=CredentialError("Venice API key is not stored"),
            ),
            mock.patch.object(components.subprocess, "run") as run,
            self.assertRaisesRegex(CredentialError, "API key is not stored"),
        ):
            components.start_components()

        run.assert_not_called()

    def test_venice_starts_only_tts_service_and_uses_key_readiness(self) -> None:
        settings = replace(
            self.settings,
            llm_provider="venice",
        )
        with (
            mock.patch.object(
                components,
                "default_user_config",
                return_value=mock.Mock(providers=settings),
            ),
            mock.patch.object(
                components, "get_venice_api_key", return_value="secret"
            ) as get_key,
            mock.patch.object(components.subprocess, "run") as run,
            mock.patch.object(components, "socket_ready", return_value=True),
        ):
            self.assertTrue(components.llm_ready())
            components.start_components()

        get_key.assert_called_with()
        run.assert_called_once_with(
            ["systemctl", "--user", "start", "voice-harness-tts.service"],
            check=True,
        )

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

    def test_stop_waits_for_active_cross_process_usage(self) -> None:
        usage_started = threading.Event()
        release_usage = threading.Event()

        def use_components() -> None:
            with components.component_usage():
                usage_started.set()
                release_usage.wait(timeout=2)

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(components, "STATE_DIR", Path(temporary)),
            mock.patch.object(components.subprocess, "run") as run,
        ):
            usage = threading.Thread(target=use_components)
            usage.start()
            self.assertTrue(usage_started.wait(timeout=1))
            stopping = threading.Thread(target=components.stop_components)
            stopping.start()
            time.sleep(0.02)
            run.assert_not_called()
            release_usage.set()
            usage.join(timeout=2)
            stopping.join(timeout=2)

        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
