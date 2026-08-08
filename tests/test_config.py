from __future__ import annotations

import unittest
from pathlib import Path

from local_voice_harness import config


class ConfigPathTests(unittest.TestCase):
    def test_job_state_is_durable_but_worker_logs_are_session_only(self) -> None:
        self.assertEqual(config.JOBS_DIR, config.DURABLE_STATE_DIR / "jobs")
        self.assertEqual(config.JOB_LOGS_DIR, config.STATE_DIR / "jobs")
        self.assertEqual(config.LEGACY_JOBS_DIR, config.STATE_DIR / "jobs")
        self.assertNotEqual(config.JOBS_DIR, config.JOB_LOGS_DIR)

    def test_xdg_state_home_uses_standard_fallback(self) -> None:
        home = Path("/home/example")

        self.assertEqual(
            config.xdg_state_home({"XDG_STATE_HOME": "/custom/state"}, home=home),
            Path("/custom/state"),
        )
        self.assertEqual(config.xdg_state_home({}, home=home), home / ".local/state")
        self.assertEqual(
            config.xdg_state_home({"XDG_STATE_HOME": "relative/state"}, home=home),
            home / ".local/state",
        )

    def test_systemd_state_directory_selects_absolute_voice_harness_entry(
        self,
    ) -> None:
        environment = {
            "STATE_DIRECTORY": (
                "/var/lib/other:/home/example/.local/state/voice-harness"
            ),
            "XDG_STATE_HOME": "/ignored",
        }

        self.assertEqual(
            config.systemd_state_directory(environment),
            Path("/home/example/.local/state/voice-harness"),
        )
        self.assertEqual(
            config.durable_state_dir(environment, home=Path("/home/example")),
            Path("/home/example/.local/state/voice-harness"),
        )

    def test_invalid_or_ambiguous_state_directory_uses_xdg_fallback(self) -> None:
        home = Path("/home/example")
        self.assertEqual(
            config.durable_state_dir(
                {
                    "STATE_DIRECTORY": "relative:/absolute/one:/absolute/two",
                    "XDG_STATE_HOME": "/custom/state",
                },
                home=home,
            ),
            Path("/custom/state/voice-harness"),
        )


if __name__ == "__main__":
    unittest.main()
