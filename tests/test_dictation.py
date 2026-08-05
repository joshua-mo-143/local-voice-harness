from __future__ import annotations

import io
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest import mock

from local_voice_harness import cli, config, dictation
from local_voice_harness.errors import HarnessError


class DictationTests(unittest.TestCase):
    def test_typed_dictation_uses_separate_audio_file(self) -> None:
        self.assertNotEqual(dictation.WAV_PATH, config.WAV_PATH)

    def test_cli_accepts_dictation_toggle(self) -> None:
        arguments = cli.parser().parse_args(["dictate", "toggle"])
        self.assertEqual(arguments.command, "dictate")
        self.assertEqual(arguments.dictation_command, "toggle")

    def test_begin_starts_dictation_recording(self) -> None:
        with mock.patch.object(dictation, "start_recording") as start:
            dictation.run("begin")
        start.assert_called_once_with()

    def test_toggle_starts_when_idle(self) -> None:
        with (
            mock.patch.object(dictation, "recording_active", return_value=False),
            mock.patch.object(dictation, "start_recording") as start,
        ):
            dictation.run("toggle")
        start.assert_called_once_with()

    def test_toggle_stops_transcribes_and_types_when_recording(self) -> None:
        with (
            mock.patch.object(dictation, "recording_active", return_value=True),
            mock.patch.object(dictation, "stop_recording") as stop,
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            mock.patch.object(dictation, "transcribe_and_type") as transcribe,
        ):
            dictation.run("toggle")
        stop.assert_called_once_with()
        transcribe.assert_called_once_with()

    def test_transcribes_typed_dictation_audio(self) -> None:
        with (
            mock.patch.object(
                dictation, "transcribe", return_value="hello"
            ) as transcribe,
            mock.patch.object(dictation, "inject") as inject,
        ):
            dictation.transcribe_and_type()
        transcribe.assert_called_once_with(dictation.WAV_PATH)
        inject.assert_called_once_with("hello")

    def test_runelite_dictation_is_rejected(self) -> None:
        with (
            mock.patch.object(
                dictation,
                "_active_window_class",
                return_value="net-runelite-client-runelite",
            ),
            self.assertRaisesRegex(HarnessError, "disabled for RuneLite"),
        ):
            dictation.inject("buying fish")

    def test_runelite_recording_is_rejected_before_capture(self) -> None:
        with (
            mock.patch.object(
                dictation,
                "_active_window_class",
                return_value="net-runelite-client-runelite",
            ),
            mock.patch.object(dictation, "socket_ready") as socket_ready,
            self.assertRaisesRegex(HarnessError, "disabled for RuneLite"),
        ):
            dictation.start_recording()
        socket_ready.assert_not_called()

    def test_stdout_injection(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "stdout"}),
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            redirect_stdout(output),
        ):
            dictation.inject("hello world")
        self.assertEqual(output.getvalue(), "hello world\n")

    def test_invalid_injection_mode_is_rejected(self) -> None:
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "invalid"}),
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            self.assertRaises(HarnessError),
        ):
            dictation.inject("hello")

    def test_xclip_does_not_capture_background_owner_output(self) -> None:
        command = ["xclip", "-selection", "clipboard"]
        completed = subprocess.CompletedProcess(command, 0)
        with (
            mock.patch.object(dictation.shutil, "which", return_value="/usr/bin/xclip"),
            mock.patch.object(
                dictation.subprocess, "run", return_value=completed
            ) as run,
        ):
            dictation._copy_to_clipboard("hello")
        run.assert_called_once_with(
            command,
            input="hello",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )

    def test_xclip_failure_is_reported(self) -> None:
        completed = subprocess.CompletedProcess([], 1)
        with (
            mock.patch.object(dictation.shutil, "which", return_value="/usr/bin/xclip"),
            mock.patch.object(dictation.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(HarnessError, "could not copy"),
        ):
            dictation._copy_to_clipboard("hello")

    def test_auto_injection_pastes_after_copying_to_xclip(self) -> None:
        operations: list[tuple[str, object]] = []
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "auto"}),
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(
                dictation, "_active_window_class", return_value="firefox"
            ),
            mock.patch.object(dictation.shutil, "which", return_value="/usr/bin/xclip"),
            mock.patch.object(
                dictation,
                "_copy_to_clipboard",
                side_effect=lambda text: operations.append(("copy", text)),
            ),
            mock.patch.object(
                dictation,
                "_xdotool",
                side_effect=lambda *args: operations.append(("xdotool", args)),
            ),
            mock.patch.object(dictation.time, "sleep"),
        ):
            dictation.inject("hello")
        self.assertEqual(
            operations,
            [
                ("copy", "hello"),
                ("xdotool", ("key", "--clearmodifiers", "ctrl+v")),
            ],
        )

    def test_auto_injection_keeps_native_herdr_path(self) -> None:
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "auto"}),
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            mock.patch.object(dictation, "_send_to_herdr", return_value=True) as send,
            mock.patch.object(dictation, "_copy_to_clipboard") as copy,
            mock.patch.object(dictation, "_xdotool") as xdotool,
        ):
            dictation.inject("hello")
        send.assert_called_once_with("hello")
        copy.assert_not_called()
        xdotool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
