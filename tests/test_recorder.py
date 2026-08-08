from __future__ import annotations

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from multiprocessing.connection import Connection
from pathlib import Path
from unittest import mock

from local_voice_harness import recorder
from local_voice_harness.errors import HarnessError


def _paths(root: Path) -> recorder.RecorderPaths:
    return recorder.RecorderPaths(
        root,
        root / "recording.wav",
        root / "recording.pid",
        root / "pw-record.log",
    )


def _cross_mode_paths(
    root: Path,
) -> tuple[recorder.RecorderPaths, recorder.RecorderPaths]:
    shared_lock = root / "recording.lock"
    manual = recorder.RecorderPaths(
        root / "manual",
        root / "manual" / "request.wav",
        root / "manual" / "recording.pid",
        root / "manual" / "pw-record.log",
        shared_lock,
    )
    dictation = recorder.RecorderPaths(
        root / "dictation",
        root / "dictation" / "recording.wav",
        root / "dictation" / "recording.pid",
        root / "dictation" / "pw-record.log",
        shared_lock,
    )
    return manual, dictation


def _write_state(paths: recorder.RecorderPaths, identity: str) -> None:
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.process.write_text(
        json.dumps({"pid": os.getpid(), "process_start": identity})
    )


def _take_recording_lock(state_dir: str, connection: Connection) -> None:
    with recorder.recording_lock(Path(state_dir)):
        connection.send("acquired")
    connection.close()


class RecorderTests(unittest.TestCase):
    def test_process_identity_includes_start_time_beyond_pid(self) -> None:
        identity = recorder.process_identity(os.getpid())

        self.assertIsNotNone(identity)
        self.assertTrue(identity and identity.isdigit())

    def test_pid_reuse_never_signals_mismatched_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            _write_state(paths, "different-start-time")
            paths.audio.write_bytes(b"private audio")

            with (
                mock.patch.object(
                    recorder, "process_identity", return_value="replacement-process"
                ),
                mock.patch.object(recorder, "_pidfd_send") as send,
            ):
                recorder.cancel_recording(paths)

            send.assert_not_called()
            self.assertFalse(paths.process.exists())
            self.assertFalse(paths.audio.exists())

    def test_stop_rejects_pid_reuse_without_signalling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            _write_state(paths, "old-process")

            with (
                mock.patch.object(
                    recorder, "process_identity", return_value="new-process"
                ),
                mock.patch.object(recorder, "_pidfd_send") as send,
                self.assertRaisesRegex(HarnessError, "not currently recording"),
            ):
                recorder.stop_recording(
                    paths, missing_message="not currently recording"
                )

            send.assert_not_called()

    def test_concurrent_starts_launch_exactly_one_recorder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            process = mock.Mock(pid=os.getpid(), returncode=None)
            process.poll.return_value = None
            barrier = threading.Barrier(3)
            errors: list[BaseException] = []

            def begin() -> None:
                barrier.wait()
                try:
                    recorder.start_recording(
                        paths,
                        source="microphone",
                        ready=lambda: True,
                    )
                except BaseException as exc:
                    errors.append(exc)

            with (
                mock.patch.object(
                    recorder.subprocess, "Popen", return_value=process
                ) as popen,
                mock.patch.object(recorder.time, "sleep"),
            ):
                threads = [threading.Thread(target=begin) for _ in range(2)]
                for thread in threads:
                    thread.start()
                barrier.wait()
                for thread in threads:
                    thread.join(2)

            self.assertFalse(errors)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            popen.assert_called_once()
            state = json.loads(paths.process.read_text())
            self.assertEqual(state["pid"], os.getpid())
            self.assertEqual(
                state["process_start"], recorder.process_identity(os.getpid())
            )

    def test_cross_mode_live_owner_blocks_start_in_both_orderings(self) -> None:
        identity = recorder.process_identity(os.getpid())
        assert identity is not None
        for owner_index in (0, 1):
            with self.subTest(owner_index=owner_index):
                with tempfile.TemporaryDirectory() as temporary:
                    modes = _cross_mode_paths(Path(temporary))
                    owner = modes[owner_index]
                    contender = modes[1 - owner_index]
                    _write_state(owner, identity)
                    owner_audio = b"owner audio"
                    contender_audio = b"contender audio"
                    owner.audio.write_bytes(owner_audio)
                    contender.state_dir.mkdir(parents=True)
                    contender.audio.write_bytes(contender_audio)

                    with (
                        mock.patch.object(recorder.subprocess, "Popen") as popen,
                        self.assertRaisesRegex(HarnessError, "another recording mode"),
                    ):
                        recorder.start_recording(
                            contender,
                            source="",
                            ready=lambda: True,
                            conflicts=(owner,),
                        )

                    popen.assert_not_called()
                    self.assertEqual(owner.audio.read_bytes(), owner_audio)
                    self.assertEqual(contender.audio.read_bytes(), contender_audio)
                    self.assertTrue(owner.process.exists())
                    self.assertFalse(contender.process.exists())

    def test_stale_cross_mode_state_is_cleaned_before_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manual, dictation = _cross_mode_paths(Path(temporary))
            manual.state_dir.mkdir(parents=True)
            manual.process.write_text(
                json.dumps({"pid": 999_999_999, "process_start": "stale"})
            )
            manual_audio = b"completed manual audio"
            manual.audio.write_bytes(manual_audio)
            process = mock.Mock(pid=os.getpid(), returncode=None)
            process.poll.return_value = None

            with (
                mock.patch.object(
                    recorder.subprocess, "Popen", return_value=process
                ) as popen,
                mock.patch.object(recorder.time, "sleep"),
            ):
                recorder.start_recording(
                    dictation,
                    source="",
                    ready=lambda: True,
                    conflicts=(manual,),
                )

            popen.assert_called_once()
            self.assertFalse(manual.process.exists())
            self.assertEqual(manual.audio.read_bytes(), manual_audio)
            self.assertTrue(dictation.process.exists())

    def test_unverifiable_cross_mode_owner_blocks_without_touching_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manual, dictation = _cross_mode_paths(Path(temporary))
            manual.state_dir.mkdir(parents=True)
            manual.process.write_text(str(os.getpid()))
            manual.audio.write_bytes(b"manual audio")
            dictation.state_dir.mkdir(parents=True)
            dictation.audio.write_bytes(b"dictation audio")

            with (
                mock.patch.object(recorder, "_pidfd_exited", return_value=False),
                mock.patch.object(
                    recorder, "_legacy_command_matches", return_value=None
                ),
                mock.patch.object(recorder.subprocess, "Popen") as popen,
                self.assertRaisesRegex(HarnessError, "cannot verify legacy"),
            ):
                recorder.start_recording(
                    dictation,
                    source="",
                    ready=lambda: True,
                    conflicts=(manual,),
                )

            popen.assert_not_called()
            self.assertEqual(manual.process.read_text(), str(os.getpid()))
            self.assertEqual(manual.audio.read_bytes(), b"manual audio")
            self.assertEqual(dictation.audio.read_bytes(), b"dictation audio")

    def test_simultaneous_cross_mode_starts_launch_exactly_one_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manual, dictation = _cross_mode_paths(Path(temporary))
            for mode in (manual, dictation):
                mode.state_dir.mkdir(parents=True)
                mode.audio.write_bytes(b"untouched contender audio")
            process = mock.Mock(pid=os.getpid(), returncode=None)
            process.poll.return_value = None
            barrier = threading.Barrier(3)
            errors: list[HarnessError] = []

            def begin(
                paths: recorder.RecorderPaths,
                conflict: recorder.RecorderPaths,
            ) -> None:
                barrier.wait()
                try:
                    recorder.start_recording(
                        paths,
                        source="",
                        ready=lambda: True,
                        conflicts=(conflict,),
                    )
                except HarnessError as exc:
                    errors.append(exc)

            with (
                mock.patch.object(
                    recorder.subprocess, "Popen", return_value=process
                ) as popen,
                mock.patch.object(recorder.time, "sleep"),
            ):
                threads = [
                    threading.Thread(target=begin, args=(manual, dictation)),
                    threading.Thread(target=begin, args=(dictation, manual)),
                ]
                for thread in threads:
                    thread.start()
                barrier.wait()
                for thread in threads:
                    thread.join(2)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            popen.assert_called_once()
            self.assertEqual(len(errors), 1)
            self.assertRegex(str(errors[0]), "another recording mode")
            self.assertEqual(
                sum(mode.process.exists() for mode in (manual, dictation)), 1
            )
            self.assertEqual(
                sum(mode.audio.exists() for mode in (manual, dictation)), 1
            )

    def test_real_legacy_process_blocks_other_mode_and_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manual, dictation = _cross_mode_paths(Path(temporary))
            manual.state_dir.mkdir(parents=True)
            command = 'exec -a pw-record "$1" -c \'import time; time.sleep(30)\' "$2"'
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    command,
                    "legacy-recorder",
                    sys.executable,
                    str(manual.audio),
                ]
            )
            try:
                deadline = time.monotonic() + 2
                while (
                    recorder._legacy_command_matches(process.pid, manual.audio)
                    is not True
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                manual.process.write_text(str(process.pid))

                with (
                    mock.patch.object(recorder.subprocess, "Popen") as popen,
                    self.assertRaisesRegex(HarnessError, "another recording mode"),
                ):
                    recorder.start_recording(
                        dictation,
                        source="",
                        ready=lambda: True,
                        conflicts=(manual,),
                    )

                popen.assert_not_called()
                migrated = json.loads(manual.process.read_text())
                self.assertEqual(migrated["pid"], process.pid)
                self.assertEqual(
                    migrated["process_start"],
                    recorder.process_identity(process.pid),
                )
            finally:
                process.terminate()
                process.wait(timeout=2)

    def test_reused_legacy_pid_state_is_removed_without_signalling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            paths.process.write_text(str(os.getpid()))

            with mock.patch.object(recorder, "_pidfd_send") as send:
                recorder.cancel_recording(paths)

            send.assert_not_called()
            self.assertFalse(paths.process.exists())

    def test_active_legacy_pw_record_state_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            command = 'exec -a pw-record "$1" -c \'import time; time.sleep(30)\' "$2"'
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    command,
                    "legacy-recorder",
                    sys.executable,
                    str(paths.audio),
                ]
            )
            try:
                deadline = time.monotonic() + 2
                while (
                    recorder._legacy_command_matches(process.pid, paths.audio)
                    is not True
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertIs(
                    recorder._legacy_command_matches(process.pid, paths.audio), True
                )
                paths.process.write_text(str(process.pid))

                self.assertTrue(recorder.recording_active(paths))

                state = json.loads(paths.process.read_text())
                self.assertEqual(state["pid"], process.pid)
                self.assertEqual(
                    state["process_start"], recorder.process_identity(process.pid)
                )
            finally:
                process.terminate()
                process.wait(timeout=2)

    def test_unverifiable_active_legacy_state_blocks_and_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            paths.process.write_text(str(os.getpid()))

            with (
                mock.patch.object(recorder, "_pidfd_exited", return_value=False),
                mock.patch.object(
                    recorder, "_legacy_command_matches", return_value=None
                ),
                self.assertRaisesRegex(HarnessError, "cannot verify legacy"),
            ):
                recorder.recording_active(paths)

            self.assertEqual(paths.process.read_text(), str(os.getpid()))

    def test_stop_escalates_and_removes_state_only_after_confirmed_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            identity = recorder.process_identity(os.getpid())
            assert identity is not None
            _write_state(paths, identity)
            paths.audio.write_bytes(b"RIFF" + b"\0" * 64)

            with (
                mock.patch.object(recorder, "_pidfd_send") as send,
                mock.patch.object(
                    recorder, "_pidfd_exited", side_effect=[False, True]
                ) as exited,
            ):
                generation = recorder.stop_recording(
                    paths, missing_message="not currently recording"
                )

            self.assertEqual(
                [call.args[1] for call in send.call_args_list],
                [signal.SIGINT, signal.SIGTERM],
            )
            self.assertEqual(exited.call_count, 2)
            self.assertFalse(paths.process.exists())
            self.assertTrue(recorder.is_generation_path(paths, generation))
            self.assertTrue(generation.exists())

    def test_stop_preserves_state_when_sigterm_does_not_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            identity = recorder.process_identity(os.getpid())
            assert identity is not None
            _write_state(paths, identity)

            with (
                mock.patch.object(recorder, "_pidfd_send"),
                mock.patch.object(
                    recorder, "_pidfd_exited", side_effect=[False, False]
                ),
                self.assertRaisesRegex(HarnessError, "ownership was preserved"),
            ):
                recorder.stop_recording(
                    paths, missing_message="not currently recording"
                )

            self.assertTrue(paths.process.exists())

    def test_cancel_preserves_state_and_audio_when_process_does_not_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            identity = recorder.process_identity(os.getpid())
            assert identity is not None
            _write_state(paths, identity)
            paths.audio.write_bytes(b"private audio")

            with (
                mock.patch.object(recorder, "_pidfd_send"),
                mock.patch.object(recorder, "_pidfd_exited", return_value=False),
                self.assertRaisesRegex(HarnessError, "ownership was preserved"),
            ):
                recorder.cancel_recording(paths)

            self.assertTrue(paths.process.exists())
            self.assertTrue(paths.audio.exists())

    def test_completed_generation_survives_new_recording_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            identity = recorder.process_identity(os.getpid())
            assert identity is not None
            _write_state(paths, identity)
            contents = b"RIFF-completed" + b"\0" * 64
            paths.audio.write_bytes(contents)

            with (
                mock.patch.object(recorder, "_pidfd_send"),
                mock.patch.object(recorder, "_pidfd_exited", return_value=True),
            ):
                generation = recorder.stop_recording(
                    paths, missing_message="not currently recording"
                )

            process = mock.Mock(pid=os.getpid(), returncode=None)
            process.poll.return_value = None
            paths.audio.write_bytes(b"stale writable data")
            with (
                mock.patch.object(recorder.subprocess, "Popen", return_value=process),
                mock.patch.object(recorder.time, "sleep"),
            ):
                recorder.start_recording(paths, source="", ready=lambda: True)

            self.assertEqual(generation.read_bytes(), contents)
            self.assertFalse(paths.audio.exists())

    def test_direct_handoff_rejects_active_recording_non_destructively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            identity = recorder.process_identity(os.getpid())
            assert identity is not None
            _write_state(paths, identity)
            contents = b"RIFF-active" + b"\0" * 64
            paths.audio.write_bytes(contents)

            with self.assertRaisesRegex(HarnessError, "still active"):
                recorder.handoff_recording(
                    paths, active_message="recording is still active"
                )

            self.assertEqual(paths.audio.read_bytes(), contents)
            self.assertTrue(paths.process.exists())

    def test_retry_accepts_only_strict_pending_generation_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))

            def write_audio(audio: Path) -> None:
                audio.write_bytes(b"RIFF" + b"\0" * 64)

            generation = recorder.write_audio_generation(
                paths,
                write_audio,
            )

            self.assertEqual(
                recorder.retry_generation(
                    paths,
                    generation,
                    active_message="recording is active",
                ),
                generation,
            )
            with self.assertRaisesRegex(HarnessError, "not a harness"):
                recorder.retry_generation(
                    paths,
                    Path(temporary) / "arbitrary.wav",
                    active_message="recording is active",
                )

    def test_recording_lock_serializes_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            receive, send = context.Pipe(duplex=False)
            child = context.Process(
                target=_take_recording_lock,
                args=(temporary, send),
            )
            with recorder.recording_lock(Path(temporary)):
                child.start()
                send.close()
                self.assertFalse(receive.poll(0.2))

            self.assertTrue(receive.poll(2))
            self.assertEqual(receive.recv(), "acquired")
            child.join(2)
            self.assertFalse(child.is_alive())
            self.assertEqual(child.exitcode, 0)
            receive.close()


if __name__ == "__main__":
    unittest.main()
