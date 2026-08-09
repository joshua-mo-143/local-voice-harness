from __future__ import annotations

import fcntl
import json
import os
import re
import select
import signal
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .errors import HarnessError


@dataclass(frozen=True)
class RecorderPaths:
    state_dir: Path
    audio: Path
    process: Path
    log: Path
    shared_lock: Path | None = None

    @property
    def lock(self) -> Path:
        return self.shared_lock or self.state_dir / "recording.lock"

    @property
    def generations(self) -> Path:
        return self.state_dir / "recordings"


@dataclass(frozen=True)
class RecorderState:
    pid: int
    identity: str | None
    mode: str


def process_identity(pid: int) -> str | None:
    """Return the Linux process start tick, which remains stable for its lifetime."""

    try:
        suffix = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return suffix[19]
    except (IndexError, OSError):
        return None


@contextmanager
def recording_lock(state_dir: Path, lock_path: Path | None = None) -> Iterator[None]:
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = state_dir.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise HarnessError(f"runtime directory is not harness-owned: {state_dir}")
    state_dir.chmod(0o700)
    descriptor = os.open(
        lock_path or state_dir / "recording.lock", os.O_CREAT | os.O_RDWR, 0o600
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _locked(paths: RecorderPaths) -> Iterator[None]:
    with recording_lock(paths.state_dir, paths.lock):
        yield


def _read_state(paths: RecorderPaths) -> RecorderState | None:
    try:
        raw = paths.process.read_text()
        try:
            legacy_pid = int(raw)
        except ValueError:
            pass
        else:
            return RecorderState(legacy_pid, None, "manual")
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None
        pid = value.get("pid")
        identity = value.get("process_start")
        mode = value.get("mode", "manual")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(identity, str)
            or not identity
            or mode not in {"manual", "vad"}
        ):
            return None
        return RecorderState(pid, identity, mode)
    except (OSError, json.JSONDecodeError):
        return None


def _legacy_command_matches(pid: int, audio_path: Path) -> bool | None:
    try:
        arguments = [
            argument.decode(errors="surrogateescape")
            for argument in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if argument
        ]
    except OSError:
        return None
    return bool(
        arguments
        and Path(arguments[0]).name == "pw-record"
        and str(audio_path) in arguments[1:]
    )


def _write_state(
    paths: RecorderPaths, pid: int, identity: str, *, mode: str = "manual"
) -> None:
    if mode not in {"manual", "vad"}:
        raise ValueError(f"unsupported recorder mode: {mode}")
    temporary = paths.process.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"pid": pid, "process_start": identity, "mode": mode},
            separators=(",", ":"),
        )
        + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, paths.process)


def is_generation_path(paths: RecorderPaths, path: Path) -> bool:
    pattern = rf"{re.escape(paths.audio.stem)}-[0-9a-f]{{32}}\.wav"
    return (
        path.is_absolute()
        and path.parent == paths.generations
        and re.fullmatch(pattern, path.name) is not None
    )


def _ensure_owned_directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise HarnessError(f"runtime directory is not harness-owned: {path}")
    path.chmod(0o700)


def _handoff_audio_locked(paths: RecorderPaths) -> Path:
    _ensure_owned_directory(paths.generations)
    generation = paths.generations / f"{paths.audio.stem}-{uuid.uuid4().hex}.wav"
    try:
        paths.audio.rename(generation)
    except FileNotFoundError as exc:
        raise HarnessError("recorded audio is missing") from exc
    try:
        metadata = generation.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 44
        ):
            raise HarnessError("no microphone audio was captured")
        generation.chmod(0o600)
        return generation
    except Exception:
        generation.unlink(missing_ok=True)
        raise


def handoff_recording(paths: RecorderPaths, *, active_message: str) -> Path:
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is not None:
            os.close(owned[1])
            raise HarnessError(active_message)
        return _handoff_audio_locked(paths)


def retry_generation(
    paths: RecorderPaths,
    generation: Path,
    *,
    active_message: str,
    conflicts: tuple[RecorderPaths, ...] = (),
) -> Path:
    with _locked(paths):
        for candidate in (paths, *conflicts):
            owned = _owned_pidfd(candidate)
            if owned is not None:
                os.close(owned[1])
                raise HarnessError(active_message)
        if not is_generation_path(paths, generation):
            raise HarnessError("retry path is not a harness recording generation")
        try:
            directory = paths.generations.lstat()
            metadata = generation.lstat()
        except OSError as exc:
            raise HarnessError("pending recording generation does not exist") from exc
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != os.getuid()
            or directory.st_mode & 0o077
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 44
        ):
            raise HarnessError("pending recording generation is not safe to retry")
        return generation


def write_audio_generation(
    paths: RecorderPaths,
    writer: Callable[[Path], None],
    *,
    conflicts: tuple[RecorderPaths, ...] = (),
) -> Path:
    with _locked(paths):
        for candidate in (paths, *conflicts):
            owned = _owned_pidfd(candidate)
            if owned is not None:
                os.close(owned[1])
                raise HarnessError("another recording is active")
        paths.audio.unlink(missing_ok=True)
        try:
            writer(paths.audio)
            return _handoff_audio_locked(paths)
        except Exception:
            paths.audio.unlink(missing_ok=True)
            raise


def _owned_pidfd(paths: RecorderPaths) -> tuple[int, int] | None:
    state = _read_state(paths)
    if state is None:
        if paths.process.exists():
            raise HarnessError(
                "recorder ownership state is invalid; refusing to replace it"
            )
        return None
    pid, expected_identity = state.pid, state.identity
    if pid <= 0:
        raise HarnessError(
            "recorder ownership state is invalid; refusing to replace it"
        )
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        paths.process.unlink(missing_ok=True)
        return None
    if expected_identity is None:
        if _pidfd_exited(pidfd, 0):
            os.close(pidfd)
            paths.process.unlink(missing_ok=True)
            return None
        identity = process_identity(pid)
        command_matches = _legacy_command_matches(pid, paths.audio)
        if identity is None or command_matches is None:
            os.close(pidfd)
            raise HarnessError(
                "cannot verify legacy recorder ownership; refusing to replace it"
            )
        if not command_matches:
            os.close(pidfd)
            paths.process.unlink(missing_ok=True)
            return None
        if _pidfd_exited(pidfd, 0):
            os.close(pidfd)
            paths.process.unlink(missing_ok=True)
            return None
        _write_state(paths, pid, identity, mode=state.mode)
        return pid, pidfd
    identity = process_identity(pid)
    if identity is None:
        os.close(pidfd)
        raise HarnessError("cannot verify recorder ownership; refusing to replace it")
    if identity != expected_identity:
        os.close(pidfd)
        paths.process.unlink(missing_ok=True)
        return None
    return pid, pidfd


def _pidfd_send(pidfd: int, sig: signal.Signals) -> None:
    signal.pidfd_send_signal(pidfd, sig)


def _pidfd_exited(pidfd: int, timeout: float) -> bool:
    readable, _, _ = select.select([pidfd], [], [], timeout)
    return bool(readable)


def recording_active(paths: RecorderPaths) -> bool:
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is None:
            return False
        os.close(owned[1])
        return True


def recording_mode(paths: RecorderPaths) -> str | None:
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is None:
            return None
        os.close(owned[1])
        state = _read_state(paths)
        if state is None:
            raise HarnessError("recorder ownership state became invalid")
        return state.mode


def request_vad_stop(paths: RecorderPaths) -> bool:
    """Ask an active VAD owner to stop; return false when the listener is idle."""

    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is None:
            return False
        _pid, pidfd = owned
        try:
            state = _read_state(paths)
            if state is None:
                raise HarnessError("recorder ownership state became invalid")
            if state.mode != "vad":
                raise HarnessError("manual dictation is already active")
            _pidfd_send(pidfd, signal.SIGTERM)
            return True
        finally:
            os.close(pidfd)


def claim_current_process(
    paths: RecorderPaths,
    *,
    mode: str,
    conflicts: tuple[RecorderPaths, ...] = (),
) -> None:
    if any(candidate.lock != paths.lock for candidate in conflicts):
        raise ValueError("conflicting recorders do not share an ownership lock")
    with _locked(paths):
        for candidate in (paths, *conflicts):
            owned = _owned_pidfd(candidate)
            if owned is not None:
                os.close(owned[1])
                raise HarnessError("another recording mode is already active")
        identity = process_identity(os.getpid())
        if identity is None:
            raise HarnessError("could not establish recorder process identity")
        paths.audio.unlink(missing_ok=True)
        _write_state(paths, os.getpid(), identity, mode=mode)


def complete_current_recording(
    paths: RecorderPaths,
    *,
    mode: str,
    cancelled: Callable[[], bool] | None = None,
    retain_owner: bool = False,
) -> Path:
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is None:
            raise HarnessError("recording ownership was lost before completion")
        pid, pidfd = owned
        os.close(pidfd)
        state = _read_state(paths)
        if pid != os.getpid() or state is None or state.mode != mode:
            raise HarnessError("recording is owned by another process")
        if cancelled is not None and cancelled():
            paths.process.unlink(missing_ok=True)
            paths.audio.unlink(missing_ok=True)
            raise HarnessError("recording was cancelled before completion")
        if not retain_owner:
            paths.process.unlink(missing_ok=True)
        return _handoff_audio_locked(paths)


def abandon_current_recording(paths: RecorderPaths, *, mode: str) -> None:
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is not None:
            pid, pidfd = owned
            os.close(pidfd)
            state = _read_state(paths)
            if pid == os.getpid() and state is not None and state.mode == mode:
                paths.process.unlink(missing_ok=True)
        paths.audio.unlink(missing_ok=True)


def any_recording_active(paths: tuple[RecorderPaths, ...]) -> bool:
    if not paths:
        return False
    lock = paths[0].lock
    if any(candidate.lock != lock for candidate in paths[1:]):
        raise ValueError("recorders do not share an ownership lock")
    with recording_lock(paths[0].state_dir, lock):
        for candidate in paths:
            owned = _owned_pidfd(candidate)
            if owned is not None:
                os.close(owned[1])
                return True
    return False


def start_recording(
    paths: RecorderPaths,
    *,
    source: str,
    ready: Callable[[], bool],
    prepare: Callable[[], None] | None = None,
    conflicts: tuple[RecorderPaths, ...] = (),
) -> None:
    if any(candidate.lock != paths.lock for candidate in conflicts):
        raise ValueError("conflicting recorders do not share an ownership lock")
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is not None:
            os.close(owned[1])
            print("already recording")
            return
        for candidate in conflicts:
            if candidate == paths:
                continue
            owned = _owned_pidfd(candidate)
            if owned is not None:
                os.close(owned[1])
                raise HarnessError("another recording mode is already active")
        if not ready():
            raise HarnessError("STT is stopped or still loading")
        if prepare is not None:
            prepare()
        paths.audio.unlink(missing_ok=True)
        command = ["pw-record"]
        if source:
            command.extend(("--target", source))
        command.extend(
            ("--channels=1", "--rate=16000", "--format=s16", str(paths.audio))
        )
        log_handle = paths.log.open("wb")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        time.sleep(0.2)
        if process.poll() is not None:
            detail = paths.log.read_text(errors="replace").strip()
            raise HarnessError(f"pw-record failed: {detail or process.returncode}")
        try:
            pidfd = os.pidfd_open(process.pid)
        except ProcessLookupError as exc:
            raise HarnessError("recorder exited during startup") from exc
        try:
            if process.poll() is not None:
                raise HarnessError("recorder exited during startup")
            identity = process_identity(process.pid)
            if identity is None:
                _pidfd_send(pidfd, signal.SIGTERM)
                raise HarnessError("could not establish recorder process identity")
            if _pidfd_exited(pidfd, 0):
                raise HarnessError("recorder exited during startup")
            try:
                _write_state(paths, process.pid, identity)
            except Exception:
                _pidfd_send(pidfd, signal.SIGTERM)
                raise
        finally:
            os.close(pidfd)


def stop_recording(paths: RecorderPaths, *, missing_message: str) -> Path:
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is None:
            raise HarnessError(missing_message)
        _pid, pidfd = owned
        try:
            try:
                _pidfd_send(pidfd, signal.SIGINT)
                if not _pidfd_exited(pidfd, 2):
                    _pidfd_send(pidfd, signal.SIGTERM)
                    if not _pidfd_exited(pidfd, 2):
                        raise HarnessError(
                            "recorder did not stop after SIGTERM; ownership was preserved"
                        )
            except ProcessLookupError as exc:
                if not _pidfd_exited(pidfd, 0):
                    raise HarnessError("recorder exit could not be confirmed") from exc
            paths.process.unlink(missing_ok=True)
            return _handoff_audio_locked(paths)
        finally:
            os.close(pidfd)


def cancel_recording(paths: RecorderPaths) -> None:
    with _locked(paths):
        owned = _owned_pidfd(paths)
        if owned is None:
            paths.audio.unlink(missing_ok=True)
            return
        pid, pidfd = owned
        state = _read_state(paths)
        if state is None:
            os.close(pidfd)
            raise HarnessError("recorder ownership state became invalid")
        try:
            _pidfd_send(pidfd, signal.SIGTERM)
        except ProcessLookupError as exc:
            if not _pidfd_exited(pidfd, 0):
                os.close(pidfd)
                raise HarnessError("recorder exit could not be confirmed") from exc

    try:
        if not _pidfd_exited(pidfd, 2):
            raise HarnessError(
                "recorder did not stop after SIGTERM; ownership was preserved"
            )
    finally:
        os.close(pidfd)

    with _locked(paths):
        current = _read_state(paths)
        if current is None and paths.process.exists():
            raise HarnessError(
                "recorder ownership changed during cancellation; state was preserved"
            )
        if current is not None and (
            current.pid,
            current.identity,
            current.mode,
        ) != (pid, state.identity, state.mode):
            return
        paths.process.unlink(missing_ok=True)
        paths.audio.unlink(missing_ok=True)
