from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .process import run_command

SCP_REMOTE = re.compile(
    r"^(?P<user>[A-Za-z0-9._-]+)@(?P<host>[A-Za-z0-9.-]+):(?P<path>[^:\s]+)$"
)

Checkpoint = Callable[[], None]
FinalizeCheckout = Callable[[Path], None]


class LocalGitError(RuntimeError):
    pass


class LocalGitOperationAmbiguous(LocalGitError):
    pass


class LocalGitRefChanged(LocalGitError):
    """A fetched moving ref no longer matches its resolved commit."""


def remote_identity(remote: str) -> str | None:
    """Return a protocol-independent host/path identity for a Git remote."""
    value = remote.strip().removesuffix(".git")
    match = SCP_REMOTE.fullmatch(value)
    if match is not None:
        host = match.group("host")
        path = match.group("path")
    else:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return None
        if parsed.scheme not in {"http", "https", "ssh"} or parsed.hostname is None:
            return None
        host = parsed.hostname
        path = parsed.path.strip("/").removeprefix("git/")
    normalized_path = path.strip("/").removesuffix(".git")
    if not normalized_path:
        return None
    return f"{host.casefold()}/{normalized_path.casefold()}"


@dataclass(frozen=True, slots=True)
class ExpectedRemote:
    url: str
    identity: str

    @classmethod
    def from_url(cls, url: str) -> ExpectedRemote:
        identity = remote_identity(url)
        if identity is None:
            raise LocalGitError("expected repository remote URL is invalid")
        return cls(url=url, identity=identity)


class LocalGitRepository:
    """Contain, verify, and atomically materialize local Git repositories."""

    def __init__(
        self,
        *,
        clone_root: Path,
        allowed_root: Path,
        git_executable: str = "git",
        lock_name: str = ".voice-harness-git.lock",
    ) -> None:
        self.clone_root = clone_root.expanduser().resolve()
        self.allowed_root = allowed_root.expanduser().resolve()
        self.git_executable = git_executable
        self.lock_name = lock_name

    def _run(
        self,
        command: list[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            process = run_command(command, timeout=timeout)
        except OSError as exc:
            raise LocalGitError(f"Git command failed: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise LocalGitOperationAmbiguous(
                "Git command timed out; its outcome is ambiguous"
            ) from exc
        if check and process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            raise LocalGitError(
                detail or f"command exited with status {process.returncode}"
            )
        return process

    def git(
        self,
        checkout: Path,
        *arguments: str,
        timeout: float = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self._run(
            [self.git_executable, "-C", str(checkout), *arguments],
            timeout=timeout,
            check=check,
        )

    def verify_checkout(
        self,
        checkout: Path,
        expected: ExpectedRemote | None = None,
        *,
        expected_label: str | None = None,
    ) -> None:
        if not (checkout / ".git").exists():
            raise LocalGitError(f"{checkout} exists but is not a Git repository")
        if expected is None:
            return
        origin = self.git(checkout, "remote", "get-url", "origin").stdout
        actual = remote_identity(origin)
        if actual is None or actual != expected.identity:
            label = expected_label or expected.identity
            raise LocalGitError(f"{checkout} exists but its origin is not {label}")

    def find_checkout(
        self,
        candidates: list[Path],
        expected: ExpectedRemote,
        *,
        expected_label: str | None = None,
    ) -> Path | None:
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                self.verify_checkout(
                    resolved,
                    expected,
                    expected_label=expected_label,
                )
            except LocalGitError:
                continue
            return resolved
        return None

    def ensure_remote(self, checkout: Path, name: str, url: str) -> None:
        current = self.git(
            checkout,
            "remote",
            "get-url",
            name,
            check=False,
        )
        action = "set-url" if current.returncode == 0 else "add"
        self.git(checkout, "remote", action, name, url)

    def current_branch(self, checkout: Path) -> str | None:
        branch = self.git(
            checkout,
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ).stdout.strip()
        return branch or None

    def checkout_remote_ref(
        self,
        checkout: Path,
        *,
        remote_url: str,
        remote_ref: str,
        branch: str,
        expected_oid: str,
        checkpoint: Checkpoint | None = None,
    ) -> str:
        """Fetch a validated remote ref and check out its expected commit."""

        checkout = checkout.resolve()
        try:
            checkout.relative_to(self.allowed_root)
        except ValueError as exc:
            raise LocalGitError(
                "Git checkout escapes the allowed project root"
            ) from exc
        expected_remote = ExpectedRemote.from_url(remote_url)
        if not remote_ref.startswith("refs/") or any(
            character.isspace() for character in remote_ref
        ):
            raise LocalGitError("pull-request remote ref is invalid")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_oid):
            raise LocalGitError("pull-request head OID is invalid")
        self.verify_checkout(checkout, expected_remote)
        self.git(checkout, "check-ref-format", "--branch", branch)
        if checkpoint is not None:
            checkpoint()
        self.git(
            checkout,
            "fetch",
            "--force",
            "--no-tags",
            "--",
            remote_url,
            remote_ref,
            timeout=180,
        )
        if checkpoint is not None:
            checkpoint()
        fetched_oid = self.git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
        if fetched_oid.casefold() != expected_oid.casefold():
            raise LocalGitRefChanged(
                "fetched pull-request head does not match GitHub metadata"
            )
        self.git(checkout, "checkout", "-B", branch, "FETCH_HEAD", timeout=180)
        if checkpoint is not None:
            checkpoint()
        checked_out_oid = self.git(checkout, "rev-parse", "HEAD").stdout.strip()
        if checked_out_oid.casefold() != expected_oid.casefold():
            raise LocalGitError("checked-out pull-request head changed unexpectedly")
        return self.current_branch(checkout) or branch

    def _validate_root(self) -> None:
        try:
            self.clone_root.relative_to(self.allowed_root)
        except ValueError as exc:
            raise LocalGitError(
                "repository clone root must be inside the allowed project root"
            ) from exc

    @contextmanager
    def _provisioning_lock(
        self, checkpoint: Checkpoint | None = None
    ) -> Iterator[None]:
        try:
            self._validate_root()
            if checkpoint is not None:
                checkpoint()
            self.clone_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if checkpoint is not None:
                checkpoint()
            lock_path = self.clone_root / self.lock_name
            if checkpoint is not None:
                checkpoint()
            with lock_path.open("a+b") as lock:
                while True:
                    if checkpoint is not None:
                        checkpoint()
                    try:
                        fcntl.flock(
                            lock.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except BlockingIOError:
                        time.sleep(0.1)
                        continue
                    break
                try:
                    if checkpoint is not None:
                        checkpoint()
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except LocalGitError:
            raise
        except OSError as exc:
            raise LocalGitError(f"Could not materialize repository: {exc}") from exc

    @staticmethod
    def _relative_destination(value: Path) -> Path:
        if value.is_absolute() or not value.parts or ".." in value.parts:
            raise LocalGitError("repository clone destination escapes its root")
        return value

    @staticmethod
    def _cleanup_temporary(temporary: Path) -> None:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)

    def observe_materialized(
        self,
        relative_destination: Path,
        *,
        expected: ExpectedRemote,
        expected_label: str | None = None,
    ) -> Path | None:
        """Observe an expected clone without creating or repairing it."""

        relative_destination = self._relative_destination(relative_destination)
        self._validate_root()
        destination = self.clone_root / relative_destination
        resolved_parent = destination.parent.resolve()
        try:
            resolved_parent.relative_to(self.clone_root)
        except ValueError as exc:
            raise LocalGitError(
                "repository clone destination escapes its root"
            ) from exc
        if not destination.exists() and not destination.is_symlink():
            return None
        if destination.is_symlink():
            try:
                destination.resolve().relative_to(self.clone_root)
            except ValueError as exc:
                raise LocalGitError(
                    "repository clone destination escapes its root"
                ) from exc
        self.verify_checkout(
            destination,
            expected,
            expected_label=expected_label,
        )
        return destination.resolve()

    def materialize(
        self,
        relative_destination: Path,
        *,
        clone_url: str,
        clone_command: tuple[str, ...] | None = None,
        expected: ExpectedRemote | None = None,
        expected_label: str | None = None,
        finalize: FinalizeCheckout | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        relative_destination = self._relative_destination(relative_destination)
        destination = self.clone_root / relative_destination
        with self._provisioning_lock(checkpoint):
            if checkpoint is not None:
                checkpoint()
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if checkpoint is not None:
                checkpoint()
            resolved_parent = destination.parent.resolve()
            try:
                resolved_parent.relative_to(self.clone_root)
            except ValueError as exc:
                raise LocalGitError(
                    "repository clone destination escapes its root"
                ) from exc
            if destination.exists():
                if destination.is_symlink():
                    try:
                        destination.resolve().relative_to(self.clone_root)
                    except ValueError as exc:
                        raise LocalGitError(
                            "repository clone destination escapes its root"
                        ) from exc
                self.verify_checkout(
                    destination,
                    expected,
                    expected_label=expected_label,
                )
                if finalize is not None:
                    if checkpoint is not None:
                        checkpoint()
                    finalize(destination)
                    if checkpoint is not None:
                        checkpoint()
                return destination.resolve()
            temporary = destination.parent / (
                f".{destination.name}.clone-{uuid.uuid4().hex}"
            )
            try:
                if checkpoint is not None:
                    checkpoint()
                command = (
                    [*clone_command, str(temporary)]
                    if clone_command is not None
                    else [
                        self.git_executable,
                        "clone",
                        "--",
                        clone_url,
                        str(temporary),
                    ]
                )
                self._run(
                    command,
                    timeout=300,
                )
                if checkpoint is not None:
                    checkpoint()
                self.verify_checkout(
                    temporary,
                    expected,
                    expected_label=expected_label,
                )
                if finalize is not None:
                    if checkpoint is not None:
                        checkpoint()
                    finalize(temporary)
                    if checkpoint is not None:
                        checkpoint()
                if checkpoint is not None:
                    checkpoint()
                os.replace(temporary, destination)
                if checkpoint is not None:
                    checkpoint()
            finally:
                self._cleanup_temporary(temporary)
            return destination.resolve()

    def _require_allowed_checkout(self, checkout: Path) -> Path:
        resolved = checkout.expanduser().resolve()
        try:
            resolved.relative_to(self.allowed_root)
        except ValueError as exc:
            raise LocalGitError(
                "Git checkout escapes the allowed project root"
            ) from exc
        self.verify_checkout(resolved)
        return resolved

    def has_unpublished_changes(self, checkout: Path) -> bool:
        checkout = self._require_allowed_checkout(checkout)
        status = self.git(checkout, "status", "--porcelain").stdout
        return bool(status.strip())

    def commit_unpublished_changes(
        self,
        checkout: Path,
        subject: str,
        *,
        confirmed: bool,
    ) -> str | None:
        if not confirmed:
            raise LocalGitError("Git commit requires explicit confirmation")
        checkout = self._require_allowed_checkout(checkout)
        message = " ".join(subject.split())
        if not message:
            raise LocalGitError("Git commit requires a non-empty subject")
        if len(message) > 72:
            raise LocalGitError("Git commit subject is too long")
        if not self.has_unpublished_changes(checkout):
            return None
        self.git(checkout, "add", "-A")
        self.git(checkout, "commit", "-m", message)
        return self.git(checkout, "rev-parse", "HEAD").stdout.strip() or None

    def push_current_branch(self, checkout: Path, *, confirmed: bool) -> str:
        if not confirmed:
            raise LocalGitError("Git push requires explicit confirmation")
        checkout = self._require_allowed_checkout(checkout)
        branch = self.current_branch(checkout)
        if not branch or branch == "HEAD":
            raise LocalGitError("Git checkout is not on a named branch")
        self.git(checkout, "check-ref-format", "--branch", branch)
        self.git(checkout, "push", "-u", "origin", branch, timeout=180)
        return branch
