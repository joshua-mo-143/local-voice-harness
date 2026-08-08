from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from ..config import GITHUB_ROOT, REPOSITORY_ROOT

REPOSITORY = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)$"
)
Checkpoint = Callable[[], None]


class GitHubError(RuntimeError):
    pass


class GitHubOperationAmbiguous(GitHubError):
    pass


@dataclass(frozen=True)
class GitHubRepository:
    name_with_owner: str
    url: str
    is_private: bool
    default_branch: str
    parent: str | None = None


@dataclass(frozen=True)
class GitHubIssue:
    owner: str
    repository: str
    number: int

    @property
    def name_with_owner(self) -> str:
        return f"{self.owner}/{self.repository}"

    @property
    def reference(self) -> str:
        return f"{self.name_with_owner}#{self.number}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.name_with_owner}/issues/{self.number}"


@dataclass(frozen=True)
class ProvisionedRepository:
    source: GitHubRepository
    fork: GitHubRepository
    checkout: Path


@dataclass(frozen=True)
class ProvisionedPullRequest:
    source: GitHubRepository
    checkout: Path
    number: int
    branch: str | None = None


@dataclass(frozen=True)
class ProvisionedIssue:
    source: GitHubRepository
    checkout: Path
    issue: GitHubIssue


class GitHubClient:
    def __init__(
        self,
        *,
        gh_executable: str = "gh",
        git_executable: str = "git",
        clone_root: Path = GITHUB_ROOT,
        allowed_root: Path = REPOSITORY_ROOT,
    ) -> None:
        self.gh_executable = gh_executable
        self.git_executable = git_executable
        self.clone_root = clone_root.expanduser().resolve()
        self.allowed_root = allowed_root.expanduser().resolve()

    @staticmethod
    def validate_repository(value: str) -> str:
        candidate = value.strip().removesuffix(".git")
        match = REPOSITORY.fullmatch(candidate)
        if match is None or match.group("repo") in {".", ".."}:
            raise GitHubError("GitHub repository must be in owner/repository form")
        return f"{match.group('owner')}/{match.group('repo')}"

    def _run(
        self,
        command: list[str],
        *,
        timeout: float = 30,
        check: bool = True,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=str(cwd) if cwd is not None else None,
            )
        except OSError as exc:
            raise GitHubError(f"GitHub command failed: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitHubOperationAmbiguous(f"GitHub command timed out: {exc}") from exc
        if check and process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            raise GitHubError(
                detail or f"command exited with status {process.returncode}"
            )
        return process

    def _repo_view(self, repository: str, *, required: bool) -> GitHubRepository | None:
        process = self._run(
            [
                self.gh_executable,
                "repo",
                "view",
                repository,
                "--json",
                "nameWithOwner,url,isPrivate,defaultBranchRef,parent",
            ],
            timeout=15,
            check=False,
        )
        if process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            if required:
                raise GitHubError(detail or f"could not inspect {repository}")
            normalized = detail.casefold()
            if any(
                marker in normalized
                for marker in (
                    "could not resolve to a repository",
                    "repository not found",
                    "http 404",
                    "status 404",
                )
            ):
                return None
            raise GitHubError(
                detail or f"could not determine whether {repository} exists"
            )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub returned malformed repository metadata") from exc
        if not isinstance(value, dict):
            raise GitHubError("GitHub returned malformed repository metadata")
        name = self.validate_repository(str(value.get("nameWithOwner") or ""))
        default = value.get("defaultBranchRef")
        default_branch = (
            str(default.get("name") or "") if isinstance(default, dict) else ""
        )
        parent = value.get("parent")
        parent_name = (
            self.validate_repository(str(parent.get("nameWithOwner") or ""))
            if isinstance(parent, dict) and parent.get("nameWithOwner")
            else None
        )
        return GitHubRepository(
            name_with_owner=name,
            url=str(value.get("url") or f"https://github.com/{name}"),
            is_private=bool(value.get("isPrivate")),
            default_branch=default_branch,
            parent=parent_name,
        )

    def inspect_public_repository(self, repository: str) -> GitHubRepository:
        source = self._repo_view(self.validate_repository(repository), required=True)
        assert source is not None
        if source.is_private:
            raise GitHubError("the focused GitHub repository is private")
        return source

    def inspect_repository(self, repository: str) -> GitHubRepository:
        source = self._repo_view(self.validate_repository(repository), required=True)
        assert source is not None
        return source

    def issue_details(self, issue: GitHubIssue) -> dict[str, object]:
        if issue.number <= 0:
            raise GitHubError("GitHub issue number must be positive")
        repository = self.validate_repository(issue.name_with_owner)
        process = self._run(
            [
                self.gh_executable,
                "issue",
                "view",
                str(issue.number),
                "--repo",
                repository,
                "--json",
                "number,title,state,author,labels,body,comments,url",
            ],
            timeout=15,
        )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub returned malformed issue metadata") from exc
        if not isinstance(value, dict):
            raise GitHubError("GitHub returned malformed issue metadata")
        return value

    def authenticated_login(self) -> str:
        process = self._run(
            [self.gh_executable, "api", "user", "--jq", ".login"],
            timeout=15,
        )
        login = process.stdout.strip()
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", login):
            raise GitHubError("GitHub CLI did not return an authenticated user")
        return login

    def prepare_public_fork(self, repository: str) -> tuple[GitHubRepository, str, str]:
        source = self.inspect_public_repository(repository)
        login = self.authenticated_login()
        _owner, repository_name = source.name_with_owner.split("/", 1)
        target = (
            source.name_with_owner
            if source.name_with_owner.split("/", 1)[0].casefold() == login.casefold()
            else f"{login}/{repository_name}"
        )
        return source, login, target

    def reconcile_fork(
        self, source: GitHubRepository, target_name: str
    ) -> GitHubRepository | None:
        fork = self._repo_view(self.validate_repository(target_name), required=False)
        if fork is None:
            return None
        if source.name_with_owner.casefold() == fork.name_with_owner.casefold():
            return fork
        accepted_parents = {
            name.casefold() for name in (source.name_with_owner, source.parent) if name
        }
        if fork.parent is None or fork.parent.casefold() not in accepted_parents:
            raise GitHubError(
                f"{target_name} exists but is not a fork of {source.name_with_owner}"
            )
        return fork

    def ensure_fork(
        self,
        source: GitHubRepository,
        login: str,
        *,
        checkpoint: Checkpoint | None = None,
        before_submit: Callable[[], None] | None = None,
    ) -> GitHubRepository:
        source_owner, repository_name = source.name_with_owner.split("/", 1)
        if source_owner.casefold() == login.casefold():
            return source
        target_name = f"{login}/{repository_name}"
        accepted_parents = {
            name.casefold() for name in (source.name_with_owner, source.parent) if name
        }
        existing = self._repo_view(target_name, required=False)
        if existing is not None:
            if (
                existing.parent is None
                or existing.parent.casefold() not in accepted_parents
            ):
                raise GitHubError(
                    f"{target_name} already exists but is not a fork of "
                    f"{source.name_with_owner}"
                )
            return existing
        if checkpoint is not None:
            checkpoint()
        if before_submit is not None:
            before_submit()
        self._run(
            [
                self.gh_executable,
                "repo",
                "fork",
                source.name_with_owner,
                "--clone=false",
            ],
            timeout=120,
        )
        fork = self._repo_view(target_name, required=True)
        assert fork is not None
        if fork.parent is None or fork.parent.casefold() not in accepted_parents:
            raise GitHubError(
                f"GitHub did not create the expected fork at {target_name}"
            )
        return fork

    @staticmethod
    def _remote_repository(remote: str) -> str | None:
        value = remote.strip().removesuffix(".git")
        if value.startswith("git@github.com:"):
            return value.removeprefix("git@github.com:")
        parsed = urlsplit(value)
        if (
            parsed.scheme in {"http", "https", "ssh"}
            and parsed.hostname
            and parsed.hostname.casefold() == "github.com"
        ):
            return parsed.path.strip("/").removeprefix("git/")
        return None

    def _git(self, checkout: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self._run(
            [self.git_executable, "-C", str(checkout), *arguments],
            timeout=60,
        )

    def _verify_checkout(self, checkout: Path, expected: GitHubRepository) -> None:
        if not (checkout / ".git").exists():
            raise GitHubError(f"{checkout} exists but is not a Git repository")
        origin = self._git(checkout, "remote", "get-url", "origin").stdout
        actual = self._remote_repository(origin)
        if actual is None or actual.casefold() != expected.name_with_owner.casefold():
            raise GitHubError(
                f"{checkout} exists but its origin is not {expected.name_with_owner}"
            )

    def _ensure_upstream(
        self, checkout: Path, source: GitHubRepository, fork: GitHubRepository
    ) -> None:
        if source.name_with_owner.casefold() == fork.name_with_owner.casefold():
            return
        current = self._run(
            [
                self.git_executable,
                "-C",
                str(checkout),
                "remote",
                "get-url",
                "upstream",
            ],
            check=False,
        )
        action = "set-url" if current.returncode == 0 else "add"
        self._git(checkout, "remote", action, "upstream", source.url)

    @contextmanager
    def _provisioning_lock(
        self, checkpoint: Checkpoint | None = None
    ) -> Iterator[None]:
        try:
            self.clone_root.relative_to(self.allowed_root)
        except ValueError as exc:
            raise GitHubError(
                "VOICE_HARNESS_GITHUB_ROOT must be inside VOICE_HARNESS_PROJECT_ROOT"
            ) from exc
        if checkpoint is not None:
            checkpoint()
        self.clone_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if checkpoint is not None:
            checkpoint()
        lock_path = self.clone_root / ".voice-harness-github.lock"
        if checkpoint is not None:
            checkpoint()
        with lock_path.open("a+b") as lock:
            if checkpoint is not None:
                checkpoint()
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if checkpoint is not None:
                    checkpoint()
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _materialize_clone(
        self,
        *,
        owner: str,
        repository_name: str,
        clone_source: str,
        verify: GitHubRepository,
        finalize: Callable[[Path], None] | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        destination = self.clone_root / owner / repository_name
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
                raise GitHubError("GitHub clone destination escapes its root") from exc
            if destination.exists():
                self._verify_checkout(destination, verify)
                if finalize is not None:
                    if checkpoint is not None:
                        checkpoint()
                    finalize(destination)
                    if checkpoint is not None:
                        checkpoint()
                return destination.resolve()
            temporary = destination.parent / (
                f".{repository_name}.clone-{uuid.uuid4().hex}"
            )
            try:
                if checkpoint is not None:
                    checkpoint()
                self._run(
                    [
                        self.gh_executable,
                        "repo",
                        "clone",
                        clone_source,
                        str(temporary),
                    ],
                    timeout=300,
                )
                if checkpoint is not None:
                    checkpoint()
                self._verify_checkout(temporary, verify)
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
                if temporary.exists():
                    shutil.rmtree(temporary)
            return destination.resolve()

    def ensure_clone(
        self,
        source: GitHubRepository,
        fork: GitHubRepository,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        source_owner, repository_name = source.name_with_owner.split("/", 1)

        def finalize(checkout: Path) -> None:
            self._ensure_upstream(checkout, source, fork)

        if checkpoint is None:
            return self._materialize_clone(
                owner=source_owner,
                repository_name=repository_name,
                clone_source=fork.name_with_owner,
                verify=fork,
                finalize=finalize,
            )
        return self._materialize_clone(
            owner=source_owner,
            repository_name=repository_name,
            clone_source=fork.name_with_owner,
            verify=fork,
            finalize=finalize,
            checkpoint=checkpoint,
        )

    def ensure_repository_clone(
        self,
        source: GitHubRepository,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        source_owner, repository_name = source.name_with_owner.split("/", 1)
        if checkpoint is None:
            return self._materialize_clone(
                owner=source_owner,
                repository_name=repository_name,
                clone_source=source.name_with_owner,
                verify=source,
            )
        return self._materialize_clone(
            owner=source_owner,
            repository_name=repository_name,
            clone_source=source.name_with_owner,
            verify=source,
            checkpoint=checkpoint,
        )

    def find_repository_checkout(
        self, source: GitHubRepository, candidates: list[Path]
    ) -> Path | None:
        for candidate in candidates:
            try:
                self._verify_checkout(candidate.resolve(), source)
            except GitHubError:
                continue
            return candidate.resolve()
        return None

    def provision_issue(
        self,
        issue: GitHubIssue,
        *,
        candidates: list[Path] | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> ProvisionedIssue:
        if issue.number <= 0:
            raise GitHubError("GitHub issue number must be positive")
        if checkpoint is not None:
            checkpoint()
        source = self.inspect_repository(issue.name_with_owner)
        if checkpoint is not None:
            checkpoint()
        checkout = self.find_repository_checkout(source, candidates or [])
        if checkout is None:
            if checkpoint is None:
                checkout = self.ensure_repository_clone(source)
            else:
                checkout = self.ensure_repository_clone(source, checkpoint=checkpoint)
        canonical_owner, canonical_repository = source.name_with_owner.split("/", 1)
        canonical_issue = GitHubIssue(
            canonical_owner, canonical_repository, issue.number
        )
        return ProvisionedIssue(
            source=source,
            checkout=checkout,
            issue=canonical_issue,
        )

    def checkout_pull_request(
        self,
        checkout: Path,
        number: int,
        *,
        branch: str,
        checkpoint: Checkpoint | None = None,
    ) -> str | None:
        if number <= 0:
            raise GitHubError("GitHub pull request number must be positive")
        if not re.fullmatch(r"voice/[a-z0-9][a-z0-9._/-]{0,100}", branch):
            raise GitHubError("invalid voice pull-request branch")
        if checkpoint is not None:
            checkpoint()
        self._run(
            [
                self.gh_executable,
                "pr",
                "checkout",
                str(number),
                "--branch",
                branch,
                "--force",
            ],
            timeout=180,
            cwd=checkout,
        )
        if checkpoint is not None:
            checkpoint()
        head = self._git(checkout, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        return head or None

    def provision_public_fork(
        self,
        repository: str,
        *,
        confirmed: bool,
        checkpoint: Checkpoint | None = None,
    ) -> ProvisionedRepository:
        if not confirmed:
            raise GitHubError("GitHub fork creation requires explicit confirmation")
        if checkpoint is not None:
            checkpoint()
        source = self.inspect_public_repository(repository)
        if checkpoint is not None:
            checkpoint()
        login = self.authenticated_login()
        if checkpoint is not None:
            checkpoint()
        if checkpoint is None:
            fork = self.ensure_fork(source, login)
        else:
            fork = self.ensure_fork(source, login, checkpoint=checkpoint)
        if checkpoint is not None:
            checkpoint()
        if checkpoint is None:
            checkout = self.ensure_clone(source, fork)
        else:
            checkout = self.ensure_clone(source, fork, checkpoint=checkpoint)
        if checkpoint is not None:
            checkpoint()
        return ProvisionedRepository(source=source, fork=fork, checkout=checkout)

    def provision_pull_request(
        self,
        repository: str,
        number: int,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> ProvisionedPullRequest:
        if number <= 0:
            raise GitHubError("GitHub pull request number must be positive")
        if checkpoint is not None:
            checkpoint()
        source = self.inspect_repository(repository)
        if checkpoint is not None:
            checkpoint()
        if checkpoint is None:
            checkout = self.ensure_repository_clone(source)
        else:
            checkout = self.ensure_repository_clone(source, checkpoint=checkpoint)
        if checkpoint is not None:
            checkpoint()
        return ProvisionedPullRequest(source=source, checkout=checkout, number=number)
