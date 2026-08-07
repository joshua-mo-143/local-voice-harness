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


class GitHubError(RuntimeError):
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
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitHubError(f"GitHub command failed: {exc}") from exc
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
            if required:
                detail = process.stderr.strip() or process.stdout.strip()
                raise GitHubError(detail or f"could not inspect {repository}")
            return None
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

    def ensure_fork(self, source: GitHubRepository, login: str) -> GitHubRepository:
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
    def _provisioning_lock(self) -> Iterator[None]:
        try:
            self.clone_root.relative_to(self.allowed_root)
        except ValueError as exc:
            raise GitHubError(
                "VOICE_HARNESS_GITHUB_ROOT must be inside VOICE_HARNESS_PROJECT_ROOT"
            ) from exc
        self.clone_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.clone_root / ".voice-harness-github.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
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
    ) -> Path:
        destination = self.clone_root / owner / repository_name
        with self._provisioning_lock():
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            resolved_parent = destination.parent.resolve()
            try:
                resolved_parent.relative_to(self.clone_root)
            except ValueError as exc:
                raise GitHubError("GitHub clone destination escapes its root") from exc
            if destination.exists():
                self._verify_checkout(destination, verify)
                if finalize is not None:
                    finalize(destination)
                return destination.resolve()
            temporary = destination.parent / (
                f".{repository_name}.clone-{uuid.uuid4().hex}"
            )
            try:
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
                self._verify_checkout(temporary, verify)
                if finalize is not None:
                    finalize(temporary)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return destination.resolve()

    def ensure_clone(self, source: GitHubRepository, fork: GitHubRepository) -> Path:
        source_owner, repository_name = source.name_with_owner.split("/", 1)
        return self._materialize_clone(
            owner=source_owner,
            repository_name=repository_name,
            clone_source=fork.name_with_owner,
            verify=fork,
            finalize=lambda checkout: self._ensure_upstream(checkout, source, fork),
        )

    def ensure_repository_clone(self, source: GitHubRepository) -> Path:
        source_owner, repository_name = source.name_with_owner.split("/", 1)
        return self._materialize_clone(
            owner=source_owner,
            repository_name=repository_name,
            clone_source=source.name_with_owner,
            verify=source,
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
    ) -> ProvisionedIssue:
        if issue.number <= 0:
            raise GitHubError("GitHub issue number must be positive")
        source = self.inspect_repository(issue.name_with_owner)
        checkout = self.find_repository_checkout(source, candidates or [])
        if checkout is None:
            checkout = self.ensure_repository_clone(source)
        canonical_owner, canonical_repository = source.name_with_owner.split("/", 1)
        canonical_issue = GitHubIssue(
            canonical_owner, canonical_repository, issue.number
        )
        return ProvisionedIssue(
            source=source,
            checkout=checkout,
            issue=canonical_issue,
        )

    def checkout_pull_request(self, checkout: Path, number: int) -> str | None:
        self._run(
            [self.gh_executable, "pr", "checkout", str(number)],
            timeout=180,
            cwd=checkout,
        )
        head = self._git(checkout, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        return head or None

    def provision_public_fork(
        self, repository: str, *, confirmed: bool
    ) -> ProvisionedRepository:
        if not confirmed:
            raise GitHubError("GitHub fork creation requires explicit confirmation")
        source = self.inspect_public_repository(repository)
        fork = self.ensure_fork(source, self.authenticated_login())
        checkout = self.ensure_clone(source, fork)
        return ProvisionedRepository(source=source, fork=fork, checkout=checkout)

    def provision_pull_request(
        self, repository: str, number: int
    ) -> ProvisionedPullRequest:
        if number <= 0:
            raise GitHubError("GitHub pull request number must be positive")
        source = self.inspect_repository(repository)
        checkout = self.ensure_repository_clone(source)
        branch = self.checkout_pull_request(checkout, number)
        return ProvisionedPullRequest(
            source=source, checkout=checkout, number=number, branch=branch
        )
