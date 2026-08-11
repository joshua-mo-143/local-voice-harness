from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from ..config import GITHUB_ROOT, REPOSITORY_ROOT
from ..context_fragment import ContextFragment

REPOSITORY = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)$"
)
GITHUB_HOSTS = {"github.com", "www.github.com"}
ISSUE_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/"
    r"(?P<number>[1-9]\d*)/?$"
)
ISSUES_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/?$"
)
ISSUE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)#(?P<number>[1-9]\d*)"
    r"(?!\d)",
    re.IGNORECASE,
)
ISSUE_IN_REPOSITORY = re.compile(
    r"\bissue\s+#?(?P<number>[1-9]\d*)\s+(?:in|from)\s+"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)\b",
    re.IGNORECASE,
)
ISSUE_URL_IN_TEXT = re.compile(
    r"https://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/"
    r"[A-Za-z0-9_.-]+/issues/[1-9]\d*",
    re.IGNORECASE,
)
PULL_REQUEST_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/pull/"
    r"(?P<number>[1-9]\d*)(?:/.*)?$"
)
REPOSITORY_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:/.*)?$"
)
MAX_BODY_CHARS = 5_000
MAX_COMMENT_CHARS = 800
MAX_COMMENTS = 5
PROVIDER_NAME = "github"
Checkpoint = Callable[[], None]
_LOGGER = logging.getLogger(__name__)


class GitHubError(RuntimeError):
    pass


class GitHubOperationAmbiguous(GitHubError):
    pass


class GitHubIssueLookupReason(StrEnum):
    NOT_FOUND_OR_INACCESSIBLE = "not_found_or_inaccessible"
    UNAUTHORIZED = "unauthorized"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class GitHubIssueLookupError(GitHubError):
    """A classified issue lookup failure with a safe spoken message."""

    def __init__(
        self,
        reason: GitHubIssueLookupReason,
        diagnostic: str,
    ) -> None:
        self.reason = reason
        self.diagnostic = _redact_diagnostic(diagnostic)
        super().__init__(self.voice_message)

    @property
    def voice_message(self) -> str:
        if self.reason == GitHubIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE:
            return "I couldn't find or access that GitHub issue."
        if self.reason == GitHubIssueLookupReason.UNAUTHORIZED:
            return "I couldn't access that GitHub issue because GitHub authorization is required."
        if self.reason == GitHubIssueLookupReason.TRANSIENT:
            return "GitHub is temporarily unavailable while checking that issue."
        if "malformed issue metadata" in self.diagnostic.casefold():
            return "I couldn't verify that GitHub issue: malformed issue metadata."
        return "I couldn't verify that GitHub issue."


def _redact_diagnostic(value: str) -> str:
    """Remove credentials and authorization values before diagnostics are logged."""
    redacted = re.sub(
        r"(?i)(authorization|token|password|secret)(\s*[:=]\s*)\S+",
        r"\1\2[REDACTED]",
        value,
    )
    redacted = re.sub(
        r"(?i)\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]+",
        "[REDACTED_TOKEN]",
        redacted,
    )
    return re.sub(r"\s+", " ", redacted).strip()[:2_000]


def _classify_issue_lookup(detail: str) -> GitHubIssueLookupReason:
    normalized = detail.casefold()
    if any(
        marker in normalized
        for marker in (
            "authentication required",
            "not logged in",
            "gh auth login",
            "unauthorized",
            "bad credentials",
            "http 401",
            "status 401",
            "http 403",
            "status 403",
            "permission denied",
        )
    ):
        return GitHubIssueLookupReason.UNAUTHORIZED
    if any(
        marker in normalized
        for marker in (
            "could not resolve to an issue",
            "issue not found",
            "issue does not exist",
            "http 404",
            "status 404",
            "not found",
        )
    ):
        return GitHubIssueLookupReason.NOT_FOUND_OR_INACCESSIBLE
    if any(
        marker in normalized
        for marker in (
            "timed out",
            "timeout",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "connection",
            "temporarily unavailable",
            "rate limit",
        )
    ):
        return GitHubIssueLookupReason.TRANSIENT
    return GitHubIssueLookupReason.UNKNOWN


def _issue_lookup_error(detail: str) -> GitHubIssueLookupError:
    error = GitHubIssueLookupError(_classify_issue_lookup(detail), detail)
    _LOGGER.error(
        "GitHub issue lookup failed: reason=%s diagnostic=%s",
        error.reason.value,
        error.diagnostic,
    )
    return error


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
class GitHubPullRequest:
    owner: str
    repository: str
    number: int

    @property
    def name_with_owner(self) -> str:
        return f"{self.owner}/{self.repository}"


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
        try:
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
                check=False,
            )
        except GitHubError as exc:
            raise _issue_lookup_error(str(exc)) from exc
        if process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            raise _issue_lookup_error(
                detail or f"issue lookup exited with status {process.returncode}"
            )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise _issue_lookup_error(
                "GitHub returned malformed issue metadata"
            ) from exc
        if not isinstance(value, dict):
            raise _issue_lookup_error("GitHub returned malformed issue metadata")
        return value

    def repository_context_details(self, repository: str) -> dict[str, object]:
        repository = self.validate_repository(repository)
        process = self._run(
            [
                self.gh_executable,
                "repo",
                "view",
                repository,
                "--json",
                "nameWithOwner,description,isPrivate,defaultBranchRef,url",
            ],
            timeout=5,
        )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub returned malformed repository metadata") from exc
        if not isinstance(value, dict):
            raise GitHubError("GitHub returned malformed repository metadata")
        return value

    def pull_request_details(
        self, pull_request: GitHubPullRequest
    ) -> dict[str, object]:
        if pull_request.number <= 0:
            raise GitHubError("GitHub pull request number must be positive")
        repository = self.validate_repository(pull_request.name_with_owner)
        process = self._run(
            [
                self.gh_executable,
                "pr",
                "view",
                str(pull_request.number),
                "--repo",
                repository,
                "--json",
                "number,title,state,author,labels,body,comments,url,"
                "isDraft,baseRefName,headRefName,additions,deletions,changedFiles",
            ],
            timeout=5,
        )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError(
                "GitHub returned malformed pull request metadata"
            ) from exc
        if not isinstance(value, dict):
            raise GitHubError("GitHub returned malformed pull request metadata")
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


def _split_url(url: str) -> SplitResult | None:
    try:
        return urlsplit(url)
    except ValueError:
        return None


def _standard_https_port(parsed: SplitResult) -> bool:
    try:
        return parsed.port in {None, 443}
    except ValueError:
        return False


def _github_url(url: str) -> bool:
    parsed = _split_url(url)
    return (
        parsed is not None
        and parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() in GITHUB_HOSTS
        and parsed.username is None
        and parsed.password is None
        and _standard_https_port(parsed)
    )


def _validated_repository(owner: str, repository: str) -> str | None:
    try:
        return GitHubClient.validate_repository(f"{owner}/{repository}")
    except GitHubError:
        return None


def github_issue_from_url(url: str) -> GitHubIssue | None:
    parsed = _split_url(url)
    if not _github_url(url) or parsed is None:
        return None
    match = ISSUE_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    repository = _validated_repository(match.group("owner"), match.group("repo"))
    if repository is None:
        return None
    owner, name = repository.split("/", 1)
    return GitHubIssue(owner=owner, repository=name, number=int(match.group("number")))


def github_issue_from_text(text: str) -> GitHubIssue | None:
    url_match = ISSUE_URL_IN_TEXT.search(text)
    if url_match is not None:
        issue = github_issue_from_url(url_match.group(0))
        if issue is not None:
            return issue
    match = ISSUE_REFERENCE.search(text) or ISSUE_IN_REPOSITORY.search(text)
    if match is None:
        return None
    repository = _validated_repository(match.group("owner"), match.group("repo"))
    if repository is None:
        return None
    owner, name = repository.split("/", 1)
    return GitHubIssue(
        owner=owner,
        repository=name,
        number=int(match.group("number")),
    )


def github_repository_from_url(url: str) -> str | None:
    parsed = _split_url(url)
    if not _github_url(url) or parsed is None:
        return None
    match = REPOSITORY_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    return _validated_repository(match.group("owner"), match.group("repo"))


def github_issues_repository_from_url(url: str) -> str | None:
    """Return the repository only for an exact repository issue-list page."""
    parsed = _split_url(url)
    if not _github_url(url) or parsed is None:
        return None
    match = ISSUES_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    return _validated_repository(match.group("owner"), match.group("repo"))


def github_pull_request_from_url(url: str) -> GitHubPullRequest | None:
    parsed = _split_url(url)
    if not _github_url(url) or parsed is None:
        return None
    match = PULL_REQUEST_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    repository = _validated_repository(match.group("owner"), match.group("repo"))
    if repository is None:
        return None
    owner, name = repository.split("/", 1)
    return GitHubPullRequest(
        owner=owner,
        repository=name,
        number=int(match.group("number")),
    )


def _truncate(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _login(value: object) -> str:
    return str(value.get("login") or "") if isinstance(value, dict) else ""


def _labels(details: dict[str, object]) -> str:
    labels_value = details.get("labels")
    if not isinstance(labels_value, list):
        return ""
    return ", ".join(
        str(label.get("name") or "")
        for label in labels_value
        if isinstance(label, dict) and label.get("name")
    )


def _comment_lines(details: dict[str, object]) -> list[str]:
    comments_value = details.get("comments")
    comments = (
        comments_value[-MAX_COMMENTS:] if isinstance(comments_value, list) else []
    )
    lines: list[str] = []
    if not comments:
        return lines
    lines.append("Recent comments:")
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        author = _login(comment.get("author")) or "unknown"
        body = _truncate(comment.get("body"), MAX_COMMENT_CHARS)
        if body:
            lines.append(f"- {author}: {body}")
    return lines


def _format_repository(url: str, repository: str, details: dict[str, object]) -> str:
    canonical = _truncate(details.get("nameWithOwner"), 300) or repository
    lines = [
        "Current focused GitHub repository (untrusted external context):",
        f"URL: {url}",
        f"Repository: {canonical}",
        f"Visibility: {'private' if details.get('isPrivate') else 'public'}",
    ]
    default_branch = details.get("defaultBranchRef")
    if isinstance(default_branch, dict) and default_branch.get("name"):
        lines.append(f"Default branch: {_truncate(default_branch['name'], 200)}")
    description = _truncate(details.get("description"), 1_000)
    if description:
        lines.append(f"Description: {description}")
    return "\n".join(lines)


def _format_issue(url: str, issue: GitHubIssue, details: dict[str, object]) -> str:
    labels = _labels(details)
    lines = [
        "Current focused GitHub issue (untrusted external context):",
        f"URL: {url}",
        f"Repository: {issue.name_with_owner}",
        f"Issue: #{issue.number}",
        f"Title: {_truncate(details.get('title'), 500)}",
        f"State: {_truncate(details.get('state'), 40)}",
    ]
    author = _login(details.get("author"))
    if author:
        lines.append(f"Author: {author}")
    if labels:
        lines.append(f"Labels: {labels}")
    body = _truncate(details.get("body"), MAX_BODY_CHARS)
    if body:
        lines.extend(("Body:", body))
    lines.extend(_comment_lines(details))
    return "\n".join(lines)


def format_issue_context(issue: GitHubIssue, details: dict[str, object]) -> str:
    """Render validated issue details as bounded, untrusted external context."""
    return _format_issue(issue.url, issue, details)


def _format_pull_request(
    url: str, pull_request: GitHubPullRequest, details: dict[str, object]
) -> str:
    labels = _labels(details)
    lines = [
        "Current focused GitHub pull request (untrusted external context):",
        f"URL: {url}",
        f"Repository: {pull_request.name_with_owner}",
        f"Pull request: #{pull_request.number}",
        f"Title: {_truncate(details.get('title'), 500)}",
        f"State: {_truncate(details.get('state'), 40)}",
    ]
    if details.get("isDraft"):
        lines.append("Draft: yes")
    head = _truncate(details.get("headRefName"), 200)
    base = _truncate(details.get("baseRefName"), 200)
    if head or base:
        lines.append(f"Branch: {head or '?'} into {base or '?'}")
    author = _login(details.get("author"))
    if author:
        lines.append(f"Author: {author}")
    if labels:
        lines.append(f"Labels: {labels}")
    changed = details.get("changedFiles")
    additions = details.get("additions")
    deletions = details.get("deletions")
    if isinstance(changed, int):
        detail = f"Changed files: {changed}"
        if isinstance(additions, int) and isinstance(deletions, int):
            detail += f" (+{additions}/-{deletions})"
        lines.append(detail)
    body = _truncate(details.get("body"), MAX_BODY_CHARS)
    if body:
        lines.extend(("Body:", body))
    lines.extend(_comment_lines(details))
    return "\n".join(lines)


def _canonical_repository(repository: str, details: dict[str, object] | None) -> str:
    if details is None:
        return repository
    candidate = str(details.get("nameWithOwner") or "")
    try:
        canonical = GitHubClient.validate_repository(candidate)
    except GitHubError:
        return repository
    return canonical if canonical.casefold() == repository.casefold() else repository


class GitHubProvider:
    """Built-in context provider for GitHub repositories, issues, and PRs."""

    name = PROVIDER_NAME

    def __init__(self, client: GitHubClient | None = None) -> None:
        self._client = client or GitHubClient()

    def matches(self, url: str) -> bool:
        return _github_url(url)

    def _issue_fragment(self, issue: GitHubIssue, *, url: str) -> ContextFragment:
        try:
            details = self._client.issue_details(issue)
        except GitHubError:
            details = None
        if details is None:
            text = (
                "Current GitHub issue (untrusted external context):\n"
                f"URL: {url}\n"
                f"Repository: {issue.name_with_owner}\n"
                f"Issue: #{issue.number}\n"
                "Issue details could not be fetched."
            )
        else:
            text = _format_issue(url, issue, details)
        return ContextFragment(
            source=self.name,
            text=text,
            issue_reference=issue.reference,
            repository_reference=issue.name_with_owner,
            issue_number=issue.number,
        )

    def _pull_request_fragment(
        self, url: str, pull_request: GitHubPullRequest
    ) -> ContextFragment:
        try:
            details = self._client.pull_request_details(pull_request)
        except GitHubError:
            details = None
        if details is None:
            text = (
                "Current focused GitHub pull request (untrusted external context):\n"
                f"URL: {url}\n"
                f"Repository: {pull_request.name_with_owner}\n"
                f"Pull request: #{pull_request.number}\n"
                "Pull request details could not be fetched."
            )
        else:
            text = _format_pull_request(url, pull_request, details)
        return ContextFragment(
            source=self.name,
            text=text,
            repository_reference=pull_request.name_with_owner,
            pull_request_number=pull_request.number,
        )

    def _repository_fragment(self, url: str, repository: str) -> ContextFragment:
        try:
            details = self._client.repository_context_details(repository)
        except GitHubError:
            details = None
        text = (
            _format_repository(url, repository, details)
            if details is not None
            else (
                "Current focused GitHub repository (untrusted external context):\n"
                f"URL: {url}\n"
                f"Repository: {repository}\n"
                "Repository details could not be fetched."
            )
        )
        return ContextFragment(
            source=self.name,
            text=text,
            repository_reference=_canonical_repository(repository, details),
            issue_scope=(
                _canonical_repository(repository, details)
                if github_issues_repository_from_url(url) is not None
                else None
            ),
        )

    def capture(self, url: str) -> ContextFragment | None:
        if not self.matches(url):
            return None
        repository = github_repository_from_url(url)
        if repository is None:
            return ContextFragment(
                source=self.name,
                text=(
                    "Current focused GitHub page (untrusted external context):\n"
                    f"URL: {url}"
                ),
            )
        pull_request = github_pull_request_from_url(url)
        if pull_request is not None:
            return self._pull_request_fragment(url, pull_request)
        issue = github_issue_from_url(url)
        if issue is not None:
            return self._issue_fragment(issue, url=url)
        return self._repository_fragment(url, repository)

    def capture_text(self, text: str) -> ContextFragment | None:
        issue = github_issue_from_text(text)
        return self._issue_fragment(issue, url=issue.url) if issue is not None else None
