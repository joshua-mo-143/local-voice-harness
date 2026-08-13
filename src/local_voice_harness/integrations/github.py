from __future__ import annotations

import json
import logging
import re
import secrets
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn
from urllib.parse import SplitResult, urlsplit

from ..context_fragment import ContextFragment
from ..diagnostic_safety import redact_diagnostic
from ..local_git import (
    ExpectedRemote,
    LocalGitError,
    LocalGitOperationAmbiguous,
    LocalGitRepository,
    remote_identity,
)
from ..process import run_command

REPOSITORY = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)/"
    r"(?P<repo>[A-Za-z0-9_.-]{1,100})$"
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
MAX_ISSUE_TITLE_CHARS = 200
MAX_ISSUE_BODY_CHARS = 10_000
ISSUE_CORRELATION_MARKER = re.compile(r"^[0-9a-f]{32}$")
ISSUE_OBSERVATION_LIMIT = 100
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
    return redact_diagnostic(value)


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
class GitHubIssueCreationPlan:
    repository: str
    title: str
    body: str
    correlation_marker: str


@dataclass(frozen=True)
class GitHubIssueCreationResult:
    issue: GitHubIssue
    url: str
    correlation_marker: str


@dataclass(frozen=True)
class GitHubPullRequest:
    owner: str
    repository: str
    number: int

    @property
    def name_with_owner(self) -> str:
        return f"{self.owner}/{self.repository}"


@dataclass(frozen=True)
class GitHubForkPlan:
    source: GitHubRepository
    login: str
    target: str


@dataclass(frozen=True)
class GitHubPullRequestCheckoutInputs:
    remote_url: str
    head_ref: str
    head_oid: str


@dataclass(frozen=True)
class GitHubPullRequestPlan:
    source: GitHubRepository
    pull_request: GitHubPullRequest
    checkout: GitHubPullRequestCheckoutInputs


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
        clone_root: Path | None = None,
        allowed_root: Path | None = None,
        timeout: float = 30,
        local_git: LocalGitRepository | None = None,
    ) -> None:
        self.gh_executable = gh_executable
        self.git_executable = git_executable
        self.clone_root = (clone_root or Path.home() / "src").expanduser().resolve()
        self.allowed_root = (allowed_root or Path.home()).expanduser().resolve()
        self.timeout = timeout
        self.local_git = local_git or LocalGitRepository(
            git_executable=git_executable,
            clone_root=self.clone_root,
            allowed_root=self.allowed_root,
            lock_name=".voice-harness-github.lock",
        )

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
        timeout: float | None = None,
        check: bool = True,
        cwd: Path | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            process = run_command(
                command,
                timeout=self.timeout if timeout is None else timeout,
                cwd=cwd,
                stdin=stdin,
            )
        except OSError as exc:
            raise GitHubError(f"GitHub command failed: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitHubOperationAmbiguous(
                "GitHub command timed out; its outcome is ambiguous because an "
                f"external side effect may already have occurred: {exc}"
            ) from exc
        if check and process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            raise GitHubError(
                detail or f"command exited with status {process.returncode}"
            )
        return process

    @staticmethod
    def validate_issue_creation_plan(plan: GitHubIssueCreationPlan) -> None:
        if not isinstance(plan.repository, str):
            raise GitHubError("GitHub issue repository must be text")
        repository = GitHubClient.validate_repository(plan.repository)
        if repository != plan.repository:
            raise GitHubError("GitHub issue plan repository is not normalized")
        if not isinstance(plan.title, str):
            raise GitHubError("GitHub issue title must be text")
        if not plan.title.strip():
            raise GitHubError("GitHub issue title must not be empty")
        if len(plan.title) > MAX_ISSUE_TITLE_CHARS:
            raise GitHubError(
                f"GitHub issue title must be at most {MAX_ISSUE_TITLE_CHARS} characters"
            )
        if not isinstance(plan.body, str):
            raise GitHubError("GitHub issue body must be text")
        if not plan.body.strip():
            raise GitHubError("GitHub issue body must not be empty")
        if len(plan.body) > MAX_ISSUE_BODY_CHARS:
            raise GitHubError(
                f"GitHub issue body must be at most {MAX_ISSUE_BODY_CHARS} characters"
            )
        if (
            not isinstance(plan.correlation_marker, str)
            or ISSUE_CORRELATION_MARKER.fullmatch(plan.correlation_marker) is None
        ):
            raise GitHubError(
                "GitHub issue correlation marker must be 32 lowercase hex characters"
            )
        if plan.correlation_marker in plan.body:
            raise GitHubError(
                "GitHub issue body must not contain its correlation marker"
            )

    @staticmethod
    def _issue_marker(correlation_marker: str) -> str:
        return f"<!-- local-voice-harness-correlation:{correlation_marker} -->"

    @staticmethod
    def _creation_result(
        plan: GitHubIssueCreationPlan, url: object
    ) -> GitHubIssueCreationResult:
        value = str(url or "").strip()
        issue = github_issue_from_url(value)
        if issue is None or value != issue.url:
            raise GitHubError("GitHub returned a non-canonical issue URL")
        if issue.name_with_owner.casefold() != plan.repository.casefold():
            raise GitHubError("GitHub created an issue in an unexpected repository")
        return GitHubIssueCreationResult(issue, value, plan.correlation_marker)

    def submit_issue(
        self,
        plan: GitHubIssueCreationPlan,
        *,
        confirmed: bool,
    ) -> GitHubIssueCreationResult:
        if not confirmed:
            raise GitHubError("GitHub issue creation requires explicit confirmation")
        self.validate_issue_creation_plan(plan)
        submitted_body = (
            f"{plan.body.rstrip()}\n\n{self._issue_marker(plan.correlation_marker)}\n"
        )
        process = self._run(
            [
                self.gh_executable,
                "issue",
                "create",
                "--repo",
                plan.repository,
                "--title",
                plan.title,
                "--body-file",
                "-",
            ],
            timeout=30,
            stdin=submitted_body,
        )
        return self._creation_result(plan, process.stdout)

    def observe_issue(
        self, plan: GitHubIssueCreationPlan
    ) -> GitHubIssueCreationResult | None:
        self.validate_issue_creation_plan(plan)
        process = self._run(
            [
                self.gh_executable,
                "api",
                "--method",
                "GET",
                f"repos/{plan.repository}/issues",
                "-f",
                "state=all",
                "-f",
                "sort=created",
                "-f",
                "direction=desc",
                "-f",
                f"per_page={ISSUE_OBSERVATION_LIMIT}",
            ],
            timeout=15,
        )
        try:
            values = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError(
                "GitHub returned malformed recent issue metadata"
            ) from exc
        if not isinstance(values, list):
            raise GitHubError("GitHub returned malformed recent issue metadata")
        marker = self._issue_marker(plan.correlation_marker)
        matches: list[GitHubIssueCreationResult] = []
        for value in values:
            if (
                not isinstance(value, dict)
                or "pull_request" in value
                or marker not in str(value.get("body") or "")
            ):
                continue
            result = self._creation_result(plan, value.get("html_url"))
            number = value.get("number")
            if not isinstance(number, int) or number != result.issue.number:
                raise GitHubError("GitHub returned malformed recent issue metadata")
            matches.append(result)
        if len(matches) > 1:
            raise GitHubError("GitHub issue correlation marker is not unique")
        return matches[0] if matches else None

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
                "isDraft,baseRefName,headRefName,headRefOid,"
                "additions,deletions,changedFiles",
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

    def repository_for_checkout(self, checkout: Path) -> str:
        """Return the validated GitHub origin for an allowed local checkout."""

        resolved = checkout.expanduser().resolve()
        try:
            resolved.relative_to(self.allowed_root)
        except ValueError as exc:
            raise GitHubError(
                "Git checkout is outside the configured project root"
            ) from exc
        if resolved == self.allowed_root:
            raise GitHubError(f"{resolved} is not an allowed Git repository")
        try:
            self.local_git.verify_checkout(resolved)
            remote = self.local_git.git(
                resolved, "remote", "get-url", "origin"
            ).stdout.strip()
        except LocalGitError as exc:
            self._raise_local_git_error(exc)
        identity = remote_identity(remote)
        prefix = "github.com/"
        if identity is None or not identity.startswith(prefix):
            raise GitHubError("Git origin is not a GitHub repository")
        return self.validate_repository(identity.removeprefix(prefix))

    @staticmethod
    def _expected_remote(repository: GitHubRepository) -> ExpectedRemote:
        if not _github_url(repository.url):
            raise GitHubError(
                f"GitHub returned an invalid clone URL for {repository.name_with_owner}"
            )
        try:
            expected = ExpectedRemote.from_url(repository.url)
        except LocalGitError as exc:
            raise GitHubError(
                f"GitHub returned an invalid clone URL for {repository.name_with_owner}"
            ) from exc
        identity = f"github.com/{repository.name_with_owner.casefold()}"
        if expected.identity != identity:
            raise GitHubError(
                f"GitHub returned an invalid clone URL for {repository.name_with_owner}"
            )
        return expected

    @staticmethod
    def _raise_local_git_error(exc: LocalGitError) -> NoReturn:
        if isinstance(exc, LocalGitOperationAmbiguous):
            raise GitHubOperationAmbiguous(str(exc)) from exc
        raise GitHubError(str(exc)) from exc

    def ensure_clone(
        self,
        source: GitHubRepository,
        fork: GitHubRepository,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        source_owner, repository_name = source.name_with_owner.split("/", 1)

        def finalize(checkout: Path) -> None:
            if source.name_with_owner.casefold() != fork.name_with_owner.casefold():
                self.local_git.ensure_remote(checkout, "upstream", source.url)

        try:
            return self.local_git.materialize(
                Path(source_owner) / repository_name,
                clone_url=fork.url,
                clone_command=(
                    self.gh_executable,
                    "repo",
                    "clone",
                    fork.name_with_owner,
                ),
                expected=self._expected_remote(fork),
                expected_label=fork.name_with_owner,
                finalize=finalize,
                checkpoint=checkpoint,
            )
        except LocalGitError as exc:
            self._raise_local_git_error(exc)

    def ensure_repository_clone(
        self,
        source: GitHubRepository,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        source_owner, repository_name = source.name_with_owner.split("/", 1)
        try:
            return self.local_git.materialize(
                Path(source_owner) / repository_name,
                clone_url=source.url,
                clone_command=(
                    self.gh_executable,
                    "repo",
                    "clone",
                    source.name_with_owner,
                ),
                expected=self._expected_remote(source),
                expected_label=source.name_with_owner,
                checkpoint=checkpoint,
            )
        except LocalGitError as exc:
            self._raise_local_git_error(exc)

    def find_repository_checkout(
        self, source: GitHubRepository, candidates: list[Path]
    ) -> Path | None:
        try:
            return self.local_git.find_checkout(
                candidates,
                self._expected_remote(source),
                expected_label=source.name_with_owner,
            )
        except LocalGitError as exc:
            self._raise_local_git_error(exc)

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
        try:
            return self.local_git.current_branch(checkout)
        except LocalGitError as exc:
            self._raise_local_git_error(exc)

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


_GITHUB_STATE_VERSION = 1
_GITHUB_STATE_SECTIONS: dict[str, dict[str, str]] = {
    "repository": {"name": "github_repository"},
    "issue": {
        "number": "github_issue",
        "url": "github_issue_url",
        "context": "github_issue_context",
    },
    "issue_creation": {
        "requested": "github_issue_create_requested",
        "confirmed": "github_issue_create_confirmed",
        "title": "github_issue_create_title",
        "body": "github_issue_create_body",
        "marker": "github_issue_create_marker",
        "operation_state": "github_issue_create_operation_state",
        "created_number": "github_issue_created_number",
        "created_url": "github_issue_created_url",
    },
    "pull_request": {
        "number": "github_pull_request",
        "worktree_state": "pull_request_worktree_state",
        "branch": "pull_request_branch",
        "worktree_error": "pull_request_worktree_error",
        "remote_url": "pull_request_remote_url",
        "head_ref": "pull_request_head_ref",
        "head_oid": "pull_request_head_oid",
    },
    "fork": {
        "requested": "fork_requested",
        "confirmed": "fork_confirmed",
        "committed": "fork_committed",
        "exists": "fork_exists",
        "dispatch_exited": "fork_dispatch_exited",
        "committed_at": "fork_committed_at",
        "operation_state": "fork_operation_state",
        "operation_source": "fork_operation_source",
        "operation_source_url": "fork_operation_source_url",
        "operation_source_parent": "fork_operation_source_parent",
        "operation_source_default_branch": "fork_operation_source_default_branch",
        "operation_source_private": "fork_operation_source_private",
        "operation_login": "fork_operation_login",
        "operation_target": "fork_operation_target",
        "repository": "fork_repository",
        "reconcile_attempts": "fork_reconcile_attempts",
        "absent_observations": "fork_absent_observations",
        "next_reconcile_at": "fork_next_reconcile_at",
        "last_reconciled_at": "fork_last_reconciled_at",
        "confirmed_absent_at": "fork_confirmed_absent_at",
        "automatic_reconcile_stopped_at": "fork_automatic_reconcile_stopped_at",
        "retained_at": "fork_retained_at",
    },
}
GITHUB_PROVIDER_STATE_FIELDS = frozenset(
    field for section in _GITHUB_STATE_SECTIONS.values() for field in section.values()
)


def load_github_provider_state(state: Mapping[str, object]) -> dict[str, object]:
    """Convert legacy flat or nested v1 GitHub provider state to flat fields."""

    if not isinstance(state, Mapping):
        raise GitHubError("GitHub provider state must be an object")
    values = dict(state)
    if not values:
        return {}
    if "version" not in values:
        unsupported = set(values) - GITHUB_PROVIDER_STATE_FIELDS
        if unsupported:
            field = sorted(unsupported)[0]
            raise GitHubError(
                f"GitHub provider state contains unsupported field {field}"
            )
        return values
    if values.get("version") != _GITHUB_STATE_VERSION:
        raise GitHubError("unsupported GitHub provider state version")
    unsupported_sections = set(values) - {"version", *_GITHUB_STATE_SECTIONS}
    if unsupported_sections:
        section = sorted(unsupported_sections)[0]
        raise GitHubError(
            f"GitHub provider state contains unsupported section {section}"
        )
    flattened: dict[str, object] = {}
    for section_name, fields in _GITHUB_STATE_SECTIONS.items():
        section_value = values.get(section_name, {})
        if not isinstance(section_value, Mapping):
            raise GitHubError(
                f"GitHub provider state {section_name} section must be an object"
            )
        section = dict(section_value)
        unsupported = set(section) - set(fields)
        if unsupported:
            field = sorted(unsupported)[0]
            raise GitHubError(
                f"GitHub provider state {section_name} contains unsupported field {field}"
            )
        for name, value in section.items():
            flattened[fields[name]] = value
    return flattened


def dump_github_provider_state(state: Mapping[str, object]) -> dict[str, object]:
    """Serialize GitHub provider fields as clearly nested durable v1 state."""

    flat = load_github_provider_state(state)
    serialized: dict[str, object] = {"version": _GITHUB_STATE_VERSION}
    for section_name, fields in _GITHUB_STATE_SECTIONS.items():
        section = {name: flat[field] for name, field in fields.items() if field in flat}
        if section:
            serialized[section_name] = section
    return serialized


class GitHubProvider:
    """Built-in context provider for GitHub repositories, issues, and PRs."""

    name = PROVIDER_NAME

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    @staticmethod
    def load_state(state: Mapping[str, object]) -> dict[str, object]:
        return load_github_provider_state(state)

    @staticmethod
    def dump_state(state: Mapping[str, object]) -> dict[str, object]:
        return dump_github_provider_state(state)

    def resolve_repository(
        self, repository: str, *, public: bool = False
    ) -> GitHubRepository:
        if public:
            return self._client.inspect_public_repository(repository)
        return self._client.inspect_repository(repository)

    @property
    def local_git(self) -> LocalGitRepository:
        """Return the generic local-repository boundary configured for this provider."""

        return self._client.local_git

    def provision_issue(
        self,
        issue: GitHubIssue,
        *,
        candidates: list[Path] | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> ProvisionedIssue:
        return self._client.provision_issue(
            issue,
            candidates=candidates,
            checkpoint=checkpoint,
        )

    def resolve_issue(self, issue: GitHubIssue) -> dict[str, object]:
        """Validate and resolve current forge metadata for an issue."""

        return self._client.issue_details(issue)

    def plan_issue_creation(
        self,
        repository: str,
        title: str,
        body: str,
        *,
        correlation_marker: str | None = None,
    ) -> GitHubIssueCreationPlan:
        if not isinstance(repository, str):
            raise GitHubError("GitHub issue repository must be text")
        if not isinstance(title, str):
            raise GitHubError("GitHub issue title must be text")
        if not isinstance(body, str):
            raise GitHubError("GitHub issue body must be text")
        plan = GitHubIssueCreationPlan(
            repository=GitHubClient.validate_repository(repository),
            title=title.strip(),
            body=body.strip(),
            correlation_marker=correlation_marker or secrets.token_hex(16),
        )
        self.validate_issue_creation_plan(plan)
        return plan

    @staticmethod
    def validate_issue_creation_plan(plan: GitHubIssueCreationPlan) -> None:
        GitHubClient.validate_issue_creation_plan(plan)

    def observe_issue(
        self, plan: GitHubIssueCreationPlan
    ) -> GitHubIssueCreationResult | None:
        self.validate_issue_creation_plan(plan)
        return self._client.observe_issue(plan)

    def observe_issue_creation(
        self, plan: GitHubIssueCreationPlan
    ) -> GitHubIssueCreationResult | None:
        return self.observe_issue(plan)

    def submit_issue(
        self,
        plan: GitHubIssueCreationPlan,
        *,
        confirmed: bool,
    ) -> GitHubIssueCreationResult:
        if not confirmed:
            raise GitHubError("GitHub issue creation requires explicit confirmation")
        self.validate_issue_creation_plan(plan)
        return self._client.submit_issue(plan, confirmed=True)

    def submit_issue_creation(
        self,
        plan: GitHubIssueCreationPlan,
        *,
        confirmed: bool,
    ) -> GitHubIssueCreationResult:
        return self.submit_issue(plan, confirmed=confirmed)

    def materialize_repository(
        self,
        source: GitHubRepository,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        return self._client.ensure_repository_clone(source, checkpoint=checkpoint)

    def materialize_fork(
        self,
        plan: GitHubForkPlan,
        fork: GitHubRepository,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        self.validate_fork_plan(
            plan,
            materialized_repository=fork.name_with_owner,
        )
        return self._client.ensure_clone(plan.source, fork, checkpoint=checkpoint)

    def plan_fork(self, repository: str) -> GitHubForkPlan:
        source, login, target = self._client.prepare_public_fork(repository)
        plan = GitHubForkPlan(source=source, login=login, target=target)
        self.validate_fork_plan(plan)
        return plan

    def validate_fork_plan(
        self,
        plan: GitHubForkPlan,
        *,
        materialized_repository: str | None = None,
    ) -> None:
        source = GitHubClient.validate_repository(plan.source.name_with_owner)
        if plan.source.is_private:
            raise GitHubError("the focused GitHub repository is private")
        try:
            source_remote = ExpectedRemote.from_url(plan.source.url)
        except LocalGitError as exc:
            raise GitHubError("GitHub fork source URL is invalid") from exc
        if source_remote.identity != f"github.com/{source.casefold()}":
            raise GitHubError("GitHub fork source URL does not match its repository")
        if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", plan.login) is None:
            raise GitHubError("GitHub fork plan has an invalid login")
        source_owner, repository_name = source.split("/", 1)
        expected_target = (
            source
            if source_owner.casefold() == plan.login.casefold()
            else f"{plan.login}/{repository_name}"
        )
        target = GitHubClient.validate_repository(plan.target)
        if target.casefold() != expected_target.casefold():
            raise GitHubError("GitHub fork plan has an invalid target")
        if (
            materialized_repository is not None
            and GitHubClient.validate_repository(materialized_repository).casefold()
            != target.casefold()
        ):
            raise GitHubError(
                "persisted GitHub fork repository does not match its target"
            )

    def refresh_fork_plan(self, plan: GitHubForkPlan) -> GitHubForkPlan:
        """Re-resolve persisted source metadata before trusting fork relationships."""

        source = self.resolve_repository(plan.source.name_with_owner, public=True)
        if source.name_with_owner.casefold() != plan.source.name_with_owner.casefold():
            raise GitHubError("GitHub fork source identity changed unexpectedly")
        refreshed = GitHubForkPlan(
            source=source,
            login=plan.login,
            target=plan.target,
        )
        self.validate_fork_plan(refreshed)
        return refreshed

    def observe_fork(self, plan: GitHubForkPlan) -> GitHubRepository | None:
        plan = self.refresh_fork_plan(plan)
        return self._client.reconcile_fork(plan.source, plan.target)

    def submit_fork(
        self,
        plan: GitHubForkPlan,
        *,
        confirmed: bool,
        checkpoint: Checkpoint | None = None,
        before_submit: Callable[[], None] | None = None,
    ) -> GitHubRepository:
        if not confirmed:
            raise GitHubError("GitHub fork creation requires explicit confirmation")
        self.validate_fork_plan(plan)
        return self._client.ensure_fork(
            plan.source,
            plan.login,
            checkpoint=checkpoint,
            before_submit=before_submit,
        )

    def pull_request_checkout_inputs(
        self,
        source: GitHubRepository,
        pull_request: GitHubPullRequest,
        details: Mapping[str, object],
    ) -> GitHubPullRequestCheckoutInputs:
        if pull_request.number <= 0:
            raise GitHubError("GitHub pull request number must be positive")
        if pull_request.name_with_owner.casefold() != source.name_with_owner.casefold():
            raise GitHubError("pull request does not belong to the resolved repository")
        head_oid = str(details.get("headRefOid") or "")
        inputs = GitHubPullRequestCheckoutInputs(
            remote_url=source.url,
            head_ref=f"refs/pull/{pull_request.number}/head",
            head_oid=head_oid.lower(),
        )
        self.validate_pull_request_checkout_inputs(
            source.name_with_owner,
            pull_request.number,
            inputs,
        )
        return inputs

    def validate_pull_request_checkout_inputs(
        self,
        repository: str,
        number: int,
        inputs: GitHubPullRequestCheckoutInputs,
    ) -> None:
        """Reject persisted checkout metadata that no longer proves GitHub identity."""

        repository = GitHubClient.validate_repository(repository)
        try:
            expected = ExpectedRemote.from_url(inputs.remote_url)
        except LocalGitError as exc:
            raise GitHubError("GitHub pull-request remote URL is invalid") from exc
        if expected.identity != f"github.com/{repository.casefold()}":
            raise GitHubError(
                "GitHub pull-request remote does not match its repository"
            )
        if inputs.head_ref != f"refs/pull/{number}/head":
            raise GitHubError("GitHub pull-request ref does not match its number")
        if re.fullmatch(r"[0-9a-f]{40}", inputs.head_oid) is None:
            raise GitHubError("GitHub returned an invalid pull request head OID")

    def plan_pull_request(self, repository: str, number: int) -> GitHubPullRequestPlan:
        if number <= 0:
            raise GitHubError("GitHub pull request number must be positive")
        source = self.resolve_repository(repository)
        owner, name = source.name_with_owner.split("/", 1)
        pull_request = GitHubPullRequest(owner, name, number)
        details = self._client.pull_request_details(pull_request)
        checkout = self.pull_request_checkout_inputs(source, pull_request, details)
        return GitHubPullRequestPlan(
            source=source,
            pull_request=pull_request,
            checkout=checkout,
        )

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
            issue_scope=issue.name_with_owner,
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
