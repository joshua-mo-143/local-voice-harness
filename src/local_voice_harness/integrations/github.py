from __future__ import annotations

import json
import logging
import re
import secrets
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn
from urllib.parse import SplitResult, quote, urlsplit

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
from ..ticket_snapshot import (
    MAX_SNAPSHOT_BODY_CHARS,
    MAX_SNAPSHOT_REVISION_CHARS,
    MAX_SNAPSHOT_TITLE_CHARS,
    TicketSnapshot,
)

REPOSITORY = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)/"
    r"(?P<repo>[A-Za-z0-9_.-]{1,100})$"
)
ISSUE_IDENTITY = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)/"
    r"(?P<repo>[A-Za-z0-9_.-]{1,100})#(?P<number>[1-9]\d*)$"
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
REPOSITORY_ISSUE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)\s+issues?\s+#?"
    r"(?P<number>[1-9]\d*)(?![A-Za-z0-9_/-])",
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
PULL_REQUEST_URL_IN_TEXT = re.compile(
    r"https://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/"
    r"[A-Za-z0-9_.-]+/pull/[1-9]\d*",
    re.IGNORECASE,
)
PULL_REQUEST_LANGUAGE = re.compile(r"\b(?:pull requests?|prs?)\b", re.IGNORECASE)
PULL_REQUEST_IN_REPOSITORY = re.compile(
    r"\b(?:pull request|pr)\s+#?(?P<number>[1-9]\d*)\s+(?:in|from|on)\s+"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)\b",
    re.IGNORECASE,
)
REPOSITORY_PULL_REQUEST = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)\s+(?:pull request|pr)\s+#?"
    r"(?P<number>[1-9]\d*)(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)
BARE_PULL_REQUEST_NUMBER = re.compile(
    r"\b(?:pull request|pr)\s+#?(?P<number>[1-9]\d*)\b",
    re.IGNORECASE,
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


class GitHubCommandStartError(GitHubError):
    """A GitHub command failed before its process could be started."""


class GitHubPreconditionError(GitHubError):
    """A read-only merge precondition failed before any merge write."""


class GitHubMergeQueueArmedError(GitHubError):
    """A pull request is accurately known to remain queued for merge."""


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
class GitHubRepoCreationPlan:
    owner: str
    slug: str
    visibility: str
    correlation_marker: str

    @property
    def name_with_owner(self) -> str:
        return f"{self.owner}/{self.slug}"


@dataclass(frozen=True)
class GitHubRepoCreationResult:
    repository: GitHubRepository
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

    @property
    def url(self) -> str:
        return f"https://github.com/{self.name_with_owner}/pull/{self.number}"


@dataclass(frozen=True)
class GitHubPullRequestCreationPlan:
    repository: str
    title: str
    body: str
    head: str
    base: str
    head_oid: str
    head_repository: str
    correlation_marker: str


@dataclass(frozen=True)
class GitHubPullRequestCreationResult:
    pull_request: GitHubPullRequest
    url: str
    correlation_marker: str


@dataclass(frozen=True)
class GitHubPullRequestMergeSnapshot:
    repository: str
    number: int
    url: str
    title: str
    default_branch: str
    base_ref: str
    head_ref: str
    head_oid: str
    state: str
    is_draft: bool
    checks: str
    review_decision: str
    mergeable: str
    merge_state_status: str
    is_stack: bool
    stack_parent_url: str | None
    merge_queue_required: bool
    auto_merge_requested: bool

    def serialize(self) -> str:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def deserialize(cls, value: str) -> GitHubPullRequestMergeSnapshot:
        try:
            details = json.loads(value)
            snapshot = cls(**details)
        except (json.JSONDecodeError, TypeError) as exc:
            raise GitHubError("GitHub pull request merge snapshot is invalid") from exc
        return snapshot


@dataclass(frozen=True)
class GitHubPullRequestMergePlan:
    repository: str
    number: int
    url: str
    correlation_marker: str
    snapshot: GitHubPullRequestMergeSnapshot
    method: str


@dataclass(frozen=True)
class GitHubPullRequestMergeResult:
    pull_request: GitHubPullRequest
    url: str
    correlation_marker: str


@dataclass(frozen=True)
class PullRequestMergeIdentity:
    repository: str
    number: int

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repository}/pull/{self.number}"

    @property
    def pull_request(self) -> GitHubPullRequest:
        owner, name = self.repository.split("/", 1)
        return GitHubPullRequest(owner=owner, repository=name, number=self.number)


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

    @staticmethod
    def validate_issue_identity(value: str) -> str:
        match = ISSUE_IDENTITY.fullmatch(value.strip())
        if match is None or match.group("repo") in {".", ".."}:
            raise GitHubError("GitHub issue must be in owner/repository#number form")
        repository = GitHubClient.validate_repository(
            f"{match.group('owner')}/{match.group('repo')}"
        )
        return f"{repository}#{int(match.group('number'))}"

    def _run(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
        check: bool = True,
        write: bool = False,
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
            raise GitHubCommandStartError(
                f"GitHub command failed to start: {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GitHubOperationAmbiguous(
                "GitHub command timed out; its outcome is ambiguous because an "
                f"external side effect may already have occurred: {exc}"
            ) from exc
        if check and process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            message = detail or f"command exited with status {process.returncode}"
            if write:
                raise GitHubOperationAmbiguous(
                    "GitHub write exited without proving its outcome; an external "
                    f"side effect may already have occurred: {message}"
                )
            raise GitHubError(message)
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
            write=True,
            stdin=submitted_body,
        )
        try:
            return self._creation_result(plan, process.stdout)
        except GitHubError as exc:
            raise GitHubOperationAmbiguous(
                "GitHub write completed without a provable result; an external "
                "side effect may already have occurred"
            ) from exc

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

    @staticmethod
    def validate_pull_request_creation_plan(
        plan: GitHubPullRequestCreationPlan,
    ) -> None:
        if not isinstance(plan.repository, str):
            raise GitHubError("GitHub pull request repository must be text")
        repository = GitHubClient.validate_repository(plan.repository)
        if repository != plan.repository:
            raise GitHubError("GitHub pull request plan repository is not normalized")
        if not isinstance(plan.title, str) or not plan.title.strip():
            raise GitHubError("GitHub pull request title must not be empty")
        if len(plan.title) > MAX_ISSUE_TITLE_CHARS:
            raise GitHubError(
                "GitHub pull request title must be at most "
                f"{MAX_ISSUE_TITLE_CHARS} characters"
            )
        if not isinstance(plan.body, str) or not plan.body.strip():
            raise GitHubError("GitHub pull request body must not be empty")
        if len(plan.body) > MAX_ISSUE_BODY_CHARS:
            raise GitHubError(
                "GitHub pull request body must be at most "
                f"{MAX_ISSUE_BODY_CHARS} characters"
            )
        if not isinstance(plan.head, str) or not plan.head.strip():
            raise GitHubError("GitHub pull request head must not be empty")
        if any(character.isspace() for character in plan.head):
            raise GitHubError("GitHub pull request head is invalid")
        if not isinstance(plan.base, str) or not plan.base.strip():
            raise GitHubError("GitHub pull request base must not be empty")
        if any(character.isspace() for character in plan.base):
            raise GitHubError("GitHub pull request base is invalid")
        if re.fullmatch(r"[0-9a-f]{40}", plan.head_oid) is None:
            raise GitHubError("GitHub pull request head OID is invalid")
        head_repository = GitHubClient.validate_repository(plan.head_repository)
        if head_repository != plan.head_repository:
            raise GitHubError("GitHub pull request head repository is not normalized")
        head_owner, _name = head_repository.split("/", 1)
        prefix = f"{head_owner}:"
        if not plan.head.startswith(prefix) or plan.head == prefix:
            raise GitHubError("GitHub pull request head must be owner-qualified")
        if (
            not isinstance(plan.correlation_marker, str)
            or ISSUE_CORRELATION_MARKER.fullmatch(plan.correlation_marker) is None
        ):
            raise GitHubError(
                "GitHub pull request correlation marker must be 32 lowercase "
                "hex characters"
            )
        if plan.correlation_marker in plan.body:
            raise GitHubError(
                "GitHub pull request body must not contain its correlation marker"
            )

    @staticmethod
    def _pull_request_creation_result(
        plan: GitHubPullRequestCreationPlan, url: object
    ) -> GitHubPullRequestCreationResult:
        value = str(url or "").strip()
        pull_request = github_pull_request_from_url(value)
        if pull_request is None or value != pull_request.url:
            raise GitHubError("GitHub returned a non-canonical pull request URL")
        if pull_request.name_with_owner.casefold() != plan.repository.casefold():
            raise GitHubError(
                "GitHub created a pull request in an unexpected repository"
            )
        return GitHubPullRequestCreationResult(
            pull_request, value, plan.correlation_marker
        )

    def submit_pull_request_creation(
        self,
        plan: GitHubPullRequestCreationPlan,
        *,
        confirmed: bool,
    ) -> GitHubPullRequestCreationResult:
        if not confirmed:
            raise GitHubError(
                "GitHub pull request creation requires explicit confirmation"
            )
        self.validate_pull_request_creation_plan(plan)
        submitted_body = (
            f"{plan.body.rstrip()}\n\n{self._issue_marker(plan.correlation_marker)}\n"
        )
        process = self._run(
            [
                self.gh_executable,
                "pr",
                "create",
                "--repo",
                plan.repository,
                "--title",
                plan.title,
                "--head",
                plan.head,
                "--base",
                plan.base,
                "--body-file",
                "-",
            ],
            timeout=30,
            write=True,
            stdin=submitted_body,
        )
        try:
            submitted = self._pull_request_creation_result(plan, process.stdout)
            observed = self.observe_pull_request_creation(plan)
            if (
                observed is None
                or observed.pull_request.number != submitted.pull_request.number
            ):
                raise GitHubError("created pull request snapshot could not be verified")
            return observed
        except GitHubError as exc:
            raise GitHubOperationAmbiguous(
                "GitHub write completed without a provable pull request result; "
                "an external side effect may already have occurred"
            ) from exc

    def observe_pull_request_creation(
        self, plan: GitHubPullRequestCreationPlan
    ) -> GitHubPullRequestCreationResult | None:
        self.validate_pull_request_creation_plan(plan)
        process = self._run(
            [
                self.gh_executable,
                "pr",
                "list",
                "--repo",
                plan.repository,
                "--state",
                "all",
                "--limit",
                str(ISSUE_OBSERVATION_LIMIT),
                "--json",
                "number,url,body,baseRefName,headRefName,headRefOid,headRepository",
            ],
            timeout=15,
        )
        try:
            values = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError(
                "GitHub returned malformed recent pull request metadata"
            ) from exc
        if not isinstance(values, list):
            raise GitHubError("GitHub returned malformed recent pull request metadata")
        marker = self._issue_marker(plan.correlation_marker)
        matches: list[GitHubPullRequestCreationResult] = []
        for value in values:
            if not isinstance(value, dict) or marker not in str(
                value.get("body") or ""
            ):
                continue
            head_repository = value.get("headRepository")
            head_repository_name = (
                str(head_repository.get("nameWithOwner") or "")
                if isinstance(head_repository, dict)
                else ""
            )
            expected_branch = plan.head.split(":", 1)[1]
            if (
                str(value.get("baseRefName") or "") != plan.base
                or str(value.get("headRefName") or "") != expected_branch
                or str(value.get("headRefOid") or "").lower() != plan.head_oid
                or head_repository_name.casefold() != plan.head_repository.casefold()
            ):
                continue
            result = self._pull_request_creation_result(plan, value.get("url"))
            number = value.get("number")
            if not isinstance(number, int) or number != result.pull_request.number:
                raise GitHubError(
                    "GitHub returned malformed recent pull request metadata"
                )
            matches.append(result)
        if len(matches) > 1:
            raise GitHubError("GitHub pull request correlation marker is not unique")
        return matches[0] if matches else None

    @staticmethod
    def validate_pull_request_merge_plan(plan: GitHubPullRequestMergePlan) -> None:
        if not isinstance(plan.repository, str):
            raise GitHubError("GitHub pull request repository must be text")
        repository = GitHubClient.validate_repository(plan.repository)
        if repository != plan.repository:
            raise GitHubError(
                "GitHub pull request merge plan repository is not normalized"
            )
        if not isinstance(plan.number, int) or plan.number < 1:
            raise GitHubError("GitHub pull request number must be positive")
        expected = f"https://github.com/{repository}/pull/{plan.number}"
        if not isinstance(plan.url, str) or plan.url != expected:
            raise GitHubError("GitHub pull request merge URL is not canonical")
        if (
            not isinstance(plan.correlation_marker, str)
            or ISSUE_CORRELATION_MARKER.fullmatch(plan.correlation_marker) is None
        ):
            raise GitHubError(
                "GitHub pull request merge correlation marker must be 32 lowercase "
                "hex characters"
            )
        if plan.method not in {"merge", "squash", "rebase"}:
            raise GitHubError("GitHub pull request merge method is invalid")
        if not isinstance(plan.snapshot, GitHubPullRequestMergeSnapshot):
            raise GitHubError("GitHub pull request merge snapshot is invalid")
        if (
            plan.snapshot.repository != plan.repository
            or plan.snapshot.number != plan.number
            or plan.snapshot.url != plan.url
        ):
            raise GitHubError("GitHub pull request merge snapshot identity changed")
        snapshot = plan.snapshot
        required_text = (
            snapshot.title,
            snapshot.default_branch,
            snapshot.base_ref,
            snapshot.head_ref,
            snapshot.state,
            snapshot.checks,
            snapshot.mergeable,
            snapshot.merge_state_status,
        )
        if not all(isinstance(value, str) and value for value in required_text):
            raise GitHubError("GitHub pull request merge snapshot is malformed")
        if (
            not isinstance(snapshot.head_oid, str)
            or re.fullmatch(r"[0-9a-f]{40}", snapshot.head_oid) is None
            or snapshot.checks not in {"passing", "pending", "failing"}
            or snapshot.review_decision
            not in {"", "APPROVED", "REVIEW_REQUIRED", "CHANGES_REQUESTED"}
            or type(snapshot.is_draft) is not bool
            or type(snapshot.is_stack) is not bool
            or type(snapshot.merge_queue_required) is not bool
            or type(snapshot.auto_merge_requested) is not bool
            or (
                snapshot.stack_parent_url is not None
                and not isinstance(snapshot.stack_parent_url, str)
            )
        ):
            raise GitHubError("GitHub pull request merge snapshot is malformed")

    @staticmethod
    def _check_state(details: dict[str, object]) -> str:
        checks = details.get("statusCheckRollup")
        if not isinstance(checks, list):
            raise GitHubError("GitHub returned malformed pull request checks")
        if not checks:
            return "passing"
        pending = False
        for check in checks:
            if not isinstance(check, dict):
                raise GitHubError("GitHub returned malformed pull request checks")
            status = str(check.get("status") or "").upper()
            conclusion = str(check.get("conclusion") or "").upper()
            if status != "COMPLETED" or not conclusion:
                pending = True
                continue
            if conclusion not in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
                return "failing"
        return "pending" if pending else "passing"

    @staticmethod
    def _merge_queue_required(rulesets: object) -> bool:
        if not isinstance(rulesets, list) or not all(
            isinstance(rule, dict) for rule in rulesets
        ):
            raise GitHubError("GitHub returned malformed repository branch rules")
        return any(rule.get("type") == "merge_queue" for rule in rulesets)

    def inspect_pull_request_merge(
        self, repository: str, number: int
    ) -> GitHubPullRequestMergeSnapshot:
        repository = self.validate_repository(repository)
        pull_process = self._run(
            [
                self.gh_executable,
                "pr",
                "view",
                str(number),
                "--repo",
                repository,
                "--json",
                (
                    "number,url,title,state,isDraft,headRefName,headRefOid,"
                    "baseRefName,mergeable,mergeStateStatus,reviewDecision,"
                    "statusCheckRollup,autoMergeRequest"
                ),
            ],
            timeout=15,
        )
        repo_process = self._run(
            [
                self.gh_executable,
                "repo",
                "view",
                repository,
                "--json",
                "nameWithOwner,defaultBranchRef",
            ],
            timeout=15,
        )
        try:
            details = json.loads(pull_process.stdout)
            repo_details = json.loads(repo_process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError(
                "GitHub returned malformed pull request metadata"
            ) from exc
        if not isinstance(details, dict) or not isinstance(repo_details, dict):
            raise GitHubError("GitHub returned malformed pull request metadata")
        canonical = str(repo_details.get("nameWithOwner") or "")
        default = repo_details.get("defaultBranchRef")
        default_branch = (
            str(default.get("name") or "") if isinstance(default, dict) else ""
        )
        base_ref = str(details.get("baseRefName") or "")
        head_ref = str(details.get("headRefName") or "")
        head_oid = str(details.get("headRefOid") or "").casefold()
        url = str(details.get("url") or "")
        if (
            canonical.casefold() != repository.casefold()
            or details.get("number") != number
            or url != f"https://github.com/{canonical}/pull/{number}"
            or not default_branch
            or not base_ref
            or not head_ref
            or re.fullmatch(r"[0-9a-f]{40}", head_oid) is None
        ):
            raise GitHubError("GitHub returned malformed pull request metadata")
        rules_process = self._run(
            [
                self.gh_executable,
                "api",
                f"repos/{repository}/rules/branches/{quote(base_ref, safe='')}",
            ],
            timeout=15,
        )
        try:
            rulesets = json.loads(rules_process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub returned malformed branch rules") from exc
        stack_parent_url: str | None = None
        is_stack = False
        if base_ref != default_branch:
            stack_process = self._run(
                [
                    self.gh_executable,
                    "pr",
                    "list",
                    "--repo",
                    canonical,
                    "--state",
                    "open",
                    "--head",
                    base_ref,
                    "--json",
                    "number,url,headRefName,baseRefName",
                ],
                timeout=15,
            )
            try:
                stack_values = json.loads(stack_process.stdout)
            except json.JSONDecodeError as exc:
                raise GitHubError("GitHub returned malformed stack metadata") from exc
            if not isinstance(stack_values, list):
                raise GitHubError("GitHub returned malformed stack metadata")
            parents = [
                value
                for value in stack_values
                if isinstance(value, dict) and value.get("headRefName") == base_ref
            ]
            if parents:
                is_stack = True
                stack_parent_url = str(parents[0].get("url") or "") or None
        return GitHubPullRequestMergeSnapshot(
            repository=canonical,
            number=number,
            url=url,
            title=str(details.get("title") or ""),
            default_branch=default_branch,
            base_ref=base_ref,
            head_ref=head_ref,
            head_oid=head_oid,
            state=str(details.get("state") or "").upper(),
            is_draft=details.get("isDraft") is True,
            checks=self._check_state(details),
            review_decision=str(details.get("reviewDecision") or "").upper(),
            mergeable=str(details.get("mergeable") or "").upper(),
            merge_state_status=str(details.get("mergeStateStatus") or "").upper(),
            is_stack=is_stack,
            stack_parent_url=stack_parent_url,
            merge_queue_required=self._merge_queue_required(rulesets),
            auto_merge_requested=details.get("autoMergeRequest") is not None,
        )

    @staticmethod
    def validate_pull_request_merge_eligibility(
        snapshot: GitHubPullRequestMergeSnapshot,
    ) -> None:
        if snapshot.state != "OPEN":
            raise GitHubPreconditionError("GitHub pull request is not open")
        if snapshot.is_draft:
            raise GitHubPreconditionError("GitHub pull request is still a draft")
        if snapshot.checks != "passing":
            raise GitHubPreconditionError(
                f"GitHub pull request checks are {snapshot.checks}"
            )
        if snapshot.review_decision in {"REVIEW_REQUIRED", "CHANGES_REQUESTED"}:
            raise GitHubPreconditionError(
                "GitHub pull request does not have eligible reviews"
            )
        if snapshot.mergeable != "MERGEABLE":
            raise GitHubPreconditionError("GitHub pull request is not mergeable")
        if snapshot.merge_state_status not in {"CLEAN", "HAS_HOOKS"}:
            raise GitHubPreconditionError(
                "GitHub pull request merge state is not eligible"
            )
        if snapshot.base_ref != snapshot.default_branch:
            if snapshot.is_stack:
                detail = (
                    f" as part of the stack rooted at {snapshot.stack_parent_url}"
                    if snapshot.stack_parent_url
                    else " as part of a GitHub stack"
                )
                raise GitHubPreconditionError(
                    f"GitHub pull request targets non-default branch "
                    f"{snapshot.base_ref}{detail}; merging it would not integrate "
                    f"the change into {snapshot.default_branch}"
                )
            raise GitHubPreconditionError(
                f"GitHub pull request targets non-default branch {snapshot.base_ref}; "
                f"merging it would not integrate the change into "
                f"{snapshot.default_branch}"
            )
        if snapshot.merge_queue_required or snapshot.auto_merge_requested:
            raise GitHubPreconditionError(
                "GitHub pull request uses a merge queue; queue enrollment requires "
                "a separate explicit authorization"
            )

    @staticmethod
    def _pull_request_merge_result(
        plan: GitHubPullRequestMergePlan, url: object
    ) -> GitHubPullRequestMergeResult:
        value = str(url or "").strip()
        pull_request = github_pull_request_from_url(value)
        if pull_request is None or value != pull_request.url:
            raise GitHubError("GitHub returned a non-canonical pull request URL")
        if pull_request.name_with_owner.casefold() != plan.repository.casefold():
            raise GitHubError(
                "GitHub merged a pull request in an unexpected repository"
            )
        if pull_request.number != plan.number:
            raise GitHubError("GitHub merged an unexpected pull request")
        return GitHubPullRequestMergeResult(
            pull_request, value, plan.correlation_marker
        )

    def submit_pull_request_merge(
        self,
        plan: GitHubPullRequestMergePlan,
        *,
        confirmed: bool,
    ) -> GitHubPullRequestMergeResult:
        if not confirmed:
            raise GitHubError(
                "GitHub pull request merge requires explicit confirmation"
            )
        self.validate_pull_request_merge_plan(plan)
        current = self.inspect_pull_request_merge(plan.repository, plan.number)
        if current != plan.snapshot:
            raise GitHubPreconditionError(
                "GitHub pull request state changed after confirmation; confirm the "
                "current state before merging"
            )
        self.validate_pull_request_merge_eligibility(current)
        self._run(
            [
                self.gh_executable,
                "pr",
                "merge",
                str(plan.number),
                "--repo",
                plan.repository,
                f"--{plan.method}",
                "--match-head-commit",
                plan.snapshot.head_oid,
            ],
            timeout=30,
            write=True,
        )
        try:
            observed = self.observe_pull_request_merge(plan)
        except GitHubMergeQueueArmedError:
            raise
        except GitHubError as exc:
            raise GitHubOperationAmbiguous(
                "GitHub write completed without a provable pull request merge; "
                "an external side effect may already have occurred"
            ) from exc
        if observed is None:
            raise GitHubOperationAmbiguous(
                "GitHub write completed without a provable pull request merge; "
                "an external side effect may already have occurred"
            )
        return observed

    def observe_pull_request_merge(
        self, plan: GitHubPullRequestMergePlan
    ) -> GitHubPullRequestMergeResult | None:
        self.validate_pull_request_merge_plan(plan)
        process = self._run(
            [
                self.gh_executable,
                "pr",
                "view",
                str(plan.number),
                "--repo",
                plan.repository,
                "--json",
                "number,url,state,mergedAt,autoMergeRequest,headRefOid",
            ],
            timeout=15,
            check=False,
        )
        if process.returncode:
            return None
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError(
                "GitHub returned malformed pull request merge metadata"
            ) from exc
        if not isinstance(value, dict):
            raise GitHubError("GitHub returned malformed pull request merge metadata")
        state = str(value.get("state") or "").casefold()
        merged_at = value.get("mergedAt")
        if value.get("autoMergeRequest") is not None and not merged_at:
            raise GitHubMergeQueueArmedError(
                "GitHub pull request is queued for merge and remains armed"
            )
        if state != "merged" and not merged_at:
            return None
        result = self._pull_request_merge_result(plan, value.get("url") or plan.url)
        number = value.get("number")
        if not isinstance(number, int) or number != result.pull_request.number:
            raise GitHubError("GitHub returned malformed pull request merge metadata")
        return result

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

    def lookup_repository(self, repository: str) -> GitHubRepository | None:
        return self._repo_view(self.validate_repository(repository), required=False)

    @staticmethod
    def validate_repo_creation_plan(plan: GitHubRepoCreationPlan) -> None:
        if not isinstance(plan.owner, str) or not plan.owner.strip():
            raise GitHubError("GitHub repository owner must be text")
        if not isinstance(plan.slug, str) or not plan.slug.strip():
            raise GitHubError("GitHub repository slug must be text")
        GitHubClient.validate_repository(plan.name_with_owner)
        if plan.visibility not in {"private", "public"}:
            raise GitHubError("GitHub repository visibility must be private or public")
        if (
            not isinstance(plan.correlation_marker, str)
            or ISSUE_CORRELATION_MARKER.fullmatch(plan.correlation_marker) is None
        ):
            raise GitHubError(
                "GitHub repository correlation marker must be 32 lowercase hex characters"
            )

    @staticmethod
    def _repo_marker(correlation_marker: str) -> str:
        return f"local-voice-harness-correlation:{correlation_marker}"

    def plan_repository_creation(
        self,
        owner: str,
        slug: str,
        visibility: str,
        *,
        correlation_marker: str | None = None,
    ) -> GitHubRepoCreationPlan:
        plan = GitHubRepoCreationPlan(
            owner=owner.strip(),
            slug=slug.strip(),
            visibility=visibility,
            correlation_marker=correlation_marker or secrets.token_hex(16),
        )
        self.validate_repo_creation_plan(plan)
        return plan

    def observe_repository_creation(
        self, plan: GitHubRepoCreationPlan
    ) -> GitHubRepoCreationResult | None:
        self.validate_repo_creation_plan(plan)
        process = self._run(
            [
                self.gh_executable,
                "repo",
                "view",
                plan.name_with_owner,
                "--json",
                "nameWithOwner,url,isPrivate,description",
            ],
            timeout=15,
            check=False,
        )
        if process.returncode:
            detail = (process.stderr.strip() or process.stdout.strip()).casefold()
            if any(
                marker in detail
                for marker in (
                    "could not resolve to a repository",
                    "repository not found",
                    "http 404",
                    "status 404",
                )
            ):
                return None
            raise GitHubError(
                detail or f"could not determine whether {plan.name_with_owner} exists"
            )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub returned malformed repository metadata") from exc
        if not isinstance(value, dict):
            raise GitHubError("GitHub returned malformed repository metadata")
        description = str(value.get("description") or "")
        if self._repo_marker(plan.correlation_marker) not in description:
            return None
        name = self.validate_repository(str(value.get("nameWithOwner") or ""))
        if name.casefold() != plan.name_with_owner.casefold():
            raise GitHubError("GitHub created a repository with an unexpected name")
        source = GitHubRepository(
            name_with_owner=name,
            url=str(value.get("url") or f"https://github.com/{name}"),
            is_private=bool(value.get("isPrivate")),
            default_branch="",
        )
        expected_private = plan.visibility == "private"
        if source.is_private != expected_private:
            raise GitHubError("GitHub created a repository with unexpected visibility")
        return GitHubRepoCreationResult(
            source,
            source.url,
            plan.correlation_marker,
        )

    def submit_repository_creation(
        self,
        plan: GitHubRepoCreationPlan,
        *,
        confirmed: bool,
    ) -> GitHubRepoCreationResult:
        if not confirmed:
            raise GitHubError(
                "GitHub repository creation requires explicit confirmation"
            )
        self.validate_repo_creation_plan(plan)
        visibility_flag = "--private" if plan.visibility == "private" else "--public"
        self._run(
            [
                self.gh_executable,
                "repo",
                "create",
                plan.name_with_owner,
                visibility_flag,
                "--description",
                self._repo_marker(plan.correlation_marker),
                "--clone=false",
            ],
            timeout=60,
            write=True,
        )
        result = self.observe_repository_creation(plan)
        if result is None:
            raise GitHubOperationAmbiguous(
                "GitHub write completed without a provable repository result; "
                "an external side effect may already have occurred"
            )
        return result

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
                    "number,title,state,author,labels,body,comments,url,updatedAt",
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

    def list_organizations(self) -> tuple[str, ...]:
        process = self._run(
            [self.gh_executable, "org", "list", "--limit", "100"],
            timeout=15,
        )
        organizations: list[str] = []
        for line in process.stdout.splitlines():
            name = line.strip()
            if not name:
                continue
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", name):
                raise GitHubError("GitHub CLI returned a malformed organization")
            organizations.append(name)
        return tuple(organizations)

    def require_organization_membership(self, organization: str) -> str:
        """Return the canonical listed membership identity, not write authorization."""

        candidate = organization.strip()
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", candidate):
            raise GitHubError("GitHub organization must be a valid login")
        organizations = self.list_organizations()
        for name in organizations:
            if name.casefold() == candidate.casefold():
                return name
        raise GitHubError(f"authenticated user is not a listed member of {candidate}")

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
            write=True,
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

    def observe_clone(self, source: str) -> Path | None:
        """Return the checkout if a prior clone already landed. Never clones."""

        repository = github_repository_from_url(source)
        if repository is None:
            try:
                repository = self.validate_repository(source)
            except GitHubError:
                repository = None
        if repository is not None:
            owner, name = repository.split("/", 1)
            destination = self.local_git.clone_root / owner / name
            if not destination.exists():
                return None
            try:
                expected = ExpectedRemote.from_url(f"https://github.com/{repository}")
                self.local_git.verify_checkout(
                    destination,
                    expected,
                    expected_label=repository,
                )
            except LocalGitError:
                return None
            return destination.resolve()
        name = source.strip().rstrip("/").rsplit("/", 1)[-1]
        if name.casefold().endswith(".git"):
            name = name[:-4]
        if not name or name.startswith("."):
            return None
        destination = self.local_git.clone_root / name
        if not destination.exists():
            return None
        try:
            self.local_git.verify_checkout(destination)
        except LocalGitError:
            return None
        return destination.resolve()

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
    match = (
        ISSUE_REFERENCE.search(text)
        or REPOSITORY_ISSUE.search(text)
        or ISSUE_IN_REPOSITORY.search(text)
    )
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


def _merge_identity(repository: str, number: int) -> PullRequestMergeIdentity | None:
    try:
        normalized = GitHubClient.validate_repository(repository)
    except GitHubError:
        return None
    if number < 1:
        return None
    return PullRequestMergeIdentity(normalized, number)


def _identity_from_groups(match: re.Match[str]) -> PullRequestMergeIdentity | None:
    return _merge_identity(
        f"{match.group('owner')}/{match.group('repo')}",
        int(match.group("number")),
    )


def pull_request_identity_from_text(text: str) -> PullRequestMergeIdentity | None:
    """Return one explicit trusted PR identity, or None if absent or ambiguous."""

    identities: list[PullRequestMergeIdentity] = []
    seen: set[tuple[str, int]] = set()

    def add(identity: PullRequestMergeIdentity | None) -> None:
        if identity is None:
            return
        key = (identity.repository.casefold(), identity.number)
        if key in seen:
            return
        seen.add(key)
        identities.append(identity)

    for match in PULL_REQUEST_URL_IN_TEXT.finditer(text):
        pull_request = github_pull_request_from_url(match.group(0))
        if pull_request is not None:
            add(_merge_identity(pull_request.name_with_owner, pull_request.number))
    for pattern in (PULL_REQUEST_IN_REPOSITORY, REPOSITORY_PULL_REQUEST):
        for match in pattern.finditer(text):
            add(_identity_from_groups(match))
    if PULL_REQUEST_LANGUAGE.search(text) is not None:
        for match in ISSUE_REFERENCE.finditer(text):
            add(_identity_from_groups(match))
    return identities[0] if len(identities) == 1 else None


def bare_pull_request_number(text: str) -> int | None:
    matches = [
        int(match.group("number")) for match in BARE_PULL_REQUEST_NUMBER.finditer(text)
    ]
    return matches[0] if len(set(matches)) == 1 else None


def resolve_pull_request_merge_identity(
    *,
    utterance: str,
    focused_repository: str | None = None,
    focused_number: int | None = None,
    conversation_repository: str | None = None,
    conversation_number: int | None = None,
) -> PullRequestMergeIdentity | None:
    """Return the unique merge identity, or None when it is missing or ambiguous."""

    explicit = pull_request_identity_from_text(utterance)
    if explicit is not None:
        return explicit
    focused = (
        _merge_identity(focused_repository, focused_number)
        if focused_repository and focused_number
        else None
    )
    conversation = (
        _merge_identity(conversation_repository, conversation_number)
        if conversation_repository and conversation_number
        else None
    )
    number = bare_pull_request_number(utterance)
    candidates: list[PullRequestMergeIdentity] = []
    if number is not None:
        if focused_repository:
            identity = _merge_identity(focused_repository, number)
            if identity is not None:
                candidates.append(identity)
        if conversation_repository:
            identity = _merge_identity(conversation_repository, number)
            if identity is not None:
                candidates.append(identity)
    else:
        candidates.extend(
            identity for identity in (focused, conversation) if identity is not None
        )
    unique: list[PullRequestMergeIdentity] = []
    seen: set[tuple[str, int]] = set()
    for identity in candidates:
        key = (identity.repository.casefold(), identity.number)
        if key in seen:
            continue
        seen.add(key)
        unique.append(identity)
    return unique[0] if len(unique) == 1 else None


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
    "pull_request_creation": {
        "requested": "github_pr_create_requested",
        "confirmed": "github_pr_create_confirmed",
        "title": "github_pr_create_title",
        "body": "github_pr_create_body",
        "marker": "github_pr_create_marker",
        "commit_subject": "github_pr_create_commit_subject",
        "base": "github_pr_create_base",
        "head_oid": "github_pr_create_head_oid",
        "published_head_oid": "github_pr_create_published_head_oid",
        "head_repository": "github_pr_create_head_repository",
        "checkout_origin": "github_pr_create_checkout_origin",
        "status_digest": "github_pr_create_status_digest",
        "operation_state": "github_pr_create_operation_state",
        "created_number": "github_pr_created_number",
        "created_url": "github_pr_created_url",
    },
    "pull_request_merge": {
        "requested": "github_pr_merge_requested",
        "confirmed": "github_pr_merge_confirmed",
        "number": "github_pr_merge_number",
        "url": "github_pr_merge_url",
        "marker": "github_pr_merge_marker",
        "snapshot": "github_pr_merge_snapshot",
        "method": "github_pr_merge_method",
        "operation_state": "github_pr_merge_operation_state",
    },
    "repo_creation": {
        "requested": "github_repo_create_requested",
        "org_requested": "github_repo_create_org_requested",
        "owner": "github_repo_create_owner",
        "confirmed": "github_repo_create_confirmed",
        "visibility": "github_repo_create_visibility",
        "marker": "github_repo_create_marker",
        "operation_state": "github_repo_create_operation_state",
        "created_url": "github_repo_created_url",
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


_GITHUB_STATE_DERIVED_FIELDS = frozenset(
    {
        "github_issue_created_number",
        "github_issue_created_url",
        "pull_request_branch",
    }
)


def dump_github_provider_state(state: Mapping[str, object]) -> dict[str, object]:
    """Serialize GitHub provider fields as clearly nested durable v1 state."""

    flat = load_github_provider_state(state)
    serialized: dict[str, object] = {"version": _GITHUB_STATE_VERSION}
    for section_name, fields in _GITHUB_STATE_SECTIONS.items():
        section = {
            name: flat[field]
            for name, field in fields.items()
            if field in flat and field not in _GITHUB_STATE_DERIVED_FIELDS
        }
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

    def repository_for_checkout(self, checkout: Path) -> str:
        return self._client.repository_for_checkout(checkout)

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

    def ticket_snapshot(self, reference: str) -> TicketSnapshot:
        """Fetch and identity-check the current GitHub issue fields."""

        match = ISSUE_REFERENCE.fullmatch(reference.strip())
        if match is None:
            raise GitHubError("GitHub ticket snapshot requires an exact issue identity")
        repository = GitHubClient.validate_repository(
            f"{match.group('owner')}/{match.group('repo')}"
        )
        owner, name = repository.split("/", 1)
        issue = GitHubIssue(owner, name, int(match.group("number")))
        details = self.resolve_issue(issue)
        number = details.get("number")
        url = str(details.get("url") or "").strip()
        title = details.get("title")
        body = details.get("body")
        revision = details.get("updatedAt")
        state = details.get("state")
        if (
            number != issue.number
            or url != issue.url
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(body, str)
            or not isinstance(revision, str)
            or not revision.strip()
            or not isinstance(state, str)
            or len(title.strip()) > MAX_SNAPSHOT_TITLE_CHARS
            or len(body) > MAX_SNAPSHOT_BODY_CHARS
            or len(revision.strip()) > MAX_SNAPSHOT_REVISION_CHARS
        ):
            raise GitHubError("GitHub returned an invalid ticket snapshot")
        return TicketSnapshot(
            provider=self.name,
            identity=issue.reference,
            provider_id=url,
            title=title.strip(),
            body=body.strip(),
            revision=revision.strip(),
            url=url,
            state=state.strip(),
        )

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

    def plan_pull_request_creation(
        self,
        repository: str,
        title: str,
        body: str,
        head: str,
        base: str,
        head_oid: str,
        head_repository: str,
        *,
        correlation_marker: str | None = None,
    ) -> GitHubPullRequestCreationPlan:
        if not isinstance(repository, str):
            raise GitHubError("GitHub pull request repository must be text")
        if not isinstance(title, str):
            raise GitHubError("GitHub pull request title must be text")
        if not isinstance(body, str):
            raise GitHubError("GitHub pull request body must be text")
        if not isinstance(head, str):
            raise GitHubError("GitHub pull request head must be text")
        plan = GitHubPullRequestCreationPlan(
            repository=GitHubClient.validate_repository(repository),
            title=title.strip(),
            body=body.strip(),
            head=head.strip(),
            base=base.strip(),
            head_oid=head_oid.strip().lower(),
            head_repository=GitHubClient.validate_repository(head_repository),
            correlation_marker=correlation_marker or secrets.token_hex(16),
        )
        self.validate_pull_request_creation_plan(plan)
        return plan

    @staticmethod
    def validate_pull_request_creation_plan(
        plan: GitHubPullRequestCreationPlan,
    ) -> None:
        GitHubClient.validate_pull_request_creation_plan(plan)

    def observe_pull_request_creation(
        self, plan: GitHubPullRequestCreationPlan
    ) -> GitHubPullRequestCreationResult | None:
        self.validate_pull_request_creation_plan(plan)
        return self._client.observe_pull_request_creation(plan)

    def submit_pull_request_creation(
        self,
        plan: GitHubPullRequestCreationPlan,
        *,
        confirmed: bool,
    ) -> GitHubPullRequestCreationResult:
        if not confirmed:
            raise GitHubError(
                "GitHub pull request creation requires explicit confirmation"
            )
        self.validate_pull_request_creation_plan(plan)
        return self._client.submit_pull_request_creation(plan, confirmed=True)

    def plan_pull_request_merge(
        self,
        repository: str,
        number: int,
        *,
        correlation_marker: str | None = None,
        snapshot: str | None = None,
        method: str = "squash",
    ) -> GitHubPullRequestMergePlan:
        if not isinstance(repository, str):
            raise GitHubError("GitHub pull request repository must be text")
        if not isinstance(number, int):
            raise GitHubError("GitHub pull request number must be an integer")
        normalized = GitHubClient.validate_repository(repository)
        observed = (
            GitHubPullRequestMergeSnapshot.deserialize(snapshot)
            if snapshot is not None
            else self._client.inspect_pull_request_merge(normalized, number)
        )
        plan = GitHubPullRequestMergePlan(
            repository=normalized,
            number=number,
            url=f"https://github.com/{normalized}/pull/{number}",
            correlation_marker=correlation_marker or secrets.token_hex(16),
            snapshot=observed,
            method=method,
        )
        self.validate_pull_request_merge_plan(plan)
        GitHubClient.validate_pull_request_merge_eligibility(observed)
        return plan

    @staticmethod
    def validate_pull_request_merge_plan(plan: GitHubPullRequestMergePlan) -> None:
        GitHubClient.validate_pull_request_merge_plan(plan)

    def observe_pull_request_merge(
        self, plan: GitHubPullRequestMergePlan
    ) -> GitHubPullRequestMergeResult | None:
        self.validate_pull_request_merge_plan(plan)
        return self._client.observe_pull_request_merge(plan)

    def confirm_pull_request_merge_state(
        self, plan: GitHubPullRequestMergePlan
    ) -> None:
        """Re-fetch and reject a stale or newly ineligible confirmation."""

        self.validate_pull_request_merge_plan(plan)
        current = self._client.inspect_pull_request_merge(plan.repository, plan.number)
        if current != plan.snapshot:
            raise GitHubPreconditionError(
                "GitHub pull request state changed after confirmation; confirm the "
                "current state before merging"
            )
        GitHubClient.validate_pull_request_merge_eligibility(current)

    def submit_pull_request_merge(
        self,
        plan: GitHubPullRequestMergePlan,
        *,
        confirmed: bool,
    ) -> GitHubPullRequestMergeResult:
        if not confirmed:
            raise GitHubError(
                "GitHub pull request merge requires explicit confirmation"
            )
        self.validate_pull_request_merge_plan(plan)
        return self._client.submit_pull_request_merge(plan, confirmed=True)

    def plan_repository_creation(
        self,
        owner: str,
        slug: str,
        visibility: str,
        *,
        correlation_marker: str | None = None,
    ) -> GitHubRepoCreationPlan:
        plan = GitHubRepoCreationPlan(
            owner=owner.strip(),
            slug=slug.strip(),
            visibility=visibility,
            correlation_marker=correlation_marker or secrets.token_hex(16),
        )
        GitHubClient.validate_repo_creation_plan(plan)
        return plan

    def authenticated_login(self) -> str:
        return self._client.authenticated_login()

    def lookup_repository(self, repository: str) -> GitHubRepository | None:
        return self._client.lookup_repository(repository)

    def observe_repository_creation(
        self, plan: GitHubRepoCreationPlan
    ) -> GitHubRepoCreationResult | None:
        return self._client.observe_repository_creation(plan)

    def submit_repository_creation(
        self,
        plan: GitHubRepoCreationPlan,
        *,
        confirmed: bool,
    ) -> GitHubRepoCreationResult:
        if not confirmed:
            raise GitHubError(
                "GitHub repository creation requires explicit confirmation"
            )
        GitHubClient.validate_repo_creation_plan(plan)
        return self._client.submit_repository_creation(plan, confirmed=True)

    def list_organizations(self) -> tuple[str, ...]:
        return self._client.list_organizations()

    def require_organization_membership(self, organization: str) -> str:
        return self._client.require_organization_membership(organization)

    def materialize_repository(
        self,
        source: GitHubRepository,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        return self._client.ensure_repository_clone(source, checkpoint=checkpoint)

    def observe_repository_materialization(
        self, source: GitHubRepository
    ) -> Path | None:
        owner, name = source.name_with_owner.split("/", 1)
        try:
            return self._client.local_git.observe_materialized(
                Path(owner) / name,
                expected=self._client._expected_remote(source),
                expected_label=source.name_with_owner,
            )
        except LocalGitError as exc:
            self._client._raise_local_git_error(exc)

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

    def owns_issue_reference(self, reference: str) -> bool:
        try:
            GitHubClient.validate_issue_identity(reference)
        except GitHubError:
            return False
        return True

    def canonicalize_issue_reference(self, reference: str) -> str:
        return GitHubClient.validate_issue_identity(reference)

    def owns_repository_reference(self, reference: str) -> bool:
        try:
            GitHubClient.validate_repository(reference)
        except GitHubError:
            return False
        return True

    def canonicalize_repository_reference(self, reference: str) -> str:
        return GitHubClient.validate_repository(reference)
