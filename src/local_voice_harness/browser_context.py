from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit

from .context_fragment import ContextFragment
from .context_providers import capture_context
from .desktop import DesktopError, get_desktop
from .focused_app_context import focused_app_context
from .integrations.github import GitHubClient, GitHubError, GitHubIssue

FIREFOX_CLASSES = {"firefox", "org.mozilla.firefox"}
GITHUB_HOSTS = {"github.com", "www.github.com"}
LINEAR_HOSTS = {"linear.app", "www.linear.app"}
ISSUE_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/(?P<number>\d+)/?$"
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
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/pull/(?P<number>\d+)"
    r"(?:/.*)?$"
)
REPOSITORY_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:/.*)?$"
)
LINEAR_ISSUE_PATH = re.compile(
    r"^/[A-Za-z0-9][A-Za-z0-9-]*/issue/"
    r"(?P<identifier>[A-Za-z][A-Za-z0-9]+-\d+)(?:/[^/?#]+)?/?$",
    re.IGNORECASE,
)
MAX_BODY_CHARS = 5_000
MAX_COMMENT_CHARS = 800
MAX_COMMENTS = 5


@dataclass(frozen=True)
class GitHubPullRequest:
    owner: str
    repository: str
    number: int


@dataclass(frozen=True)
class RequestContext:
    text: str
    focused_repository: str | None = None
    focused_issue: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_context: str | None = None
    github_pull_request: int | None = None
    linear_issue: str | None = None
    focused_app_class: str | None = None
    focused_app_context: str | None = None
    focused_app_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinearIssue:
    identifier: str


class GitHubContext(str):
    github_repository: str | None
    github_issue: int | None
    github_issue_context: str | None
    github_pull_request: int | None

    def __new__(
        cls,
        value: str,
        *,
        github_repository: str | None = None,
        github_issue: int | None = None,
        github_issue_context: str | None = None,
        github_pull_request: int | None = None,
    ) -> GitHubContext:
        instance = super().__new__(cls, value)
        instance.github_repository = github_repository
        instance.github_issue = github_issue
        instance.github_issue_context = github_issue_context
        instance.github_pull_request = github_pull_request
        return instance


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
    timeout: float = 5,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def focused_firefox_url() -> str | None:
    """Capture the focused Firefox tab URL without leaving the address bar open."""
    desktop = get_desktop()
    if desktop is None or not desktop.has_clipboard():
        return None
    window = desktop.active_window()
    if window is None or window.window_class not in FIREFOX_CLASSES:
        return None

    clipboard_existed, previous_clipboard = desktop.read_clipboard()
    captured = ""
    try:
        if not desktop.send_key("ctrl+l", window=window):
            return None
        time.sleep(0.05)
        if desktop.active_window() != window or not desktop.send_key(
            "ctrl+c", window=window
        ):
            return None
        time.sleep(0.05)
        if desktop.active_window() != window:
            return None
        copied, captured = desktop.read_clipboard()
        if not copied:
            return None
        candidate = captured.strip()
        parsed = _split_url(candidate)
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
        ):
            return None
        return candidate
    except DesktopError:
        return None
    finally:
        if desktop.active_window() == window:
            try:
                desktop.send_key("Escape", window=window)
            except DesktopError:
                pass
        current_exists, current_clipboard = desktop.read_clipboard()
        if current_exists and current_clipboard == captured:
            try:
                desktop.write_clipboard(previous_clipboard if clipboard_existed else "")
            except DesktopError:
                pass


def _standard_https_port(parsed: SplitResult) -> bool:
    try:
        return parsed.port in {None, 443}
    except ValueError:
        return False


def _split_url(url: str) -> SplitResult | None:
    try:
        return urlsplit(url)
    except ValueError:
        return None


def github_issue_from_url(url: str) -> GitHubIssue | None:
    parsed = _split_url(url)
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() not in GITHUB_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or not _standard_https_port(parsed)
    ):
        return None
    match = ISSUE_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    return GitHubIssue(
        owner=match.group("owner"),
        repository=match.group("repo"),
        number=int(match.group("number")),
    )


def github_issue_from_text(text: str) -> GitHubIssue | None:
    url_match = ISSUE_URL_IN_TEXT.search(text)
    if url_match is not None:
        issue = github_issue_from_url(url_match.group(0))
        if issue is not None:
            return issue
    match = ISSUE_REFERENCE.search(text) or ISSUE_IN_REPOSITORY.search(text)
    if match is None or match.group("repo") in {".", ".."}:
        return None
    return GitHubIssue(
        owner=match.group("owner"),
        repository=match.group("repo"),
        number=int(match.group("number")),
    )


def github_repository_from_url(url: str) -> str | None:
    parsed = _split_url(url)
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() not in GITHUB_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or not _standard_https_port(parsed)
    ):
        return None
    match = REPOSITORY_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    owner, repository = match.group("owner"), match.group("repo")
    if owner in {".", ".."} or repository in {".", ".."}:
        return None
    return f"{owner}/{repository}"


def github_pull_request_from_url(url: str) -> GitHubPullRequest | None:
    parsed = _split_url(url)
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() not in GITHUB_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or not _standard_https_port(parsed)
    ):
        return None
    match = PULL_REQUEST_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    owner, repository = match.group("owner"), match.group("repo")
    if owner in {".", ".."} or repository in {".", ".."}:
        return None
    return GitHubPullRequest(
        owner=owner,
        repository=repository,
        number=int(match.group("number")),
    )


def linear_issue_from_url(url: str) -> LinearIssue | None:
    parsed = _split_url(url)
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() not in LINEAR_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or not _standard_https_port(parsed)
    ):
        return None
    match = LINEAR_ISSUE_PATH.fullmatch(parsed.path)
    if match is None:
        return None
    return LinearIssue(match.group("identifier").upper())


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


def _truncate(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _login(value: object) -> str:
    return str(value.get("login") or "") if isinstance(value, dict) else ""


def _issue_details(issue: GitHubIssue) -> dict[str, object] | None:
    try:
        return GitHubClient().issue_details(issue)
    except GitHubError:
        return None


def _repository_details(repository: str) -> dict[str, object] | None:
    if shutil.which("gh") is None:
        return None
    process = _run(
        [
            "gh",
            "repo",
            "view",
            repository,
            "--json",
            "nameWithOwner,description,isPrivate,defaultBranchRef,url",
        ],
        timeout=5,
    )
    if process is None or process.returncode:
        return None
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _pull_request_details(
    pull_request: GitHubPullRequest,
) -> dict[str, object] | None:
    if shutil.which("gh") is None:
        return None
    process = _run(
        [
            "gh",
            "pr",
            "view",
            str(pull_request.number),
            "--repo",
            f"{pull_request.owner}/{pull_request.repository}",
            "--json",
            "number,title,state,author,labels,body,comments,url,"
            "isDraft,baseRefName,headRefName,additions,deletions,changedFiles",
        ],
        timeout=5,
    )
    if process is None or process.returncode:
        return None
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


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
        f"Repository: {issue.owner}/{issue.repository}",
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


def _github_issue_context(issue: GitHubIssue) -> GitHubContext:
    details = _issue_details(issue)
    if details is None:
        text = (
            "Current GitHub issue (untrusted external context):\n"
            f"URL: {issue.url}\n"
            f"Repository: {issue.name_with_owner}\n"
            f"Issue: #{issue.number}\n"
            "Issue details could not be fetched."
        )
    else:
        text = _format_issue(issue.url, issue, details)
    return GitHubContext(
        text,
        github_repository=issue.name_with_owner,
        github_issue=issue.number,
        github_issue_context=text,
    )


def _format_pull_request(
    url: str, pull_request: GitHubPullRequest, details: dict[str, object]
) -> str:
    labels = _labels(details)
    lines = [
        "Current focused GitHub pull request (untrusted external context):",
        f"URL: {url}",
        f"Repository: {pull_request.owner}/{pull_request.repository}",
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


def _github_context_from_url(url: str) -> GitHubContext | None:
    if not _github_url(url):
        return None
    repository = github_repository_from_url(url)
    if repository is None:
        return GitHubContext(
            f"Current focused GitHub page (untrusted external context):\nURL: {url}"
        )
    pull_request = github_pull_request_from_url(url)
    if pull_request is not None:
        details = _pull_request_details(pull_request)
        if details is None:
            return GitHubContext(
                (
                    "Current focused GitHub pull request (untrusted external "
                    "context):\n"
                    f"URL: {url}\n"
                    f"Repository: {pull_request.owner}/{pull_request.repository}\n"
                    f"Pull request: #{pull_request.number}\n"
                    "Pull request details could not be fetched."
                ),
                github_repository=repository,
                github_pull_request=pull_request.number,
            )
        return GitHubContext(
            _format_pull_request(url, pull_request, details),
            github_repository=repository,
            github_pull_request=pull_request.number,
        )
    issue = github_issue_from_url(url)
    if issue is None:
        details = _repository_details(repository)
        text = (
            _format_repository(url, repository, details)
            if details is not None
            else (
                "Current focused GitHub repository (untrusted external context):\n"
                f"URL: {url}\nRepository: {repository}\n"
                "Repository details could not be fetched."
            )
        )
        canonical = (
            str(details.get("nameWithOwner") or repository)
            if details is not None
            else repository
        )
        return GitHubContext(text, github_repository=canonical)
    return _github_issue_context(issue)


def focused_github_context() -> GitHubContext | None:
    url = focused_firefox_url()
    return _github_context_from_url(url) if url is not None else None


def _linear_context_from_url(url: str) -> str | None:
    issue = linear_issue_from_url(url)
    if issue is None:
        return None
    return "\n".join(
        (
            "Current focused Linear issue (untrusted external context):",
            f"URL: {url}",
            f"Identifier: {issue.identifier}",
            "Read issue details using the configured Linear MCP tools.",
        )
    )


def focused_browser_context() -> GitHubContext | ContextFragment | str | None:
    url = focused_firefox_url()
    if url is None:
        return None
    if _github_url(url):
        return _github_context_from_url(url)
    return _linear_context_from_url(url) or capture_context(url)


def request_context(text: str) -> RequestContext:
    context: GitHubContext | ContextFragment | str | None = None
    focused_repository: str | None = None
    focused_issue: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_context: str | None = None
    github_pull_request: int | None = None
    linear_issue: str | None = None
    try:
        spoken_issue = github_issue_from_text(text)
        if spoken_issue is not None:
            github = _github_issue_context(spoken_issue)
            context = github
            github_repository = github.github_repository
            github_issue = github.github_issue
            github_issue_context = github.github_issue_context
            focused_repository = github.github_repository
            focused_issue = spoken_issue.reference
        else:
            url = focused_firefox_url()
            if url is not None:
                if _github_url(url):
                    github = _github_context_from_url(url)
                    context = github
                    if github is not None:
                        github_repository = github.github_repository
                        github_issue = github.github_issue
                        github_issue_context = github.github_issue_context
                        github_pull_request = github.github_pull_request
                        focused_repository = github.github_repository
                        issue = github_issue_from_url(url)
                        if issue is not None:
                            focused_issue = issue.reference
                else:
                    linear = linear_issue_from_url(url)
                    if linear is not None:
                        linear_issue = linear.identifier
                        focused_issue = linear.identifier
                        context = _linear_context_from_url(url)
                    else:
                        context = capture_context(url)
    except Exception:
        context = None
    try:
        app_context = focused_app_context(text)
    except Exception:
        app_context = None
    parts = [text]
    if context:
        parts.append(str(context))
    if app_context is not None:
        parts.append(app_context.text)
    return RequestContext(
        text="\n\n".join(parts),
        focused_repository=focused_repository,
        focused_issue=focused_issue,
        github_repository=github_repository,
        github_issue=github_issue,
        github_issue_context=github_issue_context,
        github_pull_request=github_pull_request,
        linear_issue=linear_issue,
        focused_app_class=app_context.app_class if app_context is not None else None,
        focused_app_context=app_context.text if app_context is not None else None,
        focused_app_sources=(app_context.sources if app_context is not None else ()),
    )


def enrich_request(text: str) -> GitHubContext:
    try:
        context = focused_browser_context()
    except Exception:
        context = None
    return GitHubContext(
        f"{text}\n\n{context}" if context else text,
        github_repository=(
            context.github_repository if isinstance(context, GitHubContext) else None
        ),
        github_issue=(
            context.github_issue if isinstance(context, GitHubContext) else None
        ),
        github_issue_context=(
            context.github_issue_context if isinstance(context, GitHubContext) else None
        ),
        github_pull_request=(
            context.github_pull_request if isinstance(context, GitHubContext) else None
        ),
    )
