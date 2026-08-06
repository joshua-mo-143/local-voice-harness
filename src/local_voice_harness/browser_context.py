from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit

FIREFOX_CLASSES = {"firefox"}
GITHUB_HOSTS = {"github.com", "www.github.com"}
ISSUE_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/(?P<number>\d+)/?$"
)
PULL_REQUEST_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/pull/(?P<number>\d+)"
    r"(?:/.*)?$"
)
REPOSITORY_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:/.*)?$"
)
MAX_BODY_CHARS = 5_000
MAX_COMMENT_CHARS = 800
MAX_COMMENTS = 5


@dataclass(frozen=True)
class GitHubIssue:
    owner: str
    repository: str
    number: int


@dataclass(frozen=True)
class GitHubPullRequest:
    owner: str
    repository: str
    number: int


class GitHubContext(str):
    github_repository: str | None
    github_pull_request: int | None

    def __new__(
        cls,
        value: str,
        *,
        github_repository: str | None = None,
        github_pull_request: int | None = None,
    ) -> GitHubContext:
        instance = super().__new__(cls, value)
        instance.github_repository = github_repository
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


def _active_window_id() -> str | None:
    process = _run(["xdotool", "getactivewindow"])
    if process is None or process.returncode:
        return None
    window_id = process.stdout.strip()
    return window_id if window_id.isdigit() else None


def _window_class(window_id: str) -> str:
    process = _run(["xdotool", "getwindowclassname", window_id])
    return (
        process.stdout.strip().casefold()
        if process is not None and process.returncode == 0
        else ""
    )


def _read_clipboard() -> tuple[bool, str]:
    process = _run(["xclip", "-selection", "clipboard", "-out"])
    if process is None or process.returncode:
        return False, ""
    return True, process.stdout


def _write_clipboard(text: str) -> None:
    _run(["xclip", "-selection", "clipboard"], input_text=text)


def _send_key(window_id: str, key: str) -> bool:
    process = _run(["xdotool", "key", "--window", window_id, "--clearmodifiers", key])
    return process is not None and process.returncode == 0


def focused_firefox_url() -> str | None:
    """Capture the focused Firefox tab URL without leaving the address bar open."""
    if shutil.which("xdotool") is None or shutil.which("xclip") is None:
        return None
    window_id = _active_window_id()
    if window_id is None or _window_class(window_id) not in FIREFOX_CLASSES:
        return None

    clipboard_existed, previous_clipboard = _read_clipboard()
    captured = ""
    try:
        if not _send_key(window_id, "ctrl+l"):
            return None
        time.sleep(0.05)
        if _active_window_id() != window_id or not _send_key(window_id, "ctrl+c"):
            return None
        time.sleep(0.05)
        if _active_window_id() != window_id:
            return None
        copied, captured = _read_clipboard()
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
    finally:
        if _active_window_id() == window_id:
            _send_key(window_id, "Escape")
        current_exists, current_clipboard = _read_clipboard()
        if current_exists and current_clipboard == captured:
            _write_clipboard(previous_clipboard if clipboard_existed else "")


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
    if shutil.which("gh") is None:
        return None
    process = _run(
        [
            "gh",
            "issue",
            "view",
            str(issue.number),
            "--repo",
            f"{issue.owner}/{issue.repository}",
            "--json",
            "number,title,state,author,labels,body,comments,url",
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


def focused_github_context() -> GitHubContext | None:
    url = focused_firefox_url()
    if url is None or not _github_url(url):
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
    details = _issue_details(issue)
    if details is None:
        return GitHubContext(
            (
                "Current focused GitHub issue (untrusted external context):\n"
                f"URL: {url}\n"
                f"Repository: {issue.owner}/{issue.repository}\n"
                f"Issue: #{issue.number}\n"
                "Issue details could not be fetched."
            ),
            github_repository=repository,
        )
    return GitHubContext(
        _format_issue(url, issue, details), github_repository=repository
    )


def enrich_request(text: str) -> GitHubContext:
    try:
        context = focused_github_context()
    except Exception:
        context = None
    return GitHubContext(
        f"{text}\n\n{context}" if context else text,
        github_repository=(
            context.github_repository if isinstance(context, GitHubContext) else None
        ),
        github_pull_request=(
            context.github_pull_request if isinstance(context, GitHubContext) else None
        ),
    )
