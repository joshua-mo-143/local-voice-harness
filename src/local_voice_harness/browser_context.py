from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit

from .desktop import DesktopError, get_desktop

FIREFOX_CLASSES = {"firefox", "org.mozilla.firefox"}
GITHUB_HOSTS = {"github.com", "www.github.com"}
ISSUE_PATH = re.compile(
    r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/(?P<number>\d+)/?$"
)
MAX_BODY_CHARS = 5_000
MAX_COMMENT_CHARS = 800
MAX_COMMENTS = 5


@dataclass(frozen=True)
class GitHubIssue:
    owner: str
    repository: str
    number: int


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


def _format_issue(url: str, issue: GitHubIssue, details: dict[str, object]) -> str:
    labels_value = details.get("labels")
    labels = (
        ", ".join(
            str(label.get("name") or "")
            for label in labels_value
            if isinstance(label, dict) and label.get("name")
        )
        if isinstance(labels_value, list)
        else ""
    )
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

    comments_value = details.get("comments")
    comments = (
        comments_value[-MAX_COMMENTS:] if isinstance(comments_value, list) else []
    )
    if comments:
        lines.append("Recent comments:")
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            author = _login(comment.get("author")) or "unknown"
            body = _truncate(comment.get("body"), MAX_COMMENT_CHARS)
            if body:
                lines.append(f"- {author}: {body}")
    return "\n".join(lines)


def focused_github_context() -> str | None:
    url = focused_firefox_url()
    if url is None or not _github_url(url):
        return None
    issue = github_issue_from_url(url)
    if issue is None:
        return f"Current focused GitHub page (untrusted external context):\nURL: {url}"
    details = _issue_details(issue)
    if details is None:
        return (
            "Current focused GitHub issue (untrusted external context):\n"
            f"URL: {url}\n"
            f"Repository: {issue.owner}/{issue.repository}\n"
            f"Issue: #{issue.number}\n"
            "Issue details could not be fetched."
        )
    return _format_issue(url, issue, details)


def enrich_request(text: str) -> str:
    try:
        context = focused_github_context()
    except Exception:
        context = None
    return f"{text}\n\n{context}" if context else text
