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
ZENDESK_HOST = re.compile(
    r"^(?P<subdomain>[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)"
    r"\.zendesk\.com$",
    re.IGNORECASE,
)
ZENDESK_TICKET_PATH = re.compile(r"^/agent/tickets/(?P<number>\d+)/?$")
MAX_BODY_CHARS = 5_000
MAX_COMMENT_CHARS = 800
MAX_COMMENTS = 5
MAX_ZENDESK_PAGE_CHARS = 10_000


@dataclass(frozen=True)
class GitHubIssue:
    owner: str
    repository: str
    number: int


@dataclass(frozen=True)
class ZendeskTicket:
    subdomain: str
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


def zendesk_ticket_from_url(url: str) -> ZendeskTicket | None:
    parsed = _split_url(url)
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or not _standard_https_port(parsed)
    ):
        return None
    host_match = ZENDESK_HOST.fullmatch(parsed.hostname)
    path_match = ZENDESK_TICKET_PATH.fullmatch(parsed.path)
    if host_match is None or path_match is None:
        return None
    return ZendeskTicket(
        subdomain=host_match.group("subdomain").casefold(),
        number=int(path_match.group("number")),
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


def _github_context_from_url(url: str) -> str | None:
    if not _github_url(url):
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


def focused_github_context() -> str | None:
    url = focused_firefox_url()
    return _github_context_from_url(url) if url is not None else None


def _focused_firefox_page_text(expected_url: str) -> str | None:
    """Copy rendered page text after confirming the focused tab has not changed."""
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
        if not copied or captured.strip() != expected_url:
            return None

        if not _send_key(window_id, "Escape"):
            return None
        time.sleep(0.05)
        if _active_window_id() != window_id or not _send_key(window_id, "ctrl+a"):
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
        page_text = "\n".join(
            line.strip() for line in captured.splitlines() if line.strip()
        )
        return page_text or None
    finally:
        if _active_window_id() == window_id:
            _send_key(window_id, "Escape")
        current_exists, current_clipboard = _read_clipboard()
        if current_exists and current_clipboard == captured:
            _write_clipboard(previous_clipboard if clipboard_existed else "")


def _zendesk_context_from_url(url: str) -> str | None:
    ticket = zendesk_ticket_from_url(url)
    if ticket is None:
        return None
    lines = [
        "Current focused Zendesk ticket (untrusted external context):",
        f"URL: {url}",
        f"Tenant: {ticket.subdomain}",
        f"Ticket: #{ticket.number}",
    ]
    page_text = _focused_firefox_page_text(url)
    if page_text is None:
        lines.append("Rendered ticket text could not be read from the browser.")
    else:
        lines.extend(
            (
                "Rendered ticket text:",
                _truncate(page_text, MAX_ZENDESK_PAGE_CHARS),
            )
        )
    return "\n".join(lines)


def focused_zendesk_context() -> str | None:
    url = focused_firefox_url()
    return _zendesk_context_from_url(url) if url is not None else None


def focused_browser_context() -> str | None:
    url = focused_firefox_url()
    if url is None:
        return None
    if _github_url(url):
        return _github_context_from_url(url)
    return _zendesk_context_from_url(url)


def enrich_request(text: str) -> str:
    try:
        context = focused_browser_context()
    except Exception:
        context = None
    return f"{text}\n\n{context}" if context else text
