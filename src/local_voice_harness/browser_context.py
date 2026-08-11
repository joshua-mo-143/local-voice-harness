from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from .context_fragment import ContextFragment
from .context_providers import (
    capture_context,
    capture_text_context,
)
from .desktop import DesktopError, get_desktop
from .focused_app_context import focused_app_context
from .integrations import github as _github
from .integrations.registry import (
    IntegrationRegistry,
    RegistryInput,
    integration_enabled,
)
from .ticket_targets import extract_ticket_targets
from .user_config import PlatformSettings

GitHubIssue = _github.GitHubIssue
GitHubPullRequest = _github.GitHubPullRequest
github_issue_from_text = _github.github_issue_from_text
github_issue_from_url = _github.github_issue_from_url
github_pull_request_from_url = _github.github_pull_request_from_url
github_repository_from_url = _github.github_repository_from_url

FIREFOX_CLASSES = {"firefox", "org.mozilla.firefox"}


@dataclass(frozen=True)
class RequestContext:
    text: str
    focused_repository: str | None = None
    focused_issue: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_context: str | None = None
    github_pull_request: int | None = None
    external_issue_reference: str | None = None
    external_issue_source: str | None = None
    issue_scope: str | None = None
    issue_scope_source: str | None = None
    focused_app_class: str | None = None
    focused_app_context: str | None = None
    focused_app_sources: tuple[str, ...] = ()


class GitHubContext(str):
    """Compatibility rendering returned by :func:`enrich_request`."""

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
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            return None
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
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


def focused_github_context() -> ContextFragment | None:
    context = focused_browser_context()
    return context if context is not None and context.source == "github" else None


def focused_browser_context() -> ContextFragment | None:
    url = focused_firefox_url()
    return capture_context(url) if url is not None else None


def focused_herdr_github_context(
    integrations: RegistryInput,
) -> ContextFragment | None:
    """Capture GitHub issue-list scope from Herdr's focused workspace."""

    if not isinstance(integrations, IntegrationRegistry) or not integration_enabled(
        "github", integrations
    ):
        return None
    try:
        checkout = integrations.herdr_client().focused_checkout()
        if checkout is None:
            return None
        repository = integrations.github_client().repository_for_checkout(checkout)
    except Exception:
        return None
    return capture_context(f"https://github.com/{repository}/issues", integrations)


def request_context(
    text: str,
    *,
    platform: PlatformSettings,
    integrations: RegistryInput,
) -> RequestContext:
    context: ContextFragment | None = None
    try:
        context = capture_text_context(text, integrations)
        if context is None:
            url = focused_firefox_url()
            if url is not None:
                context = capture_context(url, integrations)
        if context is None and extract_ticket_targets(text).has_unresolved_scope:
            context = focused_herdr_github_context(integrations)
    except Exception:
        context = None

    try:
        app_context = focused_app_context(text, platform)
    except Exception:
        app_context = None

    focused_repository = context.repository_reference if context is not None else None
    focused_issue = context.issue_reference if context is not None else None
    github_context = (
        context if context is not None and context.source == "github" else None
    )
    github_repository = (
        github_context.repository_reference if github_context is not None else None
    )
    github_issue = github_context.issue_number if github_context is not None else None
    github_issue_context = (
        github_context.text
        if github_context is not None and github_context.issue_number is not None
        else None
    )
    github_pull_request = (
        github_context.pull_request_number if github_context is not None else None
    )
    external_issue_reference = (
        context.issue_reference
        if context is not None and github_context is None
        else None
    )
    external_issue_source = (
        context.source if external_issue_reference is not None and context else None
    )
    issue_scope = context.issue_scope if context is not None else None
    issue_scope_source = context.source if issue_scope is not None and context else None

    parts = [text]
    if context is not None:
        parts.append(context.text)
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
        external_issue_reference=external_issue_reference,
        external_issue_source=external_issue_source,
        issue_scope=issue_scope,
        issue_scope_source=issue_scope_source,
        focused_app_class=app_context.app_class if app_context is not None else None,
        focused_app_context=app_context.text if app_context is not None else None,
        focused_app_sources=(app_context.sources if app_context is not None else ()),
    )


def enrich_request(text: str) -> GitHubContext:
    try:
        context = focused_browser_context()
    except Exception:
        context = None
    if isinstance(context, GitHubContext):
        return GitHubContext(
            f"{text}\n\n{context}",
            github_repository=context.github_repository,
            github_issue=context.github_issue,
            github_issue_context=context.github_issue_context,
            github_pull_request=context.github_pull_request,
        )
    github_context = (
        context if context is not None and context.source == "github" else None
    )
    return GitHubContext(
        f"{text}\n\n{context}" if context else text,
        github_repository=(
            github_context.repository_reference if github_context is not None else None
        ),
        github_issue=(
            github_context.issue_number if github_context is not None else None
        ),
        github_issue_context=(
            github_context.text
            if github_context is not None and github_context.issue_number is not None
            else None
        ),
        github_pull_request=(
            github_context.pull_request_number if github_context is not None else None
        ),
    )
