"""Zendesk browser-context provider.

Zendesk support is an optional integration that is disabled by default on fresh
installations. This module is entirely self-contained: it owns Zendesk URL
matching and the browser page capture it depends on, so nothing else in the
harness inspects a Zendesk URL or copies a Zendesk page unless the integration
is explicitly enabled and its provider is instantiated by the registry.

All captured page text is treated as untrusted external input, never as
instructions, is size-bounded, and fails closed on any error or focus change.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit

from ..context_fragment import ContextFragment
from ..desktop import DesktopError, get_desktop

# Kept local so the provider does not depend on ``browser_context``; the wider
# harness must not reach into Zendesk internals, and Zendesk must not reach back.
FIREFOX_CLASSES = {"firefox", "org.mozilla.firefox"}

ZENDESK_HOST = re.compile(
    r"^(?P<subdomain>[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)"
    r"\.zendesk\.com$",
    re.IGNORECASE,
)
ZENDESK_TICKET_PATH = re.compile(r"^/agent/tickets/(?P<number>\d+)/?$")
ZENDESK_REFERENCE = re.compile(
    r"^(?P<subdomain>[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)#"
    r"(?P<number>\d+)$",
    re.IGNORECASE,
)
MAX_ZENDESK_PAGE_CHARS = 10_000

PROVIDER_NAME = "zendesk"


@dataclass(frozen=True)
class ZendeskTicket:
    subdomain: str
    number: int


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


def _truncate(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


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


def zendesk_reference(ticket: ZendeskTicket) -> str:
    return f"{ticket.subdomain}#{ticket.number}"


def zendesk_ticket_from_reference(reference: str) -> ZendeskTicket | None:
    candidate = reference.strip()
    ticket = zendesk_ticket_from_url(candidate)
    if ticket is not None:
        return ticket
    match = ZENDESK_REFERENCE.fullmatch(candidate)
    if match is None:
        return None
    return ZendeskTicket(
        subdomain=match.group("subdomain").casefold(),
        number=int(match.group("number")),
    )


def _focused_firefox_page_text(expected_url: str) -> str | None:
    """Copy rendered page text after confirming the focused tab has not changed."""
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
        if not copied or captured.strip() != expected_url:
            return None

        if not desktop.send_key("Escape", window=window):
            return None
        time.sleep(0.05)
        if desktop.active_window() != window or not desktop.send_key(
            "ctrl+a", window=window
        ):
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
        page_text = "\n".join(
            line.strip() for line in captured.splitlines() if line.strip()
        )
        return page_text or None
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


def zendesk_context_from_url(url: str) -> str | None:
    """Render bounded, untrusted-labelled context for a focused Zendesk ticket."""
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


class ZendeskProvider:
    """Optional context provider for focused Zendesk agent tickets."""

    name = PROVIDER_NAME

    def matches(self, url: str) -> bool:
        return zendesk_ticket_from_url(url) is not None

    def capture(self, url: str) -> ContextFragment | None:
        ticket = zendesk_ticket_from_url(url)
        text = zendesk_context_from_url(url)
        if ticket is None or text is None:
            return None
        return ContextFragment(
            source=self.name,
            text=text,
            issue_reference=zendesk_reference(ticket),
        )

    def owns_issue_reference(self, reference: str) -> bool:
        return zendesk_ticket_from_reference(reference) is not None

    def canonicalize_issue_reference(self, reference: str) -> str:
        ticket = zendesk_ticket_from_reference(reference)
        if ticket is None:
            return reference.strip()
        return zendesk_reference(ticket)
