"""Admit confirmation-gated ticket closes from trusted speech."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import HarnessError
from .ticket_targets import TicketExtraction, TicketReference, resolve_named_ticket

MISSING_TICKET_IDENTITY = "Which ticket should I close?"
_CLOSE_VERB = re.compile(r"\b(?:close|resolve)\b", re.IGNORECASE)
_TICKET_OBJECT = re.compile(
    r"\b(?:this|that|the|a|an)\s+(?:github\s+|linear\s+)?(?:ticket|issue)s?\b"
    r"|\b(?:github\s+|linear\s+)?(?:ticket|issue)s?\b",
    re.IGNORECASE,
)
_BLOCKED = re.compile(
    r"\b(?:implement|work\s+on|fix|update|edit|rewrite|revise|patch|change|"
    r"split|merge|create|file|open|review|summari[sz]e|adversarial)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TicketCloseAdmission:
    ticket: TicketReference | None

    @property
    def missing_identity_response(self) -> str:
        return MISSING_TICKET_IDENTITY


@dataclass(frozen=True, slots=True)
class TicketCloseDispatch:
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_close_requested: bool = False
    issue_key: str | None = None
    linear_ticket_close_requested: bool = False


def wants_ticket_close_context(utterance: str) -> bool:
    """Whether a close utterance may consume focused ticket context."""

    if _CLOSE_VERB.search(utterance) is None or _BLOCKED.search(utterance):
        return False
    return bool(_TICKET_OBJECT.search(utterance))


def admit_ticket_close(
    utterance: str,
    extraction: TicketExtraction,
    *,
    focused_issue: str | None,
) -> TicketCloseAdmission | None:
    """Admit a close that names one focused or spoken ticket."""

    if not wants_ticket_close_context(utterance) and not (
        _CLOSE_VERB.search(utterance)
        and not _BLOCKED.search(utterance)
        and any(
            reference.canonical or reference.scoped
            for reference in extraction.references
        )
    ):
        return None
    if _BLOCKED.search(utterance):
        return None
    if not (
        _TICKET_OBJECT.search(utterance)
        or any(
            reference.canonical or reference.scoped
            for reference in extraction.references
        )
    ):
        return None
    return TicketCloseAdmission(
        resolve_named_ticket(extraction, focused_issue=focused_issue),
    )


def close_turn_arguments(ticket: TicketReference) -> TicketCloseDispatch:
    """Return Cursor turn kwargs for one validated ticket identity."""

    if ticket.canonical is None or ticket.source is None:
        raise HarnessError("Ticket close requires a trusted ticket identity")
    if ticket.source == "github":
        repository, number = ticket.canonical.rsplit("#", 1)
        return TicketCloseDispatch(
            github_repository=repository,
            github_issue=int(number),
            github_issue_close_requested=True,
        )
    return TicketCloseDispatch(
        issue_key=ticket.canonical,
        linear_ticket_close_requested=True,
    )
