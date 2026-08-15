"""Draft and admit confirmation-gated ticket title/body updates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import BackendSettings
from .errors import HarnessError
from .llm_transport import ChatCompletionRequest, LlmTransport
from .ticket_targets import TicketExtraction, TicketReference, resolve_named_ticket

MAX_UPDATE_TITLE_CHARS = 200
MAX_UPDATE_BODY_CHARS = 10_000
MISSING_TICKET_IDENTITY = "Which ticket should I update?"
_UPDATE_VERB = re.compile(
    r"\b(?:update|edit|rewrite|revise|patch|change)\b",
    re.IGNORECASE,
)
_TICKET_OBJECT = re.compile(
    r"\b(?:this|that|the|a|an)\s+(?:github\s+|linear\s+)?(?:ticket|issue)s?\b"
    r"|\b(?:github\s+|linear\s+)?(?:ticket|issue)s?\b",
    re.IGNORECASE,
)
_TITLE_BODY = re.compile(r"\b(?:title|body|description)\b", re.IGNORECASE)
_BLOCKED = re.compile(
    r"\b(?:implement|work\s+on|fix|close|split|merge|create|file|open|"
    r"review|summari[sz]e|adversarial)\b",
    re.IGNORECASE,
)
_DRAFT_TOOL = {
    "type": "function",
    "function": {
        "name": "draft_ticket_update",
        "description": "Draft the exact replacement title and body for one ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": MAX_UPDATE_TITLE_CHARS},
                "body": {"type": "string", "maxLength": MAX_UPDATE_BODY_CHARS},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class TicketUpdateAdmission:
    ticket: TicketReference | None

    @property
    def missing_identity_response(self) -> str:
        return MISSING_TICKET_IDENTITY


@dataclass(frozen=True, slots=True)
class TicketUpdateDraft:
    title: str
    body: str


def wants_ticket_update_context(utterance: str) -> bool:
    """Whether an update utterance may consume focused ticket context."""

    if _UPDATE_VERB.search(utterance) is None or _BLOCKED.search(utterance):
        return False
    return bool(_TICKET_OBJECT.search(utterance) or _TITLE_BODY.search(utterance))


def admit_ticket_update(
    utterance: str,
    extraction: TicketExtraction,
    *,
    focused_issue: str | None,
) -> TicketUpdateAdmission | None:
    """Admit a title/body update that names one focused or spoken ticket."""

    if not wants_ticket_update_context(utterance) and not (
        _UPDATE_VERB.search(utterance)
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
        or _TITLE_BODY.search(utterance)
        or any(
            reference.canonical or reference.scoped
            for reference in extraction.references
        )
    ):
        return None
    return TicketUpdateAdmission(
        resolve_named_ticket(extraction, focused_issue=focused_issue),
    )


@dataclass(frozen=True, slots=True)
class TicketUpdateDispatch:
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_update_requested: bool = False
    issue_key: str | None = None
    linear_ticket_update_requested: bool = False


def update_turn_arguments(ticket: TicketReference) -> TicketUpdateDispatch:
    """Return Cursor turn kwargs for one validated ticket identity."""

    if ticket.canonical is None or ticket.source is None:
        raise HarnessError("Ticket update requires a trusted ticket identity")
    if ticket.source == "github":
        repository, number = ticket.canonical.rsplit("#", 1)
        return TicketUpdateDispatch(
            github_repository=repository,
            github_issue=int(number),
            github_issue_update_requested=True,
        )
    return TicketUpdateDispatch(
        issue_key=ticket.canonical,
        linear_ticket_update_requested=True,
    )


def _validated_draft(title: object, body: object) -> TicketUpdateDraft:
    if not isinstance(title, str) or not title.strip():
        raise HarnessError("Ticket update draft requires a non-empty title")
    if not isinstance(body, str) or not body.strip():
        raise HarnessError("Ticket update draft requires a non-empty body")
    title = " ".join(title.split())
    body = body.strip()
    if len(title) > MAX_UPDATE_TITLE_CHARS:
        raise HarnessError("Ticket update draft title is too long")
    if len(body) > MAX_UPDATE_BODY_CHARS:
        raise HarnessError("Ticket update draft body is too long")
    return TicketUpdateDraft(title, body)


def draft_ticket_update(
    utterance: str,
    ticket: str,
    *,
    settings: BackendSettings | None = None,
) -> TicketUpdateDraft:
    trusted_request = utterance.strip()
    if not trusted_request:
        raise HarnessError("Ticket update requires a spoken request")
    transport = LlmTransport.from_settings(settings)
    message = transport.chat_completion(
        ChatCompletionRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's trusted spoken request into an exact "
                        "replacement title and body for one existing ticket. Preserve "
                        "concrete requirements, do not invent acceptance criteria, and "
                        "do not include the ticket identity in the title. Return only "
                        "the forced tool call."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"ticket": ticket, "request": trusted_request}
                    ),
                },
            ],
            temperature=0,
            max_tokens=1024,
            stream=False,
            tools=[_DRAFT_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "draft_ticket_update"},
            },
            parallel_tool_calls=False,
        )
    )
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise HarnessError("LLM did not return a ticket update draft")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != "draft_ticket_update":
        raise HarnessError("LLM returned a malformed ticket update draft")
    try:
        arguments = json.loads(str(function.get("arguments") or "{}"))
    except json.JSONDecodeError as exc:
        raise HarnessError("LLM returned a malformed ticket update draft") from exc
    if not isinstance(arguments, dict):
        raise HarnessError("LLM returned a malformed ticket update draft")
    return _validated_draft(arguments.get("title"), arguments.get("body"))
