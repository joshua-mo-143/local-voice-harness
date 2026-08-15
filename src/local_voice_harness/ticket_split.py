"""Draft and admit confirmation-gated ticket splits into child tickets."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, replace

from .config import BackendSettings
from .errors import HarnessError
from .llm_transport import ChatCompletionRequest, LlmTransport
from .ticket_targets import TicketExtraction, TicketReference, resolve_named_ticket

MAX_SPLIT_CHILDREN = 8
MAX_SPLIT_TITLE_CHARS = 200
MAX_SPLIT_BODY_CHARS = 10_000
MISSING_TICKET_IDENTITY = "Which ticket should I split?"
PARENT_ACTIONS = frozenset({"close", "update", "none"})
CHILD_STATES = frozenset(
    {"planned", "submitted", "created", "ambiguous", "manual_required"}
)
_SPLIT_VERB = re.compile(
    r"\b(?:split|divide|break(?:\s+\w+){0,3}\s+into)\b",
    re.IGNORECASE,
)
_TICKET_OBJECT = re.compile(
    r"\b(?:this|that|the|a|an)\s+(?:github\s+|linear\s+)?(?:ticket|issue)s?\b"
    r"|\b(?:github\s+|linear\s+)?(?:ticket|issue)s?\b",
    re.IGNORECASE,
)
_BLOCKED = re.compile(
    r"\b(?:implement|work\s+on|fix|merge|review|summari[sz]e|adversarial)\b",
    re.IGNORECASE,
)
_DRAFT_TOOL = {
    "type": "function",
    "function": {
        "name": "draft_ticket_split",
        "description": (
            "Draft the exact child tickets and the parent action for one split."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "children": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_SPLIT_CHILDREN,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "maxLength": MAX_SPLIT_TITLE_CHARS,
                            },
                            "body": {
                                "type": "string",
                                "maxLength": MAX_SPLIT_BODY_CHARS,
                            },
                        },
                        "required": ["title", "body"],
                        "additionalProperties": False,
                    },
                },
                "parent_action": {
                    "type": "string",
                    "enum": ["close", "update", "none"],
                },
                "parent_title": {
                    "type": "string",
                    "maxLength": MAX_SPLIT_TITLE_CHARS,
                },
                "parent_body": {
                    "type": "string",
                    "maxLength": MAX_SPLIT_BODY_CHARS,
                },
            },
            "required": ["children", "parent_action"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class TicketSplitAdmission:
    ticket: TicketReference | None

    @property
    def missing_identity_response(self) -> str:
        return MISSING_TICKET_IDENTITY


@dataclass(frozen=True, slots=True)
class TicketSplitDispatch:
    github_repository: str | None = None
    github_issue: int | None = None
    github_issue_split_requested: bool = False
    issue_key: str | None = None
    linear_ticket_split_requested: bool = False


@dataclass(frozen=True, slots=True)
class SplitChild:
    title: str
    body: str
    marker: str
    state: str = "planned"
    created_ref: str | None = None
    created_url: str | None = None


@dataclass(frozen=True, slots=True)
class TicketSplitDraft:
    children: tuple[SplitChild, ...]
    parent_action: str
    parent_title: str | None = None
    parent_body: str | None = None


def wants_ticket_split_context(utterance: str) -> bool:
    """Whether a split utterance may consume focused ticket context."""

    if _SPLIT_VERB.search(utterance) is None or _BLOCKED.search(utterance):
        return False
    return bool(_TICKET_OBJECT.search(utterance))


def admit_ticket_split(
    utterance: str,
    extraction: TicketExtraction,
    *,
    focused_issue: str | None,
) -> TicketSplitAdmission | None:
    """Admit a split that names one focused or spoken parent ticket."""

    if not wants_ticket_split_context(utterance) and not (
        _SPLIT_VERB.search(utterance)
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
    return TicketSplitAdmission(
        resolve_named_ticket(extraction, focused_issue=focused_issue),
    )


def split_turn_arguments(ticket: TicketReference) -> TicketSplitDispatch:
    """Return Cursor turn kwargs for one validated parent identity."""

    if ticket.canonical is None or ticket.source is None:
        raise HarnessError("Ticket split requires a trusted ticket identity")
    if ticket.source == "github":
        repository, number = ticket.canonical.rsplit("#", 1)
        return TicketSplitDispatch(
            github_repository=repository,
            github_issue=int(number),
            github_issue_split_requested=True,
        )
    return TicketSplitDispatch(
        issue_key=ticket.canonical,
        linear_ticket_split_requested=True,
    )


def spoken_split_confirmation(parent: str, count: int, parent_action: str) -> str:
    """Spoken question that names the child count and parent."""

    noun = "issue" if count == 1 else "issues"
    if parent_action == "close":
        return f"Create {count} {noun} and close {parent}?"
    if parent_action == "update":
        return f"Create {count} {noun} and update {parent}?"
    return f"Create {count} {noun} from {parent}?"


def split_preview(
    parent: str,
    draft: TicketSplitDraft,
) -> str:
    """Display the exact child set and parent action before confirmation."""

    lines = [
        spoken_split_confirmation(parent, len(draft.children), draft.parent_action)
    ]
    for index, child in enumerate(draft.children, start=1):
        lines.append(f"\nChild {index} title: {child.title}\n\nBody:\n{child.body}")
    if draft.parent_action == "update":
        lines.append(
            f"\nParent title: {draft.parent_title}\n\nParent body:\n{draft.parent_body}"
        )
    elif draft.parent_action == "close":
        lines.append(f"\nParent {parent} will be closed.")
    else:
        lines.append(f"\nParent {parent} will stay open.")
    lines.append("\nSay yes to apply this set or no to cancel.")
    return "\n".join(lines)


def encode_split_children(children: tuple[SplitChild, ...] | list[SplitChild]) -> str:
    return json.dumps(
        [
            {
                "title": child.title,
                "body": child.body,
                "marker": child.marker,
                "state": child.state,
                "created_ref": child.created_ref,
                "created_url": child.created_url,
            }
            for child in children
        ],
        separators=(",", ":"),
    )


def decode_split_children(payload: str | None) -> tuple[SplitChild, ...]:
    if not payload:
        return ()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HarnessError("Ticket split children payload is malformed") from exc
    if not isinstance(value, list) or not value or len(value) > MAX_SPLIT_CHILDREN:
        raise HarnessError("Ticket split requires a bounded child set")
    return tuple(_validated_child(item) for item in value)


def replace_split_child(
    children: tuple[SplitChild, ...],
    index: int,
    **changes: object,
) -> tuple[SplitChild, ...]:
    updated = list(children)
    updated[index] = replace(updated[index], **changes)
    return tuple(updated)


def created_child_refs(children: tuple[SplitChild, ...]) -> tuple[str, ...]:
    return tuple(
        child.created_ref
        for child in children
        if child.state == "created" and child.created_ref
    )


def assign_split_markers(draft: TicketSplitDraft) -> TicketSplitDraft:
    """Replace placeholder draft markers with durable per-child markers."""

    return TicketSplitDraft(
        tuple(replace(child, marker=uuid.uuid4().hex) for child in draft.children),
        draft.parent_action,
        draft.parent_title,
        draft.parent_body,
    )


def split_parent_identity(
    *,
    github_repository: str | None,
    github_issue: int | None,
    issue_key: str | None,
) -> str:
    if github_repository and github_issue is not None:
        return f"{github_repository}#{github_issue}"
    return (issue_key or "").strip() or "the ticket"


def linear_team_key(issue_key: str) -> str:
    identifier = issue_key.strip()
    if "-" not in identifier:
        raise HarnessError("Linear ticket split requires a team key")
    return identifier.rsplit("-", 1)[0]


def split_result_message(
    parent: str,
    children: tuple[SplitChild, ...],
    *,
    parent_action: str,
    parent_state: str | None,
) -> str:
    created = created_child_refs(children)
    if created:
        child_text = "Created child tickets " + ", ".join(created) + "."
    else:
        child_text = "No child tickets were created."
    if parent_action == "close" and parent_state == "created":
        return f"{child_text} Closed {parent}."
    if parent_action == "update" and parent_state == "created":
        return f"{child_text} Updated {parent}."
    if parent_action == "close":
        return f"{child_text} {parent} was not closed."
    if parent_action == "update":
        return f"{child_text} {parent} was not updated."
    return child_text


def _validated_child(value: object) -> SplitChild:
    if not isinstance(value, dict):
        raise HarnessError("Ticket split child must be an object")
    title = value.get("title")
    body = value.get("body")
    marker = value.get("marker")
    state = value.get("state") or "planned"
    if not isinstance(title, str) or not title.strip():
        raise HarnessError("Ticket split child requires a non-empty title")
    if not isinstance(body, str) or not body.strip():
        raise HarnessError("Ticket split child requires a non-empty body")
    if not isinstance(marker, str) or not re.fullmatch(r"[0-9a-f]{32}", marker):
        raise HarnessError("Ticket split child marker is invalid")
    if state not in CHILD_STATES:
        raise HarnessError("Ticket split child state is invalid")
    created_ref = value.get("created_ref")
    created_url = value.get("created_url")
    if created_ref is not None and not isinstance(created_ref, str):
        raise HarnessError("Ticket split child reference is invalid")
    if created_url is not None and not isinstance(created_url, str):
        raise HarnessError("Ticket split child URL is invalid")
    title = " ".join(title.split())
    body = body.strip()
    if len(title) > MAX_SPLIT_TITLE_CHARS:
        raise HarnessError("Ticket split child title is too long")
    if len(body) > MAX_SPLIT_BODY_CHARS:
        raise HarnessError("Ticket split child body is too long")
    return SplitChild(
        title,
        body,
        marker,
        state,
        created_ref.strip()
        if isinstance(created_ref, str) and created_ref.strip()
        else None,
        created_url.strip()
        if isinstance(created_url, str) and created_url.strip()
        else None,
    )


def draft_ticket_split(
    utterance: str,
    ticket: str,
    *,
    settings: BackendSettings | None = None,
) -> TicketSplitDraft:
    trusted_request = utterance.strip()
    if not trusted_request:
        raise HarnessError("Ticket split requires a spoken request")
    transport = LlmTransport.from_settings(settings)
    message = transport.chat_completion(
        ChatCompletionRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's trusted spoken request into exact child "
                        "ticket drafts and one parent action. Preserve concrete "
                        "requirements, do not invent acceptance criteria, and do not "
                        "include the parent identity in child titles. Return only the "
                        "forced tool call."
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
            max_tokens=2048,
            stream=False,
            tools=[_DRAFT_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "draft_ticket_split"},
            },
            parallel_tool_calls=False,
        )
    )
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise HarnessError("LLM did not return a ticket split draft")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != "draft_ticket_split":
        raise HarnessError("LLM returned a malformed ticket split draft")
    try:
        arguments = json.loads(str(function.get("arguments") or "{}"))
    except json.JSONDecodeError as exc:
        raise HarnessError("LLM returned a malformed ticket split draft") from exc
    if not isinstance(arguments, dict):
        raise HarnessError("LLM returned a malformed ticket split draft")
    return _validated_draft(arguments)


def _validated_draft(arguments: dict[str, object]) -> TicketSplitDraft:
    raw_children = arguments.get("children")
    action = arguments.get("parent_action")
    if not isinstance(raw_children, list) or not raw_children:
        raise HarnessError("Ticket split draft requires at least one child")
    if len(raw_children) > MAX_SPLIT_CHILDREN:
        raise HarnessError("Ticket split draft has too many children")
    if action not in PARENT_ACTIONS:
        raise HarnessError("Ticket split draft parent action is invalid")
    children: list[SplitChild] = []
    for item in raw_children:
        if not isinstance(item, dict):
            raise HarnessError("Ticket split child must be an object")
        title = item.get("title")
        body = item.get("body")
        if not isinstance(title, str) or not title.strip():
            raise HarnessError("Ticket split child requires a non-empty title")
        if not isinstance(body, str) or not body.strip():
            raise HarnessError("Ticket split child requires a non-empty body")
        title = " ".join(title.split())
        body = body.strip()
        if len(title) > MAX_SPLIT_TITLE_CHARS:
            raise HarnessError("Ticket split child title is too long")
        if len(body) > MAX_SPLIT_BODY_CHARS:
            raise HarnessError("Ticket split child body is too long")
        children.append(SplitChild(title, body, marker="0" * 32))
    parent_title = arguments.get("parent_title")
    parent_body = arguments.get("parent_body")
    if action == "update":
        if not isinstance(parent_title, str) or not parent_title.strip():
            raise HarnessError("Ticket split rewrite requires a parent title")
        if not isinstance(parent_body, str) or not parent_body.strip():
            raise HarnessError("Ticket split rewrite requires a parent body")
        parent_title = " ".join(parent_title.split())
        parent_body = parent_body.strip()
    else:
        parent_title = None
        parent_body = None
    return TicketSplitDraft(tuple(children), str(action), parent_title, parent_body)
