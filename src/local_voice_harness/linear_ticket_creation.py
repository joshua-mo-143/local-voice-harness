from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import BackendSettings
from .errors import HarnessError
from .integrations.linear import LinearError, LinearIntegration
from .llm_transport import ChatCompletionRequest, LlmTransport

MAX_LINEAR_TITLE_CHARS = 255
MAX_LINEAR_DESCRIPTION_CHARS = 10_000
_TEAM_IN_UTTERANCE = re.compile(
    r"\b(?:linear\s+)?team\s+(?P<team>[A-Za-z][A-Za-z0-9]{0,15})\b",
    re.IGNORECASE,
)
_DRAFT_TOOL = {
    "type": "function",
    "function": {
        "name": "draft_linear_ticket",
        "description": "Draft one concise Linear ticket from the user's request.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": MAX_LINEAR_TITLE_CHARS},
                "description": {
                    "type": "string",
                    "maxLength": MAX_LINEAR_DESCRIPTION_CHARS,
                },
            },
            "required": ["title", "description"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class LinearTicketDraft:
    team: str
    title: str
    description: str


def team_from_utterance(utterance: str) -> str | None:
    match = _TEAM_IN_UTTERANCE.search(utterance)
    if match is None:
        return None
    try:
        return LinearIntegration.validate_team(match.group("team"))
    except LinearError:
        return None


def _validated_draft(
    team: str,
    title: object,
    description: object,
) -> LinearTicketDraft:
    try:
        team = LinearIntegration.validate_team(team)
    except LinearError as exc:
        raise HarnessError(str(exc)) from exc
    if not isinstance(title, str) or not title.strip():
        raise HarnessError("Linear ticket draft requires a non-empty title")
    if not isinstance(description, str) or not description.strip():
        raise HarnessError("Linear ticket draft requires a non-empty description")
    title = " ".join(title.split())
    description = description.strip()
    if len(title) > MAX_LINEAR_TITLE_CHARS:
        raise HarnessError("Linear ticket draft title is too long")
    if len(description) > MAX_LINEAR_DESCRIPTION_CHARS:
        raise HarnessError("Linear ticket draft description is too long")
    return LinearTicketDraft(team, title, description)


def draft_linear_ticket(
    utterance: str,
    team: str,
    *,
    settings: BackendSettings | None = None,
) -> LinearTicketDraft:
    trusted_request = utterance.strip()
    if not trusted_request:
        raise HarnessError("Linear ticket creation requires a spoken request")
    transport = LlmTransport.from_settings(settings)
    message = transport.chat_completion(
        ChatCompletionRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's trusted spoken request into one Linear "
                        "ticket. Preserve concrete requirements, do not invent "
                        "acceptance criteria, and do not include the target team in "
                        "the title. Return only the forced tool call."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"team": team, "request": trusted_request}),
                },
            ],
            temperature=0,
            max_tokens=1024,
            stream=False,
            tools=[_DRAFT_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "draft_linear_ticket"},
            },
            parallel_tool_calls=False,
        )
    )
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise HarnessError("LLM did not return a Linear ticket draft")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != "draft_linear_ticket":
        raise HarnessError("LLM returned a malformed Linear ticket draft")
    try:
        arguments = json.loads(str(function.get("arguments") or "{}"))
    except json.JSONDecodeError as exc:
        raise HarnessError("LLM returned a malformed Linear ticket draft") from exc
    if not isinstance(arguments, dict):
        raise HarnessError("LLM returned a malformed Linear ticket draft")
    return _validated_draft(
        team,
        arguments.get("title"),
        arguments.get("description"),
    )
