from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import BackendSettings
from .errors import HarnessError
from .integrations.github import GitHubClient, GitHubError
from .llm_transport import ChatCompletionRequest, LlmTransport

MAX_ISSUE_TITLE_CHARS = 200
MAX_ISSUE_BODY_CHARS = 10_000
_REPOSITORY_IN_UTTERANCE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)"
    r"(?![A-Za-z0-9_.-])"
)
_DRAFT_TOOL = {
    "type": "function",
    "function": {
        "name": "draft_github_issue",
        "description": "Draft one concise GitHub issue from the user's request.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": MAX_ISSUE_TITLE_CHARS},
                "body": {"type": "string", "maxLength": MAX_ISSUE_BODY_CHARS},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class GitHubIssueDraft:
    repository: str
    title: str
    body: str


def repository_from_utterance(utterance: str) -> str | None:
    match = _REPOSITORY_IN_UTTERANCE.search(utterance)
    if match is None:
        return None
    try:
        return GitHubClient.validate_repository(
            f"{match.group('owner')}/{match.group('repo')}"
        )
    except GitHubError:
        return None


def _validated_draft(repository: str, title: object, body: object) -> GitHubIssueDraft:
    try:
        repository = GitHubClient.validate_repository(repository)
    except GitHubError as exc:
        raise HarnessError(str(exc)) from exc
    if not isinstance(title, str) or not title.strip():
        raise HarnessError("GitHub issue draft requires a non-empty title")
    if not isinstance(body, str) or not body.strip():
        raise HarnessError("GitHub issue draft requires a non-empty body")
    title = " ".join(title.split())
    body = body.strip()
    if len(title) > MAX_ISSUE_TITLE_CHARS:
        raise HarnessError("GitHub issue draft title is too long")
    if len(body) > MAX_ISSUE_BODY_CHARS:
        raise HarnessError("GitHub issue draft body is too long")
    return GitHubIssueDraft(repository, title, body)


def draft_github_issue(
    utterance: str,
    repository: str,
    *,
    settings: BackendSettings | None = None,
) -> GitHubIssueDraft:
    trusted_request = utterance.strip()
    if not trusted_request:
        raise HarnessError("GitHub issue creation requires a spoken request")
    transport = LlmTransport.from_settings(settings)
    message = transport.chat_completion(
        ChatCompletionRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's trusted spoken request into one GitHub issue. "
                        "Preserve concrete requirements, do not invent acceptance criteria, "
                        "and do not include the target repository in the title. Return only "
                        "the forced tool call."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"repository": repository, "request": trusted_request}
                    ),
                },
            ],
            temperature=0,
            max_tokens=1024,
            stream=False,
            tools=[_DRAFT_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "draft_github_issue"},
            },
            parallel_tool_calls=False,
        )
    )
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise HarnessError("LLM did not return a GitHub issue draft")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != "draft_github_issue":
        raise HarnessError("LLM returned a malformed GitHub issue draft")
    try:
        arguments = json.loads(str(function.get("arguments") or "{}"))
    except json.JSONDecodeError as exc:
        raise HarnessError("LLM returned a malformed GitHub issue draft") from exc
    if not isinstance(arguments, dict):
        raise HarnessError("LLM returned a malformed GitHub issue draft")
    return _validated_draft(
        repository,
        arguments.get("title"),
        arguments.get("body"),
    )
