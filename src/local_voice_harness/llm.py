from __future__ import annotations

import json
import time
import urllib.request

from .config import LLM_CHAT
from .cursor.jobs import cursor_turn
from .errors import HarnessError
from .notifications import notify

SYSTEM_PROMPT = (
    "You are a fast conversational voice assistant. Answer directly in one or two short "
    "sentences, usually under 40 words. Use natural spoken language without markdown or lists. "
    "You have a Cursor coding tool with access to the user's workspace. Use it for requests "
    "requiring code inspection, file edits, shell commands, or other software-engineering work. "
    "Cursor agents are managed through Herdr and can use configured MCP servers such as Linear. "
    "Delegate requests involving code or connected services to Cursor. If a Cursor job asks a "
    "question, send the user's answer back as a reply to that job. Use the Cursor tool's status "
    "or cancel action when the user asks about or cancels a job. Never claim you lack tool access."
)
QWEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cursor",
            "description": (
                "Run a Herdr-managed Cursor agent for code, commands, engineering tasks, "
                "and configured MCP services such as Linear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "repository": {"type": "string"},
                    "agent": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["submit", "reply", "status", "cancel"],
                    },
                    "job_id": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }
]


def qwen_turn(
    text: str,
    history: list[dict[str, object]] | None = None,
    cursor_session: str | None = None,
) -> tuple[str, str | None]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or [])[-8:],
        {"role": "user", "content": text},
    ]
    for tool_round in range(3):
        payload = json.dumps(
            {
                "model": "qwen3.5-4b",
                "messages": messages,
                "tools": QWEN_TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "temperature": 0.7,
                "max_tokens": 128,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            LLM_CHAT, data=payload, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        print(
            json.dumps(
                {
                    "stage": "llm",
                    "round": tool_round + 1,
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
        )
        message = result["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            answer = str(message.get("content") or "").strip()
            if not answer:
                raise HarnessError("Qwen returned an empty response")
            return answer, cursor_session
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            if name != "cursor":
                tool_result = f"Unknown tool: {name}"
            else:
                try:
                    arguments = json.loads(str(function.get("arguments") or "{}"))
                    task = str(arguments.get("task", "")).strip()
                    repository = str(arguments.get("repository", "")).strip() or None
                    agent = str(arguments.get("agent", "")).strip() or None
                    action = str(arguments.get("action", "submit")).strip() or "submit"
                    job_id = str(arguments.get("job_id", "")).strip() or cursor_session
                except (json.JSONDecodeError, AttributeError):
                    task, repository, agent, action, job_id = "", None, None, "submit", None
                if action in {"submit", "reply"} and not task:
                    tool_result = "Cursor tool error: task must not be empty"
                else:
                    notify("Cursor is working…")
                    try:
                        tool_result, cursor_session = cursor_turn(
                            task,
                            job_id if action == "reply" else None,
                            repository=repository,
                            agent=agent,
                            action=action,
                            job_id=job_id,
                        )
                    except Exception as exc:
                        tool_result = f"Cursor tool failed: {type(exc).__name__}: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id", "")),
                    "name": name,
                    "content": tool_result,
                }
            )
    raise HarnessError("Qwen exceeded the tool-call round limit")


def qwen_response(text: str, history: list[dict[str, object]] | None = None) -> str:
    return qwen_turn(text, history)[0]
