from __future__ import annotations

import json

from .browser_context import enrich_request
from .components import component_usage, llm_ready, start_components
from .config import CURSOR_PATTERN, FORK_PATTERN, PID_PATH, STT_SOCKET, TTS_SOCKET
from .cursor.jobs import (
    DeliveryClaims,
    acknowledge_deliveries,
    cursor_turn,
    release_deliveries,
)
from .errors import HarnessError
from .ipc import socket_ready
from .llm import qwen_response
from .tts.client import stream_and_play


def respond(text: str) -> None:
    text = text.strip()
    if not text:
        raise HarnessError("request text is empty")
    delivery_claims: DeliveryClaims = []
    with component_usage():
        try:
            start_components()
            print(f"You: {text}")
            request = enrich_request(text)
            github_repository = getattr(request, "github_repository", None)
            github_pull_request = getattr(request, "github_pull_request", None)
            fork_requested = bool(FORK_PATTERN.search(text))
            github_arguments = (
                {
                    "github_repository": github_repository,
                    "fork_requested": fork_requested,
                    "github_pull_request": github_pull_request,
                }
                if github_repository or fork_requested or github_pull_request
                else {}
            )
            if CURSOR_PATTERN.match(text):
                response = cursor_turn(
                    request,
                    **github_arguments,
                    delivery_claims=delivery_claims,
                )[0]
            else:
                response = qwen_response(
                    request,
                    **github_arguments,
                    delivery_claims=delivery_claims,
                )
            print(f"Assistant: {response}")
            stream_and_play(response)
            acknowledge_deliveries(delivery_claims)
        except Exception:
            release_deliveries(delivery_claims)
            raise


def status() -> None:
    print(
        json.dumps(
            {
                "stt_ready": socket_ready(STT_SOCKET),
                "llm_ready": llm_ready(),
                "tts_ready": socket_ready(TTS_SOCKET),
                "recording": PID_PATH.exists(),
            }
        )
    )
