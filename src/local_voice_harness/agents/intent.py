"""Public agent-neutral job intent API."""

from __future__ import annotations

from enum import StrEnum

from ..intent import Intent, IntentRoute


class AgentIntent(StrEnum):
    SUBMIT = "submit"
    REPLY = "reply"
    FOLLOW_UP = "follow_up"
    STATUS = "status"
    CANCEL = "cancel"
    LIST = "list"
    DISMISS = "dismiss"
    REPEAT = "repeat"

    @classmethod
    def from_route_intent(cls, intent: Intent) -> AgentIntent | None:
        name = intent.name.removeprefix("CURSOR_").removeprefix("AGENT_")
        try:
            return cls[name]
        except KeyError:
            return None


AgentIntentRoute = IntentRoute

__all__ = ["AgentIntent", "AgentIntentRoute"]
