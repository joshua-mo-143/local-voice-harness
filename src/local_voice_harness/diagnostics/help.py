from __future__ import annotations

from ..responses import AssistantResponse

HARNESS_HELP_RECAP = (
    "I can talk with you, start a coding job, check status, cancel, list, "
    "dismiss, or repeat a job, hear more details or a digest of what you "
    "missed, consult the workspace or a pending job question, file a GitHub "
    "issue or a Linear ticket, inspect or change my settings, check whether "
    "I'm healthy, turn wake mode off, and hang up when you're done."
)


def harness_help_response() -> AssistantResponse:
    """Return the fixed spoken recap of shipped first-party voice capabilities."""

    return AssistantResponse.from_text(HARNESS_HELP_RECAP)
