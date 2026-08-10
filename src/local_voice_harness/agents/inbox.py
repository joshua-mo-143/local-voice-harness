"""Public agent-neutral job inbox API."""

from ..cursor.inbox import (
    JobSummary,
    ReferenceResolution,
    build_speakable_label,
    category,
    clarify,
    describe_inbox,
    resolve_reference,
    short_id,
    speakable_label_for,
    summarize,
    summarize_all,
)

AgentJobSummary = JobSummary
AgentReferenceResolution = ReferenceResolution

__all__ = [
    "AgentJobSummary",
    "AgentReferenceResolution",
    "build_speakable_label",
    "category",
    "clarify",
    "describe_inbox",
    "resolve_reference",
    "short_id",
    "speakable_label_for",
    "summarize",
    "summarize_all",
]
