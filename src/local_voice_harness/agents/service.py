"""Public agent-neutral service contracts."""

from ..cursor.service import (
    CursorTurnRequest,
    CursorTurnResult,
    StartJobRequest,
    acknowledge_deliveries,
    acknowledge_delivery,
    cancel_job,
    claim_delivery,
    count_jobs,
    cursor_turn,
    dismiss_announcement,
    job_status,
    list_jobs,
    mark_delivered,
    nuke_jobs,
    pending_results,
    recover_jobs,
    release_deliveries,
    release_delivery,
    repeat_announcement,
    reply_job,
    resolve_manual_reconciliation,
    start_follow_up,
    start_job,
)

AgentTurnRequest = CursorTurnRequest
AgentTurnResult = CursorTurnResult
StartAgentJobRequest = StartJobRequest
agent_turn = cursor_turn

__all__ = [
    "AgentTurnRequest",
    "AgentTurnResult",
    "StartAgentJobRequest",
    "acknowledge_deliveries",
    "acknowledge_delivery",
    "agent_turn",
    "cancel_job",
    "claim_delivery",
    "count_jobs",
    "dismiss_announcement",
    "job_status",
    "list_jobs",
    "mark_delivered",
    "nuke_jobs",
    "pending_results",
    "recover_jobs",
    "release_deliveries",
    "release_delivery",
    "repeat_announcement",
    "reply_job",
    "resolve_manual_reconciliation",
    "start_follow_up",
    "start_job",
]
