"""Public agent-neutral durable delivery API."""

from ..cursor.delivery import (
    DELIVERABLE_STATUSES,
    DELIVERY_CLAIM_SECONDS,
    DELIVERY_RENEW_SECONDS,
    DELIVERY_RETRY_SECONDS,
    DELIVERY_WINDOW,
    DeliveryClaim,
    DeliveryClaims,
    acknowledge_deliveries,
    acknowledge_delivery,
    claim_delivery,
    pending_deliveries,
    release_deliveries,
    release_delivery,
    renew_delivery,
)

AgentDeliveryClaim = DeliveryClaim
AgentDeliveryClaims = DeliveryClaims

__all__ = [
    "DELIVERABLE_STATUSES",
    "DELIVERY_CLAIM_SECONDS",
    "DELIVERY_RENEW_SECONDS",
    "DELIVERY_RETRY_SECONDS",
    "DELIVERY_WINDOW",
    "AgentDeliveryClaim",
    "AgentDeliveryClaims",
    "acknowledge_deliveries",
    "acknowledge_delivery",
    "claim_delivery",
    "pending_deliveries",
    "release_deliveries",
    "release_delivery",
    "renew_delivery",
]
