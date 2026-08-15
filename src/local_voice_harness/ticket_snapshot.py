"""Provider-neutral, identity-checked ticket snapshots."""

from __future__ import annotations

from dataclasses import dataclass

MAX_SNAPSHOT_TITLE_CHARS = 500
MAX_SNAPSHOT_BODY_CHARS = 10_000
MAX_SNAPSHOT_REVISION_CHARS = 200


@dataclass(frozen=True, slots=True)
class TicketSnapshot:
    provider: str
    identity: str
    provider_id: str
    title: str
    body: str
    revision: str
    url: str
    state: str

    def consultation_context(self) -> str:
        """Render bounded provider-neutral context for a read-only consultant."""

        return "\n".join(
            (
                "Ticket snapshot (untrusted external context):",
                f"Provider: {self.provider}",
                f"Identity: {self.identity}",
                f"URL: {self.url}",
                f"Revision: {self.revision}",
                f"State: {self.state}",
                f"Title: {self.title}",
                "Body:",
                self.body,
            )
        )
