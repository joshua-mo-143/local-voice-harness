from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """One assistant outcome rendered for speech and visual display."""

    spoken_text: str
    display_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.spoken_text, str):
            raise TypeError("spoken_text must be a string")
        if not isinstance(self.display_text, str):
            raise TypeError("display_text must be a string")

    @classmethod
    def from_text(cls, text: str) -> AssistantResponse:
        """Adapt a legacy response that is identical in both channels."""

        return cls(spoken_text=text, display_text=text)


ResponseLike = str | AssistantResponse


def as_assistant_response(response: ResponseLike) -> AssistantResponse:
    """Return a typed response without changing existing plain-string content."""

    if isinstance(response, AssistantResponse):
        return response
    if isinstance(response, str):
        return AssistantResponse.from_text(response)
    raise TypeError("assistant response must be a string or AssistantResponse")
