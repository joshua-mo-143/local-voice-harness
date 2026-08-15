from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_SPOKEN_UTTERANCE_WORDS = 12
_CLAUSE_SPLIT = re.compile(r"[.!?;]")
_SPOKEN_UTTERANCE_ACK = re.compile(r" for “[^”]+”$")


def spoken_utterance_slice(
    utterance: str,
    *,
    max_words: int = _MAX_SPOKEN_UTTERANCE_WORDS,
) -> str:
    """Return the first clause, or at most about ``max_words``, of a trusted utterance."""

    text = " ".join(utterance.split())
    if not text:
        return ""
    clause = _CLAUSE_SPLIT.split(text, maxsplit=1)[0].strip().rstrip(" ,:—-")
    words = clause.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return clause


def with_spoken_utterance_ack(sentence: str, utterance: str) -> str:
    """Name a bounded trusted-utterance slice in an existing spoken sentence."""

    spoken_slice = spoken_utterance_slice(utterance)
    if not spoken_slice:
        return sentence
    core = sentence[:-1] if sentence.endswith(".") else sentence
    return f"{core} for “{spoken_slice}.”"


def without_spoken_utterance_ack(spoken: str) -> str:
    """Drop a trailing trusted-utterance ack so another identity can own speech."""

    return _SPOKEN_UTTERANCE_ACK.sub(".", spoken)


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
