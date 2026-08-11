from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .vocabulary import Replacement


@dataclass(frozen=True, slots=True)
class TranscriptReplacement:
    spoken: str
    written: str

    def apply(self, text: str) -> str:
        return re.sub(
            rf"(?<!\w){re.escape(self.spoken)}(?!\w)",
            lambda _match: self.written,
            text,
            flags=re.IGNORECASE,
        )


def effective_replacements(
    user_replacements: Iterable[Replacement],
    static_replacements: Mapping[str, str],
) -> tuple[TranscriptReplacement, ...]:
    """Return the ordered rules that determine one normalized transcript."""

    user = tuple(user_replacements)
    overridden = {replacement.spoken.casefold() for replacement in user}
    rules = [
        TranscriptReplacement(replacement.spoken, replacement.written)
        for replacement in user
    ]
    rules.extend(
        TranscriptReplacement(spoken, written)
        for spoken, written in static_replacements.items()
        if spoken.casefold() not in overridden
    )
    return tuple(rules)


def normalize_transcript(
    text: str, replacements: Iterable[TranscriptReplacement]
) -> str:
    for replacement in replacements:
        text = replacement.apply(text)
    return text
