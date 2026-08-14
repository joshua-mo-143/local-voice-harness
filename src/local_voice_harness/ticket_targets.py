"""Deterministic extraction of explicit ticket targets from trusted user text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .integrations.github import GitHubClient, GitHubError

TicketSource = Literal["github", "linear"]
MISSING_ISSUE_SCOPE_ERROR = (
    "bare issue number requires an unambiguous issues-page scope"
)
MISSING_ISSUE_SCOPE_RESPONSE = (
    "Those issue numbers are ambiguous. Open a repository-scoped issues page "
    "or provide fully qualified references."
)

_GITHUB_URL = re.compile(
    r"https://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/"
    r"(?P<number>\d+)/?(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)
_GITHUB_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)#(?P<number>\d+)(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_GITHUB_REPOSITORY_TARGET = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/"
    r"[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)
_GITHUB_REPOSITORY_ISSUE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)\s+issues?\s+#?"
    r"(?P<number>\d+)(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)
_GITHUB_IN_REPOSITORY = re.compile(
    r"\bissue\s+#?(?P<number>\d+)\s+(?:in|from)\s+"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)\b",
    re.IGNORECASE,
)
_LINEAR_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])(?P<team>[A-Za-z][A-Za-z0-9]+)-"
    r"(?P<number>\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCOPED_PREFIX = re.compile(r"\b(?:issues?|tickets?)\s+", re.IGNORECASE)
_NUMBER_AT = re.compile(r"#?\d+")
_RANGE_SEPARATOR_AT = re.compile(
    r"(?:\s+(?:through|to)\s+|\s*[-–—]\s*)",
    re.IGNORECASE,
)
_NUMBER_WORD_AT = re.compile(
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|and)\b",
    re.IGNORECASE,
)
_SMALL_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS_NUMBER_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_LINEAR_TEAM = re.compile(r"^[A-Za-z][A-Za-z0-9]+$")
_MAX_TICKET_RANGE_SIZE = 25


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(
        start < existing_end and end > existing_start
        for existing_start, existing_end in spans
    )


def _spoken_number_at(text: str, start: int) -> tuple[int, int] | None:
    """Parse a bounded English cardinal number beginning at ``start``."""

    position = start
    end = start
    total = 0
    group = 0
    last: str | None = None
    consumed = False
    used_thousand = False
    while match := _NUMBER_WORD_AT.match(text, position):
        word = match.group(0).casefold()
        if word in _SMALL_NUMBER_WORDS:
            if last not in {None, "tens", "hundred", "scale", "and"}:
                break
            group += _SMALL_NUMBER_WORDS[word]
            last = "small"
        elif word in _TENS_NUMBER_WORDS:
            if last not in {None, "hundred", "scale", "and"}:
                break
            group += _TENS_NUMBER_WORDS[word]
            last = "tens"
        elif word == "hundred":
            if last != "small" or not 1 <= group <= 9:
                break
            group *= 100
            last = "hundred"
        elif word == "thousand":
            if not 1 <= group <= 999 or used_thousand:
                break
            total += group * 1_000
            group = 0
            last = "scale"
            used_thousand = True
        else:
            separator_end = match.end()
            next_start = separator_end
            whitespace = re.match(r"[\s-]+", text[next_start:])
            if whitespace is not None:
                next_start += whitespace.end()
            next_word = _NUMBER_WORD_AT.match(text, next_start)
            if (
                last not in {"hundred", "scale"}
                or next_word is None
                or next_word.group(0).casefold()
                not in {*_SMALL_NUMBER_WORDS, *_TENS_NUMBER_WORDS}
            ):
                break
            last = "and"

        consumed = True
        end = match.end()
        separator = re.match(r"[\s-]+", text[end:])
        if separator is None:
            break
        position = end + separator.end()

    return (total + group, end) if consumed else None


def _number_at(text: str, position: int) -> tuple[int, int] | None:
    digit = _NUMBER_AT.match(text, position)
    if digit is not None:
        return int(digit.group(0).removeprefix("#")), digit.end()
    return _spoken_number_at(text, position)


@dataclass(frozen=True, slots=True)
class _ScopedItem:
    raw: str
    position: int
    end: int
    value: int | None
    requested_count: int = 1
    error: str | None = None


def _scoped_items(text: str) -> list[_ScopedItem]:
    """Return raw text, position, end, and value for scoped ticket lists."""

    items: list[_ScopedItem] = []
    for prefix in _SCOPED_PREFIX.finditer(text):
        position = prefix.end()
        while True:
            start = position
            parsed = _number_at(text, start)
            if parsed is None:
                break
            value, end = parsed

            separator = _RANGE_SEPARATOR_AT.match(text, end)
            range_end = (
                _number_at(text, separator.end()) if separator is not None else None
            )
            if separator is not None and range_end is not None:
                final_value, final_end = range_end
                raw = text[start:final_end]
                requested_count = abs(final_value - value) + 1
                if value < 1 or final_value < 1:
                    items.append(
                        _ScopedItem(
                            raw,
                            start,
                            final_end,
                            None,
                            requested_count,
                            "ticket range endpoints must be positive",
                        )
                    )
                elif final_value < value:
                    items.append(
                        _ScopedItem(
                            raw,
                            start,
                            final_end,
                            None,
                            requested_count,
                            "ticket range must be ascending",
                        )
                    )
                elif requested_count > _MAX_TICKET_RANGE_SIZE:
                    items.append(
                        _ScopedItem(
                            raw,
                            start,
                            final_end,
                            None,
                            requested_count,
                            f"ticket range cannot exceed {_MAX_TICKET_RANGE_SIZE} tickets",
                        )
                    )
                else:
                    items.extend(
                        _ScopedItem(str(number), start, final_end, number)
                        for number in range(value, final_value + 1)
                    )
                end = final_end
            else:
                items.append(_ScopedItem(text[start:end], start, end, value))
            position = end

            spaces = re.match(r"\s*", text[position:])
            assert spaces is not None
            position += spaces.end()
            if position < len(text) and text[position] == ",":
                position += 1
                spaces = re.match(r"\s*", text[position:])
                assert spaces is not None
                position += spaces.end()
                conjunction = re.match(r"and\b\s*", text[position:], re.IGNORECASE)
                if conjunction is not None:
                    position += conjunction.end()
                continue
            conjunction = re.match(r"(?:and\b|&)\s*", text[position:], re.IGNORECASE)
            if conjunction is None:
                break
            position += conjunction.end()
    return items


@dataclass(frozen=True, slots=True)
class TicketReference:
    """One unique explicit target, or one deterministic extraction rejection."""

    raw: str
    position: int
    source: TicketSource | None
    canonical: str | None
    scoped: bool = False
    error: str | None = None

    @property
    def label(self) -> str:
        return self.canonical or self.raw


@dataclass(frozen=True, slots=True)
class TicketExtraction:
    references: tuple[TicketReference, ...]
    requested_count: int

    @property
    def batch_requested(self) -> bool:
        return self.requested_count > 1

    @property
    def has_unresolved_scope(self) -> bool:
        return any(
            reference.scoped
            and reference.canonical is None
            and reference.error == MISSING_ISSUE_SCOPE_ERROR
            for reference in self.references
        )


def _github_reference(
    raw: str,
    position: int,
    owner: str,
    repository: str,
    number_text: str,
) -> TicketReference:
    try:
        canonical_repository = GitHubClient.validate_repository(f"{owner}/{repository}")
    except GitHubError as exc:
        return TicketReference(raw, position, None, None, error=str(exc))
    number = int(number_text)
    if number < 1:
        return TicketReference(
            raw,
            position,
            "github",
            None,
            error="GitHub issue number must be positive",
        )
    return TicketReference(
        raw,
        position,
        "github",
        f"{canonical_repository}#{number}",
    )


def _linear_reference(
    raw: str, position: int, team: str, number_text: str
) -> TicketReference:
    number = int(number_text)
    if number < 1:
        return TicketReference(
            raw,
            position,
            "linear",
            None,
            error="Linear issue number must be positive",
        )
    return TicketReference(raw, position, "linear", f"{team.upper()}-{number}")


def _scoped_reference(
    raw: str,
    position: int,
    number_text: str,
    *,
    scope_source: str | None,
    scope: str | None,
) -> TicketReference:
    number = int(number_text)
    if number < 1:
        return TicketReference(
            raw,
            position,
            None,
            None,
            scoped=True,
            error="issue number must be positive",
        )
    if not scope or scope_source not in {"github", "linear"}:
        return TicketReference(
            raw,
            position,
            None,
            None,
            scoped=True,
            error=MISSING_ISSUE_SCOPE_ERROR,
        )
    if scope_source == "github":
        try:
            repository = GitHubClient.validate_repository(scope)
        except GitHubError as exc:
            return TicketReference(
                raw,
                position,
                "github",
                None,
                scoped=True,
                error=str(exc),
            )
        return TicketReference(
            raw,
            position,
            "github",
            f"{repository}#{number}",
            scoped=True,
        )
    if _LINEAR_TEAM.fullmatch(scope) is None:
        return TicketReference(
            raw,
            position,
            "linear",
            None,
            scoped=True,
            error="Linear team scope is invalid",
        )
    return TicketReference(
        raw,
        position,
        "linear",
        f"{scope.upper()}-{number}",
        scoped=True,
    )


def extract_ticket_targets(
    text: str,
    *,
    scope_source: str | None = None,
    scope: str | None = None,
) -> TicketExtraction:
    """Return an ordered, request-local set of explicit ticket references."""
    candidates: list[TicketReference] = []
    specific_spans: list[tuple[int, int]] = []

    for pattern in (
        _GITHUB_URL,
        _GITHUB_REFERENCE,
        _GITHUB_REPOSITORY_ISSUE,
        _GITHUB_IN_REPOSITORY,
    ):
        for match in pattern.finditer(text):
            candidates.append(
                _github_reference(
                    match.group(0),
                    match.start(),
                    match.group("owner"),
                    match.group("repo"),
                    match.group("number"),
                )
            )
            specific_spans.append(match.span())

    specific_spans.extend(
        match.span() for match in _GITHUB_REPOSITORY_TARGET.finditer(text)
    )

    for match in _LINEAR_REFERENCE.finditer(text):
        if _overlaps(match.start(), match.end(), specific_spans):
            continue
        candidates.append(
            _linear_reference(
                match.group(0),
                match.start(),
                match.group("team"),
                match.group("number"),
            )
        )
        specific_spans.append(match.span())

    requested_count = len(candidates)
    for item in _scoped_items(text):
        if _overlaps(item.position, item.end, specific_spans):
            continue
        requested_count += item.requested_count
        if item.error is not None:
            candidates.append(
                TicketReference(
                    item.raw,
                    item.position,
                    None,
                    None,
                    scoped=True,
                    error=item.error,
                )
            )
        else:
            assert item.value is not None
            candidates.append(
                _scoped_reference(
                    item.raw,
                    item.position,
                    str(item.value),
                    scope_source=scope_source,
                    scope=scope,
                )
            )

    candidates.sort(key=lambda candidate: candidate.position)
    unique: list[TicketReference] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = (
            candidate.canonical.casefold()
            if candidate.canonical is not None
            else f"{candidate.raw.casefold()}:{candidate.error or ''}"
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return TicketExtraction(tuple(unique), requested_count)
