"""Deterministic extraction of explicit ticket targets from trusted user text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .integrations.github import GitHubClient, GitHubError

TicketSource = Literal["github", "linear"]

_GITHUB_URL = re.compile(
    r"https://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/"
    r"(?P<number>\d+)/?(?![A-Za-z0-9_./-])",
    re.IGNORECASE,
)
_GITHUB_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)#(?P<number>\d+)(?![A-Za-z0-9_.-])",
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
_SCOPED_LIST = re.compile(
    r"\b(?:issues?|tickets?)\s+"
    r"(?P<items>#?\d+(?:(?:\s*,\s*(?:and\s+)?|\s+(?:and|&)\s+)#?\d+)*)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"#?\d+")
_LINEAR_TEAM = re.compile(r"^[A-Za-z][A-Za-z0-9]+$")


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(
        start < existing_end and end > existing_start
        for existing_start, existing_end in spans
    )


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
            error="bare issue number requires an unambiguous issues-page scope",
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

    for pattern in (_GITHUB_URL, _GITHUB_REFERENCE, _GITHUB_IN_REPOSITORY):
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

    for issue_list in _SCOPED_LIST.finditer(text):
        items_start = issue_list.start("items")
        for match in _NUMBER.finditer(issue_list.group("items")):
            start = items_start + match.start()
            end = items_start + match.end()
            if _overlaps(start, end, specific_spans):
                continue
            raw = match.group(0)
            candidates.append(
                _scoped_reference(
                    raw,
                    start,
                    raw.removeprefix("#"),
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
    return TicketExtraction(tuple(unique), len(candidates))
