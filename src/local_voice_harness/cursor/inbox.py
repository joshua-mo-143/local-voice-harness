"""Voice-accessible job inbox: speakable labels, summaries, and references.

This module is intentionally pure. It operates on already-loaded
:class:`~local_voice_harness.cursor.model.CursorJob` instances and never touches
the store, so it can be reasoned about and tested in isolation. The service
layer is responsible for loading jobs, applying the resolution decisions made
here, and persisting any state changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import (
    ACTIVE_STATUSES,
    CursorJob,
    JobStatus,
)

SHORT_ID_LENGTH = 4
_MAX_LABEL_WORDS = 6
_MAX_GROUP_ITEMS = 4
_TOKEN = re.compile(r"[a-z0-9]+")
# Words that never usefully distinguish one job reference from another.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "cancel",
        "cursor",
        "dismiss",
        "for",
        "it",
        "job",
        "jobs",
        "me",
        "my",
        "of",
        "on",
        "one",
        "please",
        "repeat",
        "status",
        "task",
        "that",
        "the",
        "this",
        "to",
        "work",
    }
)
# Category labels are ordered so summaries always read in the same order.
_CATEGORY_ACTIVE = "active"
_CATEGORY_AWAITING = "awaiting_user"
_CATEGORY_BLOCKED = "blocked"
_CATEGORY_COMPLETED = "completed"
_CATEGORY_FAILED = "failed"
_CATEGORY_CANCELLED = "cancelled"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def build_speakable_label(
    request: str,
    *,
    issue_key: str | None = None,
    github_repository: str | None = None,
    github_issue: int | None = None,
    github_pull_request: int | None = None,
) -> str:
    """Derive a short, speakable label from job metadata.

    Preference order favours the most unambiguous handle a user is likely to
    speak: an explicit external issue key, a GitHub issue or pull-request number,
    then the opening words of the request.
    """

    repository = (github_repository or "").split("/")[-1].strip()
    if issue_key:
        base = issue_key.strip()
    elif github_issue:
        base = f"issue {github_issue}"
    elif github_pull_request:
        base = f"pull request {github_pull_request}"
    else:
        base = ""
    if not base:
        words = _normalize(request).split()
        base = " ".join(words[:_MAX_LABEL_WORDS])
    label = base
    if repository and repository.casefold() not in base.casefold():
        label = f"{repository} {base}".strip()
    label = _normalize(label)
    return label or "untitled job"


def speakable_label_for(job: CursorJob) -> str:
    """Return the stored label, or derive one for legacy jobs without it."""

    if job.speakable_label:
        return job.speakable_label
    return build_speakable_label(
        job.request,
        issue_key=job.issue_key,
        github_repository=job.github_repository,
        github_issue=job.github_issue,
        github_pull_request=job.github_pull_request,
    )


def short_id(job: CursorJob) -> str:
    return job.id[:SHORT_ID_LENGTH]


def category(job: CursorJob) -> str:
    if job.status == JobStatus.AWAITING_USER:
        return _CATEGORY_AWAITING
    if job.status == JobStatus.BLOCKED:
        return _CATEGORY_BLOCKED
    if job.status in ACTIVE_STATUSES:
        return _CATEGORY_ACTIVE
    return job.status.value


@dataclass(frozen=True, slots=True)
class JobSummary:
    id: str
    short_id: str
    label: str
    status: JobStatus
    category: str
    detail: str

    def spoken(self) -> str:
        return f"{self.label} ({self.short_id})"


def _detail(job: CursorJob) -> str:
    if job.status == JobStatus.AWAITING_USER:
        return _normalize(job.question or job.result or "")
    if job.status == JobStatus.BLOCKED:
        return "Manual attention required in Herdr"
    if job.status == JobStatus.FAILED:
        return ""
    if job.status in {JobStatus.COMPLETED, JobStatus.CANCELLED}:
        return _normalize(job.result or "")
    tier = f"{job.workflow_tier.value} " if job.workflow_tier is not None else ""
    return f"{tier}{job.workflow_phase.value}".replace("_", " ")


def summarize(job: CursorJob) -> JobSummary:
    return JobSummary(
        id=job.id,
        short_id=short_id(job),
        label=speakable_label_for(job),
        status=job.status,
        category=category(job),
        detail=_detail(job),
    )


def summarize_all(jobs: list[CursorJob]) -> list[JobSummary]:
    return [summarize(job) for job in jobs]


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _group_phrase(label: str, summaries: list[JobSummary]) -> str | None:
    if not summaries:
        return None
    shown = summaries[:_MAX_GROUP_ITEMS]
    names = [item.label for item in shown]
    remainder = len(summaries) - len(shown)
    if remainder > 0:
        names.append(f"{remainder} more")
    return f"{label}: {_join(names)}"


def describe_inbox(jobs: list[CursorJob]) -> str:
    """Produce a concise spoken overview of the whole job inbox."""

    if not jobs:
        return "You have no Cursor jobs."
    summaries = summarize_all(jobs)
    grouped: dict[str, list[JobSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.category, []).append(summary)
    phrases: list[str] = []
    for label, key in (
        ("waiting for you", _CATEGORY_AWAITING),
        ("in progress", _CATEGORY_ACTIVE),
        ("blocked", _CATEGORY_BLOCKED),
        ("recently finished", _CATEGORY_COMPLETED),
        ("failed", _CATEGORY_FAILED),
        ("cancelled", _CATEGORY_CANCELLED),
    ):
        phrase = _group_phrase(label, grouped.get(key, []))
        if phrase is not None:
            phrases.append(phrase)
    count = len(jobs)
    noun = "job" if count == 1 else "jobs"
    return f"You have {count} Cursor {noun}. " + ". ".join(phrases) + "."


@dataclass(frozen=True, slots=True)
class ReferenceResolution:
    matches: tuple[JobSummary, ...]

    @property
    def unique(self) -> JobSummary | None:
        return self.matches[0] if len(self.matches) == 1 else None

    @property
    def ambiguous(self) -> bool:
        return len(self.matches) > 1


_TIER_STRONG = 2
_TIER_WEAK = 1
_TIER_NONE = 0


def _match_tier(
    summary: JobSummary, job: CursorJob, tokens: set[str], lowered: str
) -> int:
    """Rank how specifically a reference identifies a job.

    A strong tier means an unambiguous handle (short id, full id, issue key, or
    issue/PR number). A weak tier means the reference only shares descriptive
    words with the job label, which is far more likely to collide.
    """

    if summary.short_id in tokens or job.id in tokens:
        return _TIER_STRONG
    if job.issue_key:
        key = job.issue_key.casefold()
        key_tokens = set(_tokens(key))
        if key in lowered or (key_tokens and key_tokens <= tokens):
            return _TIER_STRONG
    for number in (job.github_issue, job.github_pull_request):
        if number is not None and str(number) in tokens:
            return _TIER_STRONG
    label_tokens = {
        token
        for token in _tokens(summary.label)
        if len(token) >= 3 and token not in _STOPWORDS
    }
    if label_tokens & tokens:
        return _TIER_WEAK
    return _TIER_NONE


def resolve_reference(jobs: list[CursorJob], reference: str) -> ReferenceResolution:
    """Match a spoken reference against candidate jobs.

    Strong identifier matches take precedence over descriptive word overlap, so
    naming a job by its id or issue number is decisive even when its label words
    also appear on other jobs. When only weak matches exist they are all
    returned, letting the caller ask for clarification rather than guess.
    """

    tokens = set(_tokens(reference))
    if not tokens:
        return ReferenceResolution(())
    lowered = reference.casefold()
    strong: list[JobSummary] = []
    weak: list[JobSummary] = []
    for job in jobs:
        summary = summarize(job)
        tier = _match_tier(summary, job, tokens, lowered)
        if tier == _TIER_STRONG:
            strong.append(summary)
        elif tier == _TIER_WEAK:
            weak.append(summary)
    return ReferenceResolution(tuple(strong or weak))


def clarify(summaries: list[JobSummary], action: str) -> str:
    """Build a clarification prompt listing the candidate jobs."""

    options = _join([summary.spoken() for summary in summaries])
    if not options:
        return f"I could not find a job to {action}."
    return f"There are several jobs I could {action}. Which one: {options}?"
