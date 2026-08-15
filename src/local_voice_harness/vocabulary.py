"""Local personal vocabulary and entity alias store.

This module keeps a user-owned, on-disk store of speech-recognition corrections
and spoken aliases for repositories and issues. It is intentionally local: the
store is never uploaded anywhere. Its content only leaves the machine when a
resolved alias or corrected transcription becomes part of a prompt that the user
explicitly asked an agent to run.

Storage
-------
The store is a single JSON document (schema ``version`` 3) at
``$XDG_CONFIG_HOME/voice-harness/vocabulary.json`` (default
``~/.config/voice-harness/vocabulary.json``). JSON is used instead of TOML
because it round-trips losslessly with only the standard library, so the harness
keeps its zero-runtime-dependency policy. The file is written privately
(``0o600``) with sorted keys so backups and diffs are stable.

Repository and issue aliases may also be created by a confirmation-gated spoken
path. That path writes only through :func:`add_alias` and takes ``source`` plus
``issue_reference`` or ``repository_reference`` from a trusted focused or
just-used fragment. It never parses a target from a second transcription.
Replacements and pronunciations remain CLI-only. The store still never silently
learns.

Three entry kinds are stored:

``replacements``
    Deterministic speech-to-text corrections. ``spoken`` (the recognized text)
    is replaced with ``written`` wherever it appears as a whole, whitespace- and
    case-insensitive phrase.
``aliases``
    A spoken ``phrase`` that resolves to a canonical entity ``target`` before
    routing. Each alias persists ``phrase``, provider ``source``, and that
    provider's canonical ``target``. ``kind`` remains inferred from the target
    form for display. Legacy untagged GitHub and ``TEAM-79`` aliases still load.
``pronunciations``
    A displayed ``written`` name and its user-owned ``spoken`` pronunciation.
    These entries are consumed only by the TTS speech renderer.

Normalization
-------------
Replacement sources and alias phrases are normalized by trimming, collapsing
internal whitespace to single spaces, and case-folding for comparison. Alias
phrases are stored in their case-folded form; replacement written targets and
alias targets preserve their case. Matching against transcribed text is always
case-insensitive and whitespace-flexible, and only matches whole phrases (it will
not fire inside a longer word).

Precedence and conflicts
-------------------------
When resolving text, longer alias phrases are matched before shorter ones so the
most specific alias wins. In the STT correction layer the precedence is:

1. user vocabulary replacements (highest),
2. the process-start ``dictation.replacements`` snapshot,
3. built-in defaults.

An alias phrase may map to exactly one canonical target. Adding a replacement or
alias whose key already resolves to a different value is rejected as a conflict
(pass ``force`` to overwrite) so the store never silently learns or overrides a
correction without explicit approval. A stored file that contains the same key
twice with different values is treated as ambiguous and rejected on load.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from . import config
from .errors import HarnessError
from .integrations.registry import RegistryInput, enabled_integrations
from .responses import AssistantResponse

SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = {1, 2, SCHEMA_VERSION}
_MAX_SPOKEN_ALIAS_PHRASE_CHARS = 80
_SPOKEN_ALIAS_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:call|name)\s+(?:this|that)\s+"
    r"(?:(?P<kind>repo(?:sitory)?|issue)\s+)?"
    r"(?P<phrase>.+?)\s*$",
    re.IGNORECASE,
)

_REPOSITORY = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9_.-]+)$"
)
_ISSUE = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)#(?P<number>[1-9]\d*)$"
)
_LINEAR = re.compile(
    r"^(?P<team>[A-Za-z][A-Za-z0-9]+)-(?P<number>[1-9]\d*)$",
    re.IGNORECASE,
)
_ZENDESK = re.compile(
    r"^(?P<subdomain>[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)#"
    r"(?P<number>\d+)$",
    re.IGNORECASE,
)


class VocabularyError(HarnessError):
    """A vocabulary store could not be read, validated, or updated."""


class AliasKind(StrEnum):
    REPOSITORY = "repository"
    ISSUE = "issue"
    LINEAR = "linear"


class SpokenAliasStatus(StrEnum):
    """Outcome of preparing one confirmation-gated spoken alias write."""

    READY = "ready"
    MISSING_PHRASE = "missing_phrase"
    MISSING_TARGET = "missing_target"
    INVALID_TARGET = "invalid_target"
    NO_CHANGE = "no_change"


_SPOKEN_PRONUNCIATION_PUNCTUATION = frozenset(" -'’.,+")


def _normalize_phrase(value: str) -> str:
    """Collapse whitespace and case-fold a spoken phrase for comparison."""

    collapsed = " ".join(value.split())
    if not collapsed:
        raise VocabularyError("phrase must not be empty")
    return collapsed.casefold()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _canonical_target(target: str) -> tuple[str, AliasKind]:
    candidate = target.strip()
    issue = _ISSUE.fullmatch(candidate)
    if issue is not None and issue.group("repo") not in {".", ".."}:
        owner, repo, number = issue.group("owner", "repo", "number")
        return f"{owner}/{repo}#{number}", AliasKind.ISSUE
    repository = _REPOSITORY.fullmatch(candidate.removesuffix(".git"))
    if repository is not None and repository.group("repo") not in {".", ".."}:
        owner, repo = repository.group("owner", "repo")
        return f"{owner}/{repo}", AliasKind.REPOSITORY
    linear = _LINEAR.fullmatch(candidate)
    if linear is not None:
        team, number = linear.group("team", "number")
        return f"{team.upper()}-{int(number)}", AliasKind.LINEAR
    raise VocabularyError(
        f"alias target {target!r} must be an owner/repository, "
        "owner/repository#number, or TEAM-79 Linear reference"
    )


def _legacy_alias_source(kind: AliasKind) -> str:
    return "linear" if kind == AliasKind.LINEAR else "github"


def _alias_identity(target: str) -> tuple[str, str, AliasKind]:
    """Infer canonical target, owning source, and kind from target form."""

    try:
        canonical, kind = _canonical_target(target)
    except VocabularyError:
        zendesk = _ZENDESK.fullmatch(target.strip())
        if zendesk is None:
            raise
        subdomain = zendesk.group("subdomain").casefold()
        return f"{subdomain}#{int(zendesk.group('number'))}", "zendesk", AliasKind.ISSUE
    return canonical, _legacy_alias_source(kind), kind


def _owned_canonical(integration: object, target: str) -> tuple[str, AliasKind] | None:
    for owns_name, canonicalize_name, default_kind in (
        ("owns_issue_reference", "canonicalize_issue_reference", AliasKind.ISSUE),
        (
            "owns_repository_reference",
            "canonicalize_repository_reference",
            AliasKind.REPOSITORY,
        ),
    ):
        owns = getattr(integration, owns_name, None)
        if owns is None or not owns(target):
            continue
        canonicalize = getattr(integration, canonicalize_name, None)
        canonical = str(canonicalize(target)) if canonicalize is not None else target
        canonical = canonical.strip()
        if not canonical:
            continue
        try:
            canonical, kind = _canonical_target(canonical)
        except VocabularyError:
            kind = default_kind
        return canonical, kind
    return None


def resolve_owned_alias_target(
    target: str,
    *,
    source: str | None = None,
    integrations: RegistryInput = None,
) -> tuple[str, str, AliasKind]:
    """Canonicalize ``target`` through one enabled provider ownership hook."""

    expected = source.strip().casefold() if source else None
    for integration in enabled_integrations(integrations):
        name = str(getattr(integration, "name", "")).strip().casefold()
        if not name or (expected is not None and name != expected):
            continue
        owned = _owned_canonical(integration, target)
        if owned is None:
            continue
        canonical, kind = owned
        return canonical, name, kind
    raise VocabularyError("no enabled provider owns that alias target")


def _spoken_kind_hint(value: str | None) -> AliasKind | None:
    if value is None:
        return None
    folded = value.casefold()
    if folded in {"repo", "repository"}:
        return AliasKind.REPOSITORY
    if folded == "issue":
        return AliasKind.ISSUE
    return None


def parse_spoken_alias_request(utterance: str) -> tuple[str, AliasKind | None] | None:
    """Extract a spoken alias phrase from the original utterance.

    The target is never taken from this text. A missing or empty phrase fails
    closed so a second transcription cannot become the written identity.
    """

    match = _SPOKEN_ALIAS_REQUEST.fullmatch(utterance.strip().rstrip(".?!"))
    if match is None:
        return None
    phrase = " ".join(match.group("phrase").split())
    if (
        not phrase
        or len(phrase) > _MAX_SPOKEN_ALIAS_PHRASE_CHARS
        or phrase.casefold() in {"repo", "repository", "issue"}
    ):
        return None
    return phrase, _spoken_kind_hint(match.group("kind"))


def select_trusted_alias_target(
    kind_hint: AliasKind | None,
    *,
    focused_repository: str | None = None,
    focused_issue: str | None = None,
) -> str | None:
    """Choose one trusted fragment identity. Never parse the utterance."""

    repository = focused_repository.strip() if focused_repository else None
    issue = focused_issue.strip() if focused_issue else None
    if kind_hint == AliasKind.REPOSITORY:
        return repository
    if kind_hint == AliasKind.ISSUE:
        return issue
    return issue or repository


@dataclass(frozen=True, slots=True)
class PendingSpokenAlias:
    """One validated spoken alias awaiting an explicit user decision."""

    trusted_utterance: str
    phrase: str
    target: str
    kind: AliasKind
    source: str = "github"
    existing_target: str | None = None
    replace: bool = False


@dataclass(frozen=True, slots=True)
class SpokenAliasPreparation:
    """Typed preflight result; only ``READY`` carries pending state."""

    status: SpokenAliasStatus
    pending: PendingSpokenAlias | None = None


def prepare_spoken_alias(
    utterance: str,
    *,
    focused_repository: str | None = None,
    focused_issue: str | None = None,
    source: str | None = None,
    path: Path | None = None,
    integrations: RegistryInput = None,
) -> SpokenAliasPreparation:
    """Resolve one spoken alias from a focused or just-used fragment identity."""

    parsed = parse_spoken_alias_request(utterance)
    if parsed is None:
        return SpokenAliasPreparation(SpokenAliasStatus.MISSING_PHRASE)
    phrase, kind_hint = parsed
    selected = select_trusted_alias_target(
        kind_hint,
        focused_repository=focused_repository,
        focused_issue=focused_issue,
    )
    if selected is None:
        return SpokenAliasPreparation(SpokenAliasStatus.MISSING_TARGET)
    try:
        target, resolved_source, kind = resolve_owned_alias_target(
            selected,
            source=source,
            integrations=integrations,
        )
    except VocabularyError:
        return SpokenAliasPreparation(SpokenAliasStatus.INVALID_TARGET)
    if kind_hint == AliasKind.REPOSITORY and kind != AliasKind.REPOSITORY:
        return SpokenAliasPreparation(SpokenAliasStatus.INVALID_TARGET)
    if kind_hint == AliasKind.ISSUE and kind == AliasKind.REPOSITORY:
        return SpokenAliasPreparation(SpokenAliasStatus.INVALID_TARGET)
    normalized = _normalize_phrase(phrase)
    existing = load(path).alias_for(normalized)
    if existing is not None and existing.target == target:
        return SpokenAliasPreparation(SpokenAliasStatus.NO_CHANGE)
    return SpokenAliasPreparation(
        SpokenAliasStatus.READY,
        PendingSpokenAlias(
            trusted_utterance=utterance,
            phrase=normalized,
            target=target,
            kind=kind,
            source=resolved_source,
            existing_target=None if existing is None else existing.target,
        ),
    )


def commit_spoken_alias(
    pending: PendingSpokenAlias,
    *,
    force: bool = False,
    path: Path | None = None,
    integrations: RegistryInput = None,
) -> None:
    """Write one previously confirmed alias through :func:`add_alias`."""

    add_alias(
        pending.phrase,
        pending.target,
        force=force,
        path=path,
        source=pending.source,
        integrations=integrations,
    )


def render_spoken_alias_preparation(
    preparation: SpokenAliasPreparation,
) -> AssistantResponse:
    """Speak the phrase and canonical target, or a fail-closed refusal."""

    if preparation.status == SpokenAliasStatus.NO_CHANGE:
        return AssistantResponse.from_text(
            "That alias already maps to the current target, so I didn't write anything."
        )
    if preparation.status == SpokenAliasStatus.MISSING_PHRASE:
        return AssistantResponse.from_text(
            "I couldn't hear the alias phrase, so I didn't write anything."
        )
    if preparation.status == SpokenAliasStatus.MISSING_TARGET:
        return AssistantResponse.from_text(
            "I don't have a focused or just-used repository or issue to alias, "
            "so I didn't write anything."
        )
    if preparation.status != SpokenAliasStatus.READY:
        return AssistantResponse.from_text(
            "I couldn't validate that repository alias, so I didn't write anything."
        )
    pending = preparation.pending
    assert pending is not None
    if pending.replace:
        existing = pending.existing_target or "a different target"
        spoken = (
            f"{pending.phrase} already resolves to {existing}. "
            f"Replace it with {pending.target}? "
            "Say yes to replace or no to cancel."
        )
    else:
        spoken = (
            f"Call {pending.phrase} {pending.target}? "
            "Say yes to confirm or no to cancel."
        )
    return AssistantResponse(spoken_text=spoken, display_text=spoken)


def render_spoken_alias_committed(pending: PendingSpokenAlias) -> AssistantResponse:
    """Report one completed alias write."""

    verb = "Replaced" if pending.replace else "Saved"
    spoken = f"{verb} alias {pending.phrase} as {pending.target}."
    return AssistantResponse(spoken_text=spoken, display_text=spoken)


@dataclass(frozen=True)
class Replacement:
    spoken: str
    written: str

    def apply(self, text: str) -> str:
        return _phrase_pattern(self.spoken).sub(
            lambda _match, replacement=self.written: replacement, text
        )


@dataclass(frozen=True)
class Alias:
    phrase: str
    target: str
    kind: AliasKind
    source: str = "github"


@dataclass(frozen=True)
class Pronunciation:
    written: str
    spoken: str

    def apply(self, text: str) -> str:
        return _phrase_pattern(self.written).sub(
            lambda _match, replacement=self.spoken: replacement, text
        )


@dataclass(frozen=True)
class Vocabulary:
    version: int = SCHEMA_VERSION
    replacements: tuple[Replacement, ...] = ()
    aliases: tuple[Alias, ...] = ()
    pronunciations: tuple[Pronunciation, ...] = ()

    def replacement_for(self, spoken: str) -> Replacement | None:
        folded = _normalize_phrase(spoken)
        for replacement in self.replacements:
            if replacement.spoken.casefold() == folded:
                return replacement
        return None

    def alias_for(self, phrase: str) -> Alias | None:
        folded = _normalize_phrase(phrase)
        for alias in self.aliases:
            if alias.phrase == folded:
                return alias
        return None

    def pronunciation_for(self, written: str) -> Pronunciation | None:
        folded = _normalize_phrase(written)
        for pronunciation in self.pronunciations:
            if pronunciation.written.casefold() == folded:
                return pronunciation
        return None

    def resolve_text(self, text: str) -> str:
        """Expand every known alias phrase in ``text`` to its canonical target.

        Matching happens in a single pass over the original text so that a
        substituted target is never rescanned by a shorter alias. Longer phrases
        are tried first so the most specific alias wins at each position.
        """

        if not self.aliases:
            return text
        ordered = sorted(self.aliases, key=lambda item: len(item.phrase), reverse=True)
        targets = {alias.phrase: alias.target for alias in self.aliases}
        bodies = [
            r"\s+".join(re.escape(word) for word in alias.phrase.split())
            for alias in ordered
        ]
        pattern = re.compile(rf"(?<!\w)(?:{'|'.join(bodies)})(?!\w)", re.IGNORECASE)

        def replace(match: re.Match[str]) -> str:
            return targets[" ".join(match.group(0).split()).casefold()]

        return pattern.sub(replace, text)

    def to_document(self) -> dict[str, object]:
        return {
            "version": self.version,
            "replacements": [
                {"spoken": item.spoken, "written": item.written}
                for item in sorted(self.replacements, key=lambda item: item.spoken)
            ],
            "aliases": [
                {
                    "phrase": item.phrase,
                    "source": item.source,
                    "target": item.target,
                    "kind": str(item.kind),
                }
                for item in sorted(self.aliases, key=lambda item: item.phrase)
            ],
            "pronunciations": [
                {"written": item.written, "spoken": item.spoken}
                for item in sorted(
                    self.pronunciations, key=lambda item: item.written.casefold()
                )
            ],
        }


def _require_mapping(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise VocabularyError(message)
    return value


def _parse_replacements(value: object) -> tuple[Replacement, ...]:
    if not isinstance(value, list):
        raise VocabularyError("'replacements' must be a list")
    replacements: list[Replacement] = []
    seen: dict[str, str] = {}
    for entry in value:
        record = _require_mapping(entry, "each replacement must be an object")
        spoken_raw = record.get("spoken")
        written_raw = record.get("written")
        if not isinstance(spoken_raw, str) or not isinstance(written_raw, str):
            raise VocabularyError("replacement 'spoken' and 'written' must be strings")
        spoken = " ".join(spoken_raw.split())
        written = written_raw.strip()
        if not spoken or not written:
            raise VocabularyError(
                "replacement 'spoken' and 'written' must not be empty"
            )
        folded = spoken.casefold()
        if folded in seen and seen[folded] != written:
            raise VocabularyError(
                f"replacement {spoken!r} is ambiguous: it maps to both "
                f"{seen[folded]!r} and {written!r}"
            )
        if folded in seen:
            continue
        seen[folded] = written
        replacements.append(Replacement(spoken=spoken, written=written))
    return tuple(replacements)


def _parse_aliases(value: object) -> tuple[Alias, ...]:
    if not isinstance(value, list):
        raise VocabularyError("'aliases' must be a list")
    aliases: list[Alias] = []
    seen: dict[str, str] = {}
    for entry in value:
        record = _require_mapping(entry, "each alias must be an object")
        phrase_raw = record.get("phrase")
        target_raw = record.get("target")
        if not isinstance(phrase_raw, str) or not isinstance(target_raw, str):
            raise VocabularyError("alias 'phrase' and 'target' must be strings")
        phrase = _normalize_phrase(phrase_raw)
        if not target_raw.strip():
            raise VocabularyError("alias 'target' must not be empty")
        target, inferred_source, kind = _alias_identity(target_raw)
        source_raw = record.get("source")
        if isinstance(source_raw, str) and source_raw.strip():
            source = source_raw.strip().casefold()
            if source != inferred_source:
                raise VocabularyError(
                    f"alias {phrase!r} source {source!r} does not own {target!r}"
                )
        else:
            source = inferred_source
        if phrase in seen and seen[phrase] != target:
            raise VocabularyError(
                f"alias {phrase!r} is ambiguous: it maps to both "
                f"{seen[phrase]!r} and {target!r}"
            )
        if phrase in seen:
            continue
        seen[phrase] = target
        aliases.append(Alias(phrase=phrase, target=target, kind=kind, source=source))
    return tuple(aliases)


def _validate_pronunciation(written: str, spoken: str) -> tuple[str, str]:
    if any(unicodedata.category(character) == "Cc" for character in written + spoken):
        raise VocabularyError(
            "pronunciation values must not contain control characters"
        )
    normalized = " ".join(written.split())
    utterance = " ".join(spoken.split())
    if not normalized or not utterance:
        raise VocabularyError("pronunciation 'written' and 'spoken' must not be empty")
    if len(normalized) > 200 or len(utterance) > 200:
        raise VocabularyError("pronunciation values must be at most 200 characters")
    if any(
        not (
            character.isalnum()
            or unicodedata.category(character) in {"Mn", "Mc"}
            or character in _SPOKEN_PRONUNCIATION_PUNCTUATION
        )
        for character in utterance
    ):
        raise VocabularyError(
            "pronunciation 'spoken' must contain only words and speech punctuation"
        )
    return normalized, utterance


def _parse_pronunciations(value: object) -> tuple[Pronunciation, ...]:
    if not isinstance(value, list):
        raise VocabularyError("'pronunciations' must be a list")
    pronunciations: list[Pronunciation] = []
    seen: dict[str, str] = {}
    for entry in value:
        record = _require_mapping(entry, "each pronunciation must be an object")
        written_raw = record.get("written")
        spoken_raw = record.get("spoken")
        if not isinstance(written_raw, str) or not isinstance(spoken_raw, str):
            raise VocabularyError(
                "pronunciation 'written' and 'spoken' must be strings"
            )
        written, spoken = _validate_pronunciation(written_raw, spoken_raw)
        folded = written.casefold()
        if folded in seen and seen[folded] != spoken:
            raise VocabularyError(
                f"pronunciation {written!r} is ambiguous: it maps to both "
                f"{seen[folded]!r} and {spoken!r}"
            )
        if folded in seen:
            continue
        seen[folded] = spoken
        pronunciations.append(Pronunciation(written=written, spoken=spoken))
    return tuple(pronunciations)


def _resolve_path(path: Path | None) -> Path:
    return path if path is not None else config.VOCABULARY_PATH


def load(path: Path | None = None) -> Vocabulary:
    """Load the vocabulary store, returning an empty store when absent."""

    location = _resolve_path(path)
    try:
        raw = location.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Vocabulary()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VocabularyError(f"vocabulary store {location} is not valid JSON") from exc
    document = _require_mapping(
        document, f"vocabulary store {location} must be an object"
    )
    version = document.get("version", SCHEMA_VERSION)
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise VocabularyError(
            f"unsupported vocabulary schema version {version!r}; "
            f"expected one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    return Vocabulary(
        version=SCHEMA_VERSION,
        replacements=_parse_replacements(document.get("replacements", [])),
        aliases=_parse_aliases(document.get("aliases", [])),
        pronunciations=_parse_pronunciations(document.get("pronunciations", [])),
    )


def save(vocabulary: Vocabulary, path: Path | None = None) -> Path:
    """Persist the vocabulary store atomically with private permissions."""

    location = _resolve_path(path)
    location.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(vocabulary.to_document(), indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=location.parent,
        prefix=".vocabulary-",
        suffix=".json",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
        os.chmod(handle.name, 0o600)
        os.replace(handle.name, location)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return location


def resolve_aliases(text: str, *, path: Path | None = None) -> str:
    """Expand known aliases before routing, tolerating an unreadable store."""

    try:
        vocabulary = load(path)
    except (VocabularyError, OSError):
        return text
    return vocabulary.resolve_text(text)


def add_replacement(
    spoken: str, written: str, *, force: bool = False, path: Path | None = None
) -> None:
    vocabulary = load(path)
    normalized = " ".join(spoken.split())
    target = written.strip()
    if not normalized or not target:
        raise VocabularyError("both a spoken phrase and a written form are required")
    existing = vocabulary.replacement_for(normalized)
    if existing is not None and existing.written != target and not force:
        raise VocabularyError(
            f"replacement {normalized!r} already maps to {existing.written!r}; "
            "pass --force to overwrite"
        )
    kept = tuple(
        item
        for item in vocabulary.replacements
        if item.spoken.casefold() != normalized.casefold()
    )
    updated = Vocabulary(
        version=SCHEMA_VERSION,
        replacements=(*kept, Replacement(spoken=normalized, written=target)),
        aliases=vocabulary.aliases,
        pronunciations=vocabulary.pronunciations,
    )
    save(updated, path)
    verb = "updated" if existing is not None else "added"
    print(f"{verb} replacement {normalized!r} -> {target!r}")


def add_alias(
    phrase: str,
    target: str,
    *,
    force: bool = False,
    path: Path | None = None,
    source: str | None = None,
    integrations: RegistryInput = None,
) -> None:
    vocabulary = load(path)
    normalized = _normalize_phrase(phrase)
    canonical, resolved_source, kind = resolve_owned_alias_target(
        target,
        source=source,
        integrations=integrations,
    )
    existing = vocabulary.alias_for(normalized)
    if existing is not None and existing.target != canonical and not force:
        raise VocabularyError(
            f"alias {normalized!r} already resolves to {existing.target!r}; "
            "pass --force to overwrite"
        )
    kept = tuple(item for item in vocabulary.aliases if item.phrase != normalized)
    updated = Vocabulary(
        version=SCHEMA_VERSION,
        replacements=vocabulary.replacements,
        aliases=(
            *kept,
            Alias(
                phrase=normalized,
                target=canonical,
                kind=kind,
                source=resolved_source,
            ),
        ),
        pronunciations=vocabulary.pronunciations,
    )
    save(updated, path)
    verb = "updated" if existing is not None else "added"
    print(f"{verb} {kind} alias {normalized!r} -> {canonical!r}")


def remove_replacement(spoken: str, *, path: Path | None = None) -> None:
    vocabulary = load(path)
    normalized = " ".join(spoken.split())
    if vocabulary.replacement_for(normalized) is None:
        raise VocabularyError(f"no replacement is stored for {normalized!r}")
    kept = tuple(
        item
        for item in vocabulary.replacements
        if item.spoken.casefold() != normalized.casefold()
    )
    save(
        Vocabulary(
            version=SCHEMA_VERSION,
            replacements=kept,
            aliases=vocabulary.aliases,
            pronunciations=vocabulary.pronunciations,
        ),
        path,
    )
    print(f"removed replacement {normalized!r}")


def remove_alias(phrase: str, *, path: Path | None = None) -> None:
    vocabulary = load(path)
    normalized = _normalize_phrase(phrase)
    if vocabulary.alias_for(normalized) is None:
        raise VocabularyError(f"no alias is stored for {normalized!r}")
    kept = tuple(item for item in vocabulary.aliases if item.phrase != normalized)
    save(
        Vocabulary(
            version=SCHEMA_VERSION,
            replacements=vocabulary.replacements,
            aliases=kept,
            pronunciations=vocabulary.pronunciations,
        ),
        path,
    )
    print(f"removed alias {normalized!r}")


def add_pronunciation(
    written: str,
    spoken: str,
    *,
    force: bool = False,
    path: Path | None = None,
) -> None:
    vocabulary = load(path)
    normalized, utterance = _validate_pronunciation(written, spoken)
    existing = vocabulary.pronunciation_for(normalized)
    if existing is not None and existing.spoken != utterance and not force:
        raise VocabularyError(
            f"pronunciation {normalized!r} already maps to {existing.spoken!r}; "
            "pass --force to overwrite"
        )
    kept = tuple(
        item
        for item in vocabulary.pronunciations
        if item.written.casefold() != normalized.casefold()
    )
    save(
        Vocabulary(
            replacements=vocabulary.replacements,
            aliases=vocabulary.aliases,
            pronunciations=(
                *kept,
                Pronunciation(written=normalized, spoken=utterance),
            ),
        ),
        path,
    )
    verb = "updated" if existing is not None else "added"
    print(f"{verb} pronunciation {normalized!r} -> {utterance!r}")


def remove_pronunciation(written: str, *, path: Path | None = None) -> None:
    vocabulary = load(path)
    normalized = " ".join(written.split())
    if vocabulary.pronunciation_for(normalized) is None:
        raise VocabularyError(f"no pronunciation is stored for {normalized!r}")
    kept = tuple(
        item
        for item in vocabulary.pronunciations
        if item.written.casefold() != normalized.casefold()
    )
    save(
        Vocabulary(
            replacements=vocabulary.replacements,
            aliases=vocabulary.aliases,
            pronunciations=kept,
        ),
        path,
    )
    print(f"removed pronunciation {normalized!r}")


def list_entries(kind: str | None = None, *, path: Path | None = None) -> None:
    vocabulary = load(path)
    if kind in (None, "replacement"):
        print("replacements:")
        if not vocabulary.replacements:
            print("  (none)")
        for item in sorted(vocabulary.replacements, key=lambda item: item.spoken):
            print(f"  {item.spoken!r} -> {item.written!r}")
    if kind in (None, "alias"):
        print("aliases:")
        if not vocabulary.aliases:
            print("  (none)")
        for item in sorted(vocabulary.aliases, key=lambda item: item.phrase):
            print(f"  [{item.kind}] {item.phrase!r} -> {item.target!r}")
    if kind in (None, "pronunciation"):
        print("pronunciations:")
        if not vocabulary.pronunciations:
            print("  (none)")
        for item in sorted(
            vocabulary.pronunciations, key=lambda item: item.written.casefold()
        ):
            print(f"  {item.written!r} -> {item.spoken!r}")


def export_entries(output: Path | None = None, *, path: Path | None = None) -> None:
    """Print or write a portable JSON backup of the store."""

    vocabulary = load(path)
    payload = json.dumps(vocabulary.to_document(), indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    output.write_text(payload, encoding="utf-8")
    print(f"exported vocabulary to {output}")


def import_entries(
    source: Path, *, replace: bool = False, path: Path | None = None
) -> None:
    """Merge (default) or replace the store with a backup document."""

    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise VocabularyError(f"could not read {source}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VocabularyError(f"{source} is not valid JSON") from exc
    document = _require_mapping(document, f"{source} must be a JSON object")
    incoming = Vocabulary(
        version=SCHEMA_VERSION,
        replacements=_parse_replacements(document.get("replacements", [])),
        aliases=_parse_aliases(document.get("aliases", [])),
        pronunciations=_parse_pronunciations(document.get("pronunciations", [])),
    )
    if replace:
        save(incoming, path)
        print(
            f"replaced vocabulary with {len(incoming.replacements)} replacement(s) "
            f"{len(incoming.aliases)} alias(es), and "
            f"{len(incoming.pronunciations)} pronunciation(s) from {source}"
        )
        return
    current = load(path)
    replacements = {item.spoken.casefold(): item for item in current.replacements}
    for item in incoming.replacements:
        replacements[item.spoken.casefold()] = item
    aliases = {item.phrase: item for item in current.aliases}
    for item in incoming.aliases:
        aliases[item.phrase] = item
    pronunciations = {item.written.casefold(): item for item in current.pronunciations}
    for item in incoming.pronunciations:
        pronunciations[item.written.casefold()] = item
    save(
        Vocabulary(
            version=SCHEMA_VERSION,
            replacements=tuple(replacements.values()),
            aliases=tuple(aliases.values()),
            pronunciations=tuple(pronunciations.values()),
        ),
        path,
    )
    print(
        f"imported {len(incoming.replacements)} replacement(s), "
        f"{len(incoming.aliases)} alias(es), and "
        f"{len(incoming.pronunciations)} pronunciation(s) from {source}"
    )
