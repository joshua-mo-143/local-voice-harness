"""Local personal vocabulary and entity alias store.

This module keeps a user-owned, on-disk store of speech-recognition corrections
and spoken aliases for repositories and issues. It is intentionally local: the
store is never uploaded anywhere. Its content only leaves the machine when a
resolved alias or corrected transcription becomes part of a prompt that the user
explicitly asked an agent to run.

Storage
-------
The store is a single JSON document (schema ``version`` 1) at
``$XDG_CONFIG_HOME/voice-harness/vocabulary.json`` (default
``~/.config/voice-harness/vocabulary.json``). JSON is used instead of TOML
because it round-trips losslessly with only the standard library, so the harness
keeps its zero-runtime-dependency policy. The file is written privately
(``0o600``) with sorted keys so backups and diffs are stable.

Two entry kinds are stored:

``replacements``
    Deterministic speech-to-text corrections. ``spoken`` (the recognized text)
    is replaced with ``written`` wherever it appears as a whole, whitespace- and
    case-insensitive phrase.
``aliases``
    A spoken ``phrase`` that resolves to a canonical entity ``target`` before
    routing. ``kind`` is ``repository`` for an ``owner/repo`` target or ``issue``
    for an ``owner/repo#number`` target; it is inferred from the target form.

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
2. ``DICTATION_REPLACEMENTS`` environment entries,
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
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from . import config
from .errors import HarnessError

SCHEMA_VERSION = 1

_REPOSITORY = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9_.-]+)$"
)
_ISSUE = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9_.-]+)#(?P<number>[1-9]\d*)$"
)


class VocabularyError(HarnessError):
    """A vocabulary store could not be read, validated, or updated."""


class AliasKind(StrEnum):
    REPOSITORY = "repository"
    ISSUE = "issue"


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
    raise VocabularyError(
        f"alias target {target!r} must be an owner/repository or "
        "owner/repository#number reference"
    )


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


@dataclass(frozen=True)
class Vocabulary:
    version: int = SCHEMA_VERSION
    replacements: tuple[Replacement, ...] = ()
    aliases: tuple[Alias, ...] = ()

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
                {"phrase": item.phrase, "target": item.target, "kind": str(item.kind)}
                for item in sorted(self.aliases, key=lambda item: item.phrase)
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
        target, kind = _canonical_target(target_raw)
        if phrase in seen and seen[phrase] != target:
            raise VocabularyError(
                f"alias {phrase!r} is ambiguous: it maps to both "
                f"{seen[phrase]!r} and {target!r}"
            )
        if phrase in seen:
            continue
        seen[phrase] = target
        aliases.append(Alias(phrase=phrase, target=target, kind=kind))
    return tuple(aliases)


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
    if version != SCHEMA_VERSION:
        raise VocabularyError(
            f"unsupported vocabulary schema version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return Vocabulary(
        version=SCHEMA_VERSION,
        replacements=_parse_replacements(document.get("replacements", [])),
        aliases=_parse_aliases(document.get("aliases", [])),
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
    )
    save(updated, path)
    verb = "updated" if existing is not None else "added"
    print(f"{verb} replacement {normalized!r} -> {target!r}")


def add_alias(
    phrase: str, target: str, *, force: bool = False, path: Path | None = None
) -> None:
    vocabulary = load(path)
    normalized = _normalize_phrase(phrase)
    canonical, kind = _canonical_target(target)
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
        aliases=(*kept, Alias(phrase=normalized, target=canonical, kind=kind)),
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
        ),
        path,
    )
    print(f"removed alias {normalized!r}")


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
    )
    if replace:
        save(incoming, path)
        print(
            f"replaced vocabulary with {len(incoming.replacements)} replacement(s) "
            f"and {len(incoming.aliases)} alias(es) from {source}"
        )
        return
    current = load(path)
    replacements = {item.spoken.casefold(): item for item in current.replacements}
    for item in incoming.replacements:
        replacements[item.spoken.casefold()] = item
    aliases = {item.phrase: item for item in current.aliases}
    for item in incoming.aliases:
        aliases[item.phrase] = item
    save(
        Vocabulary(
            version=SCHEMA_VERSION,
            replacements=tuple(replacements.values()),
            aliases=tuple(aliases.values()),
        ),
        path,
    )
    print(
        f"imported {len(incoming.replacements)} replacement(s) and "
        f"{len(incoming.aliases)} alias(es) from {source}"
    )
