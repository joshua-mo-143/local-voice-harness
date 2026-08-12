"""Bounded pronunciation rendering for technical values in spoken responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from .vocabulary import Pronunciation, VocabularyError, load


class SpeechTokenKind(StrEnum):
    URL = "url"
    PATH = "path"
    REPOSITORY = "repository"
    REFERENCE = "reference"
    COMMIT = "commit"
    JOB = "job"
    IDENTIFIER = "identifier"
    ACRONYM = "acronym"


@dataclass(frozen=True, slots=True)
class SpeechToken:
    kind: SpeechTokenKind
    value: str


_ACRONYMS = frozenset(
    {
        "API",
        "CLI",
        "CPU",
        "GPU",
        "HTTP",
        "HTTPS",
        "IP",
        "JSON",
        "LLM",
        "MCP",
        "PR",
        "SHA",
        "SSH",
        "STT",
        "TTS",
        "URL",
        "UUID",
    }
)
_TOKEN_PATTERN = re.compile(
    r"(?P<url>(?i:https?)://[^\s<>()]+)"
    r"|(?P<reference>(?i:\b(?:issue|pull[\s-]request|PR)\s+#?\d+\b)|"
    r"(?<!\w)#\d+\b)"
    r"|(?P<commit>(?i:\b(?:commit|SHA)\s+[0-9a-f]{7,40}\b)|"
    r"(?<!\w)(?=[0-9a-fA-F]{7,40}\b)(?=[0-9a-fA-F]*\d)"
    r"(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{7,40}\b)"
    r"|(?P<job>(?i:\bjob)\s+(?=[A-Za-z0-9-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9-]{7,63}\b)"
    r"|(?P<path>(?<!\w)(?:(?i:\b(?:open|edit|read|write|file|path))\s+"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]*[._][A-Za-z0-9_.-]+|"
    r"(?:~[A-Za-z0-9_.-]*|\.\.?)/[^\s,;:!?()[\]{}]+|"
    r"/[^\s,;:!?()[\]{}]+|(?:[A-Za-z0-9_.-]+/){2,}[A-Za-z0-9_.-]+|"
    r"(?i:src|tests?|docs?|scripts?|config|lib|app|packages?)/"
    r"[A-Za-z0-9_.-]*[._][A-Za-z0-9_.-]+))"
    r"|(?P<repository>(?:(?i:\brepo(?:sitory)?)\s+"
    r"\b[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:#\d+)?\b|"
    r"\b[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*#\d+\b))"
    r"|(?P<identifier>\b(?:[A-Za-z][A-Za-z0-9]*"
    r"(?:_[A-Za-z0-9]+)+|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|"
    r"[a-z]+(?:[A-Z][A-Za-z0-9]*)+|"
    r"[A-Z]+[a-z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)*)"
    r"(?:\.[A-Za-z][A-Za-z0-9]*)?\b)"
    r"|(?P<acronym>\b(?:" + "|".join(sorted(_ACRONYMS)) + r")\b)"
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SENTENCE = re.compile(r'^(.+?[.!?]["\']?)(?=\s|$)', re.DOTALL)
_TRAILING_PUNCTUATION = ".,;:!?"


def _spell(value: str) -> str:
    return " ".join(value)


def _speak_component(value: str) -> str:
    if not value:
        return ""
    if value.upper() in _ACRONYMS and value.isalpha():
        return _spell(value.upper())
    if "_" in value:
        return " ".join(_speak_component(part) for part in value.split("_") if part)
    if "-" in value:
        return " ".join(_speak_component(part) for part in value.split("-") if part)
    if _CAMEL_BOUNDARY.search(value):
        return " ".join(_speak_component(part) for part in _CAMEL_BOUNDARY.split(value))
    if "." in value:
        return " dot ".join(_speak_component(part) for part in value.split("."))
    return value


def _split_trailing_punctuation(value: str) -> tuple[str, str]:
    token = value.rstrip(_TRAILING_PUNCTUATION)
    return token, value[len(token) :]


class SpeechRenderer:
    """Render only recognized technical tokens and explicit local aliases."""

    def __init__(
        self,
        *,
        pronunciations: tuple[Pronunciation, ...] = (),
        local_checkout: Path | None = None,
    ) -> None:
        self.pronunciations = pronunciations
        self.local_checkout = local_checkout.resolve() if local_checkout else None

    @classmethod
    def from_local_config(
        cls,
        *,
        local_checkout: Path | None = None,
    ) -> SpeechRenderer:
        try:
            pronunciations = load().pronunciations
        except (OSError, VocabularyError):
            pronunciations = ()
        return cls(pronunciations=pronunciations, local_checkout=local_checkout)

    def tokenize(self, text: str) -> tuple[str | SpeechToken, ...]:
        """Return prose spans interleaved with recognized technical tokens."""

        parts: list[str | SpeechToken] = []
        position = 0
        for match in _TOKEN_PATTERN.finditer(text):
            if match.start() > position:
                parts.append(text[position : match.start()])
            kind = SpeechTokenKind(match.lastgroup or SpeechTokenKind.IDENTIFIER)
            parts.append(SpeechToken(kind, match.group(0)))
            position = match.end()
        if position < len(text):
            parts.append(text[position:])
        return tuple(parts)

    def render(self, text: str) -> str:
        """Return TTS-only text without mutating the source response."""

        aliased = self._apply_pronunciations(text)
        return "".join(
            part if isinstance(part, str) else self._render_token(part)
            for part in self.tokenize(aliased)
        )

    def _apply_pronunciations(self, text: str) -> str:
        ordered = sorted(
            self.pronunciations, key=lambda item: len(item.written), reverse=True
        )
        if not ordered:
            return text
        targets = {
            " ".join(item.written.split()).casefold(): item.spoken for item in ordered
        }
        bodies = [
            r"\s+".join(re.escape(word) for word in item.written.split())
            for item in ordered
        ]
        pattern = re.compile(rf"(?<!\w)(?:{'|'.join(bodies)})(?!\w)", re.IGNORECASE)
        return pattern.sub(
            lambda match: targets[" ".join(match.group(0).split()).casefold()],
            text,
        )

    def _render_token(self, token: SpeechToken) -> str:
        value, punctuation = _split_trailing_punctuation(token.value)
        if token.kind == SpeechTokenKind.URL:
            rendered = self._render_url(value)
        elif token.kind == SpeechTokenKind.PATH:
            rendered = self._render_path(value)
        elif token.kind == SpeechTokenKind.REPOSITORY:
            rendered = self._render_repository(value)
        elif token.kind == SpeechTokenKind.REFERENCE:
            rendered = self._render_reference(value)
        elif token.kind == SpeechTokenKind.COMMIT:
            identity = value.rsplit(maxsplit=1)[-1]
            rendered = f"commit {_spell(identity.lower())}"
        elif token.kind == SpeechTokenKind.JOB:
            _, identity = value.split(maxsplit=1)
            rendered = f"job {_spell(identity)}"
        elif token.kind == SpeechTokenKind.ACRONYM:
            rendered = _spell(value)
        else:
            rendered = _speak_component(value)
        return rendered + punctuation

    def _render_reference(self, value: str) -> str:
        number = re.search(r"\d+", value)
        assert number is not None
        folded = value.casefold()
        label = "pull request" if folded.startswith(("pr", "pull")) else "issue"
        return f"{label} {number.group(0)}"

    def _render_repository(self, value: str) -> str:
        context = re.match(r"(?i)repo(?:sitory)?\s+", value)
        reference_value = value[context.end() :] if context is not None else value
        reference, separator, number = reference_value.partition("#")
        owner, repository = reference.split("/", 1)
        spoken = f"{_speak_component(owner)} slash {_speak_component(repository)}"
        return f"{spoken}, issue {number}" if separator else spoken

    def _render_url(self, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            return value
        host = parsed.hostname or parsed.netloc
        spoken = "secure URL" if parsed.scheme.casefold() == "https" else "URL"
        spoken += f", {_speak_component(host)}"
        if port is not None:
            spoken += f", port {port}"
        segments = [unquote(part) for part in parsed.path.split("/") if part]
        if (
            host.casefold() in {"github.com", "www.github.com"}
            and len(segments) == 4
            and segments[2] in {"issues", "pull"}
            and segments[3].isdigit()
        ):
            repository = (
                f"{_speak_component(segments[0])} slash {_speak_component(segments[1])}"
            )
            reference = "issue" if segments[2] == "issues" else "pull request"
            service = (
                "secure GitHub"
                if parsed.scheme.casefold() == "https"
                else "GitHub over H T T P"
            )
            spoken = f"{service}, {repository}, {reference} {segments[3]}"
            segments = []
        if segments:
            spoken += ", " + " slash ".join(_speak_component(part) for part in segments)
        if parsed.query:
            query_parts = []
            for name, item in parse_qsl(parsed.query, keep_blank_values=True):
                rendered_name = _speak_component(name)
                query_parts.append(
                    f"{rendered_name} equals {_speak_component(item)}"
                    if item
                    else rendered_name
                )
            spoken += (
                ", query " + ", and ".join(query_parts)
                if query_parts
                else f", query {_speak_component(unquote(parsed.query))}"
            )
        if parsed.fragment:
            spoken += f", section {_speak_component(unquote(parsed.fragment))}"
        return spoken

    def _render_path(self, value: str) -> str:
        context = re.match(r"(?i)(open|edit|read|write|file|path)\s+", value)
        path_value = value[context.end() :] if context is not None else value
        action = (
            f"{context.group(1)} "
            if context is not None
            and context.group(1).casefold() in {"open", "edit", "read", "write"}
            else ""
        )
        try:
            expanded = Path(path_value).expanduser()
        except RuntimeError:
            expanded = Path(path_value)
        if expanded.is_absolute() and self.local_checkout is not None:
            try:
                relative = expanded.resolve().relative_to(self.local_checkout)
            except (OSError, ValueError):
                pass
            else:
                if relative == Path("."):
                    return f"{action}the local checkout"
                components = " slash ".join(
                    _speak_component(part) for part in relative.parts
                )
                return f"{action}the local checkout, {components}"
        parts = [part for part in path_value.replace("\\", "/").split("/") if part]
        components = " slash ".join(_speak_component(part) for part in parts)
        rendered = f"path {components}" if components else "the root directory"
        return action + rendered


class StreamingSpeechRenderer:
    """Buffer incomplete streamed sentences before applying speech rules."""

    def __init__(self, renderer: SpeechRenderer) -> None:
        self.renderer = renderer
        self.buffer = ""

    def feed(self, text: str) -> tuple[str, ...]:
        if not text.strip():
            return ()
        self.buffer += text
        rendered: list[str] = []
        while match := _SENTENCE.match(self.buffer):
            rendered.append(self.renderer.render(match.group(1).strip()))
            self.buffer = self.buffer[match.end() :].lstrip()
        return tuple(rendered)

    def flush(self) -> tuple[str, ...]:
        if not self.buffer:
            return ()
        text = self.buffer.strip()
        self.buffer = ""
        return (self.renderer.render(text),) if text else ()


def render_for_speech(text: str, *, local_checkout: Path | None = None) -> str:
    """Convenience entry point used by previews and foreground responses."""

    return SpeechRenderer.from_local_config(local_checkout=local_checkout).render(text)
