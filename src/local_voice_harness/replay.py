from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from .browser_context import RequestContext
from .errors import HarnessError
from .intent import Intent, IntentRoute
from .responses import AssistantResponse, ResponseLike, as_assistant_response
from .ticket_targets import (
    TicketExtraction,
    TicketReference,
    TicketSource,
    extract_ticket_targets,
)
from .transcript import TranscriptReplacement, normalize_transcript

SCHEMA_VERSION = 1
SUPPORTED_STAGES = (
    "transcript_normalization",
    "context_selection",
    "ticket_extraction",
    "intent_routing",
    "response_rendering",
)
DEFAULT_OMISSIONS = ("audio", "external_context_bodies", "model_payloads")
MAX_BUNDLE_BYTES = 64 * 1024
MAX_TRANSCRIPT_CHARS = 8_000
MAX_RESPONSE_CHARS = 8_000
MAX_IDENTIFIER_CHARS = 512
MAX_REPLACEMENTS = 256
MAX_REPLACEMENT_CHARS = 256

_AUTHORIZATION = re.compile(
    r"\bauthorization\s*[:=]\s*(?:bearer|basic)?\s*\S+", re.IGNORECASE
)
_BEARER = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|(?:aws[_-]?)?access[_-]?key(?:[_-]?id)?|"
    r"access[_-]?token|client[_-]?secret|password|passwd|session[_-]?token)\b"
    r"\s*[:=]\s*[\"']?\S{8,}",
    re.IGNORECASE,
)
_COOKIE_HEADER = re.compile(r"\b(?:cookie|set-cookie)\s*:\s*\S+", re.IGNORECASE)
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_URL = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_SECRET_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "key",
    "password",
    "signature",
    "token",
}
_SECRET_ENV_NAME = re.compile(
    r"(?:API[_-]?KEY|AUTH|CREDENTIAL|PASSWORD|SECRET|TOKEN)", re.IGNORECASE
)


class ReplayError(HarnessError):
    pass


@dataclass(frozen=True, slots=True)
class ReplayContext:
    sources: tuple[str, ...] = ()
    focused_repository: str | None = None
    focused_issue: str | None = None
    github_repository: str | None = None
    github_issue: int | None = None
    github_pull_request: int | None = None
    external_issue_reference: str | None = None
    external_issue_source: str | None = None
    issue_scope: str | None = None
    issue_scope_source: str | None = None
    focused_app_class: str | None = None
    focused_app_sources: tuple[str, ...] = ()

    @classmethod
    def from_request_context(cls, context: RequestContext) -> ReplayContext:
        sources: list[str] = []
        if (
            context.github_repository is not None
            or context.github_issue is not None
            or context.github_pull_request is not None
        ):
            sources.append("github")
        if (
            context.external_issue_source is not None
            and context.external_issue_source not in sources
        ):
            sources.append(context.external_issue_source)
        if context.focused_app_sources:
            sources.append("focused_app")
        return cls(
            sources=tuple(sources),
            focused_repository=context.focused_repository,
            focused_issue=context.focused_issue,
            github_repository=context.github_repository,
            github_issue=context.github_issue,
            github_pull_request=context.github_pull_request,
            external_issue_reference=context.external_issue_reference,
            external_issue_source=context.external_issue_source,
            issue_scope=context.issue_scope,
            issue_scope_source=context.issue_scope_source,
            focused_app_class=context.focused_app_class,
            focused_app_sources=context.focused_app_sources,
        )


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    version: int
    captured_stages: tuple[str, ...]
    omitted_inputs: tuple[str, ...]
    raw_transcript: str
    normalized_transcript: str
    replacements: tuple[TranscriptReplacement, ...]
    context: ReplayContext
    ticket_extraction: TicketExtraction
    route: IntentRoute
    response: AssistantResponse | None = None

    def to_dict(self) -> dict[str, object]:
        context = {
            "sources": list(self.context.sources),
            "focused_repository": self.context.focused_repository,
            "focused_issue": self.context.focused_issue,
            "github_repository": self.context.github_repository,
            "github_issue": self.context.github_issue,
            "github_pull_request": self.context.github_pull_request,
            "external_issue_reference": self.context.external_issue_reference,
            "external_issue_source": self.context.external_issue_source,
            "issue_scope": self.context.issue_scope,
            "issue_scope_source": self.context.issue_scope_source,
            "focused_app_class": self.context.focused_app_class,
            "focused_app_sources": list(self.context.focused_app_sources),
        }
        tickets = {
            "requested_count": self.ticket_extraction.requested_count,
            "references": [
                {
                    "raw": reference.raw,
                    "position": reference.position,
                    "source": reference.source,
                    "canonical": reference.canonical,
                    "scoped": reference.scoped,
                    "error": reference.error,
                }
                for reference in self.ticket_extraction.references
            ],
        }
        response = (
            {
                "spoken_text": self.response.spoken_text,
                "display_text": self.response.display_text,
            }
            if self.response is not None
            else None
        )
        return {
            "version": self.version,
            "captured_stages": list(self.captured_stages),
            "omitted_inputs": list(self.omitted_inputs),
            "transcript": {
                "raw": self.raw_transcript,
                "normalized": self.normalized_transcript,
                "replacements": [
                    {"spoken": rule.spoken, "written": rule.written}
                    for rule in self.replacements
                ],
            },
            "context": context,
            "ticket_extraction": tickets,
            "route": {
                "intent": self.route.intent.value,
                "confidence": self.route.confidence,
            },
            "response": response,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReplayBundle:
        document = _object(value, "replay manifest")
        _exact_keys(
            document,
            {
                "version",
                "captured_stages",
                "omitted_inputs",
                "transcript",
                "context",
                "ticket_extraction",
                "route",
                "response",
            },
            "replay manifest",
        )
        version = _integer(document["version"], "version", minimum=1)
        if version != SCHEMA_VERSION:
            raise ReplayError(
                f"unsupported replay schema version {version}; expected {SCHEMA_VERSION}"
            )
        stages = _string_list(document["captured_stages"], "captured_stages")
        if not stages or len(stages) != len(set(stages)):
            raise ReplayError("captured_stages must be a non-empty unique list")
        unsupported = set(stages) - set(SUPPORTED_STAGES)
        if unsupported:
            raise ReplayError(
                f"unsupported replay stages: {', '.join(sorted(unsupported))}"
            )
        omissions = _string_list(document["omitted_inputs"], "omitted_inputs")

        transcript = _object(document["transcript"], "transcript")
        _exact_keys(transcript, {"raw", "normalized", "replacements"}, "transcript")
        replacements_value = _list(transcript["replacements"], "replacements")
        if len(replacements_value) > MAX_REPLACEMENTS:
            raise ReplayError(f"replacements exceeds {MAX_REPLACEMENTS} entries")
        replacements: list[TranscriptReplacement] = []
        for index, item in enumerate(replacements_value):
            replacement = _object(item, f"replacements[{index}]")
            _exact_keys(replacement, {"spoken", "written"}, f"replacements[{index}]")
            replacements.append(
                TranscriptReplacement(
                    _text(
                        replacement["spoken"],
                        f"replacements[{index}].spoken",
                        MAX_REPLACEMENT_CHARS,
                    ),
                    _text(
                        replacement["written"],
                        f"replacements[{index}].written",
                        MAX_REPLACEMENT_CHARS,
                    ),
                )
            )

        context = _parse_context(document["context"])
        tickets = _parse_tickets(document["ticket_extraction"])
        route_value = _object(document["route"], "route")
        _exact_keys(route_value, {"intent", "confidence"}, "route")
        try:
            intent = Intent(_text(route_value["intent"], "route.intent", 64))
        except ValueError as exc:
            raise ReplayError("route.intent is unsupported") from exc
        confidence = _text(route_value["confidence"], "route.confidence", 16)
        if confidence not in {"high", "medium", "low"}:
            raise ReplayError("route.confidence is unsupported")

        response_value = document["response"]
        response: AssistantResponse | None = None
        if response_value is not None:
            response_object = _object(response_value, "response")
            _exact_keys(response_object, {"spoken_text", "display_text"}, "response")
            response = AssistantResponse(
                _text(
                    response_object["spoken_text"],
                    "response.spoken_text",
                    MAX_RESPONSE_CHARS,
                    allow_empty=True,
                ),
                _text(
                    response_object["display_text"],
                    "response.display_text",
                    MAX_RESPONSE_CHARS,
                    allow_empty=True,
                ),
            )
        if ("response_rendering" in stages) != (response is not None):
            raise ReplayError(
                "response_rendering must match the presence of response data"
            )

        bundle = cls(
            version=version,
            captured_stages=stages,
            omitted_inputs=omissions,
            raw_transcript=_text(
                transcript["raw"], "transcript.raw", MAX_TRANSCRIPT_CHARS
            ),
            normalized_transcript=_text(
                transcript["normalized"],
                "transcript.normalized",
                MAX_TRANSCRIPT_CHARS,
            ),
            replacements=tuple(replacements),
            context=context,
            ticket_extraction=tickets,
            route=IntentRoute(intent, confidence),
            response=response,
        )
        _validate_bundle(bundle)
        return bundle


def capture_bundle(
    raw_transcript: str,
    *,
    replacements: Sequence[TranscriptReplacement],
    context: RequestContext,
    route: IntentRoute,
    response: ResponseLike | None = None,
) -> ReplayBundle:
    normalized = normalize_transcript(raw_transcript.strip(), replacements)
    extraction = extract_ticket_targets(
        normalized,
        scope_source=context.issue_scope_source,
        scope=context.issue_scope,
    )
    rendered = as_assistant_response(response) if response is not None else None
    stages: list[str] = list(SUPPORTED_STAGES[:-1])
    omissions: list[str] = list(DEFAULT_OMISSIONS)
    if rendered is not None:
        stages.append("response_rendering")
    else:
        omissions.append("response_rendering")
    bundle = ReplayBundle(
        version=SCHEMA_VERSION,
        captured_stages=tuple(stages),
        omitted_inputs=tuple(omissions),
        raw_transcript=raw_transcript.strip(),
        normalized_transcript=normalized,
        replacements=tuple(replacements),
        context=ReplayContext.from_request_context(context),
        ticket_extraction=extraction,
        route=route,
        response=rendered,
    )
    _validate_bundle(bundle)
    return bundle


def run_replay(bundle: ReplayBundle) -> AssistantResponse | None:
    """Verify deterministic stages and return the recorded channel rendering."""

    normalized = normalize_transcript(bundle.raw_transcript, bundle.replacements)
    if normalized != bundle.normalized_transcript:
        raise ReplayError("transcript_normalization result does not match manifest")
    extraction = extract_ticket_targets(
        normalized,
        scope_source=bundle.context.issue_scope_source,
        scope=bundle.context.issue_scope,
    )
    if extraction != bundle.ticket_extraction:
        raise ReplayError("ticket_extraction result does not match manifest")
    # Context and routing are recorded nondeterministic decisions. Deliberately do
    # not call a provider, desktop integration, or LLM transport during replay.
    return bundle.response


def load_bundle(path: Path) -> ReplayBundle:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReplayError(f"could not read replay bundle {path}: {exc}") from exc
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ReplayError(f"replay bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"replay bundle {path} is not valid JSON") from exc
    return ReplayBundle.from_dict(document)


def save_bundle(bundle: ReplayBundle, path: Path) -> Path:
    _validate_bundle(bundle)
    if path.exists():
        raise ReplayError(f"refusing to overwrite replay bundle: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ReplayError(f"replay directory is not owned by this user: {path.parent}")
    payload = (
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode()
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ReplayError(f"replay bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    handle = tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=".replay-",
        suffix=".json",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
        os.chmod(handle.name, 0o600)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path


def default_bundle_path(directory: Path) -> Path:
    return directory / f"replay-{uuid.uuid4().hex}.json"


def manifest_summary(bundle: ReplayBundle) -> str:
    tickets = [
        reference.label
        for reference in bundle.ticket_extraction.references
        if reference.canonical is not None
    ]
    lines = [
        f"Replay schema: {bundle.version}",
        f"Captured stages: {', '.join(bundle.captured_stages)}",
        f"Omitted inputs: {', '.join(bundle.omitted_inputs) or 'none'}",
        f"Context sources: {', '.join(bundle.context.sources) or 'none'}",
        f"Route: {bundle.route.intent.value} ({bundle.route.confidence})",
        f"Ticket targets: {', '.join(tickets) or 'none'}",
    ]
    if bundle.response is not None:
        lines.append(
            "Response characters: "
            f"speech={len(bundle.response.spoken_text)}, "
            f"display={len(bundle.response.display_text)}"
        )
    return "\n".join(lines)


def _parse_context(value: object) -> ReplayContext:
    context = _object(value, "context")
    fields = {
        "sources",
        "focused_repository",
        "focused_issue",
        "github_repository",
        "github_issue",
        "github_pull_request",
        "external_issue_reference",
        "external_issue_source",
        "issue_scope",
        "issue_scope_source",
        "focused_app_class",
        "focused_app_sources",
    }
    _exact_keys(context, fields, "context")
    github_issue = context["github_issue"]
    pull_request = context["github_pull_request"]
    return ReplayContext(
        sources=_string_list(context["sources"], "context.sources"),
        focused_repository=_optional_text(
            context["focused_repository"], "context.focused_repository"
        ),
        focused_issue=_optional_text(context["focused_issue"], "context.focused_issue"),
        github_repository=_optional_text(
            context["github_repository"], "context.github_repository"
        ),
        github_issue=(
            _integer(github_issue, "context.github_issue", minimum=1)
            if github_issue is not None
            else None
        ),
        github_pull_request=(
            _integer(pull_request, "context.github_pull_request", minimum=1)
            if pull_request is not None
            else None
        ),
        external_issue_reference=_optional_text(
            context["external_issue_reference"],
            "context.external_issue_reference",
        ),
        external_issue_source=_optional_text(
            context["external_issue_source"], "context.external_issue_source"
        ),
        issue_scope=_optional_text(context["issue_scope"], "context.issue_scope"),
        issue_scope_source=_optional_text(
            context["issue_scope_source"], "context.issue_scope_source"
        ),
        focused_app_class=_optional_text(
            context["focused_app_class"], "context.focused_app_class"
        ),
        focused_app_sources=_string_list(
            context["focused_app_sources"], "context.focused_app_sources"
        ),
    )


def _parse_tickets(value: object) -> TicketExtraction:
    tickets = _object(value, "ticket_extraction")
    _exact_keys(tickets, {"requested_count", "references"}, "ticket_extraction")
    references_value = _list(tickets["references"], "ticket_extraction.references")
    references: list[TicketReference] = []
    for index, item in enumerate(references_value):
        reference = _object(item, f"ticket_extraction.references[{index}]")
        _exact_keys(
            reference,
            {"raw", "position", "source", "canonical", "scoped", "error"},
            f"ticket_extraction.references[{index}]",
        )
        source_value = reference["source"]
        source: TicketSource | None
        if source_value is None:
            source = None
        elif source_value == "github":
            source = "github"
        elif source_value == "linear":
            source = "linear"
        else:
            raise ReplayError(f"ticket reference {index} has an unsupported source")
        scoped = reference["scoped"]
        if not isinstance(scoped, bool):
            raise ReplayError(f"ticket reference {index}.scoped must be a boolean")
        references.append(
            TicketReference(
                raw=_text(
                    reference["raw"],
                    f"ticket_extraction.references[{index}].raw",
                    MAX_IDENTIFIER_CHARS,
                ),
                position=_integer(
                    reference["position"],
                    f"ticket_extraction.references[{index}].position",
                    minimum=0,
                ),
                source=source,
                canonical=_optional_text(
                    reference["canonical"],
                    f"ticket_extraction.references[{index}].canonical",
                ),
                scoped=scoped,
                error=_optional_text(
                    reference["error"],
                    f"ticket_extraction.references[{index}].error",
                ),
            )
        )
    return TicketExtraction(
        tuple(references),
        _integer(
            tickets["requested_count"],
            "ticket_extraction.requested_count",
            minimum=0,
        ),
    )


def _validate_bundle(bundle: ReplayBundle) -> None:
    if bundle.version != SCHEMA_VERSION:
        raise ReplayError(f"unsupported replay schema version {bundle.version}")
    if (
        not bundle.captured_stages
        or len(bundle.captured_stages) != len(set(bundle.captured_stages))
        or set(bundle.captured_stages) - set(SUPPORTED_STAGES)
    ):
        raise ReplayError("captured_stages contains unsupported or duplicate stages")
    required = set(SUPPORTED_STAGES[:4])
    if not required.issubset(bundle.captured_stages):
        raise ReplayError("replay bundle is missing a required semantic stage")
    canonical_order = tuple(
        stage for stage in SUPPORTED_STAGES if stage in bundle.captured_stages
    )
    if bundle.captured_stages != canonical_order:
        raise ReplayError("captured_stages must use canonical execution order")
    if ("response_rendering" in bundle.captured_stages) != (
        bundle.response is not None
    ):
        raise ReplayError("response_rendering must match the presence of response data")
    _text(bundle.raw_transcript, "transcript.raw", MAX_TRANSCRIPT_CHARS)
    _text(
        bundle.normalized_transcript,
        "transcript.normalized",
        MAX_TRANSCRIPT_CHARS,
    )
    if len(bundle.replacements) > MAX_REPLACEMENTS:
        raise ReplayError(f"replacements exceeds {MAX_REPLACEMENTS} entries")
    for index, replacement in enumerate(bundle.replacements):
        _text(
            replacement.spoken,
            f"replacements[{index}].spoken",
            MAX_REPLACEMENT_CHARS,
        )
        _text(
            replacement.written,
            f"replacements[{index}].written",
            MAX_REPLACEMENT_CHARS,
        )
    for index, omission in enumerate(bundle.omitted_inputs):
        _text(omission, f"omitted_inputs[{index}]", MAX_IDENTIFIER_CHARS)
    if len(bundle.omitted_inputs) != len(set(bundle.omitted_inputs)):
        raise ReplayError("omitted_inputs must not contain duplicates")
    missing_omissions = set(DEFAULT_OMISSIONS) - set(bundle.omitted_inputs)
    if missing_omissions:
        raise ReplayError(
            "omitted_inputs is missing required safeguards: "
            f"{', '.join(sorted(missing_omissions))}"
        )
    response_omitted = "response_rendering" in bundle.omitted_inputs
    if response_omitted == (bundle.response is not None):
        raise ReplayError(
            "response_rendering omission must match the absence of response data"
        )
    context_values = (
        ("focused_repository", bundle.context.focused_repository),
        ("focused_issue", bundle.context.focused_issue),
        ("github_repository", bundle.context.github_repository),
        ("external_issue_reference", bundle.context.external_issue_reference),
        ("external_issue_source", bundle.context.external_issue_source),
        ("issue_scope", bundle.context.issue_scope),
        ("issue_scope_source", bundle.context.issue_scope_source),
        ("focused_app_class", bundle.context.focused_app_class),
    )
    for name, value in context_values:
        if value is not None:
            _text(value, f"context.{name}", MAX_IDENTIFIER_CHARS)
    for name, values in (
        ("sources", bundle.context.sources),
        ("focused_app_sources", bundle.context.focused_app_sources),
    ):
        for index, value in enumerate(values):
            _text(value, f"context.{name}[{index}]", MAX_IDENTIFIER_CHARS)
    if bundle.route.confidence not in {"high", "medium", "low"}:
        raise ReplayError("route.confidence is unsupported")
    if bundle.ticket_extraction.requested_count < 0:
        raise ReplayError("ticket_extraction.requested_count must not be negative")
    for index, reference in enumerate(bundle.ticket_extraction.references):
        _text(
            reference.raw,
            f"ticket_extraction.references[{index}].raw",
            MAX_IDENTIFIER_CHARS,
        )
        if reference.position < 0:
            raise ReplayError(f"ticket reference {index}.position must not be negative")
        for name, value in (
            ("canonical", reference.canonical),
            ("error", reference.error),
        ):
            if value is not None:
                _text(
                    value,
                    f"ticket_extraction.references[{index}].{name}",
                    MAX_IDENTIFIER_CHARS,
                )
    if bundle.response is not None:
        _text(
            bundle.response.spoken_text,
            "response.spoken_text",
            MAX_RESPONSE_CHARS,
            allow_empty=True,
        )
        _text(
            bundle.response.display_text,
            "response.display_text",
            MAX_RESPONSE_CHARS,
            allow_empty=True,
        )
    for name, value in _manifest_strings(bundle.to_dict()):
        _reject_secrets(name, value)


def _manifest_strings(value: object, path: str = "manifest") -> list[tuple[str, str]]:
    strings: list[tuple[str, str]] = []
    if isinstance(value, str):
        strings.append((path, value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            strings.extend(_manifest_strings(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            strings.extend(_manifest_strings(item, f"{path}[{index}]"))
    return strings


def _reject_secrets(name: str, value: str) -> None:
    if (
        _AUTHORIZATION.search(value)
        or _BEARER.search(value)
        or _SECRET_ASSIGNMENT.search(value)
        or _COOKIE_HEADER.search(value)
        or _PRIVATE_KEY.search(value)
    ):
        raise ReplayError(f"{name} contains credential material")
    for match in _URL.finditer(value):
        try:
            parsed = urlsplit(match.group(0))
        except ValueError as exc:
            raise ReplayError(f"{name} contains an invalid URL") from exc
        if parsed.username is not None or parsed.password is not None:
            raise ReplayError(f"{name} contains a credential-bearing URL")
        query_keys = {
            key.casefold().replace("-", "_")
            for key, _query_value in parse_qsl(parsed.query, keep_blank_values=True)
        }
        if any(
            key in _SECRET_QUERY_KEYS
            or any(marker in key for marker in ("signature", "token", "secret"))
            for key in query_keys
        ):
            raise ReplayError(f"{name} contains an authenticated URL")
        path_parts = [part for part in parsed.path.split("/") if part]
        if (
            "webhooks" in {part.casefold() for part in path_parts}
            and len(path_parts) >= 3
        ) or (
            parsed.hostname in {"hooks.slack.com", "discord.com", "discordapp.com"}
            and len(path_parts) >= 3
        ):
            raise ReplayError(f"{name} contains a credential-bearing webhook URL")
    for key, secret in os.environ.items():
        if _SECRET_ENV_NAME.search(key) and len(secret) >= 8 and secret in value:
            raise ReplayError(f"{name} contains an environment secret")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value.keys()
    ):
        raise ReplayError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ReplayError(f"{name} must be a list")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ReplayError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ReplayError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _text(value: object, name: str, limit: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ReplayError(f"{name} must be a string")
    if not value and not allow_empty:
        raise ReplayError(f"{name} must not be empty")
    if len(value) > limit:
        raise ReplayError(f"{name} exceeds {limit} characters")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, MAX_IDENTIFIER_CHARS)


def _string_list(value: object, name: str) -> tuple[str, ...]:
    items = _list(value, name)
    return tuple(
        _text(item, f"{name}[{index}]", MAX_IDENTIFIER_CHARS)
        for index, item in enumerate(items)
    )


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayError(f"{name} must be an integer >= {minimum}")
    return value
