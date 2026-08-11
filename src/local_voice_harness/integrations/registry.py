"""Registry and capability boundary for optional integrations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..context_fragment import ContextFragment, ContextProvider
from ..user_config import IntegrationSettings, load_user_config
from .github import GitHubProvider
from .linear import CapabilityStatus, LinearIntegration
from .zendesk import ZendeskProvider

_INTEGRATION_FACTORIES: tuple[tuple[str, Callable[[], object]], ...] = (
    ("github_enabled", GitHubProvider),
    ("zendesk_enabled", ZendeskProvider),
    ("linear_enabled", LinearIntegration),
)


def _integration_settings() -> IntegrationSettings:
    return load_user_config().integrations


def enabled_integrations(
    integrations: IntegrationSettings | None = None,
) -> tuple[object, ...]:
    settings = integrations if integrations is not None else _integration_settings()
    return tuple(
        factory()
        for flag, factory in _INTEGRATION_FACTORIES
        if getattr(settings, flag, False)
    )


def integration_enabled(
    name: str,
    integrations: IntegrationSettings | None = None,
) -> bool:
    expected = name.strip().casefold()
    return any(
        str(getattr(integration, "name", "")).casefold() == expected
        for integration in enabled_integrations(integrations)
    )


def available_context_providers(
    integrations: IntegrationSettings | None = None,
) -> tuple[ContextProvider, ...]:
    return tuple(
        integration
        for integration in enabled_integrations(integrations)
        if isinstance(integration, ContextProvider)
    )


def capture_context(
    url: str,
    integrations: IntegrationSettings | None = None,
) -> ContextFragment | None:
    for provider in available_context_providers(integrations):
        try:
            if not provider.matches(url):
                continue
            fragment = provider.capture(url)
        except Exception:
            continue
        if fragment is not None:
            return fragment
    return None


def capture_text_context(
    text: str,
    integrations: IntegrationSettings | None = None,
) -> ContextFragment | None:
    for provider in available_context_providers(integrations):
        capture_text = getattr(provider, "capture_text", None)
        if capture_text is None:
            continue
        try:
            fragment = capture_text(text)
        except Exception:
            continue
        if fragment is not None:
            return fragment
    return None


def extract_issue_reference(
    text: str,
    integrations: IntegrationSettings | None = None,
) -> str | None:
    for integration in enabled_integrations(integrations):
        extractor = getattr(integration, "extract_issue_reference", None)
        if extractor is not None and (reference := extractor(text)):
            return str(reference)
    return None


def _integration_for_issue(
    reference: str,
    integrations: IntegrationSettings | None = None,
) -> object | None:
    for integration in enabled_integrations(integrations):
        owns = getattr(integration, "owns_issue_reference", None)
        if owns is not None and owns(reference):
            return integration
    return None


def resolve_issue_reference(
    reference: str | None,
    integrations: IntegrationSettings | None = None,
) -> str | None:
    if not reference:
        return None
    integration = _integration_for_issue(reference, integrations)
    if integration is None:
        return None
    canonicalize = getattr(integration, "canonicalize_issue_reference", None)
    return (
        str(canonicalize(reference)) if canonicalize is not None else reference.strip()
    )


def require_issue_capabilities(
    reference: str,
    integrations: IntegrationSettings | None = None,
) -> None:
    integration = _integration_for_issue(reference, integrations)
    require = getattr(integration, "require_capabilities", None)
    if require is not None:
        require()


def prompt_instructions(
    reference: str | None,
    integrations: IntegrationSettings | None = None,
) -> tuple[str, ...]:
    if not reference:
        return ()
    integration = _integration_for_issue(reference, integrations)
    contribute = getattr(integration, "prompt_instructions", None)
    return tuple(contribute(reference)) if contribute is not None else ()


def route_issue_repository(
    client: object,
    reference: str,
    repositories: list[Path],
    *,
    token: str,
    reserved: set[str],
    checkpoint: Any = None,
    integrations: IntegrationSettings | None = None,
) -> tuple[Path | None, str, str] | None:
    integration = _integration_for_issue(reference, integrations)
    route = getattr(integration, "route_repository", None)
    if route is None:
        return None
    return route(
        client,
        reference,
        repositories,
        token=token,
        reserved=reserved,
        checkpoint=checkpoint,
    )


def capability_statuses(
    integrations: IntegrationSettings | None = None,
) -> tuple[tuple[str, CapabilityStatus], ...]:
    statuses: list[tuple[str, CapabilityStatus]] = []
    for integration in enabled_integrations(integrations):
        check = getattr(integration, "capability_status", None)
        if check is not None:
            statuses.append((str(getattr(integration, "name", "unknown")), check()))
    return tuple(statuses)
