"""Registry and capability boundary for optional integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from ..context_fragment import ContextFragment, ContextProvider
from ..user_config import (
    IntegrationSettings,
    UserConfig,
    default_user_config,
    load_user_config,
)
from .github import GitHubClient, GitHubProvider
from .herdr import HerdrClient
from .linear import CapabilityStatus, LinearIntegration
from .zendesk import ZendeskProvider

_INTEGRATION_FACTORIES: tuple[tuple[str, Callable[..., object]], ...] = (
    ("github_enabled", GitHubProvider),
    ("zendesk_enabled", ZendeskProvider),
    ("linear_enabled", LinearIntegration),
)


@dataclass(frozen=True)
class IntegrationRegistry:
    """One immutable integration snapshot and its configured constructors."""

    settings: IntegrationSettings
    factories: tuple[tuple[str, Callable[[], object]], ...]
    github_client: Callable[[], GitHubClient]
    herdr_client: Callable[[], HerdrClient]


def _github_provider(factory: Callable[[], GitHubClient]) -> GitHubProvider:
    return GitHubProvider(factory())


def build_integration_registry(config: UserConfig) -> IntegrationRegistry:
    """Build integration providers and clients from one validated snapshot."""

    platform = config.platform
    github_client = partial(
        GitHubClient,
        gh_executable=str(platform.gh_bin),
        git_executable=str(platform.git_bin),
        clone_root=platform.github_root,
        allowed_root=platform.project_root,
        timeout=platform.github_timeout_seconds,
    )
    herdr_client = partial(
        HerdrClient,
        str(platform.herdr_bin),
        repository_root=platform.project_root,
        worktree_root=platform.herdr_worktree_root,
        timeout=platform.herdr_timeout_seconds,
        agent_inactivity_timeout=platform.cursor_agent_inactivity_seconds,
        agent_max_runtime=platform.cursor_agent_max_runtime_seconds,
    )
    factories = tuple(
        (
            flag,
            partial(_github_provider, github_client)
            if factory is GitHubProvider
            else factory,
        )
        for flag, factory in _INTEGRATION_FACTORIES
    )
    return IntegrationRegistry(
        settings=config.integrations,
        factories=factories,
        github_client=github_client,
        herdr_client=herdr_client,
    )


RegistryInput = IntegrationRegistry | IntegrationSettings | None


def _integration_settings() -> IntegrationSettings:
    """Resolve compatibility callers through the fail-visible user loader."""

    return load_user_config().integrations


def _registry(value: RegistryInput) -> IntegrationRegistry:
    if isinstance(value, IntegrationRegistry):
        return value
    if value is None:
        return build_integration_registry(load_user_config())
    defaults = default_user_config()
    if isinstance(value, IntegrationSettings):
        defaults = UserConfig(
            providers=defaults.providers,
            integrations=value,
            compute=defaults.compute,
            audio=defaults.audio,
            dictation=defaults.dictation,
            platform=defaults.platform,
        )
    return build_integration_registry(defaults)


def enabled_integrations(
    integrations: RegistryInput = None,
) -> tuple[object, ...]:
    registry = _registry(integrations)
    return tuple(
        factory()
        for flag, factory in registry.factories
        if getattr(registry.settings, flag, False)
    )


def integration_enabled(
    name: str,
    integrations: RegistryInput = None,
) -> bool:
    expected = name.strip().casefold()
    return any(
        str(getattr(integration, "name", "")).casefold() == expected
        for integration in enabled_integrations(integrations)
    )


def available_context_providers(
    integrations: RegistryInput = None,
) -> tuple[ContextProvider, ...]:
    return tuple(
        integration
        for integration in enabled_integrations(integrations)
        if isinstance(integration, ContextProvider)
    )


def capture_context(
    url: str,
    integrations: RegistryInput = None,
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
    integrations: RegistryInput = None,
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
    integrations: RegistryInput = None,
) -> str | None:
    for integration in enabled_integrations(integrations):
        extractor = getattr(integration, "extract_issue_reference", None)
        if extractor is not None and (reference := extractor(text)):
            return str(reference)
    return None


def _integration_for_issue(
    reference: str,
    integrations: RegistryInput = None,
) -> object | None:
    for integration in enabled_integrations(integrations):
        owns = getattr(integration, "owns_issue_reference", None)
        if owns is not None and owns(reference):
            return integration
    return None


def resolve_issue_reference(
    reference: str | None,
    integrations: RegistryInput = None,
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
    integrations: RegistryInput = None,
) -> None:
    integration = _integration_for_issue(reference, integrations)
    require = getattr(integration, "require_capabilities", None)
    if require is not None:
        require()


def prompt_instructions(
    reference: str | None,
    integrations: RegistryInput = None,
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
    integrations: RegistryInput = None,
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
    integrations: RegistryInput = None,
) -> tuple[tuple[str, CapabilityStatus], ...]:
    statuses: list[tuple[str, CapabilityStatus]] = []
    for integration in enabled_integrations(integrations):
        check = getattr(integration, "capability_status", None)
        if check is not None:
            statuses.append((str(getattr(integration, "name", "unknown")), check()))
    return tuple(statuses)
