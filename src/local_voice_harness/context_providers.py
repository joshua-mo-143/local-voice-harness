"""Built-in registry of enabled optional context providers.

This module defines the narrow contract by which optional integrations
contribute browser context. A :class:`ContextProvider` owns its own URL
matching and capture and emits a bounded, provenance-labelled
:class:`ContextFragment`; the registry decides which providers are active based
on the :class:`IntegrationSettings` flags.

Optional integrations are disabled by default on fresh installations, so a
provider is only ever instantiated when its flag is explicitly enabled through
``config.toml`` or the matching ``VOICE_HARNESS_INTEGRATION_*`` environment
variable. Existing installations preserve prior behaviour by enabling the flag.
Concrete providers register themselves in ``_PROVIDER_FACTORIES``; the mapping
starts empty and is extended as integrations are extracted into providers.

Loading configuration fails closed: if the unified configuration cannot be read
or is malformed, the registry falls back to the built-in defaults, which leave
every optional integration disabled.
"""

from __future__ import annotations

from collections.abc import Callable

from .context_fragment import ContextFragment, ContextProvider
from .user_config import IntegrationSettings, load_user_config

__all__ = [
    "ContextFragment",
    "ContextProvider",
    "available_context_providers",
    "capture_context",
]

# Each entry pairs the ``IntegrationSettings`` flag that enables a provider with
# a factory that builds it. The registry stays generic: adding an integration is
# a matter of appending its ``(flag, factory)`` pair here.
_PROVIDER_FACTORIES: tuple[tuple[str, Callable[[], ContextProvider]], ...] = ()


def _integration_settings() -> IntegrationSettings:
    try:
        return load_user_config().integrations
    except Exception:
        return IntegrationSettings()


def available_context_providers(
    integrations: IntegrationSettings | None = None,
) -> tuple[ContextProvider, ...]:
    """Return the optional context providers enabled by ``integrations``.

    When ``integrations`` is ``None`` the unified user configuration is loaded
    (falling back to safe, integration-disabled defaults on any error).
    """

    settings = integrations if integrations is not None else _integration_settings()
    providers: list[ContextProvider] = []
    for flag, factory in _PROVIDER_FACTORIES:
        if getattr(settings, flag, False):
            providers.append(factory())
    return tuple(providers)


def capture_context(
    url: str,
    integrations: IntegrationSettings | None = None,
) -> ContextFragment | None:
    """Return the first enabled provider's fragment for ``url``.

    Provider failures are isolated: an individual provider raising never breaks
    the caller, and a disabled provider is never consulted, so its URLs are not
    inspected and its pages are never copied.
    """

    for provider in available_context_providers(integrations):
        try:
            fragment = provider.capture(url)
        except Exception:
            continue
        if fragment is not None:
            return fragment
    return None
