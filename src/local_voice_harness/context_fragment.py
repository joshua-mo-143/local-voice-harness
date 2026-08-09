"""Provenance-labelled context contract.

This module defines the narrow contract shared by optional context providers.
It is deliberately a dependency-free leaf so providers can depend on the
contract without importing the registry that wires them together.

A :class:`ContextFragment` is always treated as untrusted external input,
never as instructions. Providers are responsible for bounding the size of the
text they emit and for embedding their own untrusted-source labelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ContextFragment:
    """A bounded, provenance-labelled piece of external context.

    ``source`` records which provider produced the fragment (its provenance).
    ``text`` is the already-bounded, untrusted-labelled rendering that is safe
    to append to a request. Stringifying a fragment yields its text so it can
    be composed with existing plain-string context.
    """

    source: str
    text: str

    def __str__(self) -> str:
        return self.text


@runtime_checkable
class ContextProvider(Protocol):
    """A capture source for a focused browser URL.

    Implementations must fail closed: ``capture`` returns ``None`` when the URL
    does not belong to the provider or when context cannot be gathered, and it
    must never raise for ordinary misses.
    """

    name: str

    def matches(self, url: str) -> bool:
        """Return whether ``url`` belongs to this provider."""
        ...

    def capture(self, url: str) -> ContextFragment | None:
        """Return a bounded fragment for ``url`` or ``None`` when unavailable."""
        ...
