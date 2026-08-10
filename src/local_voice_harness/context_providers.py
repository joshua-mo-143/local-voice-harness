"""Compatibility exports for the optional-integration context contract."""

from __future__ import annotations

from .context_fragment import ContextFragment, ContextProvider
from .integrations.registry import (
    available_context_providers,
    capture_context,
    capture_text_context,
)

__all__ = [
    "ContextFragment",
    "ContextProvider",
    "available_context_providers",
    "capture_context",
    "capture_text_context",
]
