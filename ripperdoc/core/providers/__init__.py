"""Provider client registry."""

from __future__ import annotations

from typing import Optional

from ripperdoc.core.config import ProviderType
from ripperdoc.core.providers.anthropic import AnthropicClient
from ripperdoc.core.providers.base import ProviderClient
from ripperdoc.core.providers.gemini import GeminiClient
from ripperdoc.core.providers.openai import OpenAIClient


def get_provider_client(provider: ProviderType) -> Optional[ProviderClient]:
    """Return a provider client for the given protocol."""
    if provider == ProviderType.ANTHROPIC:
        return AnthropicClient()
    if provider == ProviderType.OPENAI_COMPATIBLE:
        return OpenAIClient()
    if provider == ProviderType.GEMINI:
        return GeminiClient()
    return None


__all__ = [
    "ProviderClient",
    "AnthropicClient",
    "GeminiClient",
    "OpenAIClient",
    "get_provider_client",
]
