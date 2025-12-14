"""Provider client registry with optional dependencies."""

from __future__ import annotations

import importlib
from typing import Optional, TYPE_CHECKING, Type, cast

from ripperdoc.core.config import ProviderType
from ripperdoc.core.providers.base import ProviderClient
from ripperdoc.utils.log import get_logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ripperdoc.core.providers.anthropic import AnthropicClient  # noqa: F401
    from ripperdoc.core.providers.gemini import GeminiClient  # noqa: F401
    from ripperdoc.core.providers.openai import OpenAIClient  # noqa: F401

logger = get_logger()


def _load_client(module: str, cls: str, extra: str) -> Type[ProviderClient]:
    """Dynamically import a provider client, pointing users to the right extra."""
    try:
        mod = importlib.import_module(f"ripperdoc.core.providers.{module}")
        client_cls = cast(Type[ProviderClient], getattr(mod, cls, None))
        if client_cls is None:
            raise ImportError(f"{cls} not found in {module}")
        return client_cls
    except ImportError as exc:
        raise RuntimeError(
            f"{cls} requires optional dependency group '{extra}'. "
            f"Install with `pip install ripperdoc[{extra}]`."
        ) from exc


def get_provider_client(provider: ProviderType) -> Optional[ProviderClient]:
    """Return a provider client for the given protocol."""
    if provider == ProviderType.ANTHROPIC:
        return _load_client("anthropic", "AnthropicClient", "anthropic")()
    if provider == ProviderType.OPENAI_COMPATIBLE:
        return _load_client("openai", "OpenAIClient", "openai")()
    if provider == ProviderType.GEMINI:
        return _load_client("gemini", "GeminiClient", "gemini")()
    logger.warning("[providers] Unsupported provider", extra={"provider": provider})
    return None


__all__ = ["ProviderClient", "get_provider_client"]
