"""Helpers for composing session-level system prompt overrides."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional


def _normalize_prompt_text(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def select_base_system_prompt(
    *,
    agent_system_prompt: Optional[str],
    custom_system_prompt: Optional[str],
) -> Optional[str]:
    """Resolve the base system prompt with standard precedence."""
    return _normalize_prompt_text(agent_system_prompt) or _normalize_prompt_text(
        custom_system_prompt
    )


def compose_system_prompt(
    *,
    base_system_prompt: Optional[str],
    append_system_prompt: Optional[str],
    default_prompt_factory: Callable[[], str],
) -> str:
    """Compose a final system prompt from base/default and appended instructions."""
    normalized_base = _normalize_prompt_text(base_system_prompt)
    normalized_append = _normalize_prompt_text(append_system_prompt)
    if normalized_base:
        if normalized_append:
            return f"{normalized_base}\n\n{normalized_append}"
        return normalized_base

    system_prompt = default_prompt_factory()
    if normalized_append:
        return f"{system_prompt}\n\n{normalized_append}"
    return system_prompt


__all__ = ["compose_system_prompt", "select_base_system_prompt"]
