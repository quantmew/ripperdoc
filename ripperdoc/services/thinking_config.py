"""Shared thinking/reasoning configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ripperdoc.core.config import ModelProfile, ProtocolType

OPENAI_REASONING_OPTIONS: tuple[str, ...] = ("none", "low", "medium", "high")
GEMINI_3_FLASH_THINKING_OPTIONS: tuple[str, ...] = ("minimal", "low", "medium", "high")
GEMINI_3_PRO_THINKING_OPTIONS: tuple[str, ...] = ("low", "high")
GEMINI_LEVEL_OPTIONS: tuple[str, ...] = ("minimal", "low", "medium", "high")
GEMINI_LEVEL_MODES: frozenset[str] = frozenset({"gemini_level"})
GEMINI_BUDGET_MODES: frozenset[str] = frozenset({"gemini_budget"})


@dataclass(frozen=True)
class ThinkingControlSpec:
    kind: str
    label: str
    options: tuple[str, ...]
    hint: str
    vendor: str


def infer_thinking_vendor(
    protocol: ProtocolType,
    model_name: str,
    api_base: Optional[str],
    thinking_mode: Optional[str],
) -> str:
    mode = (thinking_mode or "").strip().lower()
    if mode:
        if mode in {"off", "disabled"}:
            return "disabled"
        if mode in GEMINI_LEVEL_MODES:
            return "gemini_level"
        if mode in GEMINI_BUDGET_MODES:
            return "gemini_budget"
        if mode != "auto":
            return mode

    lowered_model = (model_name or "").strip().lower()
    lowered_base = (api_base or "").strip().lower()

    if protocol == ProtocolType.GEMINI:
        return "gemini"
    if protocol == ProtocolType.OPENAI_COMPATIBLE:
        if (
            "generativelanguage.googleapis.com" in lowered_base
            or lowered_model.startswith("gemini")
            or "api.openai.com" in lowered_base
            or lowered_model.startswith("gpt-")
            or lowered_model.startswith("o1")
            or lowered_model.startswith("o3")
            or lowered_model.startswith("o4")
        ):
            return "openai"
    return "generic"


def thinking_control_spec(
    protocol: ProtocolType,
    model_name: str,
    api_base: Optional[str],
    thinking_mode: Optional[str],
) -> ThinkingControlSpec:
    vendor = infer_thinking_vendor(protocol, model_name, api_base, thinking_mode)
    if vendor == "disabled":
        return ThinkingControlSpec("none", "", (), "", vendor)
    if vendor == "openai":
        return ThinkingControlSpec(
            kind="select",
            label="Reasoning effort",
            options=OPENAI_REASONING_OPTIONS,
            hint="OpenAI reasoning uses named effort levels instead of a token budget.",
            vendor=vendor,
        )
    if vendor == "gemini_level":
        return ThinkingControlSpec(
            kind="select",
            label="Gemini thinking level",
            options=GEMINI_LEVEL_OPTIONS,
            hint="Gemini thinking level uses named levels and does not depend on model-name detection.",
            vendor=vendor,
        )
    if vendor in {"gemini", "gemini_budget"}:
        return ThinkingControlSpec(
            kind="input",
            label="Max thinking tokens",
            options=(),
            hint="Gemini thinking budget uses a numeric token budget.",
            vendor=vendor,
        )

    return ThinkingControlSpec(
        kind="input",
        label="Max thinking tokens",
        options=(),
        hint="",
        vendor=vendor,
    )


def thinking_effort_from_tokens(model_name: str, max_thinking_tokens: int) -> Optional[str]:
    if max_thinking_tokens <= 0:
        return None

    lowered_model = (model_name or "").strip().lower()
    if "gemini-3" in lowered_model and "flash" in lowered_model:
        if max_thinking_tokens <= 1024:
            return "minimal"
        if max_thinking_tokens <= 2048:
            return "low"
        if max_thinking_tokens <= 8192:
            return "medium"
        return "high"
    if "gemini-3" in lowered_model:
        return "low" if max_thinking_tokens <= 2048 else "high"
    if max_thinking_tokens <= 1024:
        return "low"
    if max_thinking_tokens <= 8192:
        return "medium"
    return "high"


def default_thinking_effort(
    *,
    model_name: str,
    thinking_effort: Optional[str],
    max_thinking_tokens: Optional[int],
) -> str:
    explicit = (thinking_effort or "").strip().lower()
    if explicit:
        return explicit
    inferred = thinking_effort_from_tokens(model_name, max(0, int(max_thinking_tokens or 0)))
    return inferred or "none"


def effort_to_budget(effort: str) -> int:
    lowered = (effort or "").strip().lower()
    if lowered == "minimal":
        return 1024
    if lowered == "low":
        return 2048
    if lowered == "medium":
        return 8192
    if lowered == "high":
        return 24576
    return 0


def format_thinking_summary(profile: ModelProfile) -> str:
    summary = profile.thinking_mode or "auto"
    if profile.thinking_effort:
        return f"{summary}:{profile.thinking_effort}"
    if profile.max_thinking_tokens is not None:
        return f"{summary}:{profile.max_thinking_tokens}"
    return summary
