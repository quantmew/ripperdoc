"""Lookup helpers for packaged model capability metadata."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, ConfigDict

from ripperdoc.utils.log import get_logger

logger = get_logger()

_CHAT_LIKE_MODES = {"chat", "responses", "completion"}
_OPENAI_LIKE_PROVIDERS = {
    "openai",
    "openrouter",
    "azure",
    "deepseek",
    "xai",
    "groq",
    "mistral",
    "together_ai",
    "fireworks_ai",
}


class ModelCatalogEntry(BaseModel):
    """Normalized model metadata loaded from packaged catalog."""

    model_config = ConfigDict(frozen=True)

    key: str
    provider: Optional[str] = None
    mode: Optional[str] = None
    supports_reasoning: Optional[bool] = None
    supports_vision: Optional[bool] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    currency: str = "USD"


def _normalize_model_name(model_name: str) -> str:
    return (model_name or "").strip().lower()


def _normalize_protocol(protocol: Any) -> str:
    value = getattr(protocol, "value", protocol)
    return str(value or "").strip().lower()


def _strip_numeric_suffix(value: str) -> str:
    if ":" not in value:
        return value
    head, tail = value.rsplit(":", 1)
    return head if tail.isdigit() else value


def _aliases_for_key(key: str) -> set[str]:
    aliases = {key}
    tail = key.rsplit("/", 1)[-1]
    aliases.add(tail)
    aliases.add(_strip_numeric_suffix(tail))
    if "." in tail:
        dot_tail = tail.split(".", 1)[1]
        aliases.add(dot_tail)
        aliases.add(_strip_numeric_suffix(dot_tail))
    return {alias for alias in aliases if alias}


def _query_variants(query: str) -> tuple[str, ...]:
    variants = {query}
    tail = query.rsplit("/", 1)[-1]
    variants.add(tail)
    variants.add(_strip_numeric_suffix(query))
    variants.add(_strip_numeric_suffix(tail))
    if "." in tail:
        dot_tail = tail.split(".", 1)[1]
        variants.add(dot_tail)
        variants.add(_strip_numeric_suffix(dot_tail))
    return tuple(variant for variant in variants if variant)


@lru_cache(maxsize=1)
def _load_catalog_entries() -> Dict[str, Dict[str, Any]]:
    payload: Any = None
    try:
        from ripperdoc.data.model_prices_and_context_window import (
            MODEL_PRICES_AND_CONTEXT_WINDOW,
        )

        payload = MODEL_PRICES_AND_CONTEXT_WINDOW
    except Exception as exc:  # pragma: no cover - import/runtime failure fallback
        logger.warning(
            "[model_catalog] Failed to load generated model metadata module: %s: %s",
            type(exc).__name__,
            exc,
        )
        return {}

    if isinstance(payload, dict):
        maybe_entries = payload.get("entries")
        if isinstance(maybe_entries, dict):
            logger.debug(
                "[model_catalog] Loaded packaged metadata",
                extra={"resource": "model_prices_and_context_window.py", "entries": len(maybe_entries)},
            )
            return {
                str(key).strip().lower(): value
                for key, value in maybe_entries.items()
                if isinstance(value, dict) and str(key).strip().lower() != "sample_spec"
            }
        logger.debug(
            "[model_catalog] Loaded packaged metadata",
            extra={"resource": "model_prices_and_context_window.py", "entries": len(payload)},
        )
        return {
            str(key).strip().lower(): value
            for key, value in payload.items()
            if isinstance(value, dict) and str(key).strip().lower() != "sample_spec"
        }
    logger.warning("[model_catalog] Generated model metadata has unsupported type: %s", type(payload).__name__)
    return {}


@lru_cache(maxsize=1)
def _alias_index() -> Dict[str, tuple[str, ...]]:
    index: Dict[str, list[str]] = {}
    for key in _load_catalog_entries():
        for alias in _aliases_for_key(key):
            index.setdefault(alias, []).append(key)
    return {alias: tuple(keys) for alias, keys in index.items()}


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field(row: Dict[str, Any], short_key: str, long_key: str) -> Any:
    if short_key in row:
        return row.get(short_key)
    return row.get(long_key)


def _protocol_bonus(provider: str, protocol: str) -> int:
    if not protocol:
        return 0
    provider_lower = provider.lower()
    if protocol == "anthropic":
        if "anthropic" in provider_lower:
            return 90
        if "bedrock" in provider_lower:
            return 30
        return 0
    if protocol == "gemini":
        if any(token in provider_lower for token in ("gemini", "vertex", "google")):
            return 90
        return 0
    if protocol == "openai_compatible":
        if provider_lower in _OPENAI_LIKE_PROVIDERS:
            return 70
        return 15 if provider_lower else 0
    return 0


def _candidate_score(
    key: str,
    row: Dict[str, Any],
    query: str,
    variants: Iterable[str],
    protocol: str,
) -> int:
    tail = key.rsplit("/", 1)[-1]
    stripped_tail = _strip_numeric_suffix(tail)
    mode = str(_field(row, "m", "mode") or "").lower()
    provider = str(_field(row, "p", "litellm_provider") or "")
    score = 0

    for variant in variants:
        if key == variant:
            score = max(score, 1_000)
        elif tail == variant:
            score = max(score, 980)
        elif stripped_tail == variant:
            score = max(score, 960)
        elif key.endswith(f"/{variant}"):
            score = max(score, 940)
        elif variant and variant in key:
            score = max(score, 760 if len(variant) >= 5 else 300)

    if mode in _CHAT_LIKE_MODES:
        score += 25
    score += _protocol_bonus(provider, protocol)

    # Prefer tighter canonical names for equal-scoring candidates.
    score -= min(len(key), 120) // 20
    return score


def _parse_entry(key: str, row: Dict[str, Any]) -> ModelCatalogEntry:
    supports_reasoning = _field(row, "r", "supports_reasoning")
    supports_vision = _field(row, "v", "supports_vision")
    return ModelCatalogEntry(
        key=key,
        provider=_field(row, "p", "litellm_provider"),
        mode=_field(row, "m", "mode"),
        supports_reasoning=supports_reasoning if isinstance(supports_reasoning, bool) else None,
        supports_vision=supports_vision if isinstance(supports_vision, bool) else None,
        max_input_tokens=_to_int(_field(row, "in", "max_input_tokens")),
        max_output_tokens=_to_int(_field(row, "out", "max_output_tokens")),
        max_tokens=_to_int(_field(row, "mx", "max_tokens")),
        input_cost_per_token=_to_float(_field(row, "ic", "input_cost_per_token")),
        output_cost_per_token=_to_float(_field(row, "oc", "output_cost_per_token")),
        currency=str(_field(row, "cur", "currency") or "USD"),
    )


@lru_cache(maxsize=4_096)
def _lookup_model_metadata_cached(model_name: str, protocol: str) -> Optional[ModelCatalogEntry]:
    if not model_name:
        return None

    entries = _load_catalog_entries()
    if not entries:
        return None

    if model_name in entries:
        return _parse_entry(model_name, entries[model_name])

    variants = _query_variants(model_name)
    candidates: list[str] = []
    seen: set[str] = set()
    aliases = _alias_index()
    for variant in variants:
        for candidate in aliases.get(variant, ()):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    if not candidates:
        for key in entries:
            if model_name in key:
                candidates.append(key)

    if not candidates:
        return None

    best_key: Optional[str] = None
    best_score = -1_000_000
    for key in candidates:
        row = entries.get(key)
        if not isinstance(row, dict):
            continue
        score = _candidate_score(key, row, model_name, variants, protocol)
        if score > best_score:
            best_key = key
            best_score = score

    if best_key is None:
        return None
    return _parse_entry(best_key, entries[best_key])


def lookup_model_metadata(model_name: str, protocol: Any = None) -> Optional[ModelCatalogEntry]:
    """Lookup model metadata by model name and optional protocol hint."""
    normalized_model = _normalize_model_name(model_name)
    normalized_protocol = _normalize_protocol(protocol)
    return _lookup_model_metadata_cached(normalized_model, normalized_protocol)


def get_catalog_size() -> int:
    """Expose entry count for diagnostics/tests."""
    return len(_load_catalog_entries())
