"""Shared MCP output size guardrails used by MCP tools."""

from __future__ import annotations

import os
from typing import Optional

from ripperdoc.utils.token_estimation import estimate_tokens

DEFAULT_MAX_MCP_OUTPUT_TOKENS = 25_000
MIN_MCP_OUTPUT_TOKENS = 1_000
DEFAULT_MCP_WARNING_FRACTION = 0.8


def _get_mcp_token_limits() -> tuple[int, int]:
    """Compute warning and hard limits for MCP output size."""
    max_tokens = os.getenv("RIPPERDOC_MCP_MAX_OUTPUT_TOKENS")
    try:
        max_tokens_int = int(max_tokens) if max_tokens else DEFAULT_MAX_MCP_OUTPUT_TOKENS
    except (TypeError, ValueError):
        max_tokens_int = DEFAULT_MAX_MCP_OUTPUT_TOKENS
    max_tokens_int = max(MIN_MCP_OUTPUT_TOKENS, max_tokens_int)

    warn_env = os.getenv("RIPPERDOC_MCP_WARNING_TOKENS")
    try:
        warn_tokens_int = (
            int(warn_env) if warn_env else int(max_tokens_int * DEFAULT_MCP_WARNING_FRACTION)
        )
    except (TypeError, ValueError):
        warn_tokens_int = int(max_tokens_int * DEFAULT_MCP_WARNING_FRACTION)
    warn_tokens_int = max(MIN_MCP_OUTPUT_TOKENS, min(warn_tokens_int, max_tokens_int))
    return warn_tokens_int, max_tokens_int


def evaluate_mcp_output_size(
    result_text: Optional[str],
    server_name: str,
    tool_name: str,
) -> tuple[Optional[str], Optional[str], int]:
    """Return (warning, error, token_estimate) for an MCP result text."""
    warn_tokens, max_tokens = _get_mcp_token_limits()
    token_estimate = estimate_tokens(result_text or "")

    if token_estimate > max_tokens:
        error_text = (
            f"MCP response from {server_name}:{tool_name} is ~{token_estimate:,} tokens, "
            f"which exceeds the configured limit of {max_tokens}. "
            "Refine the request (pagination/filtering) or raise RIPPERDOC_MCP_MAX_OUTPUT_TOKENS."
        )
        return None, error_text, token_estimate

    warning_text = None
    if result_text and token_estimate >= warn_tokens:
        line_count = result_text.count("\n") + 1
        warning_text = (
            f"WARNING: Large MCP response (~{token_estimate:,} tokens, {line_count:,} lines). "
            "This can fill the context quickly; consider pagination or filters."
        )
    return warning_text, None, token_estimate


__all__ = ["evaluate_mcp_output_size"]
