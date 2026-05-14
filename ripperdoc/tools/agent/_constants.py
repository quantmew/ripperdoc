"""Agent tool constants — tool names, legacy aliases, filter sets."""

from __future__ import annotations

import os

TOOL_NAME = "Agent"
TOOL_LEGACY_NAME = "Task"

DEFAULT_AGENT_RUN_TTL_SEC = float(os.getenv("RIPPERDOC_AGENT_RUN_TTL_SEC", "3600"))

# One-shot agents — run once, skip agentId/SendMessage trailer
ONE_SHOT_BUILTIN_AGENT_TYPES: frozenset[str] = frozenset({"explore", "plan"})

# Tool filtering constants for sub-agent tool access control
ALL_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset()
CUSTOM_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset()
ASYNC_AGENT_ALLOWED_TOOLS: frozenset[str] = frozenset()

# Auto-background timeout (0 = disabled)
AUTO_BACKGROUND_MS = 120_000
