"""Agent memory management — persistent memory per agent type.

Scopes:
- "user":    ~/.ripperdoc/agent-memory/{agentType}/
- "project": .ripperdoc/agent-memory/{agentType}/
- "local":   .ripperdoc/agent-memory-local/{agentType}/

This is a stub for future implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ripperdoc.utils.filesystem.config_paths import project_config_dir, user_config_dir
from ripperdoc.utils.log import get_logger

logger = get_logger()


def _memory_dir(
    scope: str,
    agent_type: str,
    *,
    project_path: Optional[Path] = None,
) -> Optional[Path]:
    if scope == "user":
        return user_config_dir() / "agent-memory" / agent_type
    if scope == "project":
        return project_config_dir(project_path) / "agent-memory" / agent_type
    if scope == "local":
        return project_config_dir(project_path) / "agent-memory-local" / agent_type
    return None


def load_agent_memory(
    agent_type: str,
    scope: str = "project",
    *,
    project_path: Optional[Path] = None,
) -> Optional[str]:
    """Load persistent memory for an agent type.

    Returns the memory text or None if no memory exists.
    """
    mem_dir = _memory_dir(scope, agent_type, project_path=project_path)
    if mem_dir is None:
        return None
    mem_file = mem_dir / "memory.md"
    if not mem_file.exists():
        return None
    try:
        return mem_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("[agent_memory] Failed to read memory for %s: %s", agent_type, exc)
        return None


def save_agent_memory(
    agent_type: str,
    memory: str,
    scope: str = "project",
    *,
    project_path: Optional[Path] = None,
) -> None:
    """Save persistent memory for an agent type."""
    mem_dir = _memory_dir(scope, agent_type, project_path=project_path)
    if mem_dir is None:
        return
    try:
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "memory.md").write_text(memory, encoding="utf-8")
    except OSError as exc:
        logger.debug("[agent_memory] Failed to save memory for %s: %s", agent_type, exc)
