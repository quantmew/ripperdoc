"""Query context and tool registry helpers."""

import asyncio
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ripperdoc.core.tool import Tool
from ripperdoc.utils.coerce import parse_optional_int
from ripperdoc.utils.file_watch import BoundedFileCache
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import UserMessage
from ripperdoc.utils.pending_messages import PendingMessageQueue

logger = get_logger()


class ToolRegistry:
    """Track available tools, including deferred ones, and expose search/activation helpers."""

    def __init__(self, tools: List[Tool[Any, Any]]) -> None:
        self._tool_map: Dict[str, Tool[Any, Any]] = {}
        self._order: List[str] = []
        self._deferred: set[str] = set()
        self._active: List[str] = []
        self._active_set: set[str] = set()
        self.replace_tools(tools)

    def replace_tools(self, tools: List[Tool[Any, Any]]) -> None:
        """Replace all known tools and rebuild active/deferred lists."""
        seen = set()
        self._tool_map.clear()
        self._order.clear()
        self._deferred.clear()
        self._active.clear()
        self._active_set.clear()

        for tool in tools:
            name = getattr(tool, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            self._tool_map[name] = tool
            self._order.append(name)
            try:
                deferred = tool.defer_loading()
            except (TypeError, AttributeError) as exc:
                logger.warning(
                    "[tool_registry] Tool.defer_loading failed: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"tool": getattr(tool, "name", None)},
                )
                deferred = False
            if deferred:
                self._deferred.add(name)
            else:
                self._active.append(name)
                self._active_set.add(name)

    @property
    def active_tools(self) -> List[Tool[Any, Any]]:
        """Return active (non-deferred) tools in original order."""
        return [self._tool_map[name] for name in self._order if name in self._active_set]

    @property
    def all_tools(self) -> List[Tool[Any, Any]]:
        """Return all known tools in registration order."""
        return [self._tool_map[name] for name in self._order]

    @property
    def deferred_names(self) -> set[str]:
        """Return the set of deferred tool names."""
        return set(self._deferred)

    def get(self, name: str) -> Optional[Tool[Any, Any]]:
        """Lookup a tool by name."""
        return self._tool_map.get(name)

    def is_active(self, name: str) -> bool:
        """Check if a tool is currently active."""
        return name in self._active_set

    def activate_tools(self, names: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Activate deferred tools by name."""
        activated: List[str] = []
        missing: List[str] = []

        # First pass: collect tools to activate (no mutations)
        to_activate: List[str] = []
        for raw_name in names:
            name = (raw_name or "").strip()
            if not name:
                continue
            if name in self._active_set:
                continue
            tool = self._tool_map.get(name)
            if tool:
                to_activate.append(name)
            else:
                missing.append(name)

        # Second pass: atomically update all data structures
        if to_activate:
            self._active.extend(to_activate)
            self._active_set.update(to_activate)
            self._deferred.difference_update(to_activate)
            activated.extend(to_activate)

        return activated, missing

    def iter_named_tools(self) -> Iterable[tuple[str, Tool[Any, Any]]]:
        """Yield (name, tool) for all known tools in registration order."""
        for name in self._order:
            tool = self._tool_map.get(name)
            if tool:
                yield name, tool


def _apply_skill_context_updates(
    tool_results: List[UserMessage], query_context: "QueryContext"
) -> None:
    """Update query context based on Skill tool outputs."""
    for message in tool_results:
        data = getattr(message, "tool_use_result", None)
        if not isinstance(data, dict):
            continue
        skill_name = (
            data.get("skill")
            or data.get("command_name")
            or data.get("commandName")
            or data.get("command")
        )
        if not skill_name:
            continue

        allowed_tools = data.get("allowed_tools") or data.get("allowedTools") or []
        if allowed_tools and getattr(query_context, "tool_registry", None):
            try:
                query_context.tool_registry.activate_tools(
                    [tool for tool in allowed_tools if isinstance(tool, str) and tool.strip()]
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "[query] Failed to activate tools listed in skill output: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        model_hint = data.get("model")
        if isinstance(model_hint, str) and model_hint.strip():
            logger.debug(
                "[query] Applying model hint from skill",
                extra={"skill": skill_name, "model": model_hint},
            )
            query_context.model = model_hint.strip()

        max_tokens = data.get("max_thinking_tokens")
        if max_tokens is None:
            max_tokens = data.get("maxThinkingTokens")
        parsed_max = parse_optional_int(max_tokens)
        if parsed_max is not None:
            logger.debug(
                "[query] Applying max thinking tokens from skill",
                extra={"skill": skill_name, "max_thinking_tokens": parsed_max},
            )
            query_context.max_thinking_tokens = parsed_max


def _append_hook_context(context: Dict[str, str], label: str, payload: Optional[str]) -> None:
    """Append hook-supplied context to the shared context dict."""
    if not payload:
        return
    key = f"Hook:{label}"
    existing = context.get(key)
    if existing:
        context[key] = f"{existing}\n{payload}"
    else:
        context[key] = payload


class QueryContext:
    """Context for a query session."""

    def __init__(
        self,
        tools: List[Tool[Any, Any]],
        max_thinking_tokens: int = 0,
        yolo_mode: bool = False,
        model: str = "main",
        verbose: bool = False,
        pause_ui: Optional[Callable[[], None]] = None,
        resume_ui: Optional[Callable[[], None]] = None,
        stop_hook: str = "stop",
        file_cache_max_entries: int = 500,
        file_cache_max_memory_mb: float = 50.0,
        pending_message_queue: Optional[PendingMessageQueue] = None,
        max_turns: Optional[int] = None,
        permission_mode: str = "default",
    ) -> None:
        self.tool_registry = ToolRegistry(tools)
        self.max_thinking_tokens = max_thinking_tokens
        self.yolo_mode = yolo_mode
        self.model = model
        self.verbose = verbose
        self.abort_controller = asyncio.Event()
        self.pending_message_queue: PendingMessageQueue = (
            pending_message_queue if pending_message_queue is not None else PendingMessageQueue()
        )
        # Use BoundedFileCache instead of plain Dict to prevent unbounded growth
        self.file_state_cache: BoundedFileCache = BoundedFileCache(
            max_entries=file_cache_max_entries,
            max_memory_mb=file_cache_max_memory_mb,
        )
        self.pause_ui = pause_ui
        self.resume_ui = resume_ui
        self.stop_hook = stop_hook
        self.stop_hook_active = False
        self.max_turns = max_turns
        self.permission_mode = permission_mode

    @property
    def tools(self) -> List[Tool[Any, Any]]:
        """Active tools available for the current request."""
        return self.tool_registry.active_tools

    @tools.setter
    def tools(self, tools: List[Tool[Any, Any]]) -> None:
        """Replace tool inventory and recompute active/deferred sets."""
        self.tool_registry.replace_tools(tools)

    def activate_tools(self, names: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Activate deferred tools by name."""
        return self.tool_registry.activate_tools(names)

    def all_tools(self) -> List[Tool[Any, Any]]:
        """Return all known tools (active + deferred)."""
        return self.tool_registry.all_tools

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory usage statistics for monitoring."""
        return {
            "file_cache": self.file_state_cache.stats(),
            "tool_count": len(self.tool_registry.all_tools),
            "active_tool_count": len(self.tool_registry.active_tools),
        }

    def drain_pending_messages(self) -> List[UserMessage]:
        """Drain queued messages waiting to be injected into the conversation."""
        return self.pending_message_queue.drain()

    def enqueue_user_message(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Queue a user-style message to inject once the current loop finishes."""
        self.pending_message_queue.enqueue_text(text, metadata=metadata)
