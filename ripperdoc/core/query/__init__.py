"""AI query system for Ripperdoc.

This package handles communication with AI models and manages
the query-response loop including tool execution.
"""

from ripperdoc.utils.log import get_logger

from .context import QueryContext, ToolRegistry, _append_hook_context, _apply_skill_context_updates
from .errors import _format_changed_file_notice
from .loop import (
    DEFAULT_REQUEST_TIMEOUT_SEC,
    MAX_LLM_RETRIES,
    MAX_QUERY_ITERATIONS,
    IterationResult,
    infer_thinking_mode,
    query,
    query_llm,
    _run_query_iteration,
)
from .permissions import ToolPermissionCallable, _check_tool_permissions
from .tools import (
    DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC,
    DEFAULT_TOOL_TIMEOUT_SEC,
    _execute_tools_in_parallel,
    _execute_tools_sequentially,
    _group_tool_calls_by_concurrency,
    _resolve_tool,
    _run_concurrent_tool_uses,
    _run_tool_use_generator,
    _run_tools_concurrently,
    _run_tools_serially,
)

logger = get_logger()

__all__ = [
    "DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC",
    "DEFAULT_REQUEST_TIMEOUT_SEC",
    "DEFAULT_TOOL_TIMEOUT_SEC",
    "MAX_LLM_RETRIES",
    "MAX_QUERY_ITERATIONS",
    "IterationResult",
    "QueryContext",
    "ToolPermissionCallable",
    "ToolRegistry",
    "infer_thinking_mode",
    "logger",
    "query",
    "query_llm",
    "_append_hook_context",
    "_apply_skill_context_updates",
    "_check_tool_permissions",
    "_execute_tools_in_parallel",
    "_execute_tools_sequentially",
    "_format_changed_file_notice",
    "_group_tool_calls_by_concurrency",
    "_resolve_tool",
    "_run_concurrent_tool_uses",
    "_run_query_iteration",
    "_run_tool_use_generator",
    "_run_tools_concurrently",
    "_run_tools_serially",
]
