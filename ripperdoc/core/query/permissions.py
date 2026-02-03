"""Tool permission evaluation helpers."""

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Union

from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger

from .context import QueryContext

logger = get_logger()

ToolPermissionCallable = Callable[
    [Tool[Any, Any], Any],
    Union[
        PermissionResult,
        Dict[str, Any],
        Tuple[bool, Optional[str]],
        bool,
        Awaitable[Union[PermissionResult, Dict[str, Any], Tuple[bool, Optional[str]], bool]],
    ],
]


async def _check_tool_permissions(
    tool: Tool[Any, Any],
    parsed_input: Any,
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable],
) -> tuple[bool, Optional[str], Optional[Any]]:
    """Evaluate whether a tool call is allowed."""
    try:
        if can_use_tool_fn is not None:
            decision = can_use_tool_fn(tool, parsed_input)
            if inspect.isawaitable(decision):
                decision = await decision
            if isinstance(decision, PermissionResult):
                return decision.result, decision.message, decision.updated_input
            if isinstance(decision, dict) and "result" in decision:
                return (
                    bool(decision.get("result")),
                    decision.get("message"),
                    decision.get("updated_input"),
                )
            if isinstance(decision, tuple) and len(decision) == 2:
                return bool(decision[0]), decision[1], None
            return bool(decision), None, None

        if not query_context.yolo_mode and tool.needs_permissions(parsed_input):
            loop = asyncio.get_running_loop()
            input_preview = (
                parsed_input.model_dump()
                if hasattr(parsed_input, "model_dump")
                else str(parsed_input)
            )
            prompt = f"Allow tool '{tool.name}' with input {input_preview}? [y/N]: "
            response = await loop.run_in_executor(None, lambda: input(prompt))
            return response.strip().lower() in ("y", "yes"), None, None

        return True, None, None
    except (TypeError, AttributeError, ValueError) as exc:
        logger.warning(
            f"Error checking permissions for tool '{tool.name}': {type(exc).__name__}: {exc}",
            extra={"tool": getattr(tool, "name", None), "error_type": type(exc).__name__},
        )
        return False, None, None
