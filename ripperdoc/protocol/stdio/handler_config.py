"""Configuration and hook helpers for stdio protocol handler."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ripperdoc.core.default_tools import filter_tools_by_names
from ripperdoc.core.hooks.config import HookDefinition, HookMatcher, HooksConfig, DEFAULT_HOOK_TIMEOUT
from ripperdoc.core.hooks.events import HookOutput
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.permissions import make_permission_checker
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.protocol.models import UserMessageData, UserStreamMessage
from ripperdoc.utils.memory import build_memory_instructions

from .timeouts import STDIO_HOOK_TIMEOUT_SEC

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


class StdioConfigMixin:
    _PERMISSION_MODES: set[str]

    def _normalize_permission_mode(self, mode: Any) -> str:
        """Normalize permission mode to a supported value."""
        if isinstance(mode, str) and mode in self._PERMISSION_MODES:
            return mode
        return "default"

    def _normalize_tool_list(self, value: Any) -> list[str] | None:
        """Normalize tool list inputs from SDK/CLI options."""
        if value is None:
            return None
        if isinstance(value, str):
            raw = value.strip()
            if raw == "":
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            names: list[str] = []
            for item in value:
                if item is None:
                    continue
                name = str(item).strip()
                if name:
                    names.append(name)
            return names
        return None

    def _apply_tool_filters(
        self,
        tools: list[Any],
        *,
        allowed_tools: list[str] | None,
        disallowed_tools: list[str] | None,
        tools_list: list[str] | None,
    ) -> list[Any]:
        """Apply SDK tool filters while keeping Task tool consistent."""
        if tools_list is None and allowed_tools is None and not disallowed_tools:
            return tools

        tool_names = [getattr(tool, "name", tool.__class__.__name__) for tool in tools]
        allow_set: set[str] | None = None

        if tools_list is not None:
            allow_set = set(tools_list)
        if allowed_tools is not None:
            allow_set = set(allowed_tools) if allow_set is None else allow_set & set(allowed_tools)

        if disallowed_tools:
            if allow_set is None:
                allow_set = set(tool_names)
            allow_set -= set(disallowed_tools)

        if allow_set is None:
            return tools

        return filter_tools_by_names(tools, list(allow_set))

    def _apply_permission_mode(self, mode: str) -> None:
        """Apply permission mode across query context, hooks, and permissions."""
        yolo_mode = mode == "bypassPermissions"
        if self._query_context:
            self._query_context.yolo_mode = yolo_mode
            self._query_context.permission_mode = mode

        hook_manager.set_permission_mode(mode)

        if yolo_mode:
            self._can_use_tool = None
        else:
            self._can_use_tool = make_permission_checker(self._project_path, yolo_mode=False)

    def _collect_hook_contexts(self, hook_result: Any) -> list[str]:
        contexts: list[str] = []
        additional_context = getattr(hook_result, "additional_context", None)
        if additional_context:
            contexts.append(str(additional_context))
        return contexts

    def _build_hook_notice_stream_message(
        self,
        text: str,
        hook_event: str,
        *,
        tool_name: str | None = None,
        level: str = "info",
    ) -> UserStreamMessage:
        return UserStreamMessage(
            session_id=self._session_id,
            message=UserMessageData(
                content=text,
                metadata={
                    "hook_notice": True,
                    "hook_event": hook_event,
                    "tool_name": tool_name,
                    "level": level,
                },
            ),
        )

    def _resolve_system_prompt(
        self,
        tools: list[Any],
        prompt: str,
        mcp_instructions: str | None,
        hook_instructions: list[str] | None = None,
    ) -> str:
        """Resolve the system prompt for the current session/query."""
        if self._custom_system_prompt:
            return self._custom_system_prompt

        additional_instructions: list[str] = []
        if self._skill_instructions:
            additional_instructions.append(self._skill_instructions)
        memory_instructions = build_memory_instructions()
        if memory_instructions:
            additional_instructions.append(memory_instructions)
        if hook_instructions:
            additional_instructions.extend([text for text in hook_instructions if text])

        return build_system_prompt(
            tools,
            prompt,
            {},
            additional_instructions=additional_instructions or None,
            mcp_instructions=mcp_instructions,
        )

    def _build_sdk_hook_scope(self, hooks: Any) -> HooksConfig:
        """Convert SDK hook config into a HooksConfig scope."""
        if not hooks or not isinstance(hooks, dict):
            return HooksConfig()
        parsed: dict[str, list[HookMatcher]] = {}
        for event_name, matchers in hooks.items():
            if not isinstance(matchers, list):
                continue
            parsed_matchers: list[HookMatcher] = []
            for matcher in matchers:
                if not isinstance(matcher, dict):
                    continue
                callback_ids = (
                    matcher.get("hookCallbackIds")
                    or matcher.get("hook_callback_ids")
                    or []
                )
                if not isinstance(callback_ids, list) or not callback_ids:
                    continue
                timeout = matcher.get("timeout", DEFAULT_HOOK_TIMEOUT)
                hook_defs: list[HookDefinition] = []
                for callback_id in callback_ids:
                    if not isinstance(callback_id, str) or not callback_id:
                        continue
                    hook_defs.append(
                        HookDefinition(
                            type="callback",
                            callback_id=callback_id,
                            timeout=timeout,
                        )
                    )
                if hook_defs:
                    parsed_matchers.append(
                        HookMatcher(matcher=matcher.get("matcher"), hooks=hook_defs)
                    )
            if parsed_matchers:
                parsed[event_name] = parsed_matchers
        return HooksConfig(hooks=parsed)

    def _configure_sdk_hooks(self, hooks: Any) -> None:
        """Register SDK-provided hooks for this session."""
        self._hooks = hooks or {}
        self._sdk_hook_scope = self._build_sdk_hook_scope(self._hooks)
        if self._query_context:
            self._query_context.add_hook_scope("sdk_hooks", self._sdk_hook_scope)
        if self._sdk_hook_scope and self._sdk_hook_scope.hooks:
            hook_manager.set_hook_callback(self._run_sdk_hook_callback)
        else:
            hook_manager.set_hook_callback(None)

    async def _run_sdk_hook_callback(
        self,
        callback_id: str,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        timeout: float | None,
    ) -> HookOutput:
        """Invoke an SDK hook callback via control protocol."""
        safe_input = self._sanitize_for_json(input_data)
        try:
            response = await self._send_control_request(
                {
                    "subtype": "hook_callback",
                    "callback_id": callback_id,
                    "input": safe_input,
                    "tool_use_id": tool_use_id,
                },
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[stdio] SDK hook callback timed out")
            return HookOutput.from_raw("", "", 1, timed_out=True)
        except Exception as exc:
            logger.error("[stdio] SDK hook callback failed: %s", exc)
            return HookOutput(error=str(exc), exit_code=1)

        if response is None:
            return HookOutput()
        if isinstance(response, dict):
            return HookOutput.from_raw(json.dumps(response), "", 0)
        return HookOutput()
