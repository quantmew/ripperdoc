"""AI query system for Ripperdoc.

This module handles communication with AI models and manages
the query-response loop including tool execution.
"""

import asyncio
import inspect
import json
import re
from uuid import uuid4
from json_repair import repair_json
from typing import AsyncGenerator, List, Optional, Dict, Any, Union, Iterable, Tuple
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolProgress,
    build_tool_description,
    tool_input_examples,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    MessageContent,
    UserMessage,
    AssistantMessage,
    ProgressMessage,
    create_user_message,
    create_assistant_message,
    create_progress_message,
    normalize_messages_for_api,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
)
from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.config import (
    get_global_config,
    ProviderType,
    provider_protocol,
    ModelProfile,
)
from pydantic import ValidationError
from ripperdoc.utils.session_usage import record_usage
from ripperdoc.utils.json_utils import safe_parse_json

import time


logger = get_logger()


def _safe_int(value: Any) -> int:
    """Best-effort int conversion for usage counters."""
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_usage_field(usage: Any, field: str) -> int:
    """Fetch a usage field from either a dict or object."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return _safe_int(usage.get(field))
    return _safe_int(getattr(usage, field, 0))


def _anthropic_usage_tokens(usage: Any) -> Dict[str, int]:
    """Extract token counts from an Anthropic response usage payload."""
    return {
        "input_tokens": _get_usage_field(usage, "input_tokens"),
        "output_tokens": _get_usage_field(usage, "output_tokens"),
        "cache_read_input_tokens": _get_usage_field(usage, "cache_read_input_tokens"),
        "cache_creation_input_tokens": _get_usage_field(usage, "cache_creation_input_tokens"),
    }


def _openai_usage_tokens(usage: Any) -> Dict[str, int]:
    """Extract token counts from an OpenAI-compatible response usage payload."""
    prompt_details = None
    if isinstance(usage, dict):
        prompt_details = usage.get("prompt_tokens_details")
    else:
        prompt_details = getattr(usage, "prompt_tokens_details", None)

    cache_read_tokens = _get_usage_field(prompt_details, "cached_tokens") if prompt_details else 0

    return {
        "input_tokens": _get_usage_field(usage, "prompt_tokens"),
        "output_tokens": _get_usage_field(usage, "completion_tokens"),
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": 0,
    }


def _resolve_model_profile(model: str) -> ModelProfile:
    """Resolve a model pointer to a concrete profile or raise if missing."""
    config = get_global_config()
    profile_name = getattr(config.model_pointers, model, None) or model
    model_profile = config.model_profiles.get(profile_name)
    if model_profile is None:
        fallback_profile = getattr(config.model_pointers, "main", "default")
        model_profile = config.model_profiles.get(fallback_profile)
    if not model_profile:
        raise ValueError(f"No model profile found for pointer: {model}")
    return model_profile


def _determine_tool_mode(model_profile: ModelProfile) -> str:
    """Return configured tool mode for provider."""
    if model_profile.provider != ProviderType.OPENAI_COMPATIBLE:
        return "native"
    configured = getattr(model_profile, "openai_tool_mode", "native") or "native"
    configured = configured.lower()
    # Backward compatibility: fall back to global setting if provided
    if configured not in {"native", "text"}:
        configured = getattr(get_global_config(), "openai_tool_mode", "native") or "native"
        configured = configured.lower()
    return configured if configured in {"native", "text"} else "native"

def _parse_text_mode_json_blocks(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse a JSON code block or raw JSON string into content blocks for text mode."""
    if not text or not isinstance(text, str):
        return None

    code_blocks = re.findall(
        r"```(?:\s*json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE
    )
    candidates = [blk.strip() for blk in code_blocks if blk.strip()]

    def _normalize_blocks(parsed: Any) -> Optional[List[Dict[str, Any]]]:
        raw_blocks = parsed if isinstance(parsed, list) else [parsed]
        normalized: List[Dict[str, Any]] = []
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                continue
            block_type = raw.get("type")
            if block_type == "text":
                text_value = raw.get("text") or raw.get("content")
                if isinstance(text_value, str) and text_value:
                    normalized.append({"type": "text", "text": text_value})
            elif block_type == "tool_use":
                tool_name = raw.get("tool") or raw.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                tool_use_id = raw.get("tool_use_id") or raw.get("id") or str(uuid4())
                input_value = raw.get("input") or {}
                if not isinstance(input_value, dict):
                    input_value = _normalize_tool_args(input_value)
                normalized.append(
                    {
                        "type": "tool_use",
                        "tool_use_id": str(tool_use_id),
                        "name": tool_name,
                        "input": input_value,
                    }
                )
        return normalized if normalized else None

    last_error: Optional[str] = None

    for candidate in candidates:
        if not candidate:
            continue

        parsed: Any = None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            parsed = repair_json(candidate, return_objects=True, ensure_ascii=False)

        if parsed is None or parsed == "":
            continue

        normalized = _normalize_blocks(parsed)
        if normalized:
            return normalized

        last_error = "Parsed JSON did not contain valid content blocks."

    if last_error:
        error_text = (
            f"JSON parsing failed: {last_error} "
            "Please resend a valid JSON array of content blocks inside a ```json``` code block."
        )
        return [{"type": "text", "text": error_text}]

    return None


def _tool_prompt_for_text_mode(tools: List[Tool[Any, Any]]) -> str:
    """Build a system hint describing available tools and the expected JSON format."""
    if not tools:
        return ""

    lines = [
        "You are in text-only tool mode. Tools are not auto-invoked by the API.",
        "Respond with one Markdown `json` code block containing a JSON array of content blocks.",
        'Each block must include `type`; use {"type": "text", "text": "<message>"} for text and '
        '{"type": "tool_use", "tool_use_id": "<tool_id>", "tool": "<tool_name>", "input": { ... required params ... }} '
        "for tool calls. Add multiple `tool_use` blocks if you need multiple tools.",
        "Include your natural language reply as a `text` block, followed by any `tool_use` blocks.",
        "Only include the JSON array inside the code block - no extra prose.",
        "Available tools:",
    ]

    for tool in tools:
        required_fields: List[str] = []
        try:
            for fname, finfo in getattr(tool.input_schema, "model_fields", {}).items():
                is_req = False
                if hasattr(finfo, "is_required"):
                    try:
                        is_req = bool(finfo.is_required())
                    except Exception:
                        is_req = False
                required_fields.append(f"{fname}{' (required)' if is_req else ''}")
        except Exception:
            required_fields = []

        required_str = ", ".join(required_fields) if required_fields else "see input schema"
        lines.append(f"- {tool.name}: fields {required_str}")

        schema_json = ""
        try:
            schema_json = json.dumps(tool.input_schema.model_json_schema(), ensure_ascii=False, indent=2)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "[tool_prompt] Failed to render input_schema",
                extra={"tool": getattr(tool, "name", None), "error": str(exc)},
            )
        if schema_json:
            lines.append("  input schema (JSON):")
            lines.append("  ```json")
            lines.append(f"  {schema_json}")
            lines.append("  ```")

    # Example response format
    example_blocks = [
        {"type": "text", "text": "好的，我来帮你查看一下README.md文件"},
        {"type": "tool_use", "tool_use_id": "tool_id_000001", "tool": "View", "input": {"file_path": "README.md"}},
    ]
    lines.append("Example:")
    lines.append("```json")
    lines.append(json.dumps(example_blocks, ensure_ascii=False, indent=2))
    lines.append("```")

    return "\n".join(lines)


def _text_mode_history(messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]]) -> List[Union[UserMessage, AssistantMessage]]:
    """Convert a message history into text-only form for text mode."""

    def _normalize_block(block: Any) -> Optional[Dict[str, Any]]:
        blk = MessageContent(**block) if isinstance(block, dict) else block
        btype = getattr(blk, "type", None)
        if btype == "text":
            text_val = getattr(blk, "text", None) or getattr(blk, "content", None) or ""
            return {"type": "text", "text": text_val}
        if btype == "tool_use":
            return {
                "type": "tool_use",
                "tool_use_id": getattr(blk, "tool_use_id", None) or getattr(blk, "id", None) or "",
                "tool": getattr(blk, "name", None) or "",
                "input": getattr(blk, "input", None) or {},
            }
        if btype == "tool_result":
            result_block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": getattr(blk, "tool_use_id", None) or getattr(blk, "id", None) or "",
                "text": getattr(blk, "text", None) or getattr(blk, "content", None) or "",
            }
            is_error = getattr(blk, "is_error", None)
            if is_error is not None:
                result_block["is_error"] = is_error
            return result_block
        text_val = getattr(blk, "text", None) or getattr(blk, "content", None)
        if text_val is not None:
            return {"type": "text", "text": text_val}
        return None

    converted: List[Union[UserMessage, AssistantMessage]] = []
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type == "progress" or msg_type is None:
            continue
        content = getattr(getattr(msg, "message", None), "content", None)
        text_content: Optional[str] = None
        if isinstance(content, list):
            normalized_blocks = []
            for block in content:
                # If a text block contains nested JSON content, expand it inline
                block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                block_text = getattr(block, "text", None) if hasattr(block, "text") else (
                    block.get("text") if isinstance(block, dict) else None
                )
                if block_type == "text" and isinstance(block_text, str):
                    parsed_nested = _parse_text_mode_json_blocks(block_text)
                    if parsed_nested:
                        normalized_blocks.extend(parsed_nested)
                        continue
                norm = _normalize_block(block)
                if norm:
                    normalized_blocks.append(norm)
            if normalized_blocks:
                json_payload = json.dumps(normalized_blocks, ensure_ascii=False, indent=2)
                text_content = f"```json\n{json_payload}\n```"
        elif isinstance(content, str):
            parsed_blocks = _parse_text_mode_json_blocks(content)
            if parsed_blocks:
                text_content = f"```json\n{json.dumps(parsed_blocks, ensure_ascii=False, indent=2)}\n```"
            else:
                text_content = content
        else:
            text_content = content if isinstance(content, str) else None
        if not text_content:
            continue
        if msg_type == "user":
            converted.append(create_user_message(text_content))
        elif msg_type == "assistant":
            converted.append(create_assistant_message(text_content))
    return converted


def _maybe_convert_json_block_to_tool_use(content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert any text blocks containing JSON content to structured content blocks."""
    if not content_blocks:
        return content_blocks

    new_blocks: List[Dict[str, Any]] = []
    converted_count = 0

    for block in content_blocks:
        if block.get("type") != "text":
            new_blocks.append(block)
            continue

        text = block.get("text")
        if not isinstance(text, str):
            new_blocks.append(block)
            continue

        parsed_blocks = _parse_text_mode_json_blocks(text)
        if not parsed_blocks:
            new_blocks.append(block)
            continue

        for parsed in parsed_blocks:
            if parsed.get("type") == "tool_use":
                new_blocks.append(
                    {
                        "type": "tool_use",
                        "tool_use_id": parsed.get("tool_use_id") or str(uuid4()),
                        "name": parsed.get("name") or parsed.get("tool"),
                        "input": parsed.get("input") or {},
                    }
                )
            elif parsed.get("type") == "text":
                new_blocks.append({"type": "text", "text": parsed.get("text") or ""})
        converted_count += 1

    if converted_count:
        logger.debug(
            "[query_llm] Converting JSON code block to structured content blocks",
            extra={"block_count": len(new_blocks)},
        )
    return new_blocks


def _normalize_tool_args(raw_args: Any) -> Dict[str, Any]:
    """Ensure tool arguments are returned as a dict, handling double-encoded strings."""
    candidate = raw_args

    # If the provider returns a JSON string (or a JSON string wrapped in another string),
    # try to parse up to two times before giving up.
    for _ in range(2):
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, str):
            candidate = safe_parse_json(candidate, log_error=False)
            continue
        break

    if isinstance(candidate, dict):
        return candidate

    preview = str(raw_args)
    preview = preview[:200] if len(preview) > 200 else preview
    logger.debug(
        "[query_llm] Tool arguments not a dict; defaulting to empty object",
        extra={"preview": preview},
    )
    return {}


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
            except Exception:
                logger.exception(
                    "[tool_registry] Tool.defer_loading failed",
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
        for raw_name in names:
            name = (raw_name or "").strip()
            if not name:
                continue
            if name in self._active_set:
                continue
            tool = self._tool_map.get(name)
            if tool:
                self._active.append(name)
                self._active_set.add(name)
                self._deferred.discard(name)
                activated.append(name)
            else:
                missing.append(name)
        return activated, missing

    def iter_named_tools(self) -> Iterable[tuple[str, Tool[Any, Any]]]:
        """Yield (name, tool) for all known tools in registration order."""
        for name in self._order:
            tool = self._tool_map.get(name)
            if tool:
                yield name, tool


class QueryContext:
    """Context for a query session."""

    def __init__(
        self,
        tools: List[Tool[Any, Any]],
        max_thinking_tokens: int = 0,
        safe_mode: bool = False,
        model: str = "main",
        verbose: bool = False,
    ) -> None:
        self.tool_registry = ToolRegistry(tools)
        self.max_thinking_tokens = max_thinking_tokens
        self.safe_mode = safe_mode
        self.model = model
        self.verbose = verbose
        self.abort_controller = asyncio.Event()

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


async def query_llm(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    system_prompt: str,
    tools: List[Tool[Any, Any]],
    max_thinking_tokens: int = 0,
    model: str = "main",
    abort_signal: Optional[asyncio.Event] = None,
) -> AssistantMessage:
    """Query the AI model and return the response.

    Args:
        messages: Conversation history
        system_prompt: System prompt for the model
        tools: Available tools
        max_thinking_tokens: Maximum tokens for thinking (0 = disabled)
        model: Model pointer to use
        abort_signal: Event to signal abortion

    Returns:
        AssistantMessage with the model's response
    """
    config = get_global_config()
    model_profile = _resolve_model_profile(model)

    # Normalize messages based on protocol family (Anthropic allows tool blocks; OpenAI-style prefers text-only)
    protocol = provider_protocol(model_profile.provider)
    tool_mode = _determine_tool_mode(model_profile)
    messages_for_model: List[Union[UserMessage, AssistantMessage, ProgressMessage]]
    if tool_mode == "text":
        messages_for_model = _text_mode_history(messages)
    else:
        messages_for_model = messages

    normalized_messages = normalize_messages_for_api(
        messages_for_model, protocol=protocol, tool_mode=tool_mode
    )
    logger.info(
        "[query_llm] Preparing model request",
        extra={
            "model_pointer": model,
            "provider": getattr(model_profile.provider, "value", str(model_profile.provider)),
            "model": model_profile.model,
            "normalized_messages": len(normalized_messages),
            "tool_count": len(tools),
            "max_thinking_tokens": max_thinking_tokens,
            "tool_mode": tool_mode,
        },
    )

    if protocol == "openai":
        summary_parts = []
        for idx, m in enumerate(normalized_messages):
            role = m.get("role")
            tool_calls = m.get("tool_calls")
            tc_ids = []
            if tool_calls:
                tc_ids = [tc.get("id") for tc in tool_calls]
            tool_call_id = m.get("tool_call_id")
            summary_parts.append(
                f"{idx}:{role}"
                + (f" tool_calls={tc_ids}" if tc_ids else "")
                + (f" tool_call_id={tool_call_id}" if tool_call_id else "")
            )
        logger.debug(f"[query_llm] OpenAI normalized messages: {' | '.join(summary_parts)}")

    logger.debug(
        f"[query_llm] Sending {len(normalized_messages)} messages to model pointer "
        f"'{model}' with {len(tools)} tool schemas; "
        f"max_thinking_tokens={max_thinking_tokens} protocol={protocol}"
    )

    # Make the API call
    start_time = time.time()

    try:
        # Create the appropriate client based on provider
        if model_profile.provider == ProviderType.ANTHROPIC:
            async with AsyncAnthropic(api_key=model_profile.api_key) as client:
                # Build tool schemas
                tool_schemas = []
                for tool in tools:
                    description = await build_tool_description(
                        tool, include_examples=True, max_examples=2
                    )
                    tool_schema = {
                        "name": tool.name,
                        "description": description,
                        "input_schema": tool.input_schema.model_json_schema(),
                        "defer_loading": bool(getattr(tool, "defer_loading", lambda: False)()),
                    }
                    examples = tool_input_examples(tool, limit=5)
                    if examples:
                        tool_schema["input_examples"] = examples
                    tool_schemas.append(tool_schema)

                # Make the API call
                response = await client.messages.create(
                    model=model_profile.model,
                    max_tokens=model_profile.max_tokens,
                    system=system_prompt,
                    messages=normalized_messages,  # type: ignore[arg-type]
                    tools=tool_schemas if tool_schemas else None,  # type: ignore
                    temperature=model_profile.temperature,
                )

                duration_ms = (time.time() - start_time) * 1000

                usage_tokens = _anthropic_usage_tokens(getattr(response, "usage", None))
                record_usage(model_profile.model, duration_ms=duration_ms, **usage_tokens)

                # Calculate cost (simplified, should use actual pricing)
                cost_usd = 0.0  # TODO: Implement cost calculation

                tool_use_blocks = [
                    block for block in response.content if getattr(block, "type", None) == "tool_use"
                ]
                logger.info(
                    "[query_llm] Received response from Anthropic",
                    extra={
                        "model": model_profile.model,
                        "duration_ms": round(duration_ms, 2),
                        "usage_tokens": usage_tokens,
                        "tool_use_blocks": len(tool_use_blocks),
                    },
                )

                return create_assistant_message(
                    content=content_blocks, cost_usd=cost_usd, duration_ms=duration_ms
                )

        elif model_profile.provider == ProviderType.OPENAI_COMPATIBLE:
            # OpenAI-compatible APIs (OpenAI, DeepSeek, Mistral, etc.)
            async with AsyncOpenAI(api_key=model_profile.api_key, base_url=model_profile.api_base) as client:
                # Build tool schemas for OpenAI format (unless disabled)
                openai_tools = []
                for tool in tools:
                    description = await build_tool_description(
                        tool, include_examples=True, max_examples=2
                    )
                    openai_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": description,
                                "parameters": tool.input_schema.model_json_schema(),
                            },
                        }
                    )

                # Prepare messages for OpenAI format
                openai_messages = [
                    {"role": "system", "content": system_prompt}
                ] + normalized_messages

                # Make the API call
                openai_response: Any = await client.chat.completions.create(
                    model=model_profile.model,
                    messages=openai_messages,
                    tools=openai_tools if openai_tools else None,  # type: ignore[arg-type]
                    temperature=model_profile.temperature,
                    max_tokens=model_profile.max_tokens,
                )

                duration_ms = (time.time() - start_time) * 1000
                usage_tokens = _openai_usage_tokens(getattr(openai_response, "usage", None))
                record_usage(model_profile.model, duration_ms=duration_ms, **usage_tokens)
                cost_usd = 0.0  # TODO: Implement cost calculation

                # Convert OpenAI response to our format
                content_blocks = []
                choice = openai_response.choices[0]

                logger.info(
                    "[query_llm] Received response from OpenAI-compatible provider",
                    extra={
                        "model": model_profile.model,
                        "duration_ms": round(duration_ms, 2),
                        "usage_tokens": usage_tokens,
                        "finish_reason": getattr(choice, "finish_reason", None),
                    },
                )

                if choice.message.content:
                    content_blocks.append({"type": "text", "text": choice.message.content})

                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        raw_args = getattr(tool_call.function, "arguments", None)
                        parsed_args = safe_parse_json(raw_args)
                        if parsed_args is None and raw_args:
                            arg_preview = str(raw_args)
                            arg_preview = arg_preview[:200] if len(arg_preview) > 200 else arg_preview
                            logger.debug(
                                "[query_llm] Failed to parse tool arguments; falling back to empty dict",
                                extra={
                                    "tool_call_id": getattr(tool_call, "id", None),
                                    "tool_name": getattr(tool_call.function, "name", None),
                                    "arguments_preview": arg_preview,
                                },
                            )
                        parsed_args = _normalize_tool_args(parsed_args if parsed_args is not None else raw_args)
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "tool_use_id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": parsed_args,
                            }
                        )
                else:
                    if tool_mode == "text":
                        content_blocks = _maybe_convert_json_block_to_tool_use(content_blocks)

                return create_assistant_message(
                    content=content_blocks, cost_usd=cost_usd, duration_ms=duration_ms
                )

        elif model_profile.provider == ProviderType.GEMINI:
            raise NotImplementedError("Gemini protocol is not yet supported.")
        else:
            raise NotImplementedError(f"Provider {model_profile.provider} not yet implemented")

    except Exception as e:
        # Return error message
        logger.exception(
            "Error querying AI model",
            extra={
                "model": getattr(model_profile, "model", None),
                "model_pointer": model,
                "provider": getattr(model_profile.provider, "value", None)
                if model_profile
                else None,
            },
        )
        duration_ms = (time.time() - start_time) * 1000
        error_msg = create_assistant_message(
            content=f"Error querying AI model: {str(e)}", duration_ms=duration_ms
        )
        error_msg.is_api_error_message = True
        return error_msg


async def query(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    system_prompt: str,
    context: Dict[str, str],
    query_context: QueryContext,
    can_use_tool_fn: Optional[Any] = None,
) -> AsyncGenerator[Union[UserMessage, AssistantMessage, ProgressMessage], None]:
    """Execute a query with tool support.

    This is the main query loop that:
    1. Sends messages to the AI
    2. Handles tool use responses
    3. Executes tools
    4. Recursively continues the conversation

    Args:
        messages: Conversation history
        system_prompt: Base system prompt
        context: Additional context dictionary
        query_context: Query configuration
        can_use_tool_fn: Optional function to check tool permissions

    Yields:
        Messages (user, assistant, progress) as they are generated
    """
    logger.info(
        "[query] Starting query loop",
        extra={
            "message_count": len(messages),
            "tool_count": len(query_context.tools),
            "safe_mode": query_context.safe_mode,
            "model_pointer": query_context.model,
        },
    )
    # Work on a copy so external mutations (e.g., UI appending messages while consuming)
    # do not interfere with recursion or normalization.
    messages = list(messages)
    model_profile = _resolve_model_profile(query_context.model)
    protocol = provider_protocol(model_profile.provider)
    tool_mode = _determine_tool_mode(model_profile)
    tools_for_model: List[Tool[Any, Any]] = []
    if tool_mode != "text":
        tools_for_model = query_context.all_tools()

    async def _check_permissions(
        tool: Tool[Any, Any], parsed_input: Any
    ) -> tuple[bool, Optional[str]]:
        """Check permissions for tool execution."""
        try:
            if can_use_tool_fn is not None:
                decision = can_use_tool_fn(tool, parsed_input)
                if inspect.isawaitable(decision):
                    decision = await decision
                if isinstance(decision, PermissionResult):
                    return decision.result, decision.message
                if isinstance(decision, dict) and "result" in decision:
                    return bool(decision.get("result")), decision.get("message")
                if isinstance(decision, tuple) and len(decision) == 2:
                    return bool(decision[0]), decision[1]
                return bool(decision), None

            if query_context.safe_mode and tool.needs_permissions(parsed_input):
                loop = asyncio.get_running_loop()
                input_preview = (
                    parsed_input.model_dump()
                    if hasattr(parsed_input, "model_dump")
                    else str(parsed_input)
                )
                prompt = f"Allow tool '{tool.name}' with input {input_preview}? [y/N]: "
                response = await loop.run_in_executor(None, lambda: input(prompt))
                return response.strip().lower() in ("y", "yes"), None

            return True, None
        except Exception as exc:
            # Fail closed on any errors
            logger.exception(
                f"Error checking permissions for tool '{tool.name}'",
                extra={"tool": getattr(tool, "name", None)},
            )
            return False, None

    # Build full system prompt with context
    full_system_prompt = system_prompt
    if context:
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        full_system_prompt = f"{system_prompt}\n\nContext:\n{context_str}"
    if tool_mode == "text":
        tool_hint = _tool_prompt_for_text_mode(query_context.all_tools())
        if tool_hint:
            full_system_prompt = f"{full_system_prompt}\n\n{tool_hint}"
    logger.debug(
        "[query] Built system prompt",
        extra={
            "prompt_chars": len(full_system_prompt),
            "context_entries": len(context),
            "tool_count": len(tools_for_model),
        },
    )

    assistant_message = await query_llm(
        messages,
        full_system_prompt,
        tools_for_model,
        query_context.max_thinking_tokens,
        query_context.model,
        query_context.abort_controller,
    )

    # Check for abort
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE)
        return

    yield assistant_message

    tool_block_count = 0
    if isinstance(assistant_message.message.content, list):
        tool_block_count = sum(
            1
            for block in assistant_message.message.content
            if hasattr(block, "type") and block.type == "tool_use"
        )
    logger.debug(
        f"[query] Assistant message received: "
        f"text_blocks={len(assistant_message.message.content) if isinstance(assistant_message.message.content, list) else 1}, "
        f"tool_use_blocks={tool_block_count}"
    )

    # Check for tool use
    tool_use_blocks = []
    if isinstance(assistant_message.message.content, list):
        for block in assistant_message.message.content:
            normalized_block = MessageContent(**block) if isinstance(block, dict) else block
            if hasattr(normalized_block, "type") and normalized_block.type == "tool_use":
                tool_use_blocks.append(normalized_block)

    # If no tool use, we're done
    if not tool_use_blocks:
        logger.debug("[query] No tool_use blocks; returning response to user.")
        return

    # Execute tools
    tool_results: List[UserMessage] = []
    permission_denied = False

    logger.debug(f"[query] Executing {len(tool_use_blocks)} tool_use block(s).")

    for tool_use in tool_use_blocks:
        tool_name = tool_use.name
        if not tool_name:
            continue
        tool_id = getattr(tool_use, "tool_use_id", None) or getattr(tool_use, "id", None) or ""
        tool_input = getattr(tool_use, "input", {}) or {}

        # Find the tool
        tool = query_context.tool_registry.get(tool_name)
        # Auto-activate when used so subsequent rounds list it as active.
        if tool:
            query_context.activate_tools([tool_name])  # type: ignore[list-item]

        if not tool:
            # Tool not found
            logger.warning(f"[query] Tool '{tool_name}' not found for tool_use_id={tool_id}")
            result_msg = create_user_message(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "text": f"Error: Tool '{tool_name}' not found",
                        "is_error": True,
                    }
                ]
            )
            tool_results.append(result_msg)
            yield result_msg
            continue

        # Execute the tool
        tool_context = ToolUseContext(
            safe_mode=query_context.safe_mode,
            verbose=query_context.verbose,
            permission_checker=can_use_tool_fn,
            tool_registry=query_context.tool_registry,
            abort_signal=query_context.abort_controller,
        )

        try:
            # Parse input using tool's schema
            parsed_input = tool.input_schema(**tool_input)
            logger.debug(
                f"[query] tool_use_id={tool_id} name={tool_name} parsed_input="
                f"{str(parsed_input)[:500]}"
            )

            # Validate input before execution
            validation = await tool.validate_input(parsed_input, tool_context)
            if not validation.result:
                logger.debug(
                    f"[query] Validation failed for tool_use_id={tool_id}: {validation.message}"
                )
                result_msg = create_user_message(
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "text": validation.message or "Tool input validation failed.",
                            "is_error": True,
                        }
                    ]
                )
                tool_results.append(result_msg)
                yield result_msg
                continue

            # Permission check (safe mode or custom checker)
            if query_context.safe_mode or can_use_tool_fn is not None:
                allowed, denial_message = await _check_permissions(tool, parsed_input)
                if not allowed:
                    logger.debug(
                        f"[query] Permission denied for tool_use_id={tool_id}: {denial_message}"
                    )
                    denial_text = (
                        denial_message
                        or f"User aborted the tool invocation: {tool_name}"
                    )
                    result_msg = create_user_message(
                        [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "text": denial_text,
                                "is_error": True,
                            }
                        ]
                    )
                    tool_results.append(result_msg)
                    yield result_msg
                    permission_denied = True
                    break

            # Execute tool
            async for output in tool.call(parsed_input, tool_context):
                if isinstance(output, ToolProgress):
                    # Yield progress
                    progress = create_progress_message(
                        tool_use_id=tool_id,
                        sibling_tool_use_ids=set(
                            getattr(t, "tool_use_id", None) or getattr(t, "id", None) or ""
                            for t in tool_use_blocks
                        ),
                        content=output.content,
                    )
                    yield progress
                    logger.debug(f"[query] Progress from tool_use_id={tool_id}: {output.content}")
                elif isinstance(output, ToolResult):
                    # Tool completed
                    result_content = output.result_for_assistant or str(output.data)
                    result_msg = create_user_message(
                        [{"type": "tool_result", "tool_use_id": tool_id, "text": result_content}],
                        tool_use_result=output.data,
                    )
                    tool_results.append(result_msg)
                    yield result_msg
                    logger.debug(
                        f"[query] Tool completed tool_use_id={tool_id} name={tool_name} "
                        f"result_len={len(result_content)}"
                    )

        except ValidationError as ve:
            # Surface input errors back to the assistant so it can correct the call.
            details = []
            for err in ve.errors():
                loc = err.get("loc") or []
                loc_str = ".".join(str(part) for part in loc) if loc else ""
                msg = err.get("msg") or ""
                if loc_str and msg:
                    details.append(f"{loc_str}: {msg}")
                elif msg:
                    details.append(msg)
            detail_text = "; ".join(details) or str(ve)
            error_msg = create_user_message(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "text": f"Invalid input for tool '{tool_name}': {detail_text}",
                        "is_error": True,
                    }
                ]
            )
            tool_results.append(error_msg)
            yield error_msg
            continue
        except Exception as e:
            # Tool execution failed
            logger.exception(
                f"Error executing tool '{tool_name}'",
                extra={"tool": tool_name, "tool_use_id": tool_id},
            )
            error_msg = create_user_message(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "text": f"Error executing tool: {str(e)}",
                        "is_error": True,
                    }
                ]
            )
            tool_results.append(error_msg)
            yield error_msg

        if permission_denied:
            break

    # Check for abort after tools
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE_FOR_TOOL_USE)
        return

    if permission_denied:
        # Permission was explicitly denied; return after surfacing the denial result.
        return

    # Continue conversation with tool results
    new_messages = messages + [assistant_message] + tool_results
    logger.debug(
        f"[query] Recursing with {len(new_messages)} messages after tools; "
        f"tool_results_count={len(tool_results)}"
    )

    async for msg in query(new_messages, system_prompt, context, query_context, can_use_tool_fn):
        yield msg
