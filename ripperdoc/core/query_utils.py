"""Utility helpers for query handling, tool schemas, and message normalization."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Union
from uuid import uuid4

from json_repair import repair_json  # type: ignore[import-not-found]
from pydantic import ValidationError

from ripperdoc.core.config import ModelProfile, ProviderType, get_global_config
from ripperdoc.core.tool import Tool, build_tool_description, tool_input_examples
from ripperdoc.utils.json_utils import safe_parse_json
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    AssistantMessage,
    MessageContent,
    ProgressMessage,
    UserMessage,
    create_assistant_message,
    create_user_message,
)

logger = get_logger()


def _safe_int(value: object) -> int:
    """Best-effort int conversion for usage counters."""
    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(value)
        if hasattr(value, "__int__"):
            return int(value)  # type: ignore[arg-type]
        return 0
    except (TypeError, ValueError):
        return 0


def _get_usage_field(usage: Optional[Mapping[str, Any] | object], field: str) -> int:
    """Fetch a usage field from either a dict or object."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return _safe_int(usage.get(field))
    return _safe_int(getattr(usage, field, 0))


def anthropic_usage_tokens(usage: Optional[Mapping[str, Any] | object]) -> Dict[str, int]:
    """Extract token counts from an Anthropic response usage payload."""
    return {
        "input_tokens": _get_usage_field(usage, "input_tokens"),
        "output_tokens": _get_usage_field(usage, "output_tokens"),
        "cache_read_input_tokens": _get_usage_field(usage, "cache_read_input_tokens"),
        "cache_creation_input_tokens": _get_usage_field(usage, "cache_creation_input_tokens"),
    }


def openai_usage_tokens(usage: Optional[Mapping[str, Any] | object]) -> Dict[str, int]:
    """Extract token counts from an OpenAI-compatible response usage payload."""
    prompt_details = None
    input_details = None
    output_details = None
    if isinstance(usage, dict):
        prompt_details = usage.get("prompt_tokens_details")
        input_details = usage.get("input_tokens_details")
        output_details = usage.get("output_tokens_details")
    else:
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)

    cache_read_tokens = 0
    if prompt_details:
        cache_read_tokens = _get_usage_field(prompt_details, "cached_tokens")
    if not cache_read_tokens and input_details:
        cache_read_tokens = _get_usage_field(input_details, "cached_tokens")

    input_tokens = _get_usage_field(usage, "prompt_tokens")
    if not input_tokens:
        input_tokens = _get_usage_field(usage, "input_tokens")

    output_tokens = _get_usage_field(usage, "completion_tokens")
    if not output_tokens:
        output_tokens = _get_usage_field(usage, "output_tokens")

    reasoning_tokens = _get_usage_field(output_details, "reasoning_tokens") if output_details else 0
    if reasoning_tokens:
        if output_tokens <= 0:
            output_tokens = reasoning_tokens
        elif output_tokens < reasoning_tokens:
            output_tokens = output_tokens + reasoning_tokens
        else:
            output_tokens = max(output_tokens, reasoning_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": 0,
    }


def estimate_cost_usd(model_profile: ModelProfile, usage_tokens: Dict[str, int]) -> float:
    """Compute USD cost using per-1M token pricing from the model profile."""
    input_price = getattr(model_profile, "input_cost_per_million_tokens", 0.0) or 0.0
    output_price = getattr(model_profile, "output_cost_per_million_tokens", 0.0) or 0.0

    total_input_tokens = (
        _safe_int(usage_tokens.get("input_tokens"))
        + _safe_int(usage_tokens.get("cache_read_input_tokens"))
        + _safe_int(usage_tokens.get("cache_creation_input_tokens"))
    )
    output_tokens = _safe_int(usage_tokens.get("output_tokens"))

    cost = (total_input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return float(cost)


def resolve_model_profile(model: str) -> ModelProfile:
    """Resolve a model pointer to a concrete profile, falling back to a safe default."""
    config = get_global_config()
    profile_name = getattr(config.model_pointers, model, None) or model
    model_profile = config.model_profiles.get(profile_name)
    if model_profile is None:
        fallback_profile = getattr(config.model_pointers, "main", "default")
        model_profile = config.model_profiles.get(fallback_profile)
    if not model_profile:
        logger.warning(
            "[config] No model profile found; using built-in default profile",
            extra={"model_pointer": model},
        )
        return ModelProfile(provider=ProviderType.OPENAI_COMPATIBLE, model="gpt-4o-mini")
    return model_profile


def determine_tool_mode(model_profile: ModelProfile) -> str:
    """Return configured tool mode for provider."""
    if model_profile.provider != ProviderType.OPENAI_COMPATIBLE:
        return "native"
    configured = getattr(model_profile, "openai_tool_mode", "native") or "native"
    configured = configured.lower()
    if configured not in {"native", "text"}:
        configured = getattr(get_global_config(), "openai_tool_mode", "native") or "native"
        configured = configured.lower()
    return configured if configured in {"native", "text"} else "native"


def _parse_text_mode_json_blocks(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse a JSON code block or raw JSON string into content blocks for text mode."""
    if not text or not isinstance(text, str):
        return None

    code_blocks = re.findall(r"```(?:\s*json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates = [blk.strip() for blk in code_blocks if blk.strip()]

    def _normalize_blocks(parsed: object) -> Optional[List[Dict[str, Any]]]:
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
                    except (TypeError, AttributeError):
                        is_req = False
                required_fields.append(f"{fname}{' (required)' if is_req else ''}")
        except (AttributeError, TypeError):
            required_fields = []

        required_str = ", ".join(required_fields) if required_fields else "see input schema"
        lines.append(f"- {tool.name}: fields {required_str}")

        schema_json = ""
        try:
            schema_json = json.dumps(
                tool.input_schema.model_json_schema(), ensure_ascii=False, indent=2
            )
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

    example_blocks = [
        {"type": "text", "text": "好的，我来帮你查看一下README.md文件"},
        {
            "type": "tool_use",
            "tool_use_id": "tool_id_000001",
            "tool": "Read",
            "input": {"file_path": "README.md"},
        },
    ]
    lines.append("Example:")
    lines.append("```json")
    lines.append(json.dumps(example_blocks, ensure_ascii=False, indent=2))
    lines.append("```")

    return "\n".join(lines)


def text_mode_history(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
) -> List[Union[UserMessage, AssistantMessage]]:
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
                block_type = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                block_text = (
                    getattr(block, "text", None)
                    if hasattr(block, "text")
                    else (block.get("text") if isinstance(block, dict) else None)
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
                text_content = (
                    f"```json\n{json.dumps(parsed_blocks, ensure_ascii=False, indent=2)}\n```"
                )
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


def _maybe_convert_json_block_to_tool_use(
    content_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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


def build_full_system_prompt(
    system_prompt: str, context: Dict[str, str], tool_mode: str, tools: List[Tool[Any, Any]]
) -> str:
    """Compose the final system prompt including context and tool hints."""
    full_prompt = system_prompt
    if context:
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        full_prompt = f"{system_prompt}\n\nContext:\n{context_str}"
    if tool_mode == "text":
        tool_hint = _tool_prompt_for_text_mode(tools)
        if tool_hint:
            full_prompt = f"{full_prompt}\n\n{tool_hint}"
    return full_prompt


def log_openai_messages(normalized_messages: List[Dict[str, Any]]) -> None:
    """Trace normalized messages for OpenAI calls to simplify debugging."""
    summary_parts = []
    for idx, message in enumerate(normalized_messages):
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")
        has_reasoning = "reasoning_content" in message and message.get("reasoning_content")
        ids = [tc.get("id") for tc in tool_calls] if tool_calls else []
        summary_parts.append(
            f"{idx}:{role}"
            + (f" tool_calls={ids}" if ids else "")
            + (f" tool_call_id={tool_call_id}" if tool_call_id else "")
            + (" +reasoning" if has_reasoning else "")
        )
    logger.debug(f"[query_llm] OpenAI normalized messages: {' | '.join(summary_parts)}")


async def build_anthropic_tool_schemas(tools: List[Tool[Any, Any]]) -> List[Dict[str, Any]]:
    """Render tool schemas in Anthropic format."""
    schemas = []
    for tool in tools:
        description = await build_tool_description(tool, include_examples=True, max_examples=2)
        schema: Dict[str, Any] = {
            "name": tool.name,
            "description": description,
            "input_schema": tool.input_schema.model_json_schema(),
            "defer_loading": bool(getattr(tool, "defer_loading", lambda: False)()),
        }
        examples = tool_input_examples(tool, limit=5)
        if examples:
            schema["input_examples"] = examples
        schemas.append(schema)
    return schemas


async def build_openai_tool_schemas(tools: List[Tool[Any, Any]]) -> List[Dict[str, Any]]:
    """Render tool schemas in OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        description = await build_tool_description(tool, include_examples=True, max_examples=2)
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
    return openai_tools


def content_blocks_from_anthropic_response(response: Any, tool_mode: str) -> List[Dict[str, Any]]:
    """Normalize Anthropic response content to our internal block format."""
    blocks: List[Dict[str, Any]] = []
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            blocks.append({"type": "text", "text": getattr(block, "text", "")})
        elif btype == "thinking":
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None) or "",
                    "signature": getattr(block, "signature", None),
                }
            )
        elif btype == "redacted_thinking":
            # Preserve encrypted payload for replay even if we don't display it.
            blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": getattr(block, "data", None),
                    "signature": getattr(block, "signature", None),
                }
            )
        elif btype == "tool_use":
            raw_input = getattr(block, "input", {}) or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "tool_use_id": getattr(block, "id", None) or str(uuid4()),
                    "name": getattr(block, "name", None),
                    "input": _normalize_tool_args(raw_input),
                }
            )

    if tool_mode == "text":
        blocks = _maybe_convert_json_block_to_tool_use(blocks)
    return blocks


def content_blocks_from_openai_choice(choice: Any, tool_mode: str) -> List[Dict[str, Any]]:
    """Normalize OpenAI-compatible choice to our internal block format."""
    content_blocks = []
    if getattr(choice.message, "content", None):
        content_blocks.append({"type": "text", "text": choice.message.content})

    if getattr(choice.message, "tool_calls", None):
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
    elif tool_mode == "text":
        content_blocks = _maybe_convert_json_block_to_tool_use(content_blocks)
    return content_blocks


def extract_tool_use_blocks(
    assistant_message: AssistantMessage,
) -> List[MessageContent]:
    """Return all tool_use blocks from an assistant message."""
    content = getattr(assistant_message.message, "content", None)
    if not isinstance(content, list):
        return []

    tool_blocks: List[MessageContent] = []
    for block in content:
        normalized = MessageContent(**block) if isinstance(block, dict) else block
        if getattr(normalized, "type", None) == "tool_use":
            tool_blocks.append(normalized)
    return tool_blocks


def tool_result_message(
    tool_use_id: str, text: str, is_error: bool = False, tool_use_result: Any = None
) -> UserMessage:
    """Build a user message representing a tool_result block."""
    block: Dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "text": text}
    if is_error:
        block["is_error"] = True
    return create_user_message([block], tool_use_result=tool_use_result)


def format_pydantic_errors(error: ValidationError) -> str:
    """Render a compact validation error summary."""
    details = []
    for err in error.errors():
        loc: list[Any] = list(err.get("loc") or [])
        loc_str = ".".join(str(part) for part in loc) if loc else ""
        msg = err.get("msg") or ""
        if loc_str and msg:
            details.append(f"{loc_str}: {msg}")
        elif msg:
            details.append(msg)
    return "; ".join(details) or str(error)
