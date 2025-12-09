import asyncio
import json
from typing import List, Any

from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.cli.ui.context_display import format_tokens
from ripperdoc.core.config import get_global_config, provider_protocol
from ripperdoc.core.query import QueryContext
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.utils.message_compaction import (
    get_remaining_context_tokens,
    resolve_auto_compact_enabled,
    summarize_context_usage,
)
from ripperdoc.utils.token_estimation import estimate_tokens
from ripperdoc.utils.mcp import (
    estimate_mcp_tokens,
    format_mcp_instructions,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.log import get_logger

from .base import SlashCommand

logger = get_logger()


def _handle(ui: Any, _: str) -> bool:
    logger.info(
        "[context_cmd] Rendering context summary",
        extra={"session_id": getattr(ui, "session_id", None)},
    )
    config = get_global_config()
    model_profile = get_profile_for_pointer("main")
    max_context_tokens = get_remaining_context_tokens(model_profile, config.context_token_limit)
    auto_compact_enabled = resolve_auto_compact_enabled(config)
    protocol = provider_protocol(model_profile.provider) if model_profile else "openai"

    if not ui.query_context:
        ui.query_context = QueryContext(
            tools=ui.get_default_tools(),
            safe_mode=ui.safe_mode,
            verbose=ui.verbose,
        )

    async def _load_servers() -> List[Any]:
        try:
            return await load_mcp_servers_async(ui.project_path)
        finally:
            await shutdown_mcp_runtime()

    servers = asyncio.run(_load_servers())
    mcp_instructions = format_mcp_instructions(servers)
    base_system_prompt = build_system_prompt(
        ui.query_context.tools,
        "",
        {},
        mcp_instructions=mcp_instructions,
    )
    memory_instructions = build_memory_instructions()
    memory_tokens = estimate_tokens(memory_instructions) if memory_instructions else 0
    mcp_tokens = estimate_mcp_tokens(servers) if mcp_instructions else 0

    breakdown = summarize_context_usage(
        ui.conversation_messages,
        ui.query_context.tools,
        base_system_prompt,
        max_context_tokens,
        auto_compact_enabled,
        memory_tokens=memory_tokens,
        mcp_tokens=mcp_tokens,
        protocol=protocol,
    )

    model_label = model_profile.model if model_profile else "Unknown model"
    lines = ui._context_usage_lines(breakdown, model_label, auto_compact_enabled)

    lines.append("")
    # Append a brief tool listing so users can see which tools are currently loaded.
    try:
        # Detailed MCP tool listing with token estimates.
        mcp_tools = [
            tool
            for tool in getattr(ui.query_context, "tool_registry", ui.query_context).all_tools
            if getattr(tool, "is_mcp", False) or getattr(tool, "name", "").startswith("mcp__")
        ]
        if mcp_tools:
            lines.append("   MCP tools · /mcp")
            for tool in mcp_tools[:20]:
                name = getattr(tool, "name", "unknown")
                display = name
                parts = name.split("__")
                if len(parts) >= 3 and parts[0] == "mcp":
                    server = parts[1]
                    display = "__".join(parts[2:])
                    display = f"{display} ({server})"
                try:
                    schema = tool.input_schema.model_json_schema()
                    token_est = estimate_tokens(json.dumps(schema, sort_keys=True))
                except Exception:
                    token_est = 0
                lines.append(f"     └ {display}: {format_tokens(token_est)} tokens")
            if len(mcp_tools) > 20:
                lines.append(f"     └ ... (+{len(mcp_tools) - 20} more)")
    except Exception:
        logger.exception(
            "[context_cmd] Failed to summarize MCP tools",
            extra={"session_id": getattr(ui, "session_id", None)},
        )
    for line in lines:
        ui.console.print(line)
    return True


command = SlashCommand(
    name="context",
    description="Show current conversation context summary",
    handler=_handle,
)


__all__ = ["command"]
