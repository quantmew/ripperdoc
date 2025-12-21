"""LLM callback helper for prompt-based hooks."""

from __future__ import annotations

from typing import Optional

from ripperdoc.core.hooks.executor import LLMCallback
from ripperdoc.core.query import query_llm
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import AssistantMessage, create_user_message

logger = get_logger()


def _extract_text(message: AssistantMessage) -> str:
    content = message.message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else None
            )
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return ""


def build_hook_llm_callback(
    *,
    model: str = "quick",
    max_thinking_tokens: int = 0,
    system_prompt: Optional[str] = None,
) -> LLMCallback:
    """Build an async callback for prompt hooks using the configured model."""

    async def _callback(prompt: str) -> str:
        try:
            assistant = await query_llm(
                [create_user_message(prompt)],
                system_prompt or "",
                [],
                max_thinking_tokens=max_thinking_tokens,
                model=model,
                stream=False,
            )
            return _extract_text(assistant).strip()
        except Exception as exc:
            logger.warning(
                "[hooks] Prompt hook LLM callback failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return f"Prompt hook evaluation failed: {exc}"

    return _callback

