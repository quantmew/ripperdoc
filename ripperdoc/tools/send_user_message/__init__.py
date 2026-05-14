"""SendUserMessage tool — send a one-way notification to the user."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "SendUserMessage"


class AttachmentInfo(BaseModel):
    path: str
    size: int
    is_image: bool = False


class SendUserMessageInput(BaseModel):
    message: str = Field(
        description="The notification body. Keep under 200 characters for push; longer for in-session.",
    )
    attachments: Optional[List[str]] = Field(
        default=None,
        description="Optional file paths to attach to the message.",
    )
    status: Literal["normal", "proactive"] = Field(
        default="normal",
        description=(
            "'normal' = standard message. "
            "'proactive' = background notification (user may be away)."
        ),
    )


class SendUserMessageOutput(BaseModel):
    message: str
    attachments: List[AttachmentInfo] = []
    sent_at: str
    status: str = "normal"


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}


def _resolve_attachment(path_str: str) -> Optional[AttachmentInfo]:
    p = Path(path_str)
    if not p.exists():
        return None
    size = p.stat().st_size
    is_image = p.suffix.lower() in _IMAGE_EXTENSIONS
    return AttachmentInfo(path=str(p), size=size, is_image=is_image)


class SendUserMessageTool(Tool[SendUserMessageInput, SendUserMessageOutput]):
    """Send a one-way notification to the user (no response required)."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Send a one-way notification to the user."

    @property
    def input_schema(self) -> type[SendUserMessageInput]:
        return SendUserMessageInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Sends a desktop notification to the user. "
            "Unlike AskUserQuestion, this is one-way — no response required. "
            "Use when a long task finishes while the user is away, or when there's "
            "something worth coming back for.\n\n"
            "Keep messages under 200 characters. Lead with what they'd act on.\n\n"
            "- status='proactive': background notification (may push to phone)\n"
            "- status='normal': standard in-session message\n"
            "- attachments: optional file paths to include"
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[SendUserMessageInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: SendUserMessageInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if not input_data.message.strip():
            return ValidationResult(result=False, message="Message cannot be empty.")
        if len(input_data.message) > 10000:
            return ValidationResult(
                result=False,
                message="Message too long (max 10000 characters).",
            )
        if input_data.attachments:
            for att in input_data.attachments:
                p = Path(att)
                if not p.exists():
                    return ValidationResult(
                        result=False,
                        message=f"Attachment not found: {att}",
                    )
                if p.stat().st_size > 10 * 1024 * 1024:
                    return ValidationResult(
                        result=False,
                        message=f"Attachment too large (>10MB): {att}",
                    )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: SendUserMessageOutput) -> str:
        parts = [f"Message sent ({output.status})"]
        if output.attachments:
            names = [os.path.basename(a.path) for a in output.attachments]
            parts.append(f"Attachments: {', '.join(names)}")
        return " | ".join(parts)

    def render_tool_use_message(
        self,
        input_data: SendUserMessageInput,
        _verbose: bool = False,
    ) -> str:
        preview = input_data.message[:80]
        if len(input_data.message) > 80:
            preview += "..."
        return f"[{input_data.status}] {preview}"

    async def call(
        self,
        input_data: SendUserMessageInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        resolved: List[AttachmentInfo] = []
        if input_data.attachments:
            for att_path in input_data.attachments:
                info = _resolve_attachment(att_path)
                if info:
                    resolved.append(info)

        output = SendUserMessageOutput(
            message=input_data.message,
            attachments=resolved,
            sent_at=datetime.now(timezone.utc).isoformat(),
            status=input_data.status,
        )

        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
