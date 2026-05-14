"""SendMessage tool — intra-team messaging and protocol handling."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseExample,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.tools.SendMessageTool._prompt import SEND_MESSAGE_PROMPT
from ripperdoc.utils.collaboration.team_context import (
    resolve_active_team_name,
    sender_name,
)
from ripperdoc.utils.collaboration.teams import (
    TEAM_LEAD_NAME,
    get_team,
    participant_color,
    send_team_message,
)
from ripperdoc.utils.log import get_logger


logger = get_logger()


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_recipient_name(team: Any, recipient: str) -> tuple[str, bool]:
    raw = (recipient or "").strip()
    if not raw:
        return raw, False

    candidate = raw.split("@", 1)[0].strip() if "@" in raw else raw
    if not candidate:
        return raw, False

    for member in getattr(team, "members", []):
        name = (getattr(member, "name", "") or "").strip()
        if name and name.lower() == candidate.lower():
            return name, True
    return candidate, False


class SendMessageInput(BaseModel):
    type: Literal[
        "message",
        "broadcast",
        "shutdown_request",
        "shutdown_response",
        "plan_approval_response",
    ]
    recipient: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    request_id: Optional[str] = None
    approve: Optional[bool] = None
    model_config = ConfigDict(extra="forbid")


class SendMessageRouting(BaseModel):
    sender: str
    sender_color: str = Field(serialization_alias="senderColor")
    target: str
    target_color: Optional[str] = Field(default=None, serialization_alias="targetColor")
    summary: Optional[str] = None
    content: Optional[str] = None


class SendMessageOutput(BaseModel):
    success: bool
    message: str
    recipients: Optional[List[str]] = None
    routing: Optional[SendMessageRouting] = None
    request_id: Optional[str] = None
    target: Optional[str] = None


class SendMessageTool(Tool[SendMessageInput, SendMessageOutput]):
    @property
    def name(self) -> str:
        return "SendMessage"

    async def description(self) -> str:
        return (
            "Send structured intra-team protocol messages: "
            "message, broadcast, shutdown_request, shutdown_response, plan_approval_response."
        )

    @property
    def input_schema(self) -> type[SendMessageInput]:
        return SendMessageInput

    def input_examples(self) -> list[ToolUseExample]:
        return [
            ToolUseExample(
                description="Direct teammate message",
                example={
                    "type": "message",
                    "recipient": "researcher",
                    "content": "Please investigate auth regression failures first",
                    "summary": "Investigate auth regressions",
                },
            ),
            ToolUseExample(
                description="Broadcast message",
                example={
                    "type": "broadcast",
                    "content": "Main branch broken, pause all commits",
                    "summary": "Main branch broken pause all commits",
                },
            ),
            ToolUseExample(
                description="Shutdown request",
                example={
                    "type": "shutdown_request",
                    "recipient": "tester",
                    "content": "Tasks done, please shut down",
                },
            ),
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return SEND_MESSAGE_PROMPT

    def needs_permissions(self, _input_data: Optional[SendMessageInput] = None) -> bool:
        return False

    def _require_field(self, value: Optional[str], field_name: str) -> Optional[str]:
        if not (value or "").strip():
            return f"{field_name} is required"
        return None

    def _validate_message_type(
        self, message_type: str, allowed_types: set[str]
    ) -> Optional[str]:
        if message_type not in allowed_types:
            return "Unsupported message type"
        return None

    async def validate_input(
        self,
        input_data: SendMessageInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        message_type = (input_data.type or "").strip()
        allowed_types = {
            "message",
            "broadcast",
            "shutdown_request",
            "shutdown_response",
            "plan_approval_response",
        }

        if error := self._validate_message_type(message_type, allowed_types):
            return ValidationResult(result=False, message=error)

        validators: dict[str, list[Callable[[], Optional[str]]]] = {
            "message": [
                lambda: self._require_field(input_data.recipient, "recipient"),
                lambda: self._require_field(input_data.content, "content"),
                lambda: self._require_field(input_data.summary, "summary"),
            ],
            "broadcast": [
                lambda: self._require_field(input_data.content, "content"),
                lambda: self._require_field(input_data.summary, "summary"),
            ],
            "shutdown_request": [
                lambda: self._require_field(input_data.recipient, "recipient"),
            ],
            "shutdown_response": [
                lambda: self._require_field(input_data.request_id, "request_id"),
                lambda: None if input_data.approve is not None else "approve is required",
                lambda: None if input_data.approve is not False or (input_data.content or "").strip()
                else "content is required when approve=false",
            ],
            "plan_approval_response": [
                lambda: self._require_field(input_data.request_id, "request_id"),
                lambda: None if input_data.approve is not None else "approve is required",
                lambda: self._require_field(input_data.recipient, "recipient"),
            ],
        }

        for validator in validators.get(message_type, []):
            if error := validator():
                return ValidationResult(result=False, message=error)

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: SendMessageOutput) -> str:
        return output.message

    def render_tool_use_message(self, input_data: SendMessageInput, _verbose: bool = False) -> str:
        return f"Sending {input_data.type} message"

    async def call(
        self,
        input_data: SendMessageInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        team_name = resolve_active_team_name(context)
        if not team_name:
            raise ValueError("No active team context found. Create/select a team first.")

        team = get_team(team_name)
        if team is None:
            raise ValueError(f"Team '{team_name}' not found.")

        _sender = sender_name(context, team_lead_name=TEAM_LEAD_NAME)
        _sender_color = participant_color(_sender)

        message_type = input_data.type
        if message_type == "message":
            recipient, exists = _normalize_recipient_name(team, input_data.recipient or "")
            if not exists and recipient != TEAM_LEAD_NAME:
                known = sorted(
                    {
                        member.name
                        for member in team.members
                        if (member.name or "").strip()
                    }
                )
                raise ValueError(
                    "Unknown recipient "
                    f"'{recipient}' for team '{team_name}'. "
                    + (
                        f"Known teammates: {', '.join(known)}"
                        if known
                        else "No teammates are registered in this team yet."
                    )
                )
            content = (input_data.content or "").strip()
            summary = (input_data.summary or "").strip()
            send_team_message(
                team_name=team_name,
                sender=_sender,
                recipients=[recipient],
                message_type="message",
                content=content,
                metadata={"summary": summary, "recipient": recipient},
            )
            output = SendMessageOutput(
                success=True,
                message=f"Message sent to {recipient}'s inbox",
                routing=SendMessageRouting(
                    sender=_sender,
                    sender_color=_sender_color,
                    target=f"@{recipient}",
                    target_color=participant_color(recipient),
                    summary=summary,
                    content=content,
                ),
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        if message_type == "broadcast":
            content = (input_data.content or "").strip()
            summary = (input_data.summary or "").strip()
            recipients = list(
                dict.fromkeys(
                    member.name
                    for member in team.members
                    if (member.name or "").strip() and member.name != _sender
                )
            )
            send_team_message(
                team_name=team_name,
                sender=_sender,
                recipients=recipients,
                message_type="broadcast",
                content=content,
                metadata={"summary": summary},
            )
            if recipients:
                message = f"Message broadcast to {len(recipients)} teammate(s): "
                message += ", ".join(recipients)
            else:
                message = "No teammates to broadcast to (you are the only team member)"
            output = SendMessageOutput(
                success=True,
                message=message,
                recipients=recipients,
                routing=SendMessageRouting(
                    sender=_sender,
                    sender_color=_sender_color,
                    target="@team",
                    summary=summary,
                    content=content,
                ),
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        if message_type == "shutdown_request":
            recipient, exists = _normalize_recipient_name(team, input_data.recipient or "")
            if not exists and recipient != TEAM_LEAD_NAME:
                known = sorted(
                    {
                        member.name
                        for member in team.members
                        if (member.name or "").strip()
                    }
                )
                raise ValueError(
                    "Unknown recipient "
                    f"'{recipient}' for team '{team_name}'. "
                    + (
                        f"Known teammates: {', '.join(known)}"
                        if known
                        else "No teammates are registered in this team yet."
                    )
                )
            content = (input_data.content or "").strip()
            request_id = f"req_{uuid4().hex[:10]}"
            request_payload = {
                "type": "shutdown_request",
                "requestId": request_id,
                "from": _sender,
                "reason": content or "Shutdown requested.",
                "timestamp": _iso_utc_now(),
            }
            send_team_message(
                team_name=team_name,
                sender=_sender,
                recipients=[recipient],
                message_type="shutdown_request",
                content=json.dumps(request_payload, ensure_ascii=False),
                metadata={
                    "request_id": request_id,
                    "requestId": request_id,
                    "recipient": recipient,
                    "content": content or "Shutdown requested.",
                    "reason": content or "Shutdown requested.",
                    "sender": _sender,
                    "from": _sender,
                },
            )
            output = SendMessageOutput(
                success=True,
                message=f"Shutdown request sent to {recipient}. Request ID: {request_id}",
                request_id=request_id,
                target=recipient,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        if message_type == "shutdown_response":
            request_id = (input_data.request_id or "").strip()
            approve = bool(input_data.approve)
            content = (input_data.content or "").strip()
            response_target = TEAM_LEAD_NAME
            sender_member = next(
                (member for member in team.members if (member.name or "").strip().lower() == _sender.lower()),
                None,
            )
            sender_backend = (
                (sender_member.backend_type or "").strip()
                if sender_member and sender_member.backend_type
                else "in-process"
            )
            sender_pane_id = sender_member.tmux_pane_id if sender_member else None

            if approve:
                response_payload: dict[str, Any] = {
                    "type": "shutdown_approved",
                    "requestId": request_id,
                    "from": _sender,
                    "timestamp": _iso_utc_now(),
                }
                if sender_pane_id:
                    response_payload["paneId"] = sender_pane_id
                if sender_backend:
                    response_payload["backendType"] = sender_backend
                assistant_message = (
                    f"Shutdown approved. Sent confirmation to {TEAM_LEAD_NAME}. Agent {_sender} is now exiting."
                )
            else:
                reject_reason = content or "Shutdown rejected."
                response_payload = {
                    "type": "shutdown_rejected",
                    "requestId": request_id,
                    "from": _sender,
                    "reason": reject_reason,
                    "timestamp": _iso_utc_now(),
                }
                assistant_message = (
                    f'Shutdown rejected. Reason: "{reject_reason}". Continuing to work.'
                )
            send_team_message(
                team_name=team_name,
                sender=_sender,
                recipients=[response_target],
                message_type="shutdown_response",
                content=json.dumps(response_payload, ensure_ascii=False),
                metadata={
                    "request_id": request_id,
                    "requestId": request_id,
                    "approve": approve,
                    "approved": approve,
                    "recipient": response_target,
                    "sender": _sender,
                    "from": _sender,
                    "content": content or ("Approved" if approve else "Rejected"),
                    "reason": content or ("Approved" if approve else "Rejected"),
                    "protocol_type": response_payload.get("type"),
                    "backendType": response_payload.get("backendType"),
                    "paneId": response_payload.get("paneId"),
                },
            )
            output = SendMessageOutput(
                success=True,
                message=assistant_message,
                request_id=request_id,
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
            return

        # plan_approval_response
        request_id = (input_data.request_id or "").strip()
        approve = bool(input_data.approve)
        if _sender != TEAM_LEAD_NAME:
            raise ValueError(
                "Only the team lead can approve plans. Teammates cannot approve or reject plans."
            )
        recipient, exists = _normalize_recipient_name(team, input_data.recipient or "")
        if not exists and recipient != TEAM_LEAD_NAME:
            known = sorted(
                {
                    member.name
                    for member in team.members
                    if (member.name or "").strip()
                }
            )
            raise ValueError(
                "Unknown recipient "
                f"'{recipient}' for team '{team_name}'. "
                + (
                    f"Known teammates: {', '.join(known)}"
                    if known
                    else "No teammates are registered in this team yet."
                )
            )
        content = (input_data.content or "").strip()
        if approve:
            protocol_payload: dict[str, Any] = {
                "type": "plan_approval_response",
                "requestId": request_id,
                "approved": True,
                "timestamp": _iso_utc_now(),
                "permissionMode": "default",
            }
            assistant_message = (
                f"Plan approved for {recipient}. They will receive the approval and can proceed with implementation."
            )
        else:
            feedback = content or "Plan needs revision"
            protocol_payload = {
                "type": "plan_approval_response",
                "requestId": request_id,
                "approved": False,
                "feedback": feedback,
                "timestamp": _iso_utc_now(),
            }
            assistant_message = (
                f'Plan rejected for {recipient} with feedback: "{feedback}"'
            )
        send_team_message(
            team_name=team_name,
            sender=_sender,
            recipients=[recipient],
            message_type="plan_approval_response",
            content=json.dumps(protocol_payload, ensure_ascii=False),
            metadata={
                "request_id": request_id,
                "requestId": request_id,
                "approve": approve,
                "approved": approve,
                "recipient": recipient,
                "permissionMode": protocol_payload.get("permissionMode"),
                "feedback": protocol_payload.get("feedback"),
            },
        )
        output = SendMessageOutput(
            success=True,
            message=assistant_message,
            request_id=request_id,
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
