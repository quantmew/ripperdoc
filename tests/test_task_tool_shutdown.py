"""Shutdown-response handling tests for AgentTool."""

from __future__ import annotations

from ripperdoc.utils.messaging.message_utils import tool_result_message
from ripperdoc.tools.agent import AgentTool
from ripperdoc.utils.messaging.messages import create_assistant_message
