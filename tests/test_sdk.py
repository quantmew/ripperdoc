"""Tests for the headless SDK.

Note: In-process mode has been removed. The SDK now only supports
subprocess mode. Tests for in-process mode have been removed.
"""

import asyncio

from ripperdoc.sdk import RipperdocClient, RipperdocOptions, query as sdk_query
from ripperdoc.utils.messages import create_assistant_message


async def _fake_runner(messages, system_prompt, context, query_context, permission_checker):
    del messages, system_prompt, context, query_context, permission_checker
    yield create_assistant_message("OK")


def test_sdk_imports_work():
    """Test that basic SDK imports work."""
    from ripperdoc.sdk import (
        RipperdocClient,
        RipperdocOptions,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        query,
        Message,
        UserMessage,
        AssistantMessage,
        SystemMessage,
        ResultMessage,
    )
    # Verify aliases work
    assert RipperdocClient is ClaudeSDKClient
    assert RipperdocOptions is ClaudeAgentOptions


def test_options_can_be_created():
    """Test that options can be created."""
    options = RipperdocOptions(
        allowed_tools=["Bash", "Read", "Task"],
        permission_mode="bypassPermissions",
    )
    assert options.allowed_tools == ["Bash", "Read", "Task"]
    assert options.permission_mode == "bypassPermissions"


def test_options_yolo_mode_deprecation():
    """Test that yolo_mode still works but sets permission_mode."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        options = RipperdocOptions(yolo_mode=True)
        assert options.permission_mode == "bypassPermissions"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "yolo_mode is deprecated" in str(w[0].message)


def test_client_can_be_created():
    """Test that client can be created."""
    options = RipperdocOptions()
    client = RipperdocClient(options=options)
    assert client.options == options
    assert client._connected is False


def test_client_properties():
    """Test that client properties work."""
    options = RipperdocOptions(
        model="test-model",
        user="test-user",
    )
    client = RipperdocClient(options=options)

    assert client.session_id is None
    assert client.turn_count == 0
    assert client.user == "test-user"
    assert client.history == []


def test_options_build_tools():
    """Test that build_tools works."""
    options = RipperdocOptions(
        allowed_tools=["Bash", "Read"],
    )
    tools = options.build_tools()
    tool_names = {tool.name for tool in tools}
    assert "Bash" in tool_names
    assert "Read" in tool_names
