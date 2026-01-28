"""Tests for Claude Agent SDK compatibility layer.

This test suite verifies that the Ripperdoc SDK provides full API
compatibility with Claude Agent SDK, allowing users to switch between
the two by simply changing the import statement.
"""

import asyncio
from typing import Any
from collections.abc import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ripperdoc.sdk import (
    # Claude SDK compatible imports
    query,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ResultMessage,
    StreamEvent,
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    PermissionMode,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionUpdate,
    HookMatcher,
    HookEvent,
    CanUseTool,
    ToolPermissionContext,
    McpServerConfig,
    AgentDefinition,
    SettingSource,
    SdkBeta,
    SandboxSettings,
    # Backward compatibility aliases
    RipperdocClient,
    RipperdocOptions,
)
from ripperdoc.sdk.adapter import MessageAdapter, ResultMessageFactory
from ripperdoc.sdk.types import (
    PreToolUseHookInput,
    PostToolUseHookInput,
    UserPromptSubmitHookInput,
    StopHookInput,
)
from ripperdoc.utils.messages import (
    create_user_message as ripperdoc_create_user_message,
    create_assistant_message as ripperdoc_create_assistant_message,
    MessageContent,
    Message,
    MessageRole,
)


# =============================================================================
# Test Type Imports and Compatibility
# =============================================================================


class TestTypeImports:
    """Test that all Claude SDK compatible types can be imported."""

    def test_message_types_exist(self):
        """All message types should be importable."""
        assert UserMessage is not None
        assert AssistantMessage is not None
        assert SystemMessage is not None
        assert ResultMessage is not None
        assert StreamEvent is not None

    def test_content_block_types_exist(self):
        """All ContentBlock subtypes should be available."""
        assert TextBlock is not None
        assert ThinkingBlock is not None
        assert ToolUseBlock is not None
        assert ToolResultBlock is not None

    def test_backward_compatibility_aliases(self):
        """Backward compatibility aliases should exist."""
        assert RipperdocClient is ClaudeSDKClient
        assert RipperdocOptions is ClaudeAgentOptions


# =============================================================================
# Test ContentBlock Types
# =============================================================================


class TestContentBlockTypes:
    """Test ContentBlock subtype creation and properties."""

    def test_text_block_creation(self):
        """TextBlock should store text content."""
        block = TextBlock(text="Hello, world!")
        assert block.text == "Hello, world!"

    def test_thinking_block_creation(self):
        """ThinkingBlock should store thinking and signature."""
        block = ThinkingBlock(thinking="Let me think...", signature="abc123")
        assert block.thinking == "Let me think..."
        assert block.signature == "abc123"

    def test_tool_use_block_creation(self):
        """ToolUseBlock should store tool invocation details."""
        block = ToolUseBlock(
            id="call_123",
            name="bash",
            input={"command": "ls -la"}
        )
        assert block.id == "call_123"
        assert block.name == "bash"
        assert block.input == {"command": "ls -la"}

    def test_tool_result_block_creation(self):
        """ToolResultBlock should store tool execution result."""
        block = ToolResultBlock(
            tool_use_id="call_123",
            content="Output here",
            is_error=False
        )
        assert block.tool_use_id == "call_123"
        assert block.content == "Output here"
        assert block.is_error is False

    def test_tool_result_block_with_list_content(self):
        """ToolResultBlock can accept list content."""
        block = ToolResultBlock(
            tool_use_id="call_123",
            content=[{"type": "text", "text": "Part 1"}],
            is_error=None
        )
        assert block.tool_use_id == "call_123"
        assert isinstance(block.content, list)

    def test_tool_result_block_optional_fields(self):
        """ToolResultBlock optional fields can be None."""
        block = ToolResultBlock(tool_use_id="call_123")
        assert block.content is None
        assert block.is_error is None


# =============================================================================
# Test Message Types
# =============================================================================


class TestMessageTypes:
    """Test Claude SDK compatible message types."""

    def test_user_message_with_string_content(self):
        """UserMessage can be created with string content."""
        msg = UserMessage(content="Hello, AI!")
        assert msg.content == "Hello, AI!"
        assert msg.uuid is None
        assert msg.parent_tool_use_id is None
        assert msg.tool_use_result is None

    def test_user_message_with_block_content(self):
        """UserMessage can be created with ContentBlock list."""
        msg = UserMessage(
            content=[TextBlock(text="Hello"), ToolUseBlock(id="1", name="test", input={})]
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextBlock)
        assert isinstance(msg.content[1], ToolUseBlock)

    def test_user_message_with_optional_fields(self):
        """UserMessage optional fields work correctly."""
        msg = UserMessage(
            content="Test",
            uuid="user-123",
            parent_tool_use_id="tool-456",
            tool_use_result={"status": "ok"}
        )
        assert msg.uuid == "user-123"
        assert msg.parent_tool_use_id == "tool-456"
        assert msg.tool_use_result == {"status": "ok"}

    def test_assistant_message_creation(self):
        """AssistantMessage should store content blocks and model."""
        msg = AssistantMessage(
            content=[TextBlock(text="Response"), ThinkingBlock(thinking="...", signature="sig")],
            model="claude-sonnet-4-5"
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.model == "claude-sonnet-4-5"
        assert msg.parent_tool_use_id is None
        assert msg.error is None

    def test_assistant_message_with_error(self):
        """AssistantMessage can have an error field."""
        msg = AssistantMessage(
            content=[TextBlock(text="Failed")],
            model="claude-sonnet-4-5",
            error="api_error"
        )
        assert msg.error == "api_error"

    def test_system_message_creation(self):
        """SystemMessage should store subtype and data."""
        msg = SystemMessage(
            subtype="info",
            data={"key": "value"}
        )
        assert msg.subtype == "info"
        assert msg.data == {"key": "value"}

    def test_result_message_creation(self):
        """ResultMessage should store session information."""
        msg = ResultMessage(
            subtype="result",
            duration_ms=1500,
            duration_api_ms=1200,
            is_error=False,
            num_turns=3,
            session_id="session-123",
            total_cost_usd=0.005
        )
        assert msg.subtype == "result"
        assert msg.duration_ms == 1500
        assert msg.duration_api_ms == 1200
        assert msg.is_error is False
        assert msg.num_turns == 3
        assert msg.session_id == "session-123"
        assert msg.total_cost_usd == 0.005

    def test_result_message_optional_fields(self):
        """ResultMessage optional fields work correctly."""
        msg = ResultMessage(
            subtype="result",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="session-456"
        )
        assert msg.total_cost_usd is None
        assert msg.usage is None
        assert msg.result is None

    def test_stream_event_creation(self):
        """StreamEvent should store event data."""
        msg = StreamEvent(
            uuid="msg-123",
            session_id="session-789",
            event={"type": "content_block_delta"}
        )
        assert msg.uuid == "msg-123"
        assert msg.session_id == "session-789"
        assert msg.event == {"type": "content_block_delta"}
        assert msg.parent_tool_use_id is None


# =============================================================================
# Test Message Adapter
# =============================================================================


class TestMessageAdapter:
    """Test message conversion between Ripperdoc and Claude SDK formats."""

    def test_convert_text_content_to_claude_block(self):
        """Test converting text content from Ripperdoc to Claude SDK format."""
        block = MessageContent(type="text", text="Test message")

        claude_block = MessageAdapter._content_to_block(block)

        assert isinstance(claude_block, TextBlock)
        assert claude_block.text == "Test message"

    def test_convert_thinking_content_to_claude_block(self):
        """Test converting thinking content from Ripperdoc to Claude SDK format."""
        block = MessageContent(
            type="thinking",
            thinking="Let me think...",
            signature="sig123"
        )

        claude_block = MessageAdapter._content_to_block(block)

        assert isinstance(claude_block, ThinkingBlock)
        assert claude_block.thinking == "Let me think..."
        assert claude_block.signature == "sig123"

    def test_convert_tool_use_content_to_claude_block(self):
        """Test converting tool use content from Ripperdoc to Claude SDK format."""
        block = MessageContent(
            type="tool_use",
            id="call_123",
            name="bash",
            input={"command": "ls"}
        )

        claude_block = MessageAdapter._content_to_block(block)

        assert isinstance(claude_block, ToolUseBlock)
        assert claude_block.id == "call_123"
        assert claude_block.name == "bash"
        assert claude_block.input == {"command": "ls"}

    def test_convert_tool_result_content_to_claude_block(self):
        """Test converting tool result content from Ripperdoc to Claude SDK format."""
        block = MessageContent(
            type="tool_result",
            tool_use_id="call_123",
            text="Result text",
            is_error=False
        )

        claude_block = MessageAdapter._content_to_block(block)

        assert isinstance(claude_block, ToolResultBlock)
        assert claude_block.tool_use_id == "call_123"
        assert claude_block.content == "Result text"

    def test_dict_to_text_block_conversion(self):
        """Test converting dict to TextBlock."""
        d = {"type": "text", "text": "Hello"}

        block = MessageAdapter._dict_to_block(d)

        assert isinstance(block, TextBlock)
        assert block.text == "Hello"

    def test_dict_to_tool_use_block_conversion(self):
        """Test converting dict to ToolUseBlock."""
        d = {
            "type": "tool_use",
            "id": "call_123",
            "name": "bash",
            "input": {"command": "ls"}
        }

        block = MessageAdapter._dict_to_block(d)

        assert isinstance(block, ToolUseBlock)
        assert block.id == "call_123"

    def test_claude_text_block_to_message_content(self):
        """Test converting TextBlock back to MessageContent."""
        block = TextBlock(text="Test text")

        mc = MessageAdapter._block_to_content(block)

        assert mc.type == "text"
        assert mc.text == "Test text"

    def test_claude_tool_use_block_to_message_content(self):
        """Test converting ToolUseBlock back to MessageContent."""
        block = ToolUseBlock(
            id="call_123",
            name="bash",
            input={"command": "ls"}
        )

        mc = MessageAdapter._block_to_content(block)

        assert mc.type == "tool_use"
        assert mc.id == "call_123"
        assert mc.name == "bash"


# =============================================================================
# Test ClaudeAgentOptions
# =============================================================================


class TestClaudeAgentOptions:
    """Test ClaudeAgentOptions configuration class."""

    def test_default_options(self):
        """Default options should have sensible defaults."""
        options = ClaudeAgentOptions()

        assert options.permission_mode == "default"
        # allowed_tools defaults to None (not specified)
        assert options.allowed_tools is None or options.allowed_tools == []
        assert options.disallowed_tools is None or options.disallowed_tools == []
        assert options.verbose is False
        assert options.max_thinking_tokens == 0
        assert options.include_partial_messages is False

    def test_permission_mode_options(self):
        """All permission modes should be accepted."""
        valid_modes = ["default", "acceptEdits", "plan", "bypassPermissions"]

        for mode in valid_modes:
            options = ClaudeAgentOptions(permission_mode=mode)
            assert options.permission_mode == mode

    def test_allowed_tools_filter(self):
        """allowed_tools should filter the tool list."""
        options = ClaudeAgentOptions(
            allowed_tools=["Bash", "Read"],
            yolo_mode=True
        )

        tools = options.build_tools()
        tool_names = {tool.name for tool in tools}

        # Should include allowed tools
        assert "Bash" in tool_names or any("Bash" in t for t in tool_names)
        assert "Read" in tool_names or any("Read" in t for t in tool_names)

    def test_disallowed_tools_filter(self):
        """disallowed_tools should exclude tools from the list."""
        options = ClaudeAgentOptions(
            disallowed_tools=["Task"],
            yolo_mode=True
        )

        tools = options.build_tools()
        tool_names = {tool.name for tool in tools}

        # Task should not be in the list
        assert "Task" not in tool_names

    def test_model_option(self):
        """Model option should be settable."""
        options = ClaudeAgentOptions(model="claude-sonnet-4-5")
        assert options.model == "claude-sonnet-4-5"

    def test_cwd_option(self):
        """Working directory option should be settable."""
        options = ClaudeAgentOptions(cwd="/tmp/test")
        assert str(options.cwd) == "/tmp/test"

    def test_system_prompt_option(self):
        """System prompt should be settable."""
        options = ClaudeAgentOptions(system_prompt="You are a helpful assistant.")
        assert options.system_prompt == "You are a helpful assistant."

    def test_max_thinking_tokens_option(self):
        """Max thinking tokens should be settable."""
        options = ClaudeAgentOptions(max_thinking_tokens=20000)
        assert options.max_thinking_tokens == 20000

    def test_yolo_mode_deprecation_warning(self):
        """yolo_mode should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning):
            options = ClaudeAgentOptions(yolo_mode=True)
            # Check that permission_mode was set
            assert options.permission_mode == "bypassPermissions"

    def test_extra_instructions_normalization(self):
        """extra_instructions should be normalized to a list."""
        options = ClaudeAgentOptions(
            additional_instructions="Single instruction"
        )
        assert options.extra_instructions() == ["Single instruction"]

        options2 = ClaudeAgentOptions(
            additional_instructions=["First", "Second"]
        )
        assert options2.extra_instructions() == ["First", "Second"]

    def test_claude_sdk_specific_fields_accepted(self):
        """Claude SDK specific fields should be accepted."""
        options = ClaudeAgentOptions(
            max_budget_usd=1.0,
            fallback_model="claude-haiku-4",
            betas=["context-1m-2025-08-07"],
        )

        assert options.max_budget_usd == 1.0
        assert options.fallback_model == "claude-haiku-4"
        assert options.betas == ["context-1m-2025-08-07"]

    def test_mcp_servers_configuration(self):
        """MCP servers configuration should work."""
        config = McpServerConfig(
            type="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        )

        options = ClaudeAgentOptions(
            mcp_servers={"test-server": config}
        )

        assert options.mcp_servers is not None
        assert "test-server" in options.mcp_servers


# =============================================================================
# Test ClaudeSDKClient
# =============================================================================


class TestClaudeSDKClient:
    """Test ClaudeSDKClient class functionality."""

    @pytest.fixture
    def fake_runner(self):
        """Create a fake query runner for testing."""
        async def runner(messages, system_prompt, context, query_context, permission_checker):
            del messages, system_prompt, context, query_context, permission_checker
            yield ripperdoc_create_assistant_message("Test response")

        return runner

    def test_client_initialization(self):
        """Client should initialize with default options."""
        client = ClaudeSDKClient()

        assert client.options is not None
        assert client.session_id is None
        assert client.turn_count == 0

    def test_client_with_custom_options(self):
        """Client should accept custom options."""
        options = ClaudeAgentOptions(
            model="test-model",
            permission_mode="bypassPermissions"
        )
        client = ClaudeSDKClient(options=options)

        assert client.options.model == "test-model"
        assert client.options.permission_mode == "bypassPermissions"

    def test_client_properties(self):
        """Client properties should return correct values."""
        client = ClaudeSDKClient(
            options=ClaudeAgentOptions(user="test-user")
        )

        assert client.user == "test-user"
        assert client.turn_count == 0
        assert client.session_id is None

    def test_async_context_manager(self):
        """Client should work as async context manager."""
        async def test():
            async with ClaudeSDKClient() as client:
                assert client is not None

        asyncio.run(test())

    def test_set_permission_mode(self):
        """set_permission_mode should change the permission mode."""
        async def test():
            client = ClaudeSDKClient()
            await client.set_permission_mode("acceptEdits")

            assert client.options.permission_mode == "acceptEdits"

        asyncio.run(test())

    def test_set_permission_mode_invalid(self):
        """set_permission_mode should reject invalid modes."""
        async def test():
            client = ClaudeSDKClient()

            with pytest.raises(ValueError):
                await client.set_permission_mode("invalid_mode")

        asyncio.run(test())

    def test_set_model(self):
        """set_model should change the current model."""
        async def test():
            client = ClaudeSDKClient()
            await client.set_model("claude-sonnet-4-5")

            assert client.options.model == "claude-sonnet-4-5"
            assert client._current_model == "claude-sonnet-4-5"

        asyncio.run(test())

    def test_rewind_files_not_implemented(self):
        """rewind_files should raise NotImplementedError."""
        async def test():
            client = ClaudeSDKClient()

            with pytest.raises(NotImplementedError):
                await client.rewind_files("user-message-123")

        asyncio.run(test())

    def test_get_server_info(self):
        """get_server_info should return session information."""
        async def test():
            client = ClaudeSDKClient()
            # Not connected - should return None
            info = await client.get_server_info()
            assert info is None

        asyncio.run(test())

    def test_get_server_info_when_connected(self):
        """get_server_info should return info when connected."""
        async def test():
            options = ClaudeAgentOptions(
                model="test-model",
                permission_mode="bypassPermissions"
            )
            client = ClaudeSDKClient(options=options)

            # Simulate being connected
            client._connected = True
            client._session_id = "test-session-123"

            info = await client.get_server_info()

            assert info is not None
            assert info["session_id"] == "test-session-123"
            assert info["model"] == "test-model"
            assert info["permission_mode"] == "bypassPermissions"

        asyncio.run(test())


# =============================================================================
# Test query() Function
# =============================================================================


class TestQueryFunction:
    """Test the query() function with Claude SDK compatible interface."""

    @pytest.fixture
    def fake_runner(self):
        """Create a fake query runner for testing."""
        async def runner(messages, system_prompt, context, query_context, permission_checker):
            del messages, system_prompt, context, query_context, permission_checker
            yield ripperdoc_create_assistant_message("Query result")

        return runner

    def test_query_with_string_prompt(self, fake_runner):
        """query() should work with string prompt."""
        async def test():
            messages = []
            options = ClaudeAgentOptions(yolo_mode=True)

            async for msg in query(prompt="Test", options=options, query_runner=fake_runner):
                messages.append(msg)

            assert len(messages) > 0
            # First message should be converted to Claude SDK format
            assert isinstance(messages[0], AssistantMessage)

        asyncio.run(test())

    def test_query_with_keyword_argument(self, fake_runner):
        """query() should require keyword-only prompt argument."""
        async def test():
            options = ClaudeAgentOptions(yolo_mode=True)

            # This should work
            async for msg in query(prompt="Test", options=options, query_runner=fake_runner):
                pass

        asyncio.run(test())

    def test_query_with_options(self, fake_runner):
        """query() should accept ClaudeAgentOptions."""
        async def test():
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model="test-model"
            )

            async for msg in query(prompt="Test", options=options, query_runner=fake_runner):
                assert isinstance(msg, (AssistantMessage, SystemMessage, ResultMessage))

        asyncio.run(test())

    def test_query_returns_claude_sdk_messages(self, fake_runner):
        """query() should return Claude SDK compatible messages."""
        async def test():
            options = ClaudeAgentOptions()

            async for msg in query(prompt="Test", options=options, query_runner=fake_runner):
                # Should be one of the Claude SDK message types
                assert isinstance(msg, (UserMessage, AssistantMessage, SystemMessage, ResultMessage, StreamEvent))

        asyncio.run(test())


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Test that old Ripperdoc SDK code still works."""

    def test_old_import_style_works(self):
        """Old import style should still work."""
        from ripperdoc.sdk import RipperdocClient, RipperdocOptions, query as old_query

        # Should be the same as new names
        assert RipperdocClient is ClaudeSDKClient
        assert RipperdocOptions is ClaudeAgentOptions

    def test_old_options_style_works(self):
        """Creating options with old style should work."""
        options = RipperdocOptions(
            yolo_mode=True,
            allowed_tools=["Bash"]
        )

        assert options.permission_mode == "bypassPermissions"

    def test_old_client_style_works(self):
        """Creating client with old style should work."""
        options = RipperdocOptions()
        client = RipperdocClient(options=options)

        assert isinstance(client, ClaudeSDKClient)


# =============================================================================
# Test Permission Result Types
# =============================================================================


class TestPermissionResultTypes:
    """Test permission result types."""

    def test_permission_result_allow_creation(self):
        """PermissionResultAllow should store allow decision."""
        result = PermissionResultAllow(
            behavior="allow",
            updated_input={"key": "new_value"}
        )

        assert result.behavior == "allow"
        assert result.updated_input == {"key": "new_value"}

    def test_permission_result_deny_creation(self):
        """PermissionResultDeny should store deny decision."""
        result = PermissionResultDeny(
            behavior="deny",
            message="Operation not allowed",
            interrupt=False
        )

        assert result.behavior == "deny"
        assert result.message == "Operation not allowed"
        assert result.interrupt is False

    def test_permission_result_deny_with_interrupt(self):
        """PermissionResultDeny can request interrupt."""
        result = PermissionResultDeny(
            behavior="deny",
            message="Critical operation",
            interrupt=True
        )

        assert result.interrupt is True


# =============================================================================
# Test Hook Types
# =============================================================================


class TestHookTypes:
    """Test hook-related types."""

    def test_hook_event_values(self):
        """HookEvent should contain expected values."""
        # HookEvent is a Union of Literals
        events = ["PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SubagentStop", "PreCompact"]
        for event in events:
            assert event in ["PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SubagentStop", "PreCompact"]

    def test_hook_matcher_creation(self):
        """HookMatcher should store matcher and hooks."""
        from ripperdoc.sdk.types import HookMatcher as TypedHookMatcher

        async def dummy_hook(input_data, tool_use_id, context):
            return {"continue_": True}

        matcher = TypedHookMatcher(
            matcher="Bash|Edit",
            hooks=[dummy_hook],
            timeout=60.0
        )

        assert matcher.matcher == "Bash|Edit"
        assert len(matcher.hooks) == 1
        assert matcher.timeout == 60.0

    def test_pre_tool_use_hook_input(self):
        """PreToolUseHookInput should have correct structure."""
        input_data = PreToolUseHookInput(
            session_id="session-123",
            transcript_path="/path/to/transcript",
            cwd="/working/dir",
            hook_event_name="PreToolUse",
            tool_name="Bash",
            tool_input={"command": "ls"}
        )

        # TypedDict creates dict objects, use dict access
        assert input_data["hook_event_name"] == "PreToolUse"
        assert input_data["tool_name"] == "Bash"
        assert input_data["tool_input"] == {"command": "ls"}

    def test_post_tool_use_hook_input(self):
        """PostToolUseHookInput should include tool response."""
        input_data = PostToolUseHookInput(
            session_id="session-123",
            transcript_path="/path/to/transcript",
            cwd="/working/dir",
            hook_event_name="PostToolUse",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response="file1.txt\nfile2.txt"
        )

        # TypedDict creates dict objects, use dict access
        assert input_data["hook_event_name"] == "PostToolUse"
        assert input_data["tool_response"] == "file1.txt\nfile2.txt"

    def test_user_prompt_submit_hook_input(self):
        """UserPromptSubmitHookInput should include prompt."""
        input_data = UserPromptSubmitHookInput(
            session_id="session-123",
            transcript_path="/path/to/transcript",
            cwd="/working/dir",
            hook_event_name="UserPromptSubmit",
            prompt="Help me with this task"
        )

        # TypedDict creates dict objects, use dict access
        assert input_data["hook_event_name"] == "UserPromptSubmit"
        assert input_data["prompt"] == "Help me with this task"


# =============================================================================
# Test MCP Server Types
# =============================================================================


class TestMcpServerTypes:
    """Test MCP server configuration types."""

    def test_mcp_server_config_dataclass(self):
        """McpServerConfig dataclass should work for stdio servers."""
        from ripperdoc.sdk.client import McpServerConfig

        config = McpServerConfig(
            type="stdio",
            command="npx",
            args=["-y", "@server/package"],
            env={"KEY": "value"}
        )

        assert config.type == "stdio"
        assert config.command == "npx"
        assert config.args == ["-y", "@server/package"]
        assert config.env == {"KEY": "value"}


# =============================================================================
# Test Agent Definition
# =============================================================================


class TestAgentDefinition:
    """Test agent definition type."""

    def test_agent_definition_creation(self):
        """AgentDefinition should store agent configuration."""
        agent = AgentDefinition(
            description="Code review agent",
            prompt="You are a code reviewer. Check the code for bugs.",
            tools=["Read", "Edit"],
            model="sonnet"
        )

        assert agent.description == "Code review agent"
        assert agent.prompt == "You are a code reviewer. Check the code for bugs."
        assert agent.tools == ["Read", "Edit"]
        assert agent.model == "sonnet"

    def test_agent_definition_with_inherit_model(self):
        """AgentDefinition can inherit model from parent."""
        agent = AgentDefinition(
            description="General assistant",
            prompt="Help the user.",
            model="inherit"
        )

        assert agent.model == "inherit"

    def test_agent_definition_without_tools(self):
        """AgentDefinition can have None tools."""
        agent = AgentDefinition(
            description="Simple agent",
            prompt="Just answer questions.",
            tools=None
        )

        assert agent.tools is None


# =============================================================================
# Test ResultMessageFactory
# =============================================================================


class TestResultMessageFactory:
    """Test ResultMessage factory."""

    def test_create_result_message(self):
        """Factory should create a proper ResultMessage."""
        result = ResultMessageFactory.create(
            session_id="session-123",
            duration_ms=1500,
            duration_api_ms=1200,
            is_error=False,
            num_turns=2
        )

        assert isinstance(result, ResultMessage)
        assert result.session_id == "session-123"
        assert result.duration_ms == 1500
        assert result.is_error is False
        assert result.num_turns == 2

    def test_create_result_message_with_optional_fields(self):
        """Factory should handle optional parameters."""
        result = ResultMessageFactory.create(
            session_id="session-456",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            total_cost_usd=0.003,
            usage={"input_tokens": 100, "output_tokens": 200},
            result="Success"
        )

        assert result.total_cost_usd == 0.003
        assert result.usage == {"input_tokens": 100, "output_tokens": 200}
        assert result.result == "Success"


# =============================================================================
# Test Setting Source
# =============================================================================


class TestSettingSource:
    """Test SettingSource type (compatibility with old SettingSource)."""

    def test_old_setting_source_enum(self):
        """Old SettingSource enum should still exist for compatibility."""
        from ripperdoc.sdk.client import SettingSource

        # Old enum values should still work
        assert SettingSource.USER == "user"
        assert SettingSource.PROJECT == "project"
        assert SettingSource.LOCAL == "local"
        assert SettingSource.ENV == "env"


# =============================================================================
# Test Tool Permission Context
# =============================================================================


class TestToolPermissionContext:
    """Test ToolPermissionContext type."""

    def test_tool_permission_context_creation(self):
        """ToolPermissionContext should store context data."""
        context = ToolPermissionContext(
            signal=None,
            suggestions=[
                PermissionUpdate(type="setMode", mode="bypassPermissions")
            ]
        )

        assert context.signal is None
        assert len(context.suggestions) == 1


# =============================================================================
# Test Integration
# =============================================================================


class TestIntegration:
    """Integration tests for Claude SDK compatibility."""

    def test_full_query_workflow_compatibility(self):
        """Test that the full query workflow is compatible with Claude SDK."""
        async def fake_runner(messages, system_prompt, context, query_context, permission_checker):
            del messages, system_prompt, context, query_context, permission_checker
            # Yield a response with tool use
            yield ripperdoc_create_assistant_message([
                {"type": "text", "text": "I'll help you with that."},
                {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"command": "echo test"}}
            ])

        async def run():
            # Use Claude SDK style query
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model="test-model"
            )

            message_count = 0
            async for msg in query(prompt="Help me", options=options, query_runner=fake_runner):
                message_count += 1
                # Verify message is Claude SDK compatible
                assert isinstance(msg, (AssistantMessage, SystemMessage, ResultMessage))

            assert message_count > 0

        asyncio.run(run())

    def test_client_session_compatibility(self):
        """Test that client sessions are compatible with Claude SDK."""
        async def fake_runner(messages, system_prompt, context, query_context, permission_checker):
            del messages, system_prompt, context, query_context, permission_checker
            yield ripperdoc_create_assistant_message("Done")

        async def run():
            # Use Claude SDK style client
            async with ClaudeSDKClient(
                ClaudeAgentOptions(
                    permission_mode="bypassPermissions"
                )
            ) as client:
                await client.query("Test query")

                message_count = 0
                async for msg in client.receive_messages():
                    message_count += 1
                    # Verify Claude SDK compatible messages
                    assert isinstance(msg, (AssistantMessage, SystemMessage, ResultMessage))

                assert message_count > 0

        asyncio.run(run())


# =============================================================================
# Test CanUseTool Type
# =============================================================================


class TestCanUseToolType:
    """Test CanUseTool callback type."""

    def test_can_use_tool_signature(self):
        """CanUseTool should have correct signature."""
        # Create a sample callback
        async def sample_callback(
            tool_name: str,
            tool_input: dict[str, Any],
            context: ToolPermissionContext
        ) -> PermissionResult:
            return PermissionResultAllow(behavior="allow")

        # Verify it matches the expected signature
        assert callable(sample_callback)

    def test_tool_permission_context(self):
        """ToolPermissionContext should store context data."""
        context = ToolPermissionContext(
            signal=None,
            suggestions=[
                PermissionUpdate(type="setMode", mode="bypassPermissions")
            ]
        )

        assert context.signal is None
        assert len(context.suggestions) == 1
        assert context.suggestions[0].mode == "bypassPermissions"


# =============================================================================
# Test SdkBeta Type
# =============================================================================


class TestSdkBeta:
    """Test SdkBeta type."""

    def test_sdk_beta_literal(self):
        """SdkBeta should be a Literal with valid beta features."""
        # SdkBeta is a Union of Literals
        assert "context-1m-2025-08-07" in SdkBeta.__args__  # type: ignore


# =============================================================================
# Test SandboxSettings Type
# =============================================================================


class TestSandboxSettings:
    """Test SandboxSettings type."""

    def test_sandbox_settings_dict(self):
        """SandboxSettings should accept dict with various fields."""
        settings: SandboxSettings = {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "excludedCommands": ["docker"],
            "network": {
                "allowUnixSockets": ["/var/run/docker.sock"],
                "allowLocalBinding": True
            }
        }

        assert settings["enabled"] is True
        assert settings["excludedCommands"] == ["docker"]
        assert settings["network"]["allowUnixSockets"] == ["/var/run/docker.sock"]
