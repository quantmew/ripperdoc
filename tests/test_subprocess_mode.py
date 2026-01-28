"""Tests for subprocess mode functionality.

This test module covers:
- Subprocess mode options
- Transport layer
- Query class
- Protocol handler
- Integration tests
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from dataclasses import dataclass

import pytest

from ripperdoc.sdk import ClaudeAgentOptions, RipperdocSDKClient
from ripperdoc.sdk._internal.transport import Transport
from ripperdoc.sdk._internal.query import Query
from ripperdoc.sdk._internal.message_parser import parse_message
from ripperdoc.sdk._errors import (
    CLIConnectionError,
    CLINotFoundError,
    MessageParseError,
)
from ripperdoc.sdk.types import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ContentBlock,
)


class TestMessageParser:
    """Test message parsing from CLI output."""

    def test_parse_user_message_with_string_content(self) -> None:
        """Parse user message with string content."""
        data = {
            "type": "user",
            "message": {"content": "Hello, world!"},
            "uuid": "test-uuid",
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert message.content == "Hello, world!"
        assert message.uuid == "test-uuid"

    def test_parse_user_message_with_block_content(self) -> None:
        """Parse user message with content blocks."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "ls"}}
                ]
            },
            "uuid": "test-uuid",
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Hello"
        assert isinstance(message.content[1], ToolUseBlock)
        assert message.content[1].name == "Bash"

    def test_parse_assistant_message(self) -> None:
        """Parse assistant message."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Response text"}
                ],
                "model": "claude-3-5-sonnet",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.model == "claude-3-5-sonnet"
        assert isinstance(message.content, list)
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Response text"

    def test_parse_assistant_message_with_thinking(self) -> None:
        """Parse assistant message with thinking block."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me think...",
                        "signature": "sig123"
                    },
                    {"type": "text", "text": "Answer"}
                ],
                "model": "claude-3-5-sonnet",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 2

    def test_parse_progress_message(self) -> None:
        """Parse progress message."""
        data = {
            "type": "progress",
            "tool_use_id": "tool-1",
            "content": "Processing...",
        }
        message = parse_message(data)
        assert isinstance(message, SystemMessage)
        assert message.subtype == "progress"
        assert message.data["tool_use_id"] == "tool-1"
        assert message.data["content"] == "Processing..."

    def test_parse_result_message(self) -> None:
        """Parse result message."""
        data = {
            "type": "result",
            "duration_ms": 1500,
            "duration_api_ms": 1200,
            "is_error": False,
            "num_turns": 3,
            "session_id": "sess-123",
            "total_cost_usd": 0.01,
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.duration_ms == 1500
        assert message.is_error is False
        assert message.num_turns == 3
        assert message.session_id == "sess-123"

    def test_parse_system_message(self) -> None:
        """Parse system message."""
        data = {
            "type": "system",
            "subtype": "error",
            "data": {"message": "An error occurred"},
        }
        message = parse_message(data)
        assert isinstance(message, SystemMessage)
        assert message.subtype == "error"
        # SystemMessage.data stores the entire input data dict
        # The message_parser stores the entire data dict, so we need to access it correctly
        assert "data" in message.data  # The original data has a 'data' key
        assert message.data["data"]["message"] == "An error occurred"

    def test_parse_invalid_message_type(self) -> None:
        """Parse invalid message type raises error."""
        data = {
            "type": "unknown_type",
            "data": {},
        }
        with pytest.raises(MessageParseError, match="Unknown message type"):
            parse_message(data)

    def test_parse_message_missing_type(self) -> None:
        """Parse message without type raises error."""
        data = {"data": {}}
        with pytest.raises(MessageParseError, match="missing 'type'"):
            parse_message(data)

    def test_parse_message_invalid_data_type(self) -> None:
        """Parse non-dict data raises error."""
        with pytest.raises(MessageParseError, match="Invalid message data type"):
            parse_message("not a dict")  # type: ignore


class MockTransport(Transport):
    """Mock transport for testing."""

    def __init__(self, prompt: str = "", options: Any = None) -> None:
        self._connected = False
        self._ready = False
        self._messages_to_send: list[dict[str, Any]] = []
        self._received_messages: list[str] = []
        self._prompt = prompt
        self._options = options
        self._closed = False

    async def connect(self) -> None:
        """Mock connect."""
        self._connected = True
        self._ready = True

    async def write(self, data: str) -> None:
        """Mock write."""
        self._received_messages.append(data)

    def read_messages(self):
        """Mock read messages - returns an async iterable that immediately exits."""
        # Return an async generator that immediately exits (empty stream)
        async def _reader():
            return
            yield  # This makes it an async generator but never yields anything

        return _reader()

    async def close(self) -> None:
        """Mock close."""
        self._closed = True
        self._connected = False
        self._ready = False

    def is_ready(self) -> bool:
        """Check if ready."""
        return self._ready

    async def end_input(self) -> None:
        """Mock end input."""
        pass


class TestQueryClass:
    """Test Query class for control protocol."""

    @pytest.fixture
    def mock_transport(self) -> MockTransport:
        """Create a mock transport."""
        return MockTransport()

    @pytest.mark.asyncio
    async def test_query_initialization(self, mock_transport: MockTransport) -> None:
        """Test Query initialization."""
        query = Query(
            transport=mock_transport,
            is_streaming_mode=True,
        )
        assert query.transport == mock_transport
        assert query.is_streaming_mode is True
        assert query._initialized is False

    @pytest.mark.asyncio
    async def test_query_send_message(self, mock_transport: MockTransport) -> None:
        """Test sending a message through Query."""
        query = Query(
            transport=mock_transport,
            is_streaming_mode=True,
        )
        await query.start()

        # Send a test message
        await query.send_message("test_type", {"key": "value"})

        # Close the query immediately (no need to receive messages in this test)
        await query.close()

    @pytest.mark.asyncio
    async def test_query_close(self, mock_transport: MockTransport) -> None:
        """Test Query cleanup."""
        query = Query(
            transport=mock_transport,
            is_streaming_mode=True,
        )
        await query.start()
        await query.close()

        assert query._closed is True


class TestSubprocessClient:
    """Test subprocess mode client functionality."""

    def test_client_initialization(self) -> None:
        """Test client creates subprocess components."""
        options = ClaudeAgentOptions()
        client = RipperdocSDKClient(options=options)

        # Subprocess components should be initialized
        assert hasattr(client, "_transport_options")
        assert hasattr(client, "_transport")
        assert hasattr(client, "_query")

    def test_build_transport_options(self) -> None:
        """Test building transport options from ClaudeAgentOptions."""
        options = ClaudeAgentOptions(
            model="claude-3-5-sonnet",
            permission_mode="bypassPermissions",
            allowed_tools=["Bash", "Read"],
            cwd="/test/path",
        )
        client = RipperdocSDKClient(options=options)

        transport_options = client._build_transport_options()

        assert transport_options["model"] == "claude-3-5-sonnet"
        assert transport_options["permission_mode"] == "bypassPermissions"
        assert transport_options["allowed_tools"] == ["Bash", "Read"]
        assert transport_options["cwd"] == "/test/path"

    @pytest.mark.asyncio
    async def test_subprocess_connect_mocked(self) -> None:
        """Test subprocess connection with mocked transport."""
        options = ClaudeAgentOptions()

        with patch(
            "ripperdoc.sdk.client._subprocess_transport",
            MockTransport
        ):
            client = RipperdocSDKClient(options=options)

            # Mock the Query class
            mock_query = MagicMock()
            mock_query.start = AsyncMock()
            mock_query._send_control_request = AsyncMock(return_value={
                "session_id": "test-session-123"
            })

            with patch("ripperdoc.sdk.client._query_class", return_value=mock_query):
                await client._connect_subprocess()

                assert client._session_id == "test-session-123"


class TestControlProtocol:
    """Test JSON Control Protocol messages."""

    def test_initialize_request_format(self) -> None:
        """Test initialize request format."""
        request = {
            "type": "control_request",
            "request_id": "req_1",
            "request": {
                "subtype": "initialize",
                "options": {
                    "model": "claude-3-5-sonnet",
                    "permission_mode": "default",
                }
            }
        }
        # Validate structure
        assert request["type"] == "control_request"
        assert request["request_id"] == "req_1"
        assert request["request"]["subtype"] == "initialize"
        assert "options" in request["request"]

    def test_query_request_format(self) -> None:
        """Test query request format."""
        request = {
            "type": "control_request",
            "request_id": "req_2",
            "request": {
                "subtype": "query",
                "prompt": "Hello, world!",
            }
        }
        # Validate structure
        assert request["type"] == "control_request"
        assert request["request"]["subtype"] == "query"
        assert request["request"]["prompt"] == "Hello, world!"

    def test_control_response_success_format(self) -> None:
        """Test control response success format."""
        response = {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": "req_1",
                "response": {
                    "session_id": "sess-123",
                    "tools": [{"name": "Bash"}],
                }
            }
        }
        # Validate structure
        assert response["type"] == "control_response"
        assert response["response"]["subtype"] == "success"
        assert "session_id" in response["response"]["response"]

    def test_control_response_error_format(self) -> None:
        """Test control response error format."""
        response = {
            "type": "control_response",
            "response": {
                "subtype": "error",
                "request_id": "req_1",
                "error": "Something went wrong",
            }
        }
        # Validate structure
        assert response["type"] == "control_response"
        assert response["response"]["subtype"] == "error"
        assert response["response"]["error"] == "Something went wrong"


class TestStdioCommand:
    """Test stdio CLI command."""

    def test_stdio_command_exists(self) -> None:
        """Test that stdio command can be imported."""
        from ripperdoc.cli.commands.stdio_cmd import stdio_cmd, StdioProtocolHandler
        assert stdio_cmd is not None
        assert StdioProtocolHandler is not None

    def test_protocol_handler_initialization(self) -> None:
        """Test StdioProtocolHandler initialization."""
        from ripperdoc.cli.commands.stdio_cmd import StdioProtocolHandler

        handler = StdioProtocolHandler(
            input_format="stream-json",
            output_format="stream-json",
        )
        assert handler._input_format == "stream-json"
        assert handler._output_format == "stream-json"
        assert handler._initialized is False


class TestIntegration:
    """Integration tests for subprocess mode."""

    @pytest.mark.asyncio
    async def test_full_subprocess_flow_mocked(self) -> None:
        """Test full subprocess flow with mocked components."""
        options = ClaudeAgentOptions(
            model="test-model",
            permission_mode="default",
        )

        # Mock the transport
        mock_transport = MockTransport()

        # Mock the query
        mock_query = MagicMock()
        mock_query.start = AsyncMock()
        mock_query._send_control_request = AsyncMock(return_value={
            "session_id": "test-session",
        })
        mock_query.close = AsyncMock()

        with patch("ripperdoc.sdk.client._subprocess_transport", return_value=mock_transport):
            with patch("ripperdoc.sdk.client._query_class", return_value=mock_query):
                client = RipperdocSDKClient(options=options)

                # Simulate connection
                await client._connect_subprocess()

                assert client._session_id == "test-session"
                assert client._query is not None

                # Simulate disconnect
                await client.disconnect()

                assert client._query is None


class TestBackwardCompatibility:
    """Test backward compatibility with SDK API.

    Note: In-process mode has been removed. The SDK now only supports
    subprocess mode. These tests verify that the API remains compatible.
    """

    def test_subprocess_mode_is_default(self) -> None:
        """Test that subprocess mode is the default and only mode."""
        options = ClaudeAgentOptions()
        # Client should have subprocess components initialized
        client = RipperdocSDKClient(options=options)
        assert hasattr(client, "_transport_options")
        assert client._transport is None  # Not connected yet
        assert client._query is None  # Not connected yet

    def test_options_work_with_all_parameters(self) -> None:
        """Test that options work with all parameters."""
        options = ClaudeAgentOptions(
            model="test-model",
            permission_mode="bypassPermissions",
            allowed_tools=["Bash"],
            max_turns=10,
        )

        assert options.model == "test-model"
        assert options.permission_mode == "bypassPermissions"
        assert options.allowed_tools == ["Bash"]
        assert options.max_turns == 10


class TestErrorHandling:
    """Test error handling in subprocess mode."""

    def test_message_parse_error_with_context(self) -> None:
        """Test MessageParseError includes context data."""
        data = {"type": "unknown"}
        with pytest.raises(MessageParseError) as exc_info:
            parse_message(data)

        # The error should include the data
        assert exc_info.value.data == data

    @pytest.mark.asyncio
    async def test_subprocess_connect_failure(self) -> None:
        """Test handling of subprocess connection failure."""
        options = ClaudeAgentOptions()

        # Mock transport that raises on connect
        class FailingTransport(Transport):
            def __init__(self, prompt: str = "", options: Any = None) -> None:
                self._connected = False
                self._ready = False

            async def connect(self) -> None:
                raise CLIConnectionError("Connection failed")

            async def write(self, data: str) -> None:
                pass

            def read_messages(self):
                return asyncio.Queue()

            async def close(self) -> None:
                pass

            def is_ready(self) -> bool:
                return False

            async def end_input(self) -> None:
                pass

        with patch(
            "ripperdoc.sdk.client._subprocess_transport",
            FailingTransport
        ):
            client = RipperdocSDKClient(options=options)

            with pytest.raises(CLIConnectionError):
                await client._connect_subprocess()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
