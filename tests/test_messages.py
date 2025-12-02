"""Test message utilities."""

from ripperdoc.utils.messages import (
    MessageRole,
    create_user_message,
    create_assistant_message,
    create_progress_message,
    normalize_messages_for_api,
)


def test_create_user_message():
    """Test creating a user message."""
    msg = create_user_message("Hello, AI!")

    assert msg.type == "user"
    assert msg.message.role == MessageRole.USER
    assert msg.message.content == "Hello, AI!"
    assert msg.uuid != ""


def test_create_assistant_message():
    """Test creating an assistant message."""
    msg = create_assistant_message("Hello, user!", cost_usd=0.01, duration_ms=1000)

    assert msg.type == "assistant"
    assert msg.message.role == MessageRole.ASSISTANT
    assert msg.message.content == "Hello, user!"
    assert msg.cost_usd == 0.01
    assert msg.duration_ms == 1000


def test_create_progress_message():
    """Test creating a progress message."""
    msg = create_progress_message(
        tool_use_id="test_id", sibling_tool_use_ids={"id1", "id2"}, content="Working..."
    )

    assert msg.type == "progress"
    assert msg.tool_use_id == "test_id"
    assert msg.content == "Working..."


def test_normalize_messages_for_api():
    """Test normalizing messages for API."""
    messages = [
        create_user_message("Hello"),
        create_assistant_message("Hi there"),
        create_progress_message("test_id", set(), "Progress"),
        create_user_message("How are you?"),
    ]

    normalized = normalize_messages_for_api(messages)

    # Progress messages should be filtered out
    assert len(normalized) == 3
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "assistant"
    assert normalized[2]["role"] == "user"


def test_message_with_tool_result():
    """Test creating a message with tool result."""
    tool_result = {"type": "tool_result", "tool_use_id": "test_id", "content": "Result content"}

    msg = create_user_message([tool_result])

    assert msg.type == "user"
    assert isinstance(msg.message.content, list)
    assert len(msg.message.content) == 1
