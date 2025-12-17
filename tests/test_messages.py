"""Test message utilities."""

from ripperdoc.utils.messages import (
    MessageRole,
    create_user_message,
    create_assistant_message,
    create_progress_message,
    normalize_messages_for_api,
)
from ripperdoc.core.providers.base import sanitize_tool_history


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


def test_sanitize_tool_history_keeps_multiple_tool_results():
    """Ensure tool_use blocks are retained when results arrive in later messages."""
    normalized = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "foo", "input": {}},
                {"type": "tool_use", "id": "call_2", "name": "bar", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_2", "text": "ok"}],
        },
    ]

    sanitized = sanitize_tool_history(normalized)

    tool_use_ids = [
        block["id"] for block in sanitized[0]["content"] if block.get("type") == "tool_use"
    ]
    assert set(tool_use_ids) == {"call_1", "call_2"}


def test_sanitize_tool_history_drops_unpaired_tool_use():
    """Unpaired tool_use blocks should still be removed."""
    normalized = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "foo", "input": {}},
                {"type": "tool_use", "id": "call_2", "name": "bar", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}],
        },
    ]

    sanitized = sanitize_tool_history(normalized)

    tool_use_ids = [
        block["id"] for block in sanitized[0]["content"] if block.get("type") == "tool_use"
    ]
    assert tool_use_ids == ["call_1"]


def test_normalize_messages_with_reasoning_metadata():
    """Ensure reasoning metadata is preserved for OpenAI-style messages."""
    assistant = create_assistant_message(
        [
            {
                "type": "tool_use",
                "tool_use_id": "call_reason",
                "name": "demo",
                "input": {},
            }
        ],
        metadata={"reasoning_content": "thinking..."},
    )
    user = create_user_message(
        [{"type": "tool_result", "tool_use_id": "call_reason", "text": "done"}]
    )

    normalized = normalize_messages_for_api([assistant, user], protocol="openai")
    asst_messages = [msg for msg in normalized if msg.get("role") == "assistant"]
    assert asst_messages, "assistant messages should be present"
    assert asst_messages[-1].get("reasoning_content") == "thinking..."


def test_normalize_messages_preserves_thinking_block():
    """Thinking/redacted thinking blocks should be passed through for Anthropic."""
    assistant = create_assistant_message(
        [
            {"type": "thinking", "thinking": "step 1", "signature": "sig"},
            {"type": "redacted_thinking", "data": "encrypted"},
            {"type": "text", "text": "answer"},
        ]
    )

    normalized = normalize_messages_for_api([assistant], protocol="anthropic")
    assert normalized and normalized[0].get("content")
    kinds = [blk.get("type") for blk in normalized[0]["content"]]
    assert kinds[:2] == ["thinking", "redacted_thinking"]


def test_normalize_messages_openai_skips_orphan_tool_results():
    """OpenAI protocol should skip tool_results without a preceding tool_use.

    This is critical for /compact functionality - after compaction, recent_tail
    may contain tool_result messages whose corresponding tool_use was in the
    compacted portion. OpenAI rejects these orphan tool_results with:
    "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"
    """
    # Simulate a post-compaction scenario: only a tool_result without the tool_use
    orphan_result = create_user_message(
        [{"type": "tool_result", "tool_use_id": "orphan_call", "text": "some result"}]
    )
    normalized = normalize_messages_for_api([orphan_result], protocol="openai")

    # The orphan tool_result should be filtered out, leaving no messages
    tool_messages = [msg for msg in normalized if msg.get("role") == "tool"]
    assert len(tool_messages) == 0, "Orphan tool_results should be skipped for OpenAI"


def test_normalize_messages_openai_keeps_paired_tool_results():
    """OpenAI protocol should keep tool_results that have a preceding tool_use."""
    assistant = create_assistant_message(
        [{"type": "tool_use", "id": "valid_call", "name": "demo", "input": {}}]
    )
    user = create_user_message(
        [{"type": "tool_result", "tool_use_id": "valid_call", "text": "result"}]
    )

    normalized = normalize_messages_for_api([assistant, user], protocol="openai")

    tool_messages = [msg for msg in normalized if msg.get("role") == "tool"]
    assert len(tool_messages) == 1, "Paired tool_results should be kept"
    assert tool_messages[0]["tool_call_id"] == "valid_call"
