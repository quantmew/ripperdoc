"""Tests for conversation compaction functionality."""

from ripperdoc.utils.conversation_compaction import (
    extract_tool_ids_from_message,
    get_complete_tool_pairs_tail,
)
from ripperdoc.utils.messages import create_user_message, create_assistant_message


def test_extract_tool_ids_from_assistant_message():
    """Test extracting tool_use IDs from assistant message."""
    msg = create_assistant_message(
        [
            {"type": "tool_use", "id": "call_1", "name": "foo", "input": {}},
            {"type": "tool_use", "id": "call_2", "name": "bar", "input": {}},
            {"type": "text", "text": "some text"},
        ]
    )

    use_ids, result_ids = extract_tool_ids_from_message(msg)
    assert use_ids == {"call_1", "call_2"}
    assert result_ids == set()


def test_extract_tool_ids_from_user_message():
    """Test extracting tool_result IDs from user message."""
    msg = create_user_message(
        [
            {"type": "tool_result", "tool_use_id": "call_1", "text": "result 1"},
            {"type": "tool_result", "tool_use_id": "call_2", "text": "result 2"},
        ]
    )

    use_ids, result_ids = extract_tool_ids_from_message(msg)
    assert use_ids == set()
    assert result_ids == {"call_1", "call_2"}


def test_extract_tool_ids_from_string_message():
    """Test extracting IDs from simple string message returns empty sets."""
    msg = create_user_message("Hello world")

    use_ids, result_ids = extract_tool_ids_from_message(msg)
    assert use_ids == set()
    assert result_ids == set()


def test_get_complete_tool_pairs_tail_no_tools():
    """Test tail extraction when there are no tool messages."""
    messages = [
        create_user_message("Hello"),
        create_assistant_message("Hi there"),
        create_user_message("How are you?"),
        create_assistant_message("I'm fine"),
    ]

    tail = get_complete_tool_pairs_tail(messages, 2)
    assert len(tail) == 2
    # Should be the last 2 messages
    assert tail[0].message.content == "How are you?"
    assert tail[1].message.content == "I'm fine"


def test_get_complete_tool_pairs_tail_with_paired_tools():
    """Test tail extraction when tool_use and tool_result are both in tail."""
    messages = [
        create_user_message("Hello"),
        create_assistant_message(
            [{"type": "tool_use", "id": "call_1", "name": "foo", "input": {}}]
        ),
        create_user_message([{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}]),
    ]

    tail = get_complete_tool_pairs_tail(messages, 2)
    # Both tool_use and tool_result are in last 2 messages, no expansion needed
    assert len(tail) == 2


def test_get_complete_tool_pairs_tail_expands_for_orphan_result():
    """Test that tail expands backwards to include tool_use for orphan tool_result."""
    messages = [
        create_user_message("Hello"),
        create_assistant_message(
            [{"type": "tool_use", "id": "call_1", "name": "foo", "input": {}}]
        ),
        create_user_message([{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}]),
        create_assistant_message("Done"),
    ]

    # Requesting last 2 messages would give [tool_result, "Done"]
    # But tool_result needs its tool_use, so it should expand to include it
    tail = get_complete_tool_pairs_tail(messages, 2)

    # Should include the assistant message with tool_use
    assert len(tail) >= 3

    # Verify tool_use is present
    tool_use_found = False
    for msg in tail:
        use_ids, _ = extract_tool_ids_from_message(msg)
        if "call_1" in use_ids:
            tool_use_found = True
            break
    assert tool_use_found, "tool_use message should be included in expanded tail"


def test_get_complete_tool_pairs_tail_multiple_orphans():
    """Test expansion with multiple orphan tool_results."""
    messages = [
        create_user_message("Start"),
        create_assistant_message(
            [
                {"type": "tool_use", "id": "call_1", "name": "foo", "input": {}},
                {"type": "tool_use", "id": "call_2", "name": "bar", "input": {}},
            ]
        ),
        create_user_message(
            [
                {"type": "tool_result", "tool_use_id": "call_1", "text": "ok1"},
                {"type": "tool_result", "tool_use_id": "call_2", "text": "ok2"},
            ]
        ),
        create_assistant_message("All done"),
        create_user_message("Thanks"),
    ]

    # Last 2 messages are ["All done", "Thanks"], no tools
    tail = get_complete_tool_pairs_tail(messages, 2)
    assert len(tail) == 2

    # Last 3 messages would include tool_results, need to expand
    tail = get_complete_tool_pairs_tail(messages, 3)
    # Should expand to include the tool_use message
    assert len(tail) >= 4

    # Verify both tool_uses are present
    all_use_ids = set()
    for msg in tail:
        use_ids, _ = extract_tool_ids_from_message(msg)
        all_use_ids.update(use_ids)
    assert "call_1" in all_use_ids
    assert "call_2" in all_use_ids


def test_get_complete_tool_pairs_tail_empty():
    """Test with empty message list."""
    tail = get_complete_tool_pairs_tail([], 5)
    assert tail == []


def test_get_complete_tool_pairs_tail_zero_count():
    """Test with zero target count."""
    messages = [create_user_message("Hello")]
    tail = get_complete_tool_pairs_tail(messages, 0)
    assert tail == []
