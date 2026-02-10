"""Regression test for replaying tool details on resume."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from ripperdoc.cli.ui.rich_ui.session import RichUI
from ripperdoc.utils.message_formatting import stringify_message_content
from ripperdoc.utils.messages import create_assistant_message, create_user_message
from ripperdoc.core.message_utils import tool_result_message


class DummyConsole:
    def __init__(self) -> None:
        self.calls: List[tuple[Any, dict[str, Any]]] = []

    def print(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls.append((_args, _kwargs))
        return None


@pytest.mark.asyncio
async def test_resume_replay_uses_tool_rendering(monkeypatch):
    calls: List[Dict[str, Any]] = []

    def fake_handle_assistant_message(ui, message, tool_registry, spinner=None):  # noqa: ARG001
        calls.append({"fn": "assistant", "tool_registry": dict(tool_registry)})
        # Pretend a tool was seen so last_tool_name can be forwarded.
        tool_registry["t1"] = {"name": "Read", "args": {"file_path": "README.md"}}
        return "Read"

    def fake_handle_tool_result_message(ui, message, tool_registry, last_tool_name, spinner=None):  # noqa: ARG001
        calls.append({
            "fn": "tool_result",
            "tool_registry": dict(tool_registry),
            "last_tool_name": last_tool_name,
        })

    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.handle_assistant_message",
        fake_handle_assistant_message,
    )
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.handle_tool_result_message",
        fake_handle_tool_result_message,
    )

    ui = RichUI.__new__(RichUI)
    ui.console = DummyConsole()
    ui._stringify_message_content = lambda content: stringify_message_content(content)
    prompt_history: List[str] = []
    replayed_users: List[str] = []

    ui._append_prompt_history = lambda text: prompt_history.append(text)
    ui._print_replay_user = lambda text: replayed_users.append(text)

    assistant_msg = create_assistant_message(
        [
            {
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "README.md"},
                "id": "t1",
            }
        ]
    )
    tool_msg = tool_result_message("t1", "done")
    user_msg = create_user_message("hello")

    RichUI.replay_conversation(ui, [assistant_msg, tool_msg, user_msg])

    assert [call["fn"] for call in calls] == ["assistant", "tool_result"]
    assert calls[-1]["last_tool_name"] == "Read"
    assert prompt_history == ["hello"]
    assert replayed_users == ["hello"]


def test_replay_conversation_can_limit_message_count():
    ui = RichUI.__new__(RichUI)
    ui.console = DummyConsole()
    ui._stringify_message_content = lambda content: stringify_message_content(content)
    replayed_users: List[str] = []
    prompt_history: List[str] = []
    ui._print_replay_user = lambda text: replayed_users.append(text)
    ui._append_prompt_history = lambda text: prompt_history.append(text)

    messages = [
        create_user_message("u1"),
        create_user_message("u2"),
        create_user_message("u3"),
    ]

    RichUI.replay_conversation(ui, messages, max_messages=2)

    assert replayed_users == ["u2", "u3"]
    assert prompt_history == ["u2", "u3"]
    assert any("2/3 messages" in str(call[0][0]) for call in ui.console.calls if call[0])


def test_replay_conversation_can_skip_history():
    ui = RichUI.__new__(RichUI)
    ui.console = DummyConsole()
    ui._stringify_message_content = lambda content: stringify_message_content(content)
    replayed_users: List[str] = []
    prompt_history: List[str] = []
    ui._print_replay_user = lambda text: replayed_users.append(text)
    ui._append_prompt_history = lambda text: prompt_history.append(text)

    messages = [create_user_message("u1"), create_user_message("u2")]

    RichUI.replay_conversation(ui, messages, max_messages=0)

    assert replayed_users == []
    assert prompt_history == []
    assert any("history replay skipped" in str(call[0][0]) for call in ui.console.calls if call[0])
