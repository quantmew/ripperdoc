"""Tests for query tool timeout strategy."""

from ripperdoc.core.query import tools as tools_module


def test_resolve_tool_timeout_disables_for_ask_user_question() -> None:
    assert tools_module._resolve_tool_timeout_sec("AskUserQuestion") is None
    assert (
        tools_module._resolve_tool_timeout_sec("Bash")
        == tools_module.DEFAULT_TOOL_TIMEOUT_SEC
    )


def test_resolve_concurrent_timeout_disables_when_batch_contains_ask_user_question() -> None:
    assert (
        tools_module._resolve_concurrent_timeout_sec(["Bash", "AskUserQuestion"]) is None
    )
    assert (
        tools_module._resolve_concurrent_timeout_sec(["Bash", "Read"])
        == tools_module.DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC
    )
