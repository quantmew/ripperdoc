"""Tests for built-in subagent prompts."""

from ripperdoc.core.agents import (
    BASH_TOOL_NAME,
    DEFAULT_READ_ONLY_SUBAGENT_TOOLS,
    _build_explore_agent_prompt,
    _build_plan_agent_prompt,
    _built_in_agents,
)


def test_built_in_read_only_agents_without_shell_have_no_shell_use_instructions() -> None:
    builtins = {agent.agent_type: agent for agent in _built_in_agents()}
    for agent_type in ("explore", "plan"):
        agent = builtins[agent_type]
        assert BASH_TOOL_NAME not in agent.tools
        assert f"Use {BASH_TOOL_NAME} ONLY for read-only operations" not in agent.system_prompt
        assert f"No {BASH_TOOL_NAME} tool is available in this subagent session." in agent.system_prompt


def test_read_only_prompt_builders_render_shell_rules_when_shell_tool_present() -> None:
    tools_with_shell = list(DEFAULT_READ_ONLY_SUBAGENT_TOOLS) + [BASH_TOOL_NAME]

    explore_prompt = _build_explore_agent_prompt(tools_with_shell)
    assert f"Use {BASH_TOOL_NAME} ONLY for read-only operations" in explore_prompt
    assert f"No {BASH_TOOL_NAME} tool is available in this subagent session." not in explore_prompt

    plan_prompt = _build_plan_agent_prompt(tools_with_shell)
    assert f"Use {BASH_TOOL_NAME} ONLY for read-only operations" in plan_prompt
    assert f"No {BASH_TOOL_NAME} tool is available in this subagent session." not in plan_prompt
