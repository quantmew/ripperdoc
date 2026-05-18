"""Prompt generation functions for the Agent tool."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ripperdoc.core.system_prompt import build_environment_prompt


def build_agent_listing(agents) -> str:
    """Build the agent listing block for the prompt."""
    agent_lines: List[str] = []
    for agent in agents:
        properties = (
            "Properties: access to current context; "
            if getattr(agent, "fork_context", False)
            else ""
        )
        tools_label = "All tools"
        if getattr(agent, "tools", None):
            tools_label = "All tools" if "*" in agent.tools else ", ".join(agent.tools)
        agent_lines.append(
            f"- {agent.agent_type}: {agent.when_to_use} ({properties}Tools: {tools_label})"
        )

    return "\n".join(agent_lines) or "- general-purpose (built-in)"


def build_task_tool_prompt(
    task_tool_name: str,
    file_read_tool_name: str,
    search_tool_name: str,
    code_tool_name: str,
    background_fetch_tool_name: str,
    agent_block: str,
) -> str:
    """Build the main Agent tool prompt string.

    Extracted from AgentTool.prompt() method.
    """
    return (
        f"Launch a new agent to handle complex, multi-step tasks autonomously. \n\n"
        f"The {task_tool_name} tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.\n\n"
        f"Available agent types and the tools they have access to:\n"
        f"{agent_block}\n\n"
        f"When starting a new agent with the {task_tool_name} tool, you must specify a subagent_type parameter to select which agent type to use.\n\n"
        f"When NOT to use the {task_tool_name} tool:\n"
        f"- If you want to read a specific file path, use the {file_read_tool_name} or {search_tool_name} tool instead of the {task_tool_name} tool, to find the match more quickly\n"
        f'- If you are searching for a specific class definition like "class Foo", use the {search_tool_name} tool instead of the {task_tool_name} tool, to find the match more quickly\n'
        f"- If you are searching for code within a specific file or set of 2-3 files, use the {file_read_tool_name} tool instead of the {task_tool_name} tool, to find the match more quickly\n"
        "- Other tasks that are not related to the agent descriptions above\n"
        "\n"
        "\n"
        "Usage notes:\n"
        "- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses\n"
        "- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.\n"
        f"- Use run_in_background=true to launch an agent asynchronously. The tool will return an agent_id immediately for later retrieval.\n"
        f"- Check background progress/output by calling {background_fetch_tool_name} with task_id=<agent_id>.\n"
        f"- To continue a completed agent, call {task_tool_name} with resume=<agent_id> and a new prompt.\n"
        '- Use isolation="worktree" to run the task in a dedicated git worktree under .ripperdoc/worktrees/. If the subagent makes no changes, the worktree is auto-cleaned; if changes are made, worktree path/branch are returned.\n'
        "- Provide clear, detailed prompts so the agent can work autonomously and return exactly the information you need.\n"
        "- Agents can opt into parent context by setting fork_context: true in their frontmatter. When enabled, they receive the full conversation history before the tool call.\n"
        "- The agent's outputs should generally be trusted\n"
        "- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent\n"
        "- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.\n"
        f'- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple {task_tool_name} tool use content blocks. For example, if you need to launch both a code-reviewer agent and a test-runner agent in parallel, send a single message with both tool calls.\n'
        "\n"
        "Example usage:\n"
        "\n"
        "<example_agent_descriptions>\n"
        '"code-reviewer": use this agent after you are done writing a significant piece of code\n'
        '"greeting-responder": use this agent when to respond to user greetings with a friendly joke\n'
        "</example_agent_description>\n"
        "\n"
        "<example>\n"
        'user: "Please write a function that checks if a number is prime"\n'
        "assistant: Sure let me write a function that checks if a number is prime\n"
        f"assistant: First let me use the {code_tool_name} tool to write a function that checks if a number is prime\n"
        f"assistant: I'm going to use the {code_tool_name} tool to write the following code:\n"
        "<code>\n"
        "function isPrime(n) {\n"
        "  if (n <= 1) return false\n"
        "  for (let i = 2; i * i <= n; i++) {\n"
        "    if (n % i === 0) return false\n"
        "  }\n"
        "  return true\n"
        "}\n"
        "</code>\n"
        "<commentary>\n"
        "Since a significant piece of code was written and the task was completed, now use the code-reviewer agent to review the code\n"
        "</commentary>\n"
        "assistant: Now let me use the code-reviewer agent to review the code\n"
        f"assistant: Uses the {task_tool_name} tool to launch the code-reviewer agent \n"
        "</example>\n"
        "\n"
        "<example>\n"
        'user: "Hello"\n'
        "<commentary>\n"
        "Since the user is greeting, use the greeting-responder agent to respond with a friendly joke\n"
        "</commentary>\n"
        f'assistant: "I\'m going to use the {task_tool_name} tool to launch the greeting-responder agent"\n'
        "</example>"
    )


def build_agent_prompt(
    agent_type: str,
    tools: list,
    working_dir: Optional[str],
    environment_prompt: str,
) -> str:
    """Build the subagent prompt string.

    Extracted from AgentTool._build_agent_prompt() method.
    """
    tool_names = ", ".join(tool.name for tool in tools if getattr(tool, "name", None))
    guidance = (
        "You are a specialized Ripperdoc subagent working autonomously. "
        "Execute the task completely using the allowed tools. "
        "Return a single, concise summary for the parent agent that includes what you did, "
        "important findings, and any follow-ups. Do not ask the user questions."
    )
    sections = [
        guidance,
        f"Agent type: {agent_type}",
        f"Allowed tools: {tool_names}",
        "Agent system prompt:",
        environment_prompt or "(no additional prompt)",
        build_environment_prompt(cwd=Path(working_dir) if working_dir else None),
    ]
    return "\n\n".join(sections)
