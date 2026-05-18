"""Built-in agent prompt builders and definitions for Ripperdoc subagents."""

from __future__ import annotations

from typing import Iterable, List

from ripperdoc.core.agents import (
    AgentDefinition,
    AgentLocation,
    BASH_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
    READ_TOOL_NAME,
)


GENERAL_AGENT_PROMPT = (
    "You are a general-purpose subagent for Ripperdoc. Work autonomously on the task "
    "provided by the parent agent. Use the allowed tools to research, edit files, and "
    "run commands as needed. When you finish, provide a concise report describing what "
    "you changed, what you investigated, and any follow-ups the parent agent should "
    "share with the user."
)

CODE_REVIEW_AGENT_PROMPT = (
    "You are a code review subagent. Inspect the code and summarize risks, bugs, "
    "missing tests, security concerns, and regressions. Do not make code changes. "
    "Provide clear, actionable feedback that the parent agent can relay to the user."
)


def _tool_available(agent_tools: Iterable[str], tool_name: str) -> bool:
    names = {name for name in agent_tools if isinstance(name, str)}
    return "*" in names or tool_name in names


def _format_tool_list(tool_names: List[str]) -> str:
    if not tool_names:
        return ""
    if len(tool_names) == 1:
        return tool_names[0]
    if len(tool_names) == 2:
        return f"{tool_names[0]} and {tool_names[1]}"
    return ", ".join(tool_names[:-1]) + f", and {tool_names[-1]}"


def _read_only_shell_guidelines(agent_tools: Iterable[str], prefix: str = "") -> List[str]:
    if _tool_available(agent_tools, BASH_TOOL_NAME):
        return [
            f"{prefix}- Use {BASH_TOOL_NAME} ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)",
            f"{prefix}- NEVER use {BASH_TOOL_NAME} for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification",
        ]
    return [
        f"{prefix}- No {BASH_TOOL_NAME} tool is available in this subagent session. Do not claim to run shell, git, or gh commands."
    ]


def _build_explore_agent_prompt(agent_tools: Iterable[str]) -> str:
    strengths: List[str] = []
    guidelines: List[str] = []

    if _tool_available(agent_tools, GLOB_TOOL_NAME):
        strengths.append("Rapidly finding files using glob patterns")
        guidelines.append(f"- Use {GLOB_TOOL_NAME} for broad file pattern matching")
    if _tool_available(agent_tools, GREP_TOOL_NAME):
        strengths.append("Searching code and text with powerful regex patterns")
        guidelines.append(f"- Use {GREP_TOOL_NAME} for searching file contents with regex")
    if _tool_available(agent_tools, READ_TOOL_NAME):
        strengths.append("Reading and analyzing file contents")
        guidelines.append(
            f"- Use {READ_TOOL_NAME} when you know the specific file path you need to read"
        )

    if not strengths:
        strengths.append("Analyzing existing code and reporting findings clearly")
    if not guidelines:
        guidelines.append("- Use available read-only tools to locate files and inspect code")

    guidelines.extend(_read_only_shell_guidelines(agent_tools))
    guidelines.extend(
        [
            "- Adapt your search approach based on the thoroughness level specified by the caller",
            "- Return file paths as absolute paths in your final response",
            "- For clear communication, avoid using emojis",
            "- Communicate your final report directly as a regular message - do NOT attempt to create files",
        ]
    )

    strength_lines = "\n".join(f"- {line}" for line in strengths)
    guideline_lines = "\n".join(guidelines)

    return (
        "You are a file search specialist. "
        "You excel at thoroughly navigating and exploring codebases.\n\n"
        "=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===\n"
        "This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:\n"
        "- Creating new files (no Write, touch, or file creation of any kind)\n"
        "- Modifying existing files (no Edit operations)\n"
        "- Deleting files (no rm or deletion)\n"
        "- Moving or copying files (no mv or cp)\n"
        "- Creating temporary files anywhere, including /tmp\n"
        "- Using redirect operators (>, >>, |) or heredocs to write to files\n"
        "- Running ANY commands that change system state\n\n"
        "Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access "
        "to file editing tools - attempting to edit files will fail.\n\n"
        "Your strengths:\n"
        f"{strength_lines}\n\n"
        "Guidelines:\n"
        f"{guideline_lines}\n\n"
        "NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:\n"
        "- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations\n"
        "- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files\n\n"
        "Complete the user's search request efficiently and report your findings clearly."
    )


def _build_plan_agent_prompt(agent_tools: Iterable[str]) -> str:
    explore_lines = [
        "   - Read any files provided to you in the initial prompt",
    ]
    explore_tools: List[str] = []
    if _tool_available(agent_tools, GLOB_TOOL_NAME):
        explore_tools.append(GLOB_TOOL_NAME)
    if _tool_available(agent_tools, GREP_TOOL_NAME):
        explore_tools.append(GREP_TOOL_NAME)
    if _tool_available(agent_tools, READ_TOOL_NAME):
        explore_tools.append(READ_TOOL_NAME)

    if explore_tools:
        explore_lines.append(
            f"   - Find existing patterns and conventions using {_format_tool_list(explore_tools)}"
        )
    else:
        explore_lines.append(
            "   - Find existing patterns and conventions using available read-only tools"
        )

    explore_lines.extend(
        [
            "   - Understand the current architecture",
            "   - Identify similar features as reference",
            "   - Trace through relevant code paths",
        ]
    )
    explore_lines.extend(_read_only_shell_guidelines(agent_tools, prefix="   "))
    explore_section = "\n".join(explore_lines)

    return (
        "You are a software architect and planning specialist. Your role is "
        "to explore the codebase and design implementation plans.\n\n"
        "=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===\n"
        "This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:\n"
        "- Creating new files (no Write, touch, or file creation of any kind)\n"
        "- Modifying existing files (no Edit operations)\n"
        "- Deleting files (no rm or deletion)\n"
        "- Moving or copying files (no mv or cp)\n"
        "- Creating temporary files anywhere, including /tmp\n"
        "- Using redirect operators (>, >>, |) or heredocs to write to files\n"
        "- Running ANY commands that change system state\n\n"
        "Your role is EXCLUSIVELY to explore the codebase and design implementation plans. "
        "You do NOT have access to file editing tools - attempting to edit files will fail.\n\n"
        "You will be provided with a set of requirements and optionally a perspective on how "
        "to approach the design process.\n\n"
        "## Your Process\n\n"
        "1. **Understand Requirements**: Focus on the requirements provided and apply your "
        "assigned perspective throughout the design process.\n\n"
        "2. **Explore Thoroughly**:\n"
        f"{explore_section}\n\n"
        "3. **Design Solution**:\n"
        "   - Create implementation approach based on your assigned perspective\n"
        "   - Consider trade-offs and architectural decisions\n"
        "   - Follow existing patterns where appropriate\n\n"
        "4. **Detail the Plan**:\n"
        "   - Provide step-by-step implementation strategy\n"
        "   - Identify dependencies and sequencing\n"
        "   - Anticipate potential challenges\n\n"
        "## Required Output\n\n"
        "End your response with:\n\n"
        "### Critical Files for Implementation\n"
        "List 3-5 files most critical for implementing this plan:\n"
        '- path/to/file1.ts - [Brief reason: e.g., "Core logic to modify"]\n'
        '- path/to/file2.ts - [Brief reason: e.g., "Interfaces to implement"]\n'
        '- path/to/file3.ts - [Brief reason: e.g., "Pattern to follow"]\n\n'
        "REMEMBER: You can ONLY explore and plan. You CANNOT and MUST NOT write, edit, or "
        "modify any files. You do NOT have access to file editing tools."
    )


DEFAULT_READ_ONLY_SUBAGENT_TOOLS = [READ_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME]
EXPLORE_AGENT_PROMPT = _build_explore_agent_prompt(DEFAULT_READ_ONLY_SUBAGENT_TOOLS)
PLAN_AGENT_PROMPT = _build_plan_agent_prompt(DEFAULT_READ_ONLY_SUBAGENT_TOOLS)


def _built_in_agents() -> List[AgentDefinition]:
    read_only_tools = list(DEFAULT_READ_ONLY_SUBAGENT_TOOLS)
    return [
        AgentDefinition(
            agent_type="general-purpose",
            when_to_use=(
                "General-purpose agent for multi-step coding tasks, deep searches, and "
                "investigations that need their own context window."
            ),
            tools=["*"],
            system_prompt=GENERAL_AGENT_PROMPT,
            location=AgentLocation.BUILT_IN,
            color="cyan",
        ),
        AgentDefinition(
            agent_type="code-reviewer",
            when_to_use=(
                "Run after implementing non-trivial code changes to review for correctness, "
                "testing gaps, security issues, and regressions."
            ),
            tools=list(read_only_tools),
            system_prompt=CODE_REVIEW_AGENT_PROMPT,
            location=AgentLocation.BUILT_IN,
            color="yellow",
        ),
        AgentDefinition(
            agent_type="explore",
            when_to_use=(
                "Fast agent specialized for exploring codebases. Use this when you need to quickly find "
                'files by patterns (eg. "src/components/**/*.tsx"), search code for keywords (eg. "API endpoints"), '
                'or answer questions about the codebase (eg. "how do API endpoints work?"). When calling this agent, '
                'specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, '
                'or "very thorough" for comprehensive analysis across multiple locations and naming conventions.'
            ),
            tools=list(read_only_tools),
            system_prompt=_build_explore_agent_prompt(read_only_tools),
            location=AgentLocation.BUILT_IN,
            color="green",
            model="main",
            disallowed_tools=["Task", "ExitPlanMode", "Edit", "Write", "NotebookEdit"],
            omit_agent_md=True,
        ),
        AgentDefinition(
            agent_type="plan",
            when_to_use=(
                "Software architect agent for designing implementation plans. Use this when "
                "you need to plan the implementation strategy for a task. Returns step-by-step "
                "plans, identifies critical files, and considers architectural trade-offs."
            ),
            tools=list(read_only_tools),
            system_prompt=_build_plan_agent_prompt(read_only_tools),
            location=AgentLocation.BUILT_IN,
            color="blue",
            model=None,
            disallowed_tools=["Task", "ExitPlanMode", "Edit", "Write", "NotebookEdit"],
            omit_agent_md=True,
        ),
        AgentDefinition(
            agent_type="verification",
            when_to_use="Verify code correctness after implementation.",
            tools=list(read_only_tools),
            system_prompt="You are a verification agent. Review code for correctness, edge cases, and test coverage.",
            location=AgentLocation.BUILT_IN,
            color="magenta",
        ),
    ]
