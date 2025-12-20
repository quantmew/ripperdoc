"""Enter plan mode tool for complex task planning.

This tool allows the AI to request entering plan mode for complex tasks
that require careful exploration and design before implementation.
"""

from __future__ import annotations

from textwrap import dedent
from typing import AsyncGenerator, Optional

from pydantic import BaseModel

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "EnterPlanMode"
ASK_USER_QUESTION_TOOL = "AskUserQuestion"

ENTER_PLAN_MODE_PROMPT = dedent(
    """\
    Use this tool when you encounter a complex task that requires careful planning and exploration before implementation. This tool transitions you into plan mode where you can thoroughly explore the codebase and design an implementation approach.

    ## When to Use This Tool

    Use EnterPlanMode when ANY of these conditions apply:

    1. **Multiple Valid Approaches**: The task can be solved in several different ways, each with trade-offs
       - Example: "Add caching to the API" - could use Redis, in-memory, file-based, etc.
       - Example: "Improve performance" - many optimization strategies possible

    2. **Significant Architectural Decisions**: The task requires choosing between architectural patterns
       - Example: "Add real-time updates" - WebSockets vs SSE vs polling
       - Example: "Implement state management" - Redux vs Context vs custom solution

    3. **Large-Scale Changes**: The task touches many files or systems
       - Example: "Refactor the authentication system"
       - Example: "Migrate from REST to GraphQL"

    4. **Unclear Requirements**: You need to explore before understanding the full scope
       - Example: "Make the app faster" - need to profile and identify bottlenecks
       - Example: "Fix the bug in checkout" - need to investigate root cause

    5. **User Input Needed**: You'll need to ask clarifying questions before starting
       - If you would use {ask_tool} to clarify the approach, consider EnterPlanMode instead
       - Plan mode lets you explore first, then present options with context

    ## When NOT to Use This Tool

    Do NOT use EnterPlanMode for:
    - Simple, straightforward tasks with obvious implementation
    - Small bug fixes where the solution is clear
    - Adding a single function or small feature
    - Tasks you're already confident how to implement
    - Research-only tasks (use the Task tool with explore agent instead)

    ## What Happens in Plan Mode

    In plan mode, you'll:
    1. Thoroughly explore the codebase using Glob, Grep, and Read tools
    2. Understand existing patterns and architecture
    3. Design an implementation approach
    4. Present your plan to the user for approval
    5. Use {ask_tool} if you need to clarify approaches
    6. Exit plan mode with ExitPlanMode when ready to implement

    ## Examples

    ### GOOD - Use EnterPlanMode:
    User: "Add user authentication to the app"
    - This requires architectural decisions (session vs JWT, where to store tokens, middleware structure)

    User: "Optimize the database queries"
    - Multiple approaches possible, need to profile first, significant impact

    User: "Implement dark mode"
    - Architectural decision on theme system, affects many components

    ### BAD - Don't use EnterPlanMode:
    User: "Fix the typo in the README"
    - Straightforward, no planning needed

    User: "Add a console.log to debug this function"
    - Simple, obvious implementation

    User: "What files handle routing?"
    - Research task, not implementation planning

    ## Important Notes

    - This tool REQUIRES user approval - they must consent to entering plan mode
    - Be thoughtful about when to use it - unnecessary plan mode slows down simple tasks
    - If unsure whether to use it, err on the side of starting implementation
    - You can always ask the user "Would you like me to plan this out first?"
    """
).format(ask_tool=ASK_USER_QUESTION_TOOL)


PLAN_MODE_INSTRUCTIONS = dedent(
    """\
    In plan mode, you should:
    1. Thoroughly explore the codebase to understand existing patterns
    2. Identify similar features and architectural approaches
    3. Consider multiple approaches and their trade-offs
    4. Use AskUserQuestion if you need to clarify the approach
    5. Design a concrete implementation strategy
    6. When ready, use ExitPlanMode to present your plan for approval

    Remember: DO NOT write or edit any files yet. This is a read-only exploration and planning phase."""
)


class EnterPlanModeToolInput(BaseModel):
    """Input for the EnterPlanMode tool.

    This tool takes no input parameters - it simply requests to enter plan mode.
    """

    pass


class EnterPlanModeToolOutput(BaseModel):
    """Output from the EnterPlanMode tool."""

    message: str
    entered: bool = True


class EnterPlanModeTool(Tool[EnterPlanModeToolInput, EnterPlanModeToolOutput]):
    """Tool for entering plan mode for complex tasks."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return (
            "Requests permission to enter plan mode for complex tasks "
            "requiring exploration and design"
        )

    @property
    def input_schema(self) -> type[EnterPlanModeToolInput]:
        return EnterPlanModeToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ENTER_PLAN_MODE_PROMPT

    def user_facing_name(self) -> str:
        return ""

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(
        self,
        input_data: Optional[EnterPlanModeToolInput] = None,  # noqa: ARG002
    ) -> bool:
        return True

    async def validate_input(
        self,
        input_data: EnterPlanModeToolInput,
        context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        """Validate that this tool is not being used in an agent context."""
        if context and context.agent_id:
            return ValidationResult(
                result=False,
                message="EnterPlanMode tool cannot be used in agent contexts",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: EnterPlanModeToolOutput) -> str:
        """Render the tool output for the AI assistant."""
        if not output.entered:
            return "User declined to enter plan mode. Continue with normal implementation."
        return f"{output.message}\n\n{PLAN_MODE_INSTRUCTIONS}"

    def render_tool_use_message(
        self,
        input_data: EnterPlanModeToolInput,
        verbose: bool = False,  # noqa: ARG002
    ) -> str:
        """Render the tool use message for display."""
        return "Requesting to enter plan mode"

    async def call(
        self,
        input_data: EnterPlanModeToolInput,  # noqa: ARG002
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        """Execute the tool to enter plan mode."""
        if context.agent_id:
            output = EnterPlanModeToolOutput(
                message="EnterPlanMode tool cannot be used in agent contexts",
                entered=False,
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
            return

        output = EnterPlanModeToolOutput(
            message=(
                "Entered plan mode. You should now focus on exploring "
                "the codebase and designing an implementation approach."
            ),
            entered=True,
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
