"""Exit plan mode tool for presenting implementation plans.

This tool allows the AI to exit plan mode and present an implementation
plan to the user for approval before starting to code.
"""

from __future__ import annotations

from textwrap import dedent
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "ExitPlanMode"

EXIT_PLAN_MODE_PROMPT = dedent(
    """\
    Use this tool when you are in plan mode and have finished writing your plan to the plan file and are ready for user approval.

    ## How This Tool Works
    - You should have already written your plan to the plan file specified in the plan mode system message
    - This tool does NOT take the plan content as a parameter - it will read the plan from the file you wrote
    - This tool simply signals that you're done planning and ready for the user to review and approve
    - The user will see the contents of your plan file when they review it

    ## When to Use This Tool
    IMPORTANT: Only use this tool when the task requires planning the implementation steps of a task that requires writing code. For research tasks where you're gathering information, searching files, reading files or in general trying to understand the codebase - do NOT use this tool.

    ## Handling Ambiguity in Plans
    Before using this tool, ensure your plan is clear and unambiguous. If there are multiple valid approaches or unclear requirements:
    1. Use the AskUserQuestion tool to clarify with the user
    2. Ask about specific implementation choices (e.g., architectural patterns, which library to use)
    3. Clarify any assumptions that could affect the implementation
    4. Edit your plan file to incorporate user feedback
    5. Only proceed with ExitPlanMode after resolving ambiguities and updating the plan file

    ## Examples

    1. Initial task: "Search for and understand the implementation of vim mode in the codebase" - Do not use the exit plan mode tool because you are not planning the implementation steps of a task.
    2. Initial task: "Help me implement yank mode for vim" - Use the exit plan mode tool after you have finished planning the implementation steps of the task.
    3. Initial task: "Add a new feature to handle user authentication" - If unsure about auth method (OAuth, JWT, etc.), use AskUserQuestion first, then use exit plan mode tool after clarifying the approach.
    """
)


class ExitPlanModeToolInput(BaseModel):
    """Input for the ExitPlanMode tool."""

    plan: str = Field(
        description="The plan you came up with, that you want to run by the user for approval. "
        "Supports markdown. The plan should be pretty concise."
    )


class ExitPlanModeToolOutput(BaseModel):
    """Output from the ExitPlanMode tool."""

    plan: str
    is_agent: bool = False


class ExitPlanModeTool(Tool[ExitPlanModeToolInput, ExitPlanModeToolOutput]):
    """Tool for exiting plan mode and presenting a plan for approval."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Prompts the user to exit plan mode and start coding"

    @property
    def input_schema(self) -> type[ExitPlanModeToolInput]:
        return ExitPlanModeToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return EXIT_PLAN_MODE_PROMPT

    def user_facing_name(self) -> str:
        return "Exit plan mode"

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(
        self,
        input_data: Optional[ExitPlanModeToolInput] = None,  # noqa: ARG002
    ) -> bool:
        return True

    async def validate_input(
        self,
        input_data: ExitPlanModeToolInput,
        context: Optional[ToolUseContext] = None,  # noqa: ARG002
    ) -> ValidationResult:
        """Validate that plan is not empty."""
        if not input_data.plan or not input_data.plan.strip():
            return ValidationResult(
                result=False,
                message="Plan cannot be empty",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ExitPlanModeToolOutput) -> str:
        """Render the tool output for the AI assistant."""
        return f"Exit plan mode and start coding now. Plan:\n{output.plan}"

    def render_tool_use_message(
        self,
        input_data: ExitPlanModeToolInput,
        verbose: bool = False,  # noqa: ARG002
    ) -> str:
        """Render the tool use message for display."""
        plan = input_data.plan
        snippet = f"{plan[:77]}..." if len(plan) > 80 else plan
        return f"Share plan for approval: {snippet}"

    async def call(
        self,
        input_data: ExitPlanModeToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        """Execute the tool to exit plan mode."""
        # Invoke the exit plan mode callback if available
        if context.on_exit_plan_mode:
            try:
                context.on_exit_plan_mode()
            except (RuntimeError, ValueError, TypeError):
                logger.debug("[exit_plan_mode_tool] Failed to call on_exit_plan_mode")

        is_agent = bool(context.agent_id)
        output = ExitPlanModeToolOutput(
            plan=input_data.plan,
            is_agent=is_agent,
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
