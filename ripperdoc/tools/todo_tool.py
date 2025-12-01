"""Todo management tools."""

from __future__ import annotations

from typing import AsyncGenerator, Optional
from textwrap import dedent
from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ValidationResult,
)
from ripperdoc.utils.todo import (
    TodoItem,
    TodoPriority,
    TodoStatus,
    format_todo_lines,
    format_todo_summary,
    get_next_actionable,
    load_todos,
    set_todos,
    summarize_todos,
    validate_todos,
)

TODO_WRITE_PROMPT = dedent(
    """\
    Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user. It also helps the user understand the progress of the task and overall progress of their requests.

    ## When to Use This Tool
    Use this tool proactively in these scenarios:

    1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
    2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
    3. User explicitly requests todo list - When the user directly asks you to use the todo list
    4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
    5. After receiving new instructions - Immediately capture user requirements as todos
    6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
    7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

    ## When NOT to Use This Tool

    Skip using this tool when:
    1. There is only a single, straightforward task
    2. The task is trivial and tracking it provides no organizational benefit
    3. The task can be completed in less than 3 trivial steps
    4. The task is purely conversational or informational

    NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

    ## Task States and Management

    1. Task States:
       - pending: Task not yet started
       - in_progress: Currently working on (limit to ONE task at a time)
       - completed: Task finished successfully

    2. Task Management:
       - Update task status in real-time as you work
       - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
       - Only have ONE task in_progress at any time
       - Complete current tasks before starting new ones
       - Remove tasks that are no longer relevant from the list entirely

    3. Task Completion Requirements:
       - ONLY mark a task as completed when you have FULLY accomplished it
       - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
       - When blocked, create a new task describing what needs to be resolved
       - Never mark a task as completed if tests are failing, implementation is partial, errors are unresolved, or needed files are missing

    4. Task Breakdown:
       - Create specific, actionable items
       - Break complex tasks into smaller, manageable steps
       - Use clear, descriptive task names

    When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.
    """
)


class TodoInputItem(BaseModel):
    """Single todo input payload."""

    content: str = Field(description="Task description")
    status: TodoStatus = Field(default="pending", description="pending|in_progress|completed")
    priority: TodoPriority = Field(default="medium", description="high|medium|low")
    id: str = Field(description="Unique identifier for the task")


class TodoWriteToolInput(BaseModel):
    """Input for updating the todo list."""

    todos: list[TodoInputItem] = Field(description="Complete todo list to persist")


class TodoReadToolInput(BaseModel):
    """Input for reading the todo list."""

    status: Optional[list[TodoStatus]] = Field(
        default=None, description="Filter by status; omit for all todos"
    )
    limit: int = Field(
        default=0,
        description="Optional limit for the number of todos to return; 0 returns all matches",
    )
    next_only: bool = Field(
        default=False,
        description="Return only the next actionable todo (in_progress first, then pending)",
    )


class TodoToolOutput(BaseModel):
    """Common output for todo operations."""

    todos: list[TodoItem]
    summary: str
    stats: dict
    next_todo: Optional[TodoItem] = None


class TodoWriteTool(Tool[TodoWriteToolInput, TodoToolOutput]):
    """Create or update the todo list."""

    @property
    def name(self) -> str:
        return "TodoWrite"

    async def description(self) -> str:
        return "Create and update a structured task list for the current session."

    @property
    def input_schema(self) -> type[TodoWriteToolInput]:
        return TodoWriteToolInput

    async def prompt(self, safe_mode: bool = False) -> str:
        return TODO_WRITE_PROMPT

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[TodoWriteToolInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: TodoWriteToolInput,
        context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        todos = [TodoItem(**todo.model_dump()) for todo in input_data.todos]
        ok, message = validate_todos(todos)
        if not ok:
            return ValidationResult(result=False, message=message)
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TodoToolOutput) -> str:
        return output.summary

    def render_tool_use_message(
        self,
        input_data: TodoWriteToolInput,
        verbose: bool = False,
    ) -> str:
        return f"Updating todo list with {len(input_data.todos)} item(s)"

    async def call(
        self,
        input_data: TodoWriteToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        try:
            todos = [TodoItem(**todo.model_dump()) for todo in input_data.todos]
            updated = set_todos(todos)
            summary = format_todo_summary(updated)
            lines = format_todo_lines(updated)
            result_text = "\n".join([summary, *lines]) if lines else summary
            output = TodoToolOutput(
                todos=updated,
                summary=summary,
                stats=summarize_todos(updated),
                next_todo=get_next_actionable(updated),
            )
            yield ToolResult(data=output, result_for_assistant=result_text)
        except Exception as exc:
            error = f"Error updating todos: {exc}"
            yield ToolResult(
                data=TodoToolOutput(
                    todos=[],
                    summary=error,
                    stats={"total": 0, "by_status": {}, "by_priority": {}},
                    next_todo=None,
                ),
                result_for_assistant=error,
            )


class TodoReadTool(Tool[TodoReadToolInput, TodoToolOutput]):
    """Read the todo list and pick the next task."""

    @property
    def name(self) -> str:
        return "TodoRead"

    async def description(self) -> str:
        return (
            "Reads the stored todo list for this project so you can review tasks, "
            "pick the next item to execute, and update progress."
        )

    @property
    def input_schema(self) -> type[TodoReadToolInput]:
        return TodoReadToolInput

    async def prompt(self, safe_mode: bool = False) -> str:
        return (
            "Use TodoRead to fetch the current todo list before making progress or when you need "
            "to confirm the next action. You can request only the next actionable item or filter "
            "by status."
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[TodoReadToolInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: TodoReadToolInput,
        context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if input_data.limit < 0:
            return ValidationResult(result=False, message="limit cannot be negative")
        if input_data.status:
            invalid = [
                status
                for status in input_data.status
                if status not in ("pending", "in_progress", "completed")
            ]
            if invalid:
                return ValidationResult(
                    result=False,
                    message=f"Invalid status values: {', '.join(invalid)}",
                )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TodoToolOutput) -> str:
        return output.summary

    def render_tool_use_message(
        self,
        input_data: TodoReadToolInput,
        verbose: bool = False,
    ) -> str:
        if input_data.next_only:
            return "Reading next actionable todo"
        return "Reading todo list"

    async def call(
        self,
        input_data: TodoReadToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        all_todos = load_todos()
        filtered = all_todos

        if input_data.status:
            allowed = set(input_data.status)
            filtered = [todo for todo in all_todos if todo.status in allowed]

        display = filtered
        next_todo = get_next_actionable(filtered)

        if input_data.next_only:
            display = [next_todo] if next_todo else []

        if input_data.limit and input_data.limit > 0:
            display = display[: input_data.limit]

        if not all_todos:
            summary = "No todos stored yet."
        elif input_data.next_only:
            summary = (
                f"Next actionable todo: {next_todo.content} (id: {next_todo.id}, status: {next_todo.status})."
                if next_todo
                else "No actionable todos (none pending or in_progress)."
            )
        else:
            summary = format_todo_summary(filtered)

        lines = format_todo_lines(display)
        result_text = "\n".join([summary, *lines]) if lines else summary
        output = TodoToolOutput(
            todos=display,
            summary=summary,
            stats=summarize_todos(filtered),
            next_todo=next_todo,
        )
        yield ToolResult(data=output, result_for_assistant=result_text)
