"""TodoRead tool — reads the todo list and picks the next task."""

from __future__ import annotations

from typing import AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.collaboration.todo import (
    TodoItem,
    TodoStatus,
    format_todo_lines,
    format_todo_summary,
    get_next_actionable,
    load_todos,
    summarize_todos,
)


logger = get_logger()


class TodoReadToolInput(BaseModel):
    """Input for reading the todo list."""

    status: Optional[List[TodoStatus]] = Field(
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

    todos: List[TodoItem]
    summary: str
    stats: Dict
    next_todo: Optional[TodoItem] = None


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

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Get only the next actionable todo",
                example={"next_only": True},
            ),
            ToolUseExample(
                description="List recent completed tasks with a limit",
                example={"status": ["completed"], "limit": 5},
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.todo_read._prompt import TODO_READ_PROMPT
        return TODO_READ_PROMPT


    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[TodoReadToolInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: TodoReadToolInput,
        _context: Optional[ToolUseContext] = None,
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
        _verbose: bool = False,
    ) -> str:
        if input_data.next_only:
            return "Reading next actionable todo"
        return "Reading todo list"

    async def call(
        self,
        input_data: TodoReadToolInput,
        _context: ToolUseContext,
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
