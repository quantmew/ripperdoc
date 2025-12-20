"""Todo management tools."""

from __future__ import annotations

from typing import AsyncGenerator, List, Optional
from textwrap import dedent
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

logger = get_logger()

DEFAULT_ACTION = "Edit"

TODO_WRITE_PROMPT = dedent(
    f"""\
    Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
    It also helps the user understand the progress of the task and overall progress of their requests.

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

    ## Examples of When to Use the Todo List

    <example>
    User: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!
    Assistant: I'll help add a dark mode toggle to your application settings. Let me create a todo list to track this implementation.
    *Creates todo list with the following items:*
    1. Create dark mode toggle component in Settings page
    2. Add dark mode state management (context/store)
    3. Implement CSS-in-JS styles for dark theme
    4. Update existing components to support theme switching
    5. Run tests and build process, addressing any failures or errors that occur
    *Begins working on the first task*

    <reasoning>
    The assistant used the todo list because:
    1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes
    2. The user explicitly requested tests and build be run afterward
    3. The assistant inferred that tests and build need to pass by adding "Ensure tests and build succeed" as the final task
    </reasoning>
    </example>

    <example>
    User: Help me rename the function getCwd to getCurrentWorkingDirectory across my project
    Assistant: Let me first search through your codebase to find all occurrences of 'getCwd'.
    *Uses grep or search tools to locate all instances of getCwd in the codebase*
    Assistant: I've found 15 instances of 'getCwd' across 8 different files. Let me create a todo list to track these changes.
    *Creates todo list with specific items for each file that needs updating*

    <reasoning>
    The assistant used the todo list because:
    1. First, the assistant searched to understand the scope of the task
    2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps
    3. The todo list helps ensure every instance is tracked and updated systematically
    4. This approach prevents missing any occurrences and maintains code consistency
    </reasoning>
    </example>


    <example>
    User: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.
    Assistant: I'll help implement these features. First, let's add all the features to the todo list.
    *Creates a todo list breaking down each feature into specific tasks based on the project architecture*
    Assistant: Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.

    <reasoning>
    The assistant used the todo list because:
    1. The user provided multiple complex features to implement in a comma separated list
    2. The todo list helps organize these large features into manageable tasks
    3. This approach allows for tracking progress across the entire implementation
    </reasoning>
    </example>

    <example>
    User: Can you help optimize my React application? It's rendering slowly and has performance issues.</user>
    Assistant: I'll help optimize your React application. First, let me examine your codebase to identify potential performance bottlenecks.
    *Reviews component structure, render patterns, state management, and data fetching*
    Assistant: After analyzing your codebase, I've identified several performance issues. Let me create a todo list to track our optimization efforts.
    *Creates todo list with items like: 1) Implement memoization for expensive calculations in ProductList, 2) Add virtualization for long lists in Dashboard, 3) Optimize image loading in Gallery component, 4) Fix state update loops in ShoppingCart, 5) Review bundle size and implement code splitting*
    Let's start by implementing memoization for the expensive calculations in your ProductList component.</assistant>

    <reasoning>
    The assistant used the todo list because:
    1. First, the assistant examined the codebase to identify specific performance issues
    2. Based on this analysis, it identified multiple optimization opportunities
    3. Performance optimization is a non-trivial task requiring multiple steps
    4. The todo list helps methodically track improvements across different components
    5. This systematic approach ensures all performance bottlenecks are addressed
    </reasoning>
    </example>

    ## Examples of When NOT to Use the Todo List

    <example>
    User: How do I print 'Hello World' in Python?
    Assistant: In Python, you can print "Hello World" with this simple code:

    python
    print("Hello World")

    This will output the text "Hello World" to the console when executed.</assistant>

    <reasoning>
    The assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.
    </reasoning>
    </example>

    <example>
    User: What does the git status command do?
    Assistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.

    <reasoning>
    The assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.
    </reasoning>
    </example>

    <example>
    User: Can you add a comment to the calculateTotal function to explain what it does?
    Assistant: Sure, let me add a comment to the calculateTotal function to explain what it does.
    * Uses the ${DEFAULT_ACTION} tool to add a comment to the calculateTotal function *

    <reasoning>
    The assistant did not use the todo list because this is a single, straightforward task confined to one location in the code. Adding a comment doesn't require tracking multiple steps or systematic organization.
    </reasoning>
    </example>

    <example>
    User: Run npm install for me and tell me what happens.
    Assistant: I'll run the npm install command for you.

    *Executes: npm install*

    The command completed successfully. Here's the output:
    [Output of npm install command]

    All dependencies have been installed according to your package.json file.

    <reasoning>
    The assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.
    </reasoning>
    </example>

    ## Task States and Management

    1. **Task States**: Use these states to track progress:
    - pending: Task not yet started
    - in_progress: Currently working on (limit to ONE task at a time)
    - completed: Task finished successfully

    2. **Task Management**:
    - Update task status in real-time as you work
    - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
    - Only have ONE task in_progress at any time
    - Complete current tasks before starting new ones
    - Remove tasks that are no longer relevant from the list entirely

    3. **Task Completion Requirements**:
    - ONLY mark a task as completed when you have FULLY accomplished it
    - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
    - When blocked, create a new task describing what needs to be resolved
    - Never mark a task as completed if:
        - Tests are failing
        - Implementation is partial
        - You encountered unresolved errors
        - You couldn't find necessary files or dependencies

    4. **Task Breakdown**:
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
        return "Update the todo list for the current session. To be used proactively and often to track progress and pending tasks."

    @property
    def input_schema(self) -> type[TodoWriteToolInput]:
        return TodoWriteToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Seed a three-step plan",
                example={
                    "todos": [
                        {
                            "id": "plan",
                            "content": "Review existing API docs",
                            "status": "pending",
                            "priority": "medium",
                        },
                        {
                            "id": "impl",
                            "content": "Implement new endpoint",
                            "status": "pending",
                            "priority": "high",
                        },
                        {
                            "id": "tests",
                            "content": "Add integration tests",
                            "status": "pending",
                            "priority": "high",
                        },
                    ]
                },
            ),
            ToolUseExample(
                description="Update a single task already in progress",
                example={
                    "todos": [
                        {
                            "id": "bugfix-123",
                            "content": "Fix login redirect loop",
                            "status": "in_progress",
                            "priority": "high",
                        }
                    ]
                },
            ),
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TODO_WRITE_PROMPT

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, _input_data: Optional[TodoWriteToolInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: TodoWriteToolInput,
        _context: Optional[ToolUseContext] = None,
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
        _verbose: bool = False,
    ) -> str:
        return f"Updating todo list with {len(input_data.todos)} item(s)"

    async def call(
        self,
        input_data: TodoWriteToolInput,
        _context: ToolUseContext,
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
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("[todo_tool] Error updating todos: %s: %s", type(exc).__name__, exc)
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

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return (
            "Use TodoRead to fetch the current todo list before making progress or when you need "
            "to confirm the next action. You can request only the next actionable item or filter "
            "by status."
        )

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
