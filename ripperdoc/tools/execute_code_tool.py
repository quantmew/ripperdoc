"""Execute Python code tool for programmatic agents."""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.programmatic_executor import execute_programmatic_code
from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolProgress,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()


class ExecuteCodeInput(BaseModel):
    """Input for executing Python code."""

    code: str = Field(
        description=(
            "Python code to execute. Use 'ctx.tool_call(tool_name, params)' for tool calls. "
            "Use 'ctx.log(msg)' for logging and 'ctx.set_result(value)' for output."
        )
    )


class ExecuteCodeOutput(BaseModel):
    """Output from code execution."""

    success: bool
    result: Any = None
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    tool_call_count: int = 0


class ExecuteCodeTool(Tool[ExecuteCodeInput, ExecuteCodeOutput]):
    """Tool for executing Python code with programmatic tool access.

    This tool allows agents to write and execute Python code that can call
    other tools via ctx.tool_call(). Useful for complex data processing,
    loops, and conditional logic.
    """

    def __init__(
        self,
        tools: Optional[Dict[str, Tool[Any, Any]]] = None,
        working_directory: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._tools = tools or {}
        self._working_directory = working_directory

    @property
    def name(self) -> str:
        return "ExecuteCode"

    async def description(self) -> str:
        return (
            "Execute Python code with access to tools via ctx.tool_call(). "
            "Use for loops, data processing, and complex logic."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return ExecuteCodeInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        del yolo_mode
        return """\
Execute Python code with programmatic access to tools via `ctx.tool_call()`.

## Context Object (ctx)

```python
ctx.tool_call("ToolName", {"param": "value"})  # Call tools (returns dict)
ctx.log("message")                              # Log progress
ctx.set_result(data)                            # Set final result
ctx.get_available_tools()                       # List available tools
ctx.get_tool_schema("ToolName")                 # Get tool schema
```

## Common Tools

| Tool | Key Params | Key Returns |
|------|-----------|-------------|
| Glob | pattern | matches (list), count |
| Read | file_path | content |
| Grep | pattern, output_mode | matches (list of {file, line_number, content}) |
| LS | path | entries (list), tree |
| Bash | command | stdout, stderr, exit_code |
| Write | file_path, content | success |
| Edit | file_path, old_string, new_string | diff |

## Allowed Imports

json, re, math, collections, itertools, functools, datetime, copy, typing, etc.

**Cannot** import os/subprocess/sys directly - use ctx.tool_call() instead.

## Example

```python
files = ctx.tool_call("Glob", {"pattern": "**/*.py"})["matches"]
for f in files[:5]:
    content = ctx.tool_call("Read", {"file_path": f})["content"]
    if "TODO" in content:
        ctx.log(f"Found TODO in {f}")
ctx.set_result({"processed": len(files)})
```
"""

    def is_read_only(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[ExecuteCodeInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: ExecuteCodeInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        if not input_data.code or not input_data.code.strip():
            return ValidationResult(result=False, message="code field cannot be empty")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ExecuteCodeOutput) -> str:
        if output.success:
            parts = []
            if output.logs:
                parts.append("Logs:\n" + "\n".join(output.logs))
            if output.result is not None:
                import json
                try:
                    result_str = json.dumps(output.result, ensure_ascii=False, indent=2)
                except (TypeError, ValueError):
                    result_str = str(output.result)
                parts.append(f"Result:\n{result_str}")
            return "\n\n".join(parts) if parts else "Execution completed successfully."
        else:
            return f"Execution failed: {output.error}"

    def render_tool_use_message(
        self, input_data: ExecuteCodeInput, verbose: bool = False
    ) -> str:
        lines = input_data.code.strip().split("\n")
        preview = lines[0][:50] + "..." if len(lines[0]) > 50 else lines[0]
        if len(lines) > 1:
            preview += f" (+{len(lines) - 1} more lines)"
        return f"Execute code: {preview}"

    def set_tools(self, tools: Dict[str, Tool[Any, Any]]) -> None:
        """Set the available tools for code execution."""
        self._tools = tools

    def set_working_directory(self, directory: str) -> None:
        """Set the working directory for code execution."""
        self._working_directory = directory

    async def call(
        self, input_data: ExecuteCodeInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        yield ToolProgress(content="Executing Python code...")

        # Log the code being executed
        logger.info(
            "[execute_code] Running code:\n%s",
            input_data.code,
        )

        # Execute the code
        result = await execute_programmatic_code(
            code=input_data.code,
            tools=self._tools,
            tool_context=context,
            working_directory=self._working_directory or os.getcwd(),
            timeout_seconds=300.0,
        )

        output = ExecuteCodeOutput(
            success=result.success,
            result=result.result,
            logs=result.logs,
            error=result.error,
            duration_ms=result.duration_ms,
            tool_call_count=result.tool_call_count,
        )

        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
