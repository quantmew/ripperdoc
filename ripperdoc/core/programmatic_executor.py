"""Programmatic execution environment for Python-based tool calling.

This module provides a sandboxed Python execution environment where agents can
execute Python code that programmatically calls tools, rather than using LLM-based
tool calling. This reduces latency and allows complex data processing.
"""

from __future__ import annotations

import ast
import asyncio
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ripperdoc.utils.log import get_logger

# Import mock modules from dedicated package
from ripperdoc.core.programmatic_mocks import (
    FORBIDDEN_MODULES,
    ALLOWED_MODULES,
    MOCK_MODULES,
    MOCK_SYS,
    MOCK_SUBPROCESS,
    MOCK_SHUTIL,
    MOCK_IO,
    MOCK_TEMPFILE,
    MOCK_SOCKET,
    create_dynamic_mock_os,
    create_dynamic_mock_glob,
    create_dynamic_mock_pathlib,
)

if TYPE_CHECKING:
    from ripperdoc.core.tool import Tool, ToolUseContext

logger = get_logger()


class SecurityViolationError(Exception):
    """Raised when code attempts a forbidden operation."""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when code execution exceeds the timeout."""
    pass


@dataclass
class ProgrammaticResult:
    """Result of programmatic code execution."""

    success: bool
    result: Any = None
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    tool_call_count: int = 0


class ProgrammaticContext:
    """Context object provided to programmatic code execution.

    This class provides a safe interface for programmatic agents to:
    - Call tools via tool_call()
    - Log progress messages
    - Set final results
    - Access environment information

    All I/O operations MUST go through tool_call() - direct file/network
    access is not allowed.
    """

    def __init__(
        self,
        tools: Dict[str, "Tool[Any, Any]"],
        tool_context: "ToolUseContext",
        working_directory: str,
        timeout_seconds: float = 300.0,
    ):
        self._tools = tools
        self._tool_context = tool_context
        self._working_directory = working_directory
        self._timeout_seconds = timeout_seconds
        self._start_time = time.time()

        # Execution state
        self._logs: List[str] = []
        self._result: Any = None
        self._result_set: bool = False
        self._tool_call_count: int = 0
        self._cancelled: bool = False

    def log(self, message: str) -> None:
        """Log a progress message."""
        timestamp = time.time() - self._start_time
        self._logs.append(f"[{timestamp:.2f}s] {message}")
        logger.debug(
            "[programmatic] %s",
            message,
            extra={"timestamp": timestamp},
        )

    def set_result(self, result: Any) -> None:
        """Set the final result to return."""
        self._result = result
        self._result_set = True

    def get_result(self) -> Any:
        """Get the current result."""
        return self._result

    def get_logs(self) -> List[str]:
        """Get all logged messages."""
        return list(self._logs)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get a tool's input and output schema as dict.

        Args:
            tool_name: Name of the tool (e.g., "Glob", "Read")

        Returns:
            Dict with 'input_schema' and 'output_schema' fields describing
            the tool's parameters and return value structure.

        Example:
            schema = ctx.get_tool_schema("Glob")
            # schema["input_schema"] shows required parameters
            # schema["output_schema"] shows return value fields
        """
        tool = self._tools.get(tool_name)
        if not tool:
            available = ", ".join(sorted(self._tools.keys()))
            raise ValueError(
                f"Tool '{tool_name}' not available. Available tools: {available}"
            )

        result: Dict[str, Any] = {"tool_name": tool_name}

        # Get input schema
        input_schema = tool.input_schema
        if hasattr(input_schema, "model_json_schema"):
            result["input_schema"] = input_schema.model_json_schema()
        elif hasattr(input_schema, "schema"):
            result["input_schema"] = input_schema.schema()
        else:
            result["input_schema"] = {}

        # Try to get output schema from type hints
        try:
            import typing
            if "call" in dir(tool):
                # Try to extract return type from call method
                call_hints = typing.get_type_hints(tool.call)
                if "return" in call_hints:
                    return_type = call_hints["return"]
                    # Check if it's a generic with output type
                    if hasattr(return_type, "__origin__"):
                        args = getattr(return_type, "__args__", ())
                        for arg in args:
                            if hasattr(arg, "model_json_schema"):
                                result["output_schema"] = arg.model_json_schema()
                                break
                            elif hasattr(arg, "schema"):
                                result["output_schema"] = arg.schema()
                                break
        except Exception:
            pass

        if "output_schema" not in result:
            result["output_schema"] = {"note": "Schema not available, check tool documentation"}

        return result

    def get_working_directory(self) -> str:
        """Get the current working directory."""
        return self._working_directory

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self._start_time

    def get_remaining_time(self) -> float:
        """Get remaining time before timeout in seconds."""
        elapsed = self.get_elapsed_time()
        return max(0.0, self._timeout_seconds - elapsed)

    def is_timeout(self) -> bool:
        """Check if execution has timed out."""
        return self.get_remaining_time() <= 0

    def cancel(self) -> None:
        """Cancel the execution."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if execution has been cancelled."""
        return self._cancelled

    def tool_call(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call a tool by name with the given parameters (synchronous).

        Args:
            tool_name: Name of the tool to call (e.g., "Glob", "Read", "Grep")
            params: Dictionary of parameters to pass to the tool

        Returns:
            The tool's result data

        Raises:
            ValueError: If tool is not available
            ExecutionTimeoutError: If execution has timed out
            SecurityViolationError: If the call is blocked for security reasons
        """
        # Check for cancellation and timeout
        if self._cancelled:
            raise RuntimeError("Execution was cancelled")
        if self.is_timeout():
            raise ExecutionTimeoutError(
                f"Execution timed out after {self._timeout_seconds}s"
            )

        # Validate tool exists
        tool = self._tools.get(tool_name)
        if not tool:
            available = ", ".join(sorted(self._tools.keys()))
            raise ValueError(
                f"Tool '{tool_name}' not available. Available tools: {available}"
            )

        self._tool_call_count += 1
        self.log(f"Calling {tool_name} with {_summarize_params(params)}")

        # Run the async tool call in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._async_tool_call(tool, tool_name, params)
            )
        finally:
            loop.close()

    async def _async_tool_call(
        self, tool: "Tool[Any, Any]", tool_name: str, params: Dict[str, Any]
    ) -> Any:
        """Internal async implementation of tool_call."""
        try:
            # Parse input using the tool's schema
            input_schema = tool.input_schema
            parsed_input = input_schema(**params)

            # Validate input
            validation = await tool.validate_input(parsed_input, self._tool_context)
            if not validation.result:
                raise ValueError(f"Invalid input for {tool_name}: {validation.message}")

            # Execute the tool
            result = None
            async for output in tool.call(parsed_input, self._tool_context):
                # Get the final ToolResult
                if hasattr(output, "data"):
                    result = output.data

            # Convert Pydantic models to dict for easier use in programmatic code
            if result is not None:
                if hasattr(result, "model_dump"):
                    # Pydantic v2
                    result = result.model_dump()
                elif hasattr(result, "dict"):
                    # Pydantic v1
                    result = result.dict()

            return result

        except Exception as exc:
            self.log(f"Tool {tool_name} failed: {exc}")
            raise


def _summarize_params(params: Dict[str, Any], max_len: int = 100) -> str:
    """Create a short summary of parameters."""
    import json
    try:
        s = json.dumps(params, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s
    except (TypeError, ValueError):
        return str(params)[:max_len]


class CodeValidator(ast.NodeVisitor):
    """AST visitor that validates code for security violations.

    Note: Modules in MOCK_MODULES are allowed because they have safe mock replacements.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".")[0]
            # Allow modules that have mock replacements
            if module in MOCK_MODULES:
                continue  # Safe - will use mock module
            if module in FORBIDDEN_MODULES:
                self.errors.append(f"Import of forbidden module: {module}")
            elif module not in ALLOWED_MODULES:
                self.errors.append(
                    f"Import of unrecognized module: {module}. "
                    f"Only these modules are allowed: {', '.join(sorted(ALLOWED_MODULES))}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split(".")[0]
            # Allow modules that have mock replacements
            if module in MOCK_MODULES:
                self.generic_visit(node)
                return  # Safe - will use mock module
            if module in FORBIDDEN_MODULES:
                self.errors.append(f"Import from forbidden module: {module}")
            elif module not in ALLOWED_MODULES:
                self.errors.append(
                    f"Import from unrecognized module: {module}. "
                    f"Only these modules are allowed: {', '.join(sorted(ALLOWED_MODULES))}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for forbidden function calls
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in ("exec", "eval", "compile", "open", "input", "__import__"):
                self.errors.append(f"Call to forbidden function: {func.id}")
        # Note: We don't block calls to mock module methods here
        # because the mock modules will raise PermissionError at runtime
        # for dangerous operations, giving better error messages
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Check for dangerous attribute access
        if node.attr in ("__class__", "__bases__", "__subclasses__", "__mro__",
                         "__globals__", "__code__", "__builtins__"):
            self.errors.append(f"Access to dangerous attribute: {node.attr}")
        self.generic_visit(node)


def _add_line_numbers(code: str) -> str:
    """Add line numbers to each line of code for error reporting.

    Example output:
       1│ import json
       2│ result = ctx.tool_call("LS", {"path": "."})
       3│ ctx.log(result)
    """
    lines = code.split("\n")
    width = len(str(len(lines)))
    numbered_lines = []
    for i, line in enumerate(lines, start=1):
        numbered_lines.append(f"{i:>{width}}│ {line}")
    return "\n".join(numbered_lines)


def _fix_traceback_line_numbers(traceback_str: str, offset: int = 1) -> str:
    """Fix line numbers in traceback to account for wrapper function.

    The code is wrapped in `def __programmatic_main__():` which adds 1 line,
    so line numbers in traceback are off by 1. This function adjusts them.

    Args:
        traceback_str: The traceback string to fix
        offset: Number of lines added by wrapper (default 1)
    """
    import re

    def replace_line_number(match: re.Match[str]) -> str:
        prefix = match.group(1)
        line_num = int(match.group(2))
        suffix = match.group(3)
        # Adjust line number, but don't go below 1
        adjusted = max(1, line_num - offset)
        return f"{prefix}{adjusted}{suffix}"

    # Match patterns like:
    # - 'File "<programmatic>", line 7'
    # - 'line 7, in __programmatic_main__'
    pattern = r'(File "<programmatic>", line |, line )(\d+)(,|$)'
    return re.sub(pattern, replace_line_number, traceback_str)


def validate_code(code: str) -> List[str]:
    """Validate code for security issues.

    Returns:
        List of error messages. Empty list means code is safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    validator = CodeValidator()
    validator.visit(tree)
    return validator.errors


async def execute_programmatic_code(
    code: str,
    tools: Dict[str, "Tool[Any, Any]"],
    tool_context: "ToolUseContext",
    working_directory: Optional[str] = None,
    timeout_seconds: float = 300.0,
) -> ProgrammaticResult:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute (async function body)
        tools: Dictionary of available tools
        tool_context: Context for tool execution
        working_directory: Working directory for the execution
        timeout_seconds: Maximum execution time

    Returns:
        ProgrammaticResult with success status, result, logs, etc.
    """
    start_time = time.time()

    # Validate code
    errors = validate_code(code)
    if errors:
        numbered_code = _add_line_numbers(code)
        return ProgrammaticResult(
            success=False,
            error="Code validation failed:\n"
            + "\n".join(f"- {e}" for e in errors)
            + f"\n\nCode:\n{numbered_code}",
            duration_ms=(time.time() - start_time) * 1000,
        )

    # Create context
    ctx = ProgrammaticContext(
        tools=tools,
        tool_context=tool_context,
        working_directory=working_directory or os.getcwd(),
        timeout_seconds=timeout_seconds,
    )

    # Prepare safe globals
    safe_globals = _build_safe_globals(ctx)

    # Wrap code in function (synchronous - tool_call handles async internally)
    wrapped_code = _wrap_in_function(code)

    try:
        # Compile the code
        compiled = compile(wrapped_code, "<programmatic>", "exec")

        # Execute to define the function
        exec(compiled, safe_globals)

        # Get the defined function
        main_func = safe_globals.get("__programmatic_main__")
        if not main_func:
            return ProgrammaticResult(
                success=False,
                error="Failed to create execution function",
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Execute with timeout using concurrent.futures
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(main_func)
            try:
                result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                ctx.cancel()  # Signal cancellation
                return ProgrammaticResult(
                    success=False,
                    error=f"Execution timed out after {timeout_seconds}s",
                    logs=ctx.get_logs(),
                    duration_ms=(time.time() - start_time) * 1000,
                    tool_call_count=ctx._tool_call_count,
                )

        # Get result
        final_result = ctx.get_result() if ctx._result_set else result

        return ProgrammaticResult(
            success=True,
            result=final_result,
            logs=ctx.get_logs(),
            duration_ms=(time.time() - start_time) * 1000,
            tool_call_count=ctx._tool_call_count,
        )

    except SecurityViolationError as e:
        numbered_code = _add_line_numbers(code)
        return ProgrammaticResult(
            success=False,
            error=f"Security violation: {e}\n\nCode:\n{numbered_code}",
            logs=ctx.get_logs(),
            duration_ms=(time.time() - start_time) * 1000,
            tool_call_count=ctx._tool_call_count,
        )
    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            numbered_code = _add_line_numbers(code)
            return ProgrammaticResult(
                success=False,
                error=f"Execution was cancelled\n\nCode:\n{numbered_code}",
                logs=ctx.get_logs(),
                duration_ms=(time.time() - start_time) * 1000,
                tool_call_count=ctx._tool_call_count,
            )
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Fix line numbers in traceback (offset by 1 due to wrapper function)
        tb = _fix_traceback_line_numbers(tb)
        numbered_code = _add_line_numbers(code)
        return ProgrammaticResult(
            success=False,
            error=f"Execution error: {e}\n\nCode:\n{numbered_code}\n\nTraceback:\n{tb}",
            logs=ctx.get_logs(),
            duration_ms=(time.time() - start_time) * 1000,
            tool_call_count=ctx._tool_call_count,
        )


def _wrap_in_function(code: str) -> str:
    """Wrap code in a function definition."""
    # Dedent and normalize
    code = textwrap.dedent(code).strip()

    # Indent the code
    indented = textwrap.indent(code, "    ")

    # Wrap in regular function (not async - tool_call is now synchronous)
    wrapped = f"def __programmatic_main__():\n{indented}\n"

    return wrapped


def _create_safe_import(
    ctx: "ProgrammaticContext",
    dynamic_os: Any,
    dynamic_glob: Any,
    dynamic_pathlib: Any,
) -> Any:
    """Create a safe import function with access to dynamic mock modules.

    This factory creates an import function that:
    - Returns mock modules for dangerous modules (os, sys, subprocess, etc.)
    - Returns dynamic mocks that can call Glob tool for filesystem operations
    - Blocks forbidden modules
    - Allows safe modules like json, re, math, etc.
    """
    def _safe_import(
        name: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        locals_dict: Optional[Dict[str, Any]] = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        """Safe import function that only allows whitelisted modules."""
        # Get the top-level module name
        top_module = name.split(".")[0]

        # Special cases for dynamic mocks with ctx access
        if top_module == "os":
            logger.debug("[programmatic] Returning dynamic mock os module")
            return dynamic_os
        if top_module == "glob":
            logger.debug("[programmatic] Returning dynamic mock glob module")
            return dynamic_glob
        if top_module == "pathlib":
            logger.debug("[programmatic] Returning dynamic mock pathlib module")
            return dynamic_pathlib

        # Check if this is a mock-able forbidden module
        if top_module in MOCK_MODULES:
            logger.debug(
                "[programmatic] Returning mock module for: %s",
                top_module,
            )
            return MOCK_MODULES[top_module]

        # For other forbidden modules without mocks, raise error
        if top_module in FORBIDDEN_MODULES:
            raise ImportError(
                f"Import of forbidden module: {top_module}. "
                f"Use ctx.tool_call() for I/O operations."
            )

        # For unrecognized modules, raise error
        if top_module not in ALLOWED_MODULES:
            raise ImportError(
                f"Import of unrecognized module: {top_module}. "
                f"Use ctx.tool_call() for I/O operations."
            )

        # Import the allowed module using the real __import__
        import builtins
        return builtins.__import__(name, globals_dict, locals_dict, fromlist, level)

    return _safe_import


def _build_safe_globals(ctx: ProgrammaticContext) -> Dict[str, Any]:
    """Build a restricted globals dictionary for code execution."""
    import json
    import re
    import math
    import statistics
    import collections
    import itertools
    import functools
    import operator
    import string
    import datetime
    import copy
    import typing
    import dataclasses
    import enum
    import random
    import hashlib
    import base64
    import uuid
    import fnmatch

    # Create dynamic mocks with ctx access for filesystem operations
    dynamic_os = create_dynamic_mock_os(ctx)
    dynamic_glob = create_dynamic_mock_glob(ctx)
    dynamic_pathlib = create_dynamic_mock_pathlib(ctx)

    # Create safe import function with access to dynamic mocks
    safe_import = _create_safe_import(ctx, dynamic_os, dynamic_glob, dynamic_pathlib)

    # Safe builtins
    safe_builtins = {
        # Types
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "bytes": bytes,
        "bytearray": bytearray,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "frozenset": frozenset,
        "type": type,
        "object": object,

        # Functions
        "abs": abs,
        "all": all,
        "any": any,
        "ascii": ascii,
        "bin": bin,
        "callable": callable,
        "chr": chr,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "format": format,
        "getattr": getattr,
        "hasattr": hasattr,
        "hash": hash,
        "hex": hex,
        "id": id,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": lambda *args, **kwargs: ctx.log(" ".join(str(a) for a in args)),
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "setattr": setattr,
        "slice": slice,
        "sorted": sorted,
        "sum": sum,
        "zip": zip,

        # Import function (restricted)
        "__import__": safe_import,

        # Exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "ImportError": ImportError,

        # Constants
        "True": True,
        "False": False,
        "None": None,
    }

    return {
        "__builtins__": safe_builtins,
        "__name__": "__programmatic__",
        "__doc__": None,

        # Context object
        "ctx": ctx,

        # Safe modules (pre-imported for convenience)
        "json": json,
        "re": re,
        "math": math,
        "statistics": statistics,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
        "operator": operator,
        "string": string,
        "datetime": datetime,
        "copy": copy,
        "typing": typing,
        "dataclasses": dataclasses,
        "enum": enum,
        "random": random,
        "hashlib": hashlib,
        "base64": base64,
        "uuid": uuid,
        "fnmatch": fnmatch,

        # Async support
        "asyncio": asyncio,

        # Mock modules (safe replacements for dangerous modules)
        # These allow code like "import os" to work without errors,
        # but dangerous operations will raise PermissionError
        # Use dynamic mocks that can call Glob tool for filesystem operations
        "os": dynamic_os,
        "sys": MOCK_SYS,
        "subprocess": MOCK_SUBPROCESS,
        "shutil": MOCK_SHUTIL,
        "pathlib": dynamic_pathlib,
        "io": MOCK_IO,
        "glob": dynamic_glob,
        "tempfile": MOCK_TEMPFILE,
        "socket": MOCK_SOCKET,
    }
