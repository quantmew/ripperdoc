"""Base Tool interface for Ripperdoc.

This module provides the abstract base class for all tools in the system.
Tools are the primary way that the AI agent interacts with the environment.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional, TypeVar, Generic, Union
from pydantic import BaseModel, ConfigDict
from enum import Enum


class ToolResult(BaseModel):
    """Result from a tool execution."""

    type: str = "result"
    data: Any
    result_for_assistant: Optional[str] = None


class ToolProgress(BaseModel):
    """Progress update from a tool execution."""

    type: str = "progress"
    content: Any
    normalized_messages: list = []
    tools: list = []


class ToolUseContext(BaseModel):
    """Context for tool execution."""

    message_id: Optional[str] = None
    agent_id: Optional[str] = None
    safe_mode: bool = False
    verbose: bool = False
    permission_checker: Optional[Any] = None
    read_file_timestamps: Dict[str, float] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationResult(BaseModel):
    """Result of input validation."""

    result: bool
    message: Optional[str] = None
    error_code: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput")
ToolOutput = Union[ToolResult, ToolProgress]


class Tool(ABC, Generic[TInput, TOutput]):
    """Abstract base class for all tools.

    Each tool must implement the core methods for describing itself,
    validating input, and executing the tool's functionality.
    """

    def __init__(self) -> None:
        self._name = self.__class__.__name__

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the tool's name."""
        pass

    @abstractmethod
    async def description(self) -> str:
        """Get the tool's description for the AI model."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Get the Pydantic model for input validation."""
        pass

    @abstractmethod
    async def prompt(self, safe_mode: bool = False) -> str:
        """Get the system prompt for this tool."""
        pass

    def user_facing_name(self) -> str:
        """Get the user-facing name of the tool."""
        return self.name

    async def is_enabled(self) -> bool:
        """Check if this tool is enabled."""
        return True

    def is_read_only(self) -> bool:
        """Check if this tool only reads data (doesn't modify state)."""
        return False

    def is_concurrency_safe(self) -> bool:
        """Check if this tool can be run concurrently with others."""
        return self.is_read_only()

    def needs_permissions(self, input_data: Optional[TInput] = None) -> bool:
        """Check if this tool needs permission to execute."""
        return not self.is_read_only()

    async def validate_input(
        self, input_data: TInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        """Validate the input before execution."""
        return ValidationResult(result=True)

    @abstractmethod
    def render_result_for_assistant(self, output: TOutput) -> str:
        """Render the tool output for the AI assistant."""
        pass

    @abstractmethod
    def render_tool_use_message(self, input_data: TInput, verbose: bool = False) -> str:
        """Render the tool use message for display."""
        pass

    @abstractmethod
    async def call(
        self, input_data: TInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Execute the tool with the given input.

        Yields progress updates and finally the result.
        """
        pass
        # This is an abstract method, subclasses must implement
        yield ToolResult(data=None)  # type: ignore


def create_tool_schema(tool: Tool[Any, Any]) -> Dict[str, Any]:
    """Create a JSON schema for the tool that can be sent to the AI model."""
    return {
        "name": tool.name,
        "description": "",  # Will be populated async
        "input_schema": tool.input_schema.model_json_schema(),
    }
