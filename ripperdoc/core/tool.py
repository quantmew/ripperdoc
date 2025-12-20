"""Base Tool interface for Ripperdoc.

This module provides the abstract base class for all tools in the system.
Tools are the primary way that the AI agent interacts with the environment.
"""

import json
from abc import ABC, abstractmethod
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, TypeVar, Generic, Union
from pydantic import BaseModel, ConfigDict, Field, SkipValidation
from ripperdoc.utils.file_watch import FileSnapshot
from ripperdoc.utils.log import get_logger


logger = get_logger()


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
    yolo_mode: bool = False
    verbose: bool = False
    permission_checker: Optional[Any] = None
    read_file_timestamps: Dict[str, float] = Field(default_factory=dict)
    # SkipValidation prevents Pydantic from copying the dict during validation,
    # ensuring Read and Edit tools share the same cache instance
    file_state_cache: Annotated[Dict[str, FileSnapshot], SkipValidation] = Field(
        default_factory=dict
    )
    tool_registry: Optional[Any] = None
    abort_signal: Optional[Any] = None
    # UI control callbacks for tools that need user interaction
    pause_ui: Optional[Any] = Field(default=None, description="Callback to pause UI spinner")
    resume_ui: Optional[Any] = Field(default=None, description="Callback to resume UI spinner")
    # Plan mode control callback
    on_exit_plan_mode: Optional[Any] = Field(
        default=None, description="Callback invoked when exiting plan mode"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationResult(BaseModel):
    """Result of input validation."""

    result: bool
    message: Optional[str] = None
    error_code: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


class ToolUseExample(BaseModel):
    """Example of how to call a tool to guide the model."""

    example: Dict[str, Any] = Field(validation_alias="input")
    description: Optional[str] = None
    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=True,
        serialization_aliases={"example": "input"},  # type: ignore[typeddict-item]
    )


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
    async def prompt(self, yolo_mode: bool = False) -> str:
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

    def defer_loading(self) -> bool:
        """Whether this tool should be omitted from the initial tool list.

        Deferred tools can be surfaced later via tool search to save tokens
        when there are many available tools.
        """
        return False

    def input_examples(self) -> List[ToolUseExample]:
        """Optional examples that demonstrate correct tool usage."""
        return []

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


async def build_tool_description(
    tool: Tool[Any, Any], *, include_examples: bool = False, max_examples: int = 3
) -> str:
    """Return the tool description with optional input examples appended."""

    description_text = await tool.description()

    if not include_examples:
        return description_text

    examples = tool.input_examples()
    if not examples:
        return description_text

    try:
        parts = []
        for idx, example in enumerate(examples[:max_examples], start=1):
            payload = json.dumps(example.example, ensure_ascii=False, indent=2)
            prefix = f"{idx}."
            if example.description:
                parts.append(f"{prefix} {example.description}\n{payload}")
            else:
                parts.append(f"{prefix} {payload}")

        if parts:
            return f"{description_text}\n\nInput examples:\n" + "\n\n".join(parts)
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        logger.warning(
            "[tool] Failed to build input example section: %s: %s",
            type(exc).__name__,
            exc,
            extra={"tool": getattr(tool, "name", None)},
        )
        return description_text

    return description_text


def tool_input_examples(tool: Tool[Any, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Return raw input example objects formatted for the model."""
    results: List[Dict[str, Any]] = []
    examples = tool.input_examples()
    if not examples:
        return results
    for example in examples[:limit]:
        try:
            results.append(example.example)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning(
                "[tool] Failed to format tool input example: %s: %s",
                type(exc).__name__,
                exc,
                extra={"tool": getattr(tool, "name", None)},
            )
            continue
    return results
