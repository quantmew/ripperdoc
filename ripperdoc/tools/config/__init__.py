"""Config tool — get or set Ripperdoc configuration settings."""

from __future__ import annotations

from typing import AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "Config"

# Supported settings with their types and descriptions
_SUPPORTED_SETTINGS: Dict[str, dict] = {
    "theme": {
        "type": "string",
        "source": "global",
        "options": ["dark", "light", "ansi"],
        "description": "Color theme for the UI",
    },
    "verbose": {
        "type": "boolean",
        "source": "global",
        "description": "Show detailed debug output",
    },
    "yolo_mode": {
        "type": "boolean",
        "source": "global",
        "description": "Auto-approve all tool calls without asking",
    },
    "auto_compact_enabled": {
        "type": "boolean",
        "source": "global",
        "description": "Auto-compact when context is full",
    },
    "auto_memory_enabled": {
        "type": "boolean",
        "source": "global",
        "description": "Enable auto-memory",
    },
    "default_thinking_tokens": {
        "type": "number",
        "source": "global",
        "description": "Default thinking token budget (0 = disabled)",
    },
    "model": {
        "type": "string",
        "source": "global",
        "description": "Current main model profile name",
    },
}


class ConfigToolInput(BaseModel):
    setting: str = Field(
        description='The setting key (e.g., "theme", "verbose", "yolo_mode")',
    )
    value: Optional[Union[str, bool, int, float]] = Field(
        default=None,
        description="The new value. Omit to get current value.",
    )


class ConfigToolOutput(BaseModel):
    success: bool
    operation: Optional[str] = None
    setting: Optional[str] = None
    value: Optional[Union[str, bool, int, float]] = None
    previous_value: Optional[Union[str, bool, int, float]] = None
    new_value: Optional[Union[str, bool, int, float]] = None
    error: Optional[str] = None


class ConfigTool(Tool[ConfigToolInput, ConfigToolOutput]):
    """Get or set Ripperdoc configuration settings."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Get or set Ripperdoc configuration settings."

    @property
    def input_schema(self) -> type[ConfigToolInput]:
        return ConfigToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.config._prompt import CONFIG_PROMPT
        return CONFIG_PROMPT


    def is_read_only(self) -> bool:
        return True  # Overridden dynamically in check_permissions

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[ConfigToolInput] = None) -> bool:
        # Reading is auto-allowed; writing requires permission
        if input_data is None:
            return True
        return input_data.value is not None

    async def validate_input(
        self,
        input_data: ConfigToolInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if input_data.setting not in _SUPPORTED_SETTINGS:
            valid = ", ".join(sorted(_SUPPORTED_SETTINGS.keys()))
            return ValidationResult(
                result=False,
                message=f'Unknown setting: "{input_data.setting}". Valid: {valid}',
            )

        if input_data.value is not None:
            cfg = _SUPPORTED_SETTINGS[input_data.setting]
            # Check options
            options = cfg.get("options")
            if options and str(input_data.value) not in options:
                return ValidationResult(
                    result=False,
                    message=f'Invalid value "{input_data.value}". Options: {", ".join(options)}',
                )

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ConfigToolOutput) -> str:
        if output.error:
            return f"Error: {output.error}"
        if output.operation == "get":
            return f"{output.setting} = {output.value}"
        if output.operation == "set":
            return f"Set {output.setting} to {output.new_value}"
        return output.error or "Unknown operation"

    def render_tool_use_message(self, input_data: ConfigToolInput, _verbose: bool = False) -> str:
        if input_data.value is not None:
            return f"Config set: {input_data.setting} = {input_data.value}"
        return f"Config get: {input_data.setting}"

    async def call(
        self,
        input_data: ConfigToolInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        from ripperdoc.core.config import get_global_config, save_global_config

        setting = input_data.setting
        cfg = _SUPPORTED_SETTINGS[setting]

        # GET operation
        if input_data.value is None:
            config = get_global_config()
            current = getattr(config, setting, None)
            output = ConfigToolOutput(
                success=True,
                operation="get",
                setting=setting,
                value=current,
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
            return

        # SET operation
        config = get_global_config()
        previous = getattr(config, setting, None)

        # Coerce boolean values
        final_value = input_data.value
        if cfg["type"] == "boolean":
            if isinstance(final_value, str):
                final_value = final_value.lower().strip() == "true"
            if not isinstance(final_value, bool):
                output = ConfigToolOutput(
                    success=False,
                    operation="set",
                    setting=setting,
                    error=f"{setting} requires true or false.",
                )
                yield ToolResult(
                    data=output,
                    result_for_assistant=self.render_result_for_assistant(output),
                )
                return

        try:
            setattr(config, setting, final_value)
            save_global_config(config)
        except Exception as exc:
            output = ConfigToolOutput(
                success=False,
                operation="set",
                setting=setting,
                error=f"{type(exc).__name__}: {exc}",
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
            return

        output = ConfigToolOutput(
            success=True,
            operation="set",
            setting=setting,
            previous_value=previous,
            new_value=final_value,
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
