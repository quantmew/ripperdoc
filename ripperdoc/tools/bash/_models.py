"""Data models for Bash tool."""

from __future__ import annotations

from typing import List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ripperdoc.utils.shell.bash_constants import (
    get_bash_default_timeout_ms,
    get_bash_max_timeout_ms,
    get_bash_max_output_length,
)

DEFAULT_TIMEOUT_MS = get_bash_default_timeout_ms()
MAX_BASH_TIMEOUT_MS = get_bash_max_timeout_ms()
MAX_OUTPUT_CHARS = get_bash_max_output_length()


class BashToolInput(BaseModel):
    """Input schema for BashTool."""

    command: str = Field(description="The bash command to execute")
    description: Optional[str] = Field(
        default=None,
        description=(
            "Clear, concise description of what this command does in active voice. "
            "Prefer 5-10 words for simple commands."
        ),
    )
    timeout: Optional[int] = Field(
        default=None,
        description=(
            f"Timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS}ms ≈ {DEFAULT_TIMEOUT_MS / 1000:.0f}s; "
            f"max: {MAX_BASH_TIMEOUT_MS}ms)"
        ),
    )
    shell_executable: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("shell_executable", "shellExecutable"),
        serialization_alias="shellExecutable",
        description="Optional shell path to use instead of the default shell.",
    )
    run_in_background: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("run_in_background", "runInBackground"),
        serialization_alias="runInBackground",
        description="If true, run the command in the background and return immediately with a task id.",
    )
    sandbox: Optional[bool] = Field(
        default=None,
        description="If true, request sandboxed execution (read-only).",
    )
    dangerously_disable_sandbox: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("dangerously_disable_sandbox", "dangerouslyDisableSandbox"),
        serialization_alias="dangerouslyDisableSandbox",
        description="If true, override sandbox mode and run without sandbox restrictions.",
    )
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True, extra="ignore")


class BashToolOutput(BaseModel):
    """Output from bash command execution."""

    stdout: str
    stderr: str
    exit_code: int
    command: str
    duration_ms: float = 0.0
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    background_task_id: Optional[str] = None
    is_truncated: bool = False
    original_length: Optional[int] = None
    exit_code_meaning: Optional[str] = None
    return_code_interpretation: Optional[str] = None
    summary: Optional[str] = None
    interrupted: bool = False
    is_image: bool = False
    sandbox: Optional[bool] = None
    is_error: bool = False
    truncation_details: List[str] = Field(default_factory=list)
