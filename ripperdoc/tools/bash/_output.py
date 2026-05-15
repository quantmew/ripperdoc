"""Output formatting for Bash tool."""

from __future__ import annotations

from typing import Any, List, Optional

from ripperdoc.utils.shell.exit_code_handlers import interpret_exit_code
from ripperdoc.utils.shell.output_utils import (
    format_duration,
    get_last_n_lines,
    is_output_large,
    sanitize_output,
    trim_blank_lines,
    truncate_output,
)
from ripperdoc.tools.bash._models import BashToolOutput, MAX_OUTPUT_CHARS


def render_result_for_assistant(output: BashToolOutput) -> str:
    """Format output for the AI."""
    result_parts = []

    if output.stdout:
        result_parts.append(f"stdout:\n{output.stdout}")

    if output.stderr:
        result_parts.append(f"stderr:\n{output.stderr}")

    exit_code_text = f"exit code: {output.exit_code}"
    meaning = output.exit_code_meaning or output.return_code_interpretation
    if meaning:
        exit_code_text += f" ({meaning})"

    timing = ""
    if output.duration_ms:
        timing = f" ({format_duration(output.duration_ms)}"
        if output.timeout_ms:
            timing += f" / timeout {output.timeout_ms / 1000:.0f}s"
        timing += ")"
    elif output.timeout_ms:
        timing = f" (timeout {output.timeout_ms / 1000:.0f}s)"

    result_parts.append(f"{exit_code_text}{timing}")

    if output.is_truncated and output.original_length:
        result_parts.append(
            f"Note: Output was truncated (original length: {output.original_length} chars)"
        )
        if output.truncation_details:
            result_parts.append(
                "Truncation details:\n" + "\n".join(output.truncation_details)
            )

    if output.interrupted:
        result_parts.append("Command was interrupted (timeout or termination).")

    if output.background_task_id:
        result_parts.append(f"Background task id: {output.background_task_id}")

    return "\n\n".join(result_parts)


def build_final_output(
    command: str,
    stdout_lines: list[str],
    stderr_lines: list[str],
    exit_code: int,
    duration_ms: float,
    timeout_ms: int,
    timeout_seconds: float,
    timed_out: bool,
    sandbox_requested: bool,
    original_command: str,
) -> BashToolOutput:
    """Build the final output from execution results."""
    raw_stdout = "".join(stdout_lines)
    raw_stderr = "".join(stderr_lines)

    if timed_out:
        timeout_msg = f"Command timed out after {timeout_seconds} seconds"
        raw_stderr = f"{raw_stderr.rstrip()}\n{timeout_msg}" if raw_stderr else timeout_msg
        exit_code = -1

    raw_stdout = sanitize_output(raw_stdout)
    raw_stderr = sanitize_output(raw_stderr)
    trimmed_stdout = trim_blank_lines(raw_stdout)
    trimmed_stderr = trim_blank_lines(raw_stderr)

    exit_result = interpret_exit_code(
        original_command, exit_code, trimmed_stdout, trimmed_stderr
    )

    summary = None
    combined_output = "\n".join([part for part in (trimmed_stdout, trimmed_stderr) if part])
    if combined_output and is_output_large(combined_output):
        summary = get_last_n_lines(combined_output, 20)

    stdout_result = truncate_output(trimmed_stdout, max_chars=MAX_OUTPUT_CHARS)
    stderr_result = truncate_output(trimmed_stderr, max_chars=MAX_OUTPUT_CHARS)
    is_image = stdout_result.get("is_image", False) or stderr_result.get("is_image", False)
    truncation_details: list[str] = []
    stdout_notice = _build_truncation_notice("stdout", stdout_result)
    if stdout_notice:
        truncation_details.append(stdout_notice)
    stderr_notice = _build_truncation_notice("stderr", stderr_result)
    if stderr_notice:
        truncation_details.append(stderr_notice)

    is_truncated = stdout_result["is_truncated"] or stderr_result["is_truncated"]
    original_length = None
    if is_truncated:
        original_length = stdout_result.get("original_length", 0) + stderr_result.get(
            "original_length", 0
        )

    return BashToolOutput(
        stdout=stdout_result["truncated_content"],
        stderr=stderr_result["truncated_content"],
        exit_code=exit_code,
        command=command,
        duration_ms=duration_ms,
        timeout_ms=timeout_ms,
        is_truncated=is_truncated,
        original_length=original_length,
        exit_code_meaning=exit_result.semantic_meaning,
        return_code_interpretation=exit_result.semantic_meaning,
        summary=summary,
        interrupted=timed_out,
        is_image=is_image,
        sandbox=sandbox_requested,
        is_error=exit_result.is_error or timed_out,
        truncation_details=truncation_details,
    )


def _build_truncation_notice(stream_name: str, truncation: dict[str, Any]) -> Optional[str]:
    """Build a machine-readable truncation message."""
    if not truncation.get("is_truncated"):
        return None

    omitted_chars = int(truncation.get("omitted_chars") or 0)
    kept_start = int(truncation.get("kept_start_chars") or 0)
    kept_end = int(truncation.get("kept_end_chars") or 0)
    start_line = truncation.get("omitted_start_line")
    end_line = truncation.get("omitted_end_line")
    start_char = truncation.get("omitted_start_char")
    end_char = truncation.get("omitted_end_char")

    location_parts: list[str] = []
    if isinstance(start_line, int) and start_line > 0:
        if isinstance(end_line, int) and end_line >= start_line:
            if start_line == end_line:
                location_parts.append(f"line {start_line}")
            else:
                location_parts.append(f"lines {start_line}-{end_line}")
    if (
        isinstance(start_char, int)
        and isinstance(end_char, int)
        and start_char > 0
        and end_char >= start_char
    ):
        location_parts.append(f"chars {start_char}-{end_char}")

    location_text = ", ".join(location_parts) if location_parts else "middle segment"
    return (
        f"{stream_name}: omitted {omitted_chars} chars at {location_text}; "
        f"kept head {kept_start} chars and tail {kept_end} chars."
    )


def build_background_launch_output(
    *,
    effective_command: str,
    task_id: str,
    start_time: float,
    sandbox_requested: bool,
    status_message: str,
) -> BashToolOutput:
    """Build output for a background task launch."""
    import asyncio
    return BashToolOutput(
        stdout="",
        stderr=status_message,
        exit_code=0,
        command=effective_command,
        duration_ms=(asyncio.get_running_loop().time() - start_time) * 1000.0,
        timeout_ms=0,
        background_task_id=task_id,
        sandbox=sandbox_requested,
        return_code_interpretation=None,
        summary=f"Command running in background with ID: {task_id}",
        interrupted=False,
        is_image=False,
    )
