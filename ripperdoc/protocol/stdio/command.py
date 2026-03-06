"""CLI entrypoints for stdio protocol support."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any, Callable

import click
from ripperdoc import __version__
from ripperdoc.protocol.models import DEFAULT_PROTOCOL_VERSION
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.sessions.session_history import (
    load_session_messages,
    list_session_summaries,
)

from .handler import StdioProtocolHandler

logger = get_logger()


def _coerce_print_messages_for_query(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert stored session messages to sampling request payloads."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        raw_message = (
            message.model_dump(by_alias=True, mode="json")
            if hasattr(message, "model_dump")
            else {}
        )
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("type", ""))
        if role not in {"user", "assistant"}:
            continue

        message_payload = raw_message.get("message") if isinstance(raw_message.get("message"), dict) else {}
        if not isinstance(message_payload, dict):
            continue

        entry: dict[str, Any] = {
            "role": role,
            "content": message_payload.get("content", ""),
        }
        parent_tool_use_id = raw_message.get("parent_tool_use_id")
        if parent_tool_use_id is not None:
            entry["parent_tool_use_id"] = parent_tool_use_id
        if role == "assistant":
            model_name = raw_message.get("model")
            if model_name is not None:
                entry["model"] = str(model_name)
        converted.append(entry)
    return converted


def _coerce_stream_payload_message_to_query(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce stream-json stdin payload entries into sampling messages."""
    normalized: list[dict[str, Any]] = []

    item_type = str(message.get("type") or message.get("role") or "").strip()
    role = item_type if item_type in {"user", "assistant"} else ""

    if message.get("type") == "control_request":
        request = message.get("request")
        if isinstance(request, dict) and str(request.get("subtype")).strip() == "query":
            prompt = str(request.get("prompt", "")).strip()
            if prompt:
                normalized.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                )
        return normalized

    if role:
        message_payload = message.get("message", message)
        if not isinstance(message_payload, dict):
            return normalized
        content = message_payload.get("content")
        if isinstance(content, str):
            content = content.strip()
            if not content:
                return normalized
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            normalized_content: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type") or "").strip()
                if block_type != "text":
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    normalized_content.append(
                        {
                            "type": "text",
                            "text": text.strip(),
                        }
                    )
            if not normalized_content:
                return normalized
            content = normalized_content
        else:
            return normalized

        entry: dict[str, Any] = {
            "role": role,
            "content": content,
        }
        parent_tool_use_id = message_payload.get("parent_tool_use_id") or message.get(
            "parent_tool_use_id"
        )
        if parent_tool_use_id is not None:
            entry["parent_tool_use_id"] = parent_tool_use_id
        if role == "assistant":
            model_name = message_payload.get("model") or message.get("model")
            if model_name is not None:
                entry["model"] = str(model_name)
        normalized.append(entry)

    return normalized


def _coerce_stream_messages_for_query(
    payload: Any,
) -> list[dict[str, Any]]:
    """Convert a JSON stream-json payload into sampling messages."""
    normalized: list[dict[str, Any]] = []
    if payload is None:
        return normalized
    if isinstance(payload, list):
        for item in payload:
            normalized.extend(_coerce_stream_messages_for_query(item))
        return normalized

    if not isinstance(payload, dict):
        return normalized

    if "messages" in payload and isinstance(payload["messages"], (list, dict)):
        normalized.extend(_coerce_stream_messages_for_query(payload["messages"]))

    normalized.extend(_coerce_stream_payload_message_to_query(payload))
    return normalized


async def _read_stream_json_messages_from_stdin() -> list[dict[str, Any]]:
    """Read non-tty stdin and parse newline-separated or JSON payload stream."""
    stdin_stream = sys.stdin
    try:
        if stdin_stream.isatty():
            return []
    except Exception:
        return []

    try:
        raw_payload = await asyncio.get_running_loop().run_in_executor(None, stdin_stream.read)
    except (OSError, ValueError):
        return []

    if not isinstance(raw_payload, str):
        return []

    text = raw_payload.strip()
    if not text:
        return []

    parsed_payload: list[Any] = []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            parsed_payload.extend(loaded)
        elif isinstance(loaded, dict):
            parsed_payload.append(loaded)
        elif loaded is not None:
            return []
    except json.JSONDecodeError:
        for line in raw_payload.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, list):
                parsed_payload.extend(loaded)
            elif isinstance(loaded, dict):
                parsed_payload.append(loaded)

    return _coerce_stream_messages_for_query(parsed_payload)


def _install_stdio_shutdown_signal_handlers(
    cancel_main_task: Callable[[], None],
) -> Callable[[], None]:
    """Translate process termination signals into cooperative task cancellation."""
    installed: list[tuple[signal.Signals, Any]] = []

    def _handle_signal(signum: int, _frame: Any) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        logger.info("[stdio] Received %s; cancelling stdio task", signal_name)
        cancel_main_task()

    for signal_name in ("SIGTERM", "SIGINT"):
        raw_signal = getattr(signal, signal_name, None)
        if raw_signal is None:
            continue
        try:
            previous_handler = signal.getsignal(raw_signal)
            signal.signal(raw_signal, _handle_signal)
        except (OSError, RuntimeError, ValueError):
            continue
        installed.append((raw_signal, previous_handler))

    def _restore() -> None:
        for raw_signal, previous_handler in installed:
            try:
                signal.signal(raw_signal, previous_handler)
            except (OSError, RuntimeError, ValueError):
                continue

    return _restore


async def _run_stdio_with_signal_handling(**kwargs: Any) -> None:
    """Run stdio mode while converting SIGTERM/SIGINT into graceful cancellation."""
    main_task = asyncio.create_task(run_stdio(**kwargs))

    def _cancel_main_task() -> None:
        if not main_task.done():
            main_task.cancel()

    restore_signal_handlers = _install_stdio_shutdown_signal_handlers(_cancel_main_task)
    try:
        await main_task
    except asyncio.CancelledError:
        if not main_task.cancelled():
            raise
    finally:
        restore_signal_handlers()


@click.command(name="stdio")
@click.option(
    "--input-format",
    type=click.Choice(["stream-json", "auto"]),
    default="stream-json",
    help="Input format for messages.",
)
@click.option(
    "--output-format",
    type=click.Choice(["stream-json", "json"]),
    default="stream-json",
    help="Output format for messages.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model profile for the current session.",
)
@click.option(
    "--permission-mode",
    type=click.Choice(["default", "acceptEdits", "plan", "dontAsk", "bypassPermissions"]),
    default="default",
    help="Permission mode for tool usage.",
)
@click.option(
    "--max-turns",
    type=int,
    default=None,
    help="Maximum number of conversation turns.",
)
@click.option(
    "--system-prompt",
    type=str,
    default=None,
    help="System prompt to use for the session.",
)
@click.option(
    "--append-system-prompt",
    type=str,
    default=None,
    help="Additional instructions to append to the system prompt.",
)
@click.option(
    "--print",
    "-p",
    is_flag=True,
    help="Print mode (for single prompt queries).",
)
@click.option(
    "--resume-session-at",
    type=str,
    default=None,
    help="When resuming, only messages up to and including the assistant message.",
    hidden=True,
)
@click.option(
    "--rewind-files",
    type=str,
    default=None,
    help="Restore files to state at the specified user message and exit (requires --resume).",
    hidden=True,
)
@click.option(
    "--sdk-url",
    type=str,
    default=None,
    help="Use remote WebSocket endpoint for SDK I/O streaming.",
    hidden=True,
)
@click.option(
    "--replay-user-messages",
    is_flag=True,
    help=(
        "Re-emit user messages from stdin back on stdout for acknowledgment "
        "(only works with --input-format=stream-json and --output-format=stream-json)."
    ),
)
@click.option(
    "--",
    "prompt",
    type=str,
    default=None,
    help="Direct prompt (for print mode).",
)
def stdio_cmd(
    input_format: str,
    output_format: str,
    model: str | None,
    permission_mode: str,
    max_turns: int | None,
    system_prompt: str | None,
    append_system_prompt: str | None,
    print: bool,
    resume_session_at: str | None,
    rewind_files: str | None,
    sdk_url: str | None,
    replay_user_messages: bool,
    prompt: str | None,
) -> None:
    """Stdio mode for SDK subprocess communication.

    This command enables Ripperdoc to communicate with SDKs via JSON Control
    Protocol over stdin/stdout. It's designed for subprocess architecture where
    the SDK manages the CLI process.

    The protocol supports:
    - control_request/control_response for protocol management
    - Message streaming for query results
    - Bidirectional communication for hooks and permissions

    Example:
        ripperdoc stdio --output-format stream-json
    """
    # Set up async event loop
    asyncio.run(
        _run_stdio_with_signal_handling(
            input_format=input_format,
            output_format=output_format,
            model=model,
            permission_mode=permission_mode,
            max_turns=max_turns,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            print_mode=print,
            resume_session_at=resume_session_at,
            rewind_files=rewind_files,
            sdk_url=sdk_url,
            replay_user_messages=replay_user_messages,
            prompt=prompt,
        )
    )


async def run_stdio(
    input_format: str,
    output_format: str,
    model: str | None,
    permission_mode: str,
    max_turns: int | None,
    system_prompt: str | None,
    print_mode: bool,
    session_id: str | None = None,
    resume_session: str | None = None,
    continue_session: bool = False,
    resume_session_at: str | None = None,
    rewind_files: str | None = None,
    fork_session: bool = False,
    project_path: str | Path | None = None,
    sdk_url: str | None = None,
    replay_user_messages: bool = False,
    prompt: str | None = None,
    append_system_prompt: str | None = None,
    default_options: dict[str, Any] | None = None,
) -> None:
    """Async entry point for stdio command."""
    if sdk_url and input_format != "stream-json":
        raise click.ClickException(
            "Error: --sdk-url requires both --input-format=stream-json and --output-format=stream-json."
        )
    if sdk_url and output_format != "stream-json":
        raise click.ClickException(
            "Error: --sdk-url requires both --input-format=stream-json and --output-format=stream-json."
        )
    if replay_user_messages and (
        input_format != "stream-json" or output_format != "stream-json"
    ):
        raise click.ClickException(
            "Error: --replay-user-messages requires both --input-format=stream-json and --output-format=stream-json."
        )

    request_default_options = dict(default_options or {})
    if session_id is not None:
        request_default_options["session_id"] = session_id
    if sdk_url is not None:
        request_default_options["sdk_url"] = sdk_url
    if replay_user_messages:
        request_default_options["replay_user_messages"] = True
    if system_prompt is not None:
        request_default_options["system_prompt"] = system_prompt
    if append_system_prompt is not None:
        request_default_options["append_system_prompt"] = append_system_prompt

    handler = StdioProtocolHandler(
        input_format=input_format,
        output_format=output_format,
        default_options=request_default_options,
    )

    if print_mode:
        if output_format == "stream-json" and not bool(request_default_options.get("verbose")):
            raise click.ClickException(
                "Error: When using --print, --output-format=stream-json requires --verbose"
            )
        if resume_session_at and not resume_session:
            raise click.ClickException("Error: --resume-session-at requires --resume")
        if rewind_files and not resume_session:
            raise click.ClickException("Error: --rewind-files requires --resume")

        prompt_input = prompt.strip() if isinstance(prompt, str) else ""
        if rewind_files and prompt_input:
            raise click.ClickException(
                "Error: --rewind-files is a standalone operation and cannot be used with a prompt"
            )

        project_root = Path(project_path) if project_path is not None else Path.cwd()
        effective_session_id = str(request_default_options.get("session_id") or "")
        resumed_messages: list[Any] = []
        if resume_session or continue_session:
            summaries = list_session_summaries(project_root)
            target_session_id: str | None = None
            if resume_session:
                match = next(
                    (
                        session
                        for session in summaries
                        if session.session_id.startswith(resume_session)
                    ),
                    None,
                )
                if match is None:
                    raise click.ClickException(f"No session found matching '{resume_session}'.")
                target_session_id = match.session_id
            elif summaries:
                target_session_id = summaries[0].session_id

            if target_session_id is not None:
                resumed_messages = load_session_messages(project_root, target_session_id)
                if not fork_session:
                    effective_session_id = target_session_id

        request_default_options["session_id"] = effective_session_id

        if resume_session_at:
            target_index = next(
                (
                    index
                    for index, message in enumerate(resumed_messages)
                    if str(getattr(message, "uuid", "")).strip() == resume_session_at
                ),
                -1,
            )
            if target_index < 0:
                raise click.ClickException(
                    f"No message found with message.uuid of: {resume_session_at}"
                )
            resumed_messages = resumed_messages[: target_index + 1]

        if rewind_files:
            rewind_target = next(
                (
                    message
                    for message in resumed_messages
                    if str(getattr(message, "uuid", "")).strip() == rewind_files
                ),
                None,
            )
            if rewind_target is None or getattr(rewind_target, "type", None) != "user":
                raise click.ClickException(
                    f"Error: --rewind-files requires a user message UUID, but {rewind_files} is not a user message in this session"
                )
            print(f"Files rewound to state at message {rewind_files}")
            return

        stream_messages = []
        if input_format == "stream-json":
            stream_messages = await _read_stream_json_messages_from_stdin()

        if (
            not prompt_input
            and not stream_messages
            and resume_session is None
            and sdk_url is None
        ):
            raise click.ClickException(
                "Error: Input must be provided either through stdin or as a prompt argument when using --print"
            )

        query_messages = _coerce_print_messages_for_query(resumed_messages)
        if stream_messages:
            query_messages.extend(stream_messages)
        elif prompt_input:
            query_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_input,
                        }
                    ],
                }
            )

        request_options: dict[str, Any] = dict(request_default_options)
        if model is not None:
            request_options["model"] = model
        if permission_mode is not None:
            request_options["permission_mode"] = permission_mode
        if max_turns is not None:
            request_options["max_turns"] = max_turns
        if system_prompt is not None:
            request_options["system_prompt"] = system_prompt
        if append_system_prompt is not None:
            request_options["append_system_prompt"] = append_system_prompt
        request_options["sdk_can_use_tool"] = False

        request = {
            "protocolVersion": DEFAULT_PROTOCOL_VERSION,
            "capabilities": {
                "sampling": {
                    "tools": True,
                }
            },
            "clientInfo": {
                "name": "ripperdoc.cli",
                "version": __version__,
            },
            "_meta": {
                "ripperdoc_options": request_options,
            },
        }

        # Mock request_id for print mode
        request_id = "print_query"

        # Initialize
        await handler._handle_initialize(request, request_id)

        query_request = {
            "messages": query_messages,
            "maxTokens": 1024,
        }
        await handler._handle_query(query_request, f"{request_id}_query")
        await handler.flush_output()
        stop_transport = getattr(handler, "_stop_sdk_transport", None)
        if callable(stop_transport):
            maybe_awaitable = stop_transport()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
        return

    # Otherwise, run the stdio protocol loop
    await handler.run()


__all__ = ["stdio_cmd", "run_stdio"]
