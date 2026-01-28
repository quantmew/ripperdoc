"""Subprocess transport implementation using anyio for clean async I/O.

This module implements stdio transport for communicating with the Ripperdoc CLI subprocess.
It uses anyio for better async/await patterns and portability.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import suppress
from pathlib import Path as PathLib
from typing import TYPE_CHECKING, Any

import anyio
from anyio.abc import Process
from anyio.streams.text import TextReceiveStream, TextSendStream

from ripperdoc.sdk._errors import (
    CLIConnectionError,
    CLINotFoundError,
    ProcessError as RipperdocProcessError,
)
from ripperdoc.sdk._internal.transport import Transport

# Type hint only - not imported at runtime to avoid circular import
if TYPE_CHECKING:
    from ripperdoc.sdk.client import ClaudeAgentOptions as _ClaudeAgentOptions
else:
    _ClaudeAgentOptions = object

logger = logging.getLogger(__name__)

# Default buffer size limit
_DEFAULT_MAX_BUFFER_SIZE = 1024 * 1024  # 1MB

# Platform-specific command line length limits
_CMD_LENGTH_LIMIT = 8000 if platform.system() == "Windows" else 100000


class SubprocessCLITransport(Transport):
    """Stdio subprocess transport using anyio for clean async I/O.

    This transport starts a Ripperdoc CLI subprocess and communicates
    with it through stdin/stdout using JSON messages separated by newlines.

    Key features:
    - Uses anyio for cross-event-loop compatibility
    - TextReceiveStream/TextSendStream for clean text I/O
    - Write lock for thread-safe writes
    - Proper resource cleanup
    - Exit error tracking
    """

    def __init__(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: _ClaudeAgentOptions | dict[str, Any],
    ):
        """Initialize the subprocess transport.

        Args:
            prompt: Either a string prompt or an async iterable of message dicts.
            options: Configuration options for the CLI. Can be a ClaudeAgentOptions
                     object or a dict (to avoid circular imports).
        """
        self._prompt = prompt
        self._is_streaming = not isinstance(prompt, str)

        # Convert options to dict for uniform access
        if isinstance(options, dict):
            self._options = options
        else:
            # Extract attributes from object
            self._options = {
                'cli_path': getattr(options, 'cli_path', None),
                'cwd': getattr(options, 'cwd', None),
                'model': getattr(options, 'model', None),
                'permission_mode': getattr(options, 'permission_mode', 'default'),
                'max_turns': getattr(options, 'max_turns', None),
                'system_prompt': getattr(options, 'system_prompt', None),
                'env': getattr(options, 'env', {}),
                'stderr': getattr(options, 'stderr', None),
                'max_buffer_size': getattr(options, 'max_buffer_size', None),
            }

        # Find CLI path
        cli_path = self._options.get('cli_path')
        self._cli_path = (
            str(cli_path) if cli_path is not None else self._find_cli()
        )

        # Working directory
        cwd = self._options.get('cwd')
        self._cwd = str(cwd) if cwd else None

        # Process and streams
        self._process: Process | None = None
        self._stdout_stream: TextReceiveStream | None = None
        self._stdin_stream: TextSendStream | None = None
        self._stderr_stream: TextReceiveStream | None = None
        self._stderr_task: anyio.Task | None = None

        # State tracking
        self._ready = False
        self._exit_error: Exception | None = None
        self._write_lock = anyio.Lock()
        max_buffer_size = self._options.get('max_buffer_size')
        self._max_buffer_size = (
            max_buffer_size
            if max_buffer_size is not None
            else _DEFAULT_MAX_BUFFER_SIZE
        )

        # Temporary files for long command lines
        self._temp_files: list[str] = []

    def _find_cli(self) -> str:
        """Find Ripperdoc CLI binary.

        Returns:
            Path to the CLI executable.

        Raises:
            CLINotFoundError: If CLI cannot be found.
        """
        # Check system PATH first
        if cli := shutil.which("ripperdoc"):
            return cli

        # Common installation locations
        locations = [
            PathLib.home() / ".local" / "bin" / "ripperdoc",
            PathLib("/usr/local") / "bin" / "ripperdoc",
            PathLib.home() / ".bin" / "ripperdoc",
        ]

        for path in locations:
            if path.exists() and path.is_file():
                return str(path)

        raise CLINotFoundError(
            "Ripperdoc CLI not found. Install with:\n"
            "  pip install -e ripperdoc[cli]\n"
            "\nOr provide the path via ClaudeAgentOptions:\n"
            "  ClaudeAgentOptions(cli_path='/path/to/ripperdoc')"
        )

    def _build_command(self) -> list[str]:
        """Build the CLI command with all arguments.

        Returns:
            List of command line arguments.
        """
        cmd = [self._cli_path, "stdio", "--output-format", "stream-json"]

        # Add basic options
        model = self._options.get('model')
        if model:
            cmd.extend(["--model", model])

        permission_mode = self._options.get('permission_mode', 'default')
        if permission_mode:
            cmd.extend(["--permission-mode", permission_mode])

        max_turns = self._options.get('max_turns')
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])

        system_prompt = self._options.get('system_prompt')
        if system_prompt:
            if isinstance(system_prompt, str):
                cmd.extend(["--system-prompt", system_prompt])

        # Handle streaming vs string mode
        if self._is_streaming:
            cmd.extend(["--input-format", "stream-json"])
        else:
            cmd.extend(["--print", "--", str(self._prompt)])

        return cmd

    async def connect(self) -> None:
        """Start the subprocess and establish communication.

        Raises:
            CLIConnectionError: If the process fails to start.
            CLINotFoundError: If the CLI cannot be found.
        """
        if self._process:
            return  # Already connected

        cmd = self._build_command()

        # Prepare environment
        env_dict = self._options.get('env', {})
        env = {**os.environ, **env_dict}
        env["RIPPERDOC_ENTRYPOINT"] = "sdk-py-stdio"

        try:
            # Merge environment variables
            process_env = {
                **os.environ,
                **env_dict,
                "RIPPERDOC_ENTRYPOINT": "sdk-py-stdio",
            }

            # Start the process
            self._process = await anyio.open_process(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._cwd,
                env=process_env,
            )

            # Setup streams
            if self._process.stdout:
                self._stdout_stream = TextReceiveStream(self._process.stdout)

            # Setup stderr stream
            if self._process.stderr:
                self._stderr_stream = TextReceiveStream(self._process.stderr)
                self._stderr_task = anyio.create_task()
                await self._stderr_task.start(self._handle_stderr)

            # Setup stdin for streaming mode
            if self._is_streaming and self._process.stdin:
                self._stdin_stream = TextSendStream(self._process.stdin)
            elif not self._is_streaming and self._process.stdin:
                # String mode: close stdin immediately
                await self._process.stdin.aclose()

            self._ready = True
            logger.info(f"Connected to Ripperdoc CLI: {' '.join(cmd)}")

        except FileNotFoundError as e:
            # Check if error is from CLI or working directory
            if self._cwd and not PathLib(self._cwd).exists():
                error = CLIConnectionError(
                    f"Working directory does not exist: {self._cwd}"
                )
                self._exit_error = error
                raise error from e

            error = CLINotFoundError(f"Ripperdoc CLI not found at: {self._cli_path}")
            self._exit_error = error
            raise error from e

        except Exception as e:
            error = CLIConnectionError(f"Failed to start Ripperdoc CLI: {e}")
            self._exit_error = error
            raise error from e

    async def _handle_stderr(self) -> None:
        """Handle stderr stream - log lines."""
        if not self._stderr_stream:
            return

        try:
            async for line in self._stderr_stream:
                line_str = line.rstrip()
                if not line_str:
                    continue

                # Invoke stderr callback if provided
                stderr_callback = self._options.get('stderr')
                if stderr_callback:
                    stderr_callback(line_str)

                logger.debug(f"[CLI stderr] {line_str}")

        except anyio.ClosedResourceError:
            pass  # Stream closed
        except Exception:
            pass  # Ignore other errors

    async def write(self, data: str) -> None:
        """Write data to the subprocess stdin.

        Args:
            data: String data to write (typically JSON + newline).

        Raises:
            CLIConnectionError: If write fails.
        """
        async with self._write_lock:
            # All checks inside lock to prevent TOCTOU races
            if not self._ready or not self._stdin_stream:
                raise CLIConnectionError("Transport not ready for writing")

            if self._process and self._process.returncode is not None:
                raise CLIConnectionError(
                    f"Cannot write to terminated process (exit code: {self._process.returncode})"
                )

            if self._exit_error:
                raise CLIConnectionError(
                    f"Cannot write to process that exited with error: {self._exit_error}"
                ) from self._exit_error

            try:
                await self._stdin_stream.send(data)
            except Exception as e:
                self._ready = False
                self._exit_error = CLIConnectionError(f"Failed to write to process: {e}")
                raise self._exit_error from e

    async def end_input(self) -> None:
        """End the input stream by closing stdin."""
        async with self._write_lock:
            if self._stdin_stream:
                with suppress(Exception):
                    await self._stdin_stream.aclose()
                self._stdin_stream = None

    def read_messages(self) -> AsyncIterable[dict[str, Any]]:
        """Read and parse JSON messages from stdout.

        Yields:
            Parsed JSON message dictionaries.

        Raises:
            CLIConnectionError: If reading fails.
        """
        return self._read_messages_impl()

    async def _read_messages_impl(self) -> AsyncIterator[dict[str, Any]]:
        """Internal implementation of message reading."""
        if not self._process or not self._stdout_stream:
            raise CLIConnectionError("Not connected")

        json_buffer = ""

        try:
            async for line in self._stdout_stream:
                line_str = line.strip()
                if not line_str:
                    continue

                # Handle JSON that may span multiple lines
                # or have embedded newlines in strings
                json_lines = line_str.split("\n")

                for json_line in json_lines:
                    json_line = json_line.strip()
                    if not json_line:
                        continue

                    # Accumulate JSON until we can parse it
                    json_buffer += json_line

                    if len(json_buffer) > self._max_buffer_size:
                        buffer_length = len(json_buffer)
                        json_buffer = ""
                        raise CLIConnectionError(
                            f"JSON message exceeded buffer size ({buffer_length} > {self._max_buffer_size})"
                        )

                    try:
                        # Try to parse - if successful, clear buffer
                        data = json.loads(json_buffer)
                        json_buffer = ""
                        yield data

                    except json.JSONDecodeError:
                        # Keep buffering - might be incomplete JSON
                        continue

        except anyio.ClosedResourceError:
            pass  # Stream closed normally
        except GeneratorExit:
            pass  # Client disconnected

        finally:
            # Check process exit status
            try:
                returncode = await self._process.wait()
            except Exception:
                returncode = -1

            # Handle non-zero exit codes
            if returncode != 0:
                error = RipperdocProcessError(
                    f"CLI process exited with code {returncode}",
                    exit_code=returncode,
                )
                self._exit_error = error

                # Only raise if we didn't get a clean disconnect
                if self._ready:
                    raise error

    async def close(self) -> None:
        """Close the transport and clean up all resources."""
        # Clean up temporary files first
        for temp_file in self._temp_files:
            with suppress(Exception):
                PathLib(temp_file).unlink(missing_ok=True)
        self._temp_files.clear()

        if not self._process:
            self._ready = False
            return

        # Close stderr task
        if self._stderr_task:
            with suppress(Exception):
                self._stderr_task.cancel()
                with suppress(anyio.TimeoutError):
                    await self._stderr_task
            self._stderr_task = None

        # Close stdin stream (with lock to prevent race with write)
        async with self._write_lock:
            self._ready = False
            if self._stdin_stream:
                with suppress(Exception):
                    await self._stdin_stream.aclose()
                self._stdin_stream = None

        # Close streams
        if self._stderr_stream:
            with suppress(Exception):
                await self._stderr_stream.aclose()
            self._stderr_stream = None

        if self._stdout_stream:
            with suppress(Exception):
                await self._stdout_stream.aclose()
            self._stdout_stream = None

        # Terminate process
        if self._process.returncode is None:
            with suppress(Exception):
                self._process.terminate()
                with suppress(Exception):
                    await self._process.wait()

        self._process = None
        self._exit_error = None

    def is_ready(self) -> bool:
        """Check if the transport is ready for communication.

        Returns:
            True if ready for communication.
        """
        return (
            self._ready
            and self._process is not None
            and self._process.returncode is None
        )


__all__ = ["SubprocessCLITransport"]
