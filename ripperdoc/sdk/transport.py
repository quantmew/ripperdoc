"""Transport abstraction layer for Ripperdoc SDK.

This module provides the transport abstraction that allows the SDK to communicate
with the Ripperdoc CLI through different mechanisms (stdio, websocket, etc.).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any, Callable


class Transport(ABC):
    """Abstract base class for transport implementations.

    Transport defines how the SDK communicates with the underlying service.
    Different implementations can use subprocess (stdio), websocket, HTTP, etc.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection."""

    @abstractmethod
    async def write(self, data: str) -> None:
        """Write data to the transport.

        Args:
            data: String data to write (typically JSON)
        """

    @abstractmethod
    def read_messages(self) -> AsyncIterable[dict[str, Any]]:
        """Read messages from the transport.

        Yields:
            Parsed message dictionaries
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the transport is ready for communication."""

    @abstractmethod
    async def end_input(self) -> None:
        """Signal that no more input will be sent."""


class StdioTransportConfig:
    """Configuration for stdio transport."""

    def __init__(
        self,
        cli_path: str | Path | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ):
        """Initialize stdio transport configuration.

        Args:
            cli_path: Path to the Ripperdoc CLI executable.
                     If None, uses "ripperdoc" from PATH.
            args: Additional command line arguments.
            env: Environment variables for the subprocess.
            cwd: Working directory for the subprocess.
        """
        self.cli_path = cli_path
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd


class StdioTransport(Transport):
    """Stdio transport implementation for subprocess communication.

    This transport communicates with the Ripperdoc CLI subprocess through
    standard input/output using JSON messages separated by newlines.
    """

    def __init__(
        self,
        config: StdioTransportConfig | None = None,
    ):
        """Initialize the stdio transport.

        Args:
            config: Transport configuration. If None, uses defaults.
        """
        self.config = config or StdioTransportConfig()
        self.process: subprocess.Popen[bytes] | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._message_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._write_queue: asyncio.Queue[str] | None = None
        self._write_task: asyncio.Task[None] | None = None
        self._connected = False

    async def connect(self) -> None:
        """Start the CLI subprocess and establish communication."""
        if self._connected:
            return

        # Build command
        cmd = [str(self.config.cli_path or "ripperdoc")]
        cmd.extend(self.config.args)

        # Prepare environment
        env = os.environ.copy()
        env.update(self.config.env)
        env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py-stdio"

        # Start subprocess
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(self.config.cwd) if self.config.cwd else None,
            bufsize=1,  # Line buffered
        )

        # Initialize queues
        self._message_queue = asyncio.Queue()
        self._write_queue = asyncio.Queue()

        # Start reader task
        self._read_task = asyncio.create_task(self._read_loop())
        self._write_task = asyncio.create_task(self._write_loop())

        self._connected = True

    async def _read_loop(self) -> None:
        """Read messages from subprocess stdout."""
        if not self.process or not self.process.stdout:
            return

        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                )
                if not line:
                    break  # EOF

                try:
                    message = json.loads(line.decode("utf-8").strip())
                    if self._message_queue:
                        await self._message_queue.put(message)
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    pass
        except Exception as e:
            if self._message_queue:
                await self._message_queue.put({
                    "type": "error",
                    "error": str(e),
                })
        finally:
            if self._message_queue:
                await self._message_queue.put({"type": "eof"})

    async def _write_loop(self) -> None:
        """Write messages to subprocess stdin."""
        if not self.process or not self.process.stdin:
            return

        try:
            while True:
                message = await self._write_queue.get()
                if message is None:  # Sentinel to stop
                    break

                data = (message + "\n").encode("utf-8")
                self.process.stdin.write(data)
                self.process.stdin.flush()
        except BrokenPipeError:
            # Process ended
            pass
        finally:
            if self.process and self.process.stdin:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass

    async def write(self, data: str) -> None:
        """Write data to the subprocess.

        Args:
            data: String data to write (typically JSON)
        """
        if not self._connected or not self._write_queue:
            raise RuntimeError("Transport not connected")

        await self._write_queue.put(data)

    def read_messages(self) -> AsyncIterable[dict[str, Any]]:
        """Read messages from the subprocess.

        Yields:
            Parsed message dictionaries
        """
        if not self._connected or not self._message_queue:
            raise RuntimeError("Transport not connected")

        async def message_generator() -> AsyncIterator[dict[str, Any]]:
            while True:
                message = await self._message_queue.get()
                if message.get("type") == "eof":
                    break
                yield message

        return message_generator()

    async def close(self) -> None:
        """Close the connection and cleanup subprocess."""
        if not self._connected:
            return

        self._connected = False

        # Stop write loop
        if self._write_queue:
            await self._write_queue.put(None)

        # Wait for tasks to complete
        if self._write_task:
            self._write_task.cancel()
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        # Terminate subprocess
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass

            self.process = None

    def is_ready(self) -> bool:
        """Check if the transport is ready for communication."""
        return (
            self._connected
            and self.process is not None
            and self.process.poll() is None
        )

    async def end_input(self) -> None:
        """Signal that no more input will be sent."""
        # For stdio, we can close stdin to signal EOF
        if self.process and self.process.stdin:
            try:
                self.process.stdin.close()
            except Exception:
                pass


class InProcessTransport(Transport):
    """In-process transport for testing and development.

    This transport executes queries directly in the current process
    without spawning a subprocess. Useful for testing and scenarios
    where subprocess overhead is undesirable.
    """

    def __init__(
        self,
        query_handler: Callable | None = None,
    ):
        """Initialize the in-process transport.

        Args:
            query_handler: Optional custom query handler.
                           If None, uses the default core query.
        """
        self._query_handler = query_handler
        self._message_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize the transport."""
        self._message_queue = asyncio.Queue()
        self._connected = True

    async def write(self, data: str) -> None:
        """Process a query request.

        Args:
            data: JSON string containing the query request
        """
        if not self._connected:
            raise RuntimeError("Transport not connected")

        try:
            request = json.loads(data)
            # Handle the request (implementation depends on query_handler)
            if self._query_handler:
                await self._query_handler(request, self._message_queue)
        except Exception as e:
            if self._message_queue:
                await self._message_queue.put({
                    "type": "error",
                    "error": str(e),
                })

    def read_messages(self) -> AsyncIterable[dict[str, Any]]:
        """Read messages from the queue.

        Yields:
            Parsed message dictionaries
        """
        if not self._connected or not self._message_queue:
            raise RuntimeError("Transport not connected")

        async def message_generator() -> AsyncIterator[dict[str, Any]]:
            while True:
                message = await self._message_queue.get()
                if message.get("type") == "eof":
                    break
                yield message

        return message_generator()

    async def close(self) -> None:
        """Close the transport."""
        self._connected = False
        if self._message_queue:
            await self._message_queue.put({"type": "eof"})

    def is_ready(self) -> bool:
        """Check if the transport is ready."""
        return self._connected

    async def end_input(self) -> None:
        """Signal that no more input will be sent."""
        if self._message_queue:
            await self._message_queue.put({"type": "eof"})


__all__ = [
    "Transport",
    "StdioTransport",
    "StdioTransportConfig",
    "InProcessTransport",
]
