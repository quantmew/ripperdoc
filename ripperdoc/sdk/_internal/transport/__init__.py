"""Transport implementations for Ripperdoc SDK.

This module provides the transport abstraction layer that allows the SDK
to communicate with the Ripperdoc CLI through different mechanisms.

The Transport interface is low-level and handles raw I/O. Higher-level
components like Query build on top of this to implement the control protocol.
"""

import abc
from collections.abc import AsyncIterator
from typing import Any


class Transport(abc.ABC):
    """Abstract transport for Ripperdoc communication.

    This interface defines the contract for different transport implementations
    (stdio, websocket, HTTP, etc.). Each transport handles the low-level
    details of reading and writing messages.

    Note: This is an internal API. The Query class builds on top of Transport
    to implement the control protocol and message routing.
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """Connect the transport and prepare for communication.

        For subprocess transports, this starts the process.
        For network transports, this establishes the connection.
        """

    @abc.abstractmethod
    async def write(self, data: str) -> None:
        """Write raw data to the transport.

        Args:
            data: Raw string data to write (typically JSON + newline)
        """

    @abc.abstractmethod
    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Read and parse messages from the transport.

        Yields:
            Parsed JSON messages from the transport
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the transport connection and clean up resources."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Check if transport is ready for communication.

        Returns:
            True if transport is ready to send/receive messages
        """

    @abc.abstractmethod
    async def end_input(self) -> None:
        """End the input stream (close stdin for process transports)."""


__all__ = ["Transport"]
