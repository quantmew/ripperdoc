"""Compatibility tests for SDK websocket connect header argument names."""

from __future__ import annotations

import pytest

from ripperdoc.protocol.stdio.handler_io import _SDKWebSocketTransport


@pytest.mark.asyncio
async def test_open_websocket_prefers_additional_headers() -> None:
    transport = _SDKWebSocketTransport("ws://localhost:3000/v2/session_ingress/ws/session-1")

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def connect(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"url": url, **kwargs})
            return object()

    fake = FakeClient()
    await transport._open_websocket(fake)

    assert len(fake.calls) == 1
    assert "additional_headers" in fake.calls[0]
    assert "extra_headers" not in fake.calls[0]


@pytest.mark.asyncio
async def test_open_websocket_falls_back_to_extra_headers_on_type_error() -> None:
    transport = _SDKWebSocketTransport("ws://localhost:3000/v2/session_ingress/ws/session-1")

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.first = True

        async def connect(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"url": url, **kwargs})
            if self.first:
                self.first = False
                raise TypeError("connect() got an unexpected keyword argument 'additional_headers'")
            return object()

    fake = FakeClient()
    await transport._open_websocket(fake)

    assert len(fake.calls) == 2
    assert "additional_headers" in fake.calls[0]
    assert "extra_headers" in fake.calls[1]
