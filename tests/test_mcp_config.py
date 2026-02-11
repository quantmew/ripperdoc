import asyncio
import pytest

from ripperdoc.utils import mcp


@pytest.mark.parametrize(
    "raw_command,raw_args,expected_cmd,expected_args",
    [
        ("npx -y @upstash/context7-mcp", None, "npx", ["-y", "@upstash/context7-mcp"]),
        ("python3", ["-m", "server"], "python3", ["-m", "server"]),
        (["node", "server.js"], None, "node", ["server.js"]),
        (None, None, None, []),
    ],
)
def test_normalize_command(raw_command, raw_args, expected_cmd, expected_args):
    cmd, args = mcp._normalize_command(raw_command, raw_args)
    assert cmd == expected_cmd
    assert args == expected_args


@pytest.mark.asyncio
async def test_mcp_runtime_load_is_non_blocking_and_reports_connecting(monkeypatch, tmp_path):
    await mcp.shutdown_mcp_runtime()
    mcp._mcp_circuit_states.clear()

    monkeypatch.setattr(mcp, "MCP_AVAILABLE", True)
    monkeypatch.setattr(
        mcp,
        "_load_server_configs",
        lambda _project_path: {
            "slow": mcp.McpServerInfo(name="slow", command="slow-server"),
        },
    )

    async def _fake_connect_server(self, config):
        await asyncio.sleep(0.06)
        return mcp.McpServerInfo(
            name=config.name,
            type=config.type,
            command=config.command,
            args=config.args,
            status="connected",
            tools=[mcp.McpToolInfo(name="ping")],
        )

    monkeypatch.setattr(mcp.McpRuntime, "_connect_server", _fake_connect_server)

    try:
        initial = await mcp.load_mcp_servers_async(tmp_path)
        assert len(initial) == 1
        assert initial[0].name == "slow"
        assert initial[0].status == "connecting"

        await asyncio.sleep(0.1)
        after = await mcp.load_mcp_servers_async(tmp_path)
        assert after[0].status == "connected"
        assert [tool.name for tool in after[0].tools] == ["ping"]
    finally:
        await mcp.shutdown_mcp_runtime()
        mcp._mcp_circuit_states.clear()


@pytest.mark.asyncio
async def test_mcp_timeout_marks_failed_and_opens_circuit_breaker(monkeypatch, tmp_path):
    await mcp.shutdown_mcp_runtime()
    mcp._mcp_circuit_states.clear()

    monkeypatch.setenv("RIPPERDOC_MCP_CONNECT_TIMEOUT_SEC", "0.02")
    monkeypatch.setenv("RIPPERDOC_MCP_CIRCUIT_BREAKER_FAILURES", "1")
    monkeypatch.setenv("RIPPERDOC_MCP_CIRCUIT_BREAKER_COOLDOWN_SEC", "60")
    monkeypatch.setattr(mcp, "MCP_AVAILABLE", True)
    monkeypatch.setattr(
        mcp,
        "_load_server_configs",
        lambda _project_path: {
            "timeout-server": mcp.McpServerInfo(name="timeout-server", command="slow-server"),
        },
    )

    connect_calls = {"count": 0}

    async def _very_slow_connect(self, config):
        connect_calls["count"] += 1
        await asyncio.sleep(0.2)
        return mcp.McpServerInfo(name=config.name, status="connected")

    monkeypatch.setattr(mcp.McpRuntime, "_connect_server", _very_slow_connect)

    try:
        first = await mcp.load_mcp_servers_async(tmp_path)
        assert first[0].status == "connecting"

        await asyncio.sleep(0.06)
        failed = await mcp.load_mcp_servers_async(tmp_path)
        assert failed[0].status == "failed"
        assert "timed out" in (failed[0].error or "").lower()
        assert connect_calls["count"] == 1

        await mcp.shutdown_mcp_runtime()

        await mcp.load_mcp_servers_async(tmp_path)
        await asyncio.sleep(0.02)
        reopened = await mcp.load_mcp_servers_async(tmp_path)
        assert reopened[0].status == "failed"
        assert "circuit breaker open" in (reopened[0].error or "").lower()
        assert connect_calls["count"] == 1
    finally:
        await mcp.shutdown_mcp_runtime()
        mcp._mcp_circuit_states.clear()
