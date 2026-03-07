import asyncio

import pytest

from ripperdoc.tools import dynamic_mcp_tool
from ripperdoc.utils import lsp, mcp


def test_sync_dynamic_mcp_loader_never_boots_runtime(monkeypatch):
    monkeypatch.setattr(dynamic_mcp_tool, "get_existing_mcp_runtime", lambda **_: None)
    monkeypatch.setattr(dynamic_mcp_tool.asyncio, "run", lambda _coro: (_ for _ in ()).throw(AssertionError("asyncio.run should not be used")))

    assert dynamic_mcp_tool.load_dynamic_mcp_tools_sync() == []


def test_sync_mcp_server_loader_shuts_down_runtime(monkeypatch):
    shutdown_calls = {"count": 0}

    async def _fake_load(*_args, **_kwargs):
        return ["server-a"]

    async def _fake_shutdown():
        shutdown_calls["count"] += 1

    monkeypatch.setattr(mcp, "load_mcp_servers_async", _fake_load)
    monkeypatch.setattr(mcp, "shutdown_mcp_runtime", _fake_shutdown)

    assert mcp.load_mcp_servers() == ["server-a"]
    assert shutdown_calls["count"] == 1


@pytest.mark.asyncio
async def test_shutdown_mcp_runtime_discards_foreign_global_runtime(monkeypatch, tmp_path):
    await mcp.shutdown_mcp_runtime()

    class ForeignRuntime:
        _closed = False

        def belongs_to_loop(self, _loop):
            return False

        async def aclose(self):
            raise AssertionError("foreign runtime must not be awaited")

    monkeypatch.setattr(mcp, "_global_runtime", ForeignRuntime())

    try:
        await mcp.shutdown_mcp_runtime()
        assert mcp._global_runtime is None

        runtime = await mcp.ensure_mcp_runtime(tmp_path)
        assert runtime.project_path == tmp_path
        assert runtime.belongs_to_loop(asyncio.get_running_loop())
    finally:
        await mcp.shutdown_mcp_runtime()


@pytest.mark.asyncio
async def test_shutdown_lsp_manager_discards_foreign_global_runtime(monkeypatch, tmp_path):
    await lsp.shutdown_lsp_manager()

    class ForeignManager:
        _closed = False

        def belongs_to_loop(self, _loop):
            return False

        async def shutdown(self):
            raise AssertionError("foreign runtime must not be awaited")

    monkeypatch.setattr(lsp, "_global_runtime", ForeignManager())

    try:
        await lsp.shutdown_lsp_manager()
        assert lsp._global_runtime is None

        manager = await lsp.ensure_lsp_manager(tmp_path)
        assert manager.project_path == tmp_path
        assert manager.belongs_to_loop(asyncio.get_running_loop())
    finally:
        await lsp.shutdown_lsp_manager()
