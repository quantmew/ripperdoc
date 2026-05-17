# MCP Implementation Alignment Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement.

**Goal:** Restructure Ripperdoc's MCP implementation to strictly mirror Claude Code's architecture.

**Architecture:** Split the monolithic `utils/mcp/__init__.py` (1550 lines) into `services/mcp/` package with modules mirroring the reference's `services/mcp/*.ts`. Restructure `tools/mcp/` from a single `__init__.py` into individual tool directories matching `tools/*Mcp*/`. Add missing modules (normalization, envExpansion, headersHelper, oauthPort). Deduplicate `mcp/_tool.py` (404 lines dead code).

**Tech Stack:** Python 3.10+, asyncio, pydantic

---

### Task 1: Create `ripperdoc/services/mcp/` package structure

**Files:**
- Create: `ripperdoc/services/mcp/__init__.py`
- Create: `ripperdoc/services/mcp/types.py`
- Create: `ripperdoc/services/mcp/normalization.py`
- Create: `ripperdoc/services/mcp/mcp_string_utils.py`
- Create: `ripperdoc/services/mcp/env_expansion.py`
- Create: `ripperdoc/services/mcp/config.py`
- Create: `ripperdoc/services/mcp/client.py`
- Create: `ripperdoc/services/mcp/utils.py`
- Modify: `ripperdoc/utils/mcp/__init__.py` → re-export shim

**Step 1: Create `__init__.py`**

```python
"""MCP service layer — connection management, config loading, auth."""
```

**Step 2: Create `types.py`**

Mirror `mcp/types.ts`:
- `ConfigScope` enum (local, user, project, dynamic, enterprise, claudeai, managed)
- `TransportType` enum (stdio, sse, sse-ide, http, ws, sdk)
- `McpToolInfo` dataclass
- `McpResourceInfo` dataclass
- `McpServerInfo` dataclass (with scope, headers, instructions, capabilities, etc.)
- `StdioServerConfig`, `SSEServerConfig`, `HTTPServerConfig`, etc. (typed configs)

**Step 3: Create `normalization.py`**

Mirror `mcp/normalization.ts`:
- `normalize_name_for_mcp(name: str) -> str`

**Step 4: Create `mcp_string_utils.py`**

Mirror `mcp/mcpStringUtils.ts`:
- `mcp_info_from_string(tool_string: str) -> Optional[dict]`
- `build_mcp_tool_name(server_name: str, tool_name: str) -> str`
- `get_mcp_prefix(server_name: str) -> str`

**Step 5: Create `env_expansion.py`**

Mirror `mcp/envExpansion.ts`:
- `expand_env_vars_in_string(value: str) -> ExpandedResult`

**Step 6: Create `config.py`**

Mirror `mcp/config.ts`:
- `load_json_file(path: Path) -> Dict`
- `normalize_command(command, args) -> tuple`
- `parse_server(name, raw) -> McpServerInfo`
- `parse_servers(data) -> Dict[str, McpServerInfo]`
- `load_server_configs(project_path) -> Dict[str, McpServerInfo]`
- `load_mcp_server_configs(project_path) -> Dict[str, McpServerInfo]`
- `parse_mcp_server_configs(raw) -> Dict[str, McpServerInfo]`
- `project_scope_key(project_path) -> str`
- `set_mcp_runtime_overrides(...)`
- `clear_mcp_runtime_overrides(...)`

**Step 7: Create `client.py`**

Mirror `mcp/client.ts` (core connection logic):
- `_SdkMcpSession` (minimal SDK client)
- `McpRuntime` class — the main connection manager
- `McpCircuitState` and circuit breaker logic
- `connect`, `_connect_server`, `_connect_server_with_policy`, `aclose`
- `server_snapshot()` method
- stderr log management

**Step 8: Create `utils.py`**

Mirror `mcp/utils.ts`:
- `format_mcp_instructions(servers) -> str`
- `estimate_mcp_tokens(servers) -> int`
- `find_mcp_resource(servers, server_name, uri) -> Optional[McpResourceInfo]`

**Step 9: Update `utils/mcp/__init__.py` → re-export shim**

Keep backward compatibility by re-exporting all public symbols from `services/mcp/`:
```python
from ripperdoc.services.mcp.types import (
    McpToolInfo, McpResourceInfo, McpServerInfo, ...
)
from ripperdoc.services.mcp.config import (
    load_mcp_server_configs, parse_mcp_server_configs, ...
)
from ripperdoc.services.mcp.client import (
    McpRuntime, ensure_mcp_runtime, shutdown_mcp_runtime, ...
)
from ripperdoc.services.mcp.utils import (
    format_mcp_instructions, estimate_mcp_tokens, find_mcp_resource, ...
)
from ripperdoc.services.mcp.mcp_string_utils import (
    mcp_info_from_string, build_mcp_tool_name, get_mcp_prefix, ...
)
from ripperdoc.services.mcp.normalization import (
    normalize_name_for_mcp, ...
)
from ripperdoc.services.mcp.env_expansion import (
    expand_env_vars_in_string, ...
)
```

### Task 2: Separate MCP tools into individual directories

**Files:**
- Create: `ripperdoc/tools/mcp_tool/` (for MCPTool — dynamic tool invocation)
- Create: `ripperdoc/tools/list_mcp_servers_tool/`
- Create: `ripperdoc/tools/list_mcp_resources_tool/`
- Create: `ripperdoc/tools/read_mcp_resource_tool/`
- Keep: `ripperdoc/tools/mcp/dynamic_mcp.py` (DynamicMcpTool wrapper)
- Keep: `ripperdoc/tools/mcp/mcp_output_limits.py`
- Delete: `ripperdoc/tools/mcp/_tool.py` (dead code, 404 lines)
- Update: `ripperdoc/tools/mcp/__init__.py` → re-export shim

### Task 3: Clean up and verify

- Remove the dead `_tool.py`
- Run tests to verify nothing broke
- Fix any import issues
