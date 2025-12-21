"""LSP configuration loader and client manager."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from ripperdoc.utils.git_utils import get_git_root, is_git_repository
from ripperdoc.utils.log import get_logger


logger = get_logger()


def _float_env(name: str, default: str) -> float:
    value = os.getenv(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


DEFAULT_REQUEST_TIMEOUT = _float_env("RIPPERDOC_LSP_TIMEOUT", "20")

LANGUAGE_ID_BY_EXT: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascriptreact",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".json": "json",
    ".jsonc": "jsonc",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".lua": "lua",
    ".sql": "sql",
    ".sh": "shellscript",
    ".bash": "shellscript",
    ".zsh": "shellscript",
    ".ps1": "powershell",
    ".dart": "dart",
    ".vue": "vue",
    ".svelte": "svelte",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".fs": "fsharp",
    ".r": "r",
}


def guess_language_id(file_path: Path) -> Optional[str]:
    """Guess LSP language id from file extension."""
    return LANGUAGE_ID_BY_EXT.get(file_path.suffix.lower())


def file_path_to_uri(file_path: Path) -> str:
    """Convert a filesystem path to file:// URI."""
    return file_path.resolve().as_uri()


def uri_to_path(uri: str) -> Optional[Path]:
    """Convert a file:// URI to a filesystem path."""
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return Path(path)


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
        return {}
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load JSON", extra={"path": str(path)})
        return {}


def _normalize_command(raw_command: Any, raw_args: Any) -> Tuple[Optional[str], List[str]]:
    """Normalize LSP server command/args."""
    args: List[str] = []
    if isinstance(raw_args, list):
        args = [str(a) for a in raw_args]

    if isinstance(raw_command, list):
        tokens = [str(t) for t in raw_command if str(t)]
        if not tokens:
            return None, args
        return tokens[0], tokens[1:] + args

    if not isinstance(raw_command, str):
        return None, args

    command_str = raw_command.strip()
    if not command_str:
        return None, args

    if not args and (" " in command_str or "\t" in command_str):
        try:
            tokens = shlex.split(command_str)
        except ValueError:
            tokens = [command_str]
        if tokens:
            return tokens[0], tokens[1:]

    return command_str, args


def _normalize_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _normalize_extensions(raw: Any) -> List[str]:
    extensions = _normalize_str_list(raw)
    normalized: List[str] = []
    for ext in extensions:
        if not ext:
            continue
        ext = ext.strip()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext.lower())
    return normalized


def _normalize_extension_language_map(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, str] = {}
    for raw_ext, raw_lang in raw.items():
        ext = str(raw_ext).strip()
        lang = str(raw_lang).strip()
        if not ext or not lang:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        result[ext.lower()] = lang.lower()
    return result


@dataclass
class LspServerConfig:
    name: str
    command: Optional[str]
    args: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    extensions: List[str] = field(default_factory=list)
    extension_language_map: Dict[str, str] = field(default_factory=dict)
    root_patterns: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


def _parse_server_config(name: str, raw: Dict[str, Any]) -> Optional[LspServerConfig]:
    command, args = _normalize_command(raw.get("command"), raw.get("args"))
    if not command:
        return None

    languages = _normalize_str_list(
        raw.get("languages")
        or raw.get("languageIds")
        or raw.get("languageId")
        or raw.get("language")
    )
    extensions = _normalize_extensions(
        raw.get("extensions") or raw.get("fileExtensions") or raw.get("file_extensions")
    )
    extension_language_map = _normalize_extension_language_map(
        raw.get("extensionToLanguage") or raw.get("extension_to_language")
    )
    root_patterns = _normalize_str_list(
        raw.get("rootPatterns") or raw.get("root_patterns") or raw.get("rootPattern")
    )

    env = raw.get("env") if isinstance(raw.get("env"), dict) else {}
    env = {str(k): str(v) for k, v in env.items()} if env else {}

    initialization_options = (
        raw.get("initializationOptions") if isinstance(raw.get("initializationOptions"), dict) else {}
    )
    settings = raw.get("settings") if isinstance(raw.get("settings"), dict) else {}

    if extension_language_map:
        for ext in extension_language_map.keys():
            if ext not in extensions:
                extensions.append(ext)
        if not languages:
            languages = list({lang for lang in extension_language_map.values() if lang})

    if not languages and not extensions and not extension_language_map:
        name_hint = str(name).strip().lower()
        if name_hint:
            languages = [name_hint]

    return LspServerConfig(
        name=name,
        command=command,
        args=[str(a) for a in args] if args else [],
        languages=[lang.lower() for lang in languages],
        extensions=extensions,
        extension_language_map=extension_language_map,
        root_patterns=root_patterns,
        env=env,
        initialization_options=initialization_options or {},
        settings=settings or {},
    )


def _parse_servers(data: Dict[str, Any]) -> Dict[str, LspServerConfig]:
    servers: Dict[str, LspServerConfig] = {}
    for key in ("servers", "lspServers"):
        raw_servers = data.get(key)
        if not isinstance(raw_servers, dict):
            continue
        for name, raw in raw_servers.items():
            if not isinstance(raw, dict):
                continue
            server_name = str(name).strip()
            if not server_name:
                continue
            parsed = _parse_server_config(server_name, raw)
            if parsed:
                servers[server_name] = parsed
    if servers:
        return servers

    for name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        if not any(
            key in raw
            for key in (
                "command",
                "args",
                "extensionToLanguage",
                "extension_to_language",
                "languages",
                "languageIds",
                "languageId",
                "language",
                "extensions",
                "fileExtensions",
                "file_extensions",
            )
        ):
            continue
        server_name = str(name).strip()
        if not server_name:
            continue
        parsed = _parse_server_config(server_name, raw)
        if parsed:
            servers[server_name] = parsed
    return servers


def load_lsp_server_configs(project_path: Optional[Path] = None) -> Dict[str, LspServerConfig]:
    project_path = project_path or Path.cwd()
    candidates = [
        Path.home() / ".ripperdoc" / "lsp.json",
        Path.home() / ".lsp.json",
        project_path / ".ripperdoc" / "lsp.json",
        project_path / ".lsp.json",
    ]

    merged: Dict[str, LspServerConfig] = {}
    for path in candidates:
        data = _load_json_file(path)
        merged.update(_parse_servers(data))

    logger.debug(
        "[lsp] Loaded LSP server configs",
        extra={
            "project_path": str(project_path),
            "server_count": len(merged),
            "candidates": [str(path) for path in candidates],
        },
    )
    return merged


def _resolve_workspace_root(file_path: Path, root_patterns: List[str]) -> Path:
    if root_patterns:
        for parent in [file_path.parent] + list(file_path.parents):
            for pattern in root_patterns:
                if not pattern:
                    continue
                candidate = parent / pattern
                if any(ch in pattern for ch in "*?[]"):
                    matches = list(parent.glob(pattern))
                    if matches:
                        return parent
                    continue
                if candidate.exists():
                    return parent

    if is_git_repository(file_path.parent):
        git_root = get_git_root(file_path.parent)
        if git_root:
            return git_root

    return file_path.parent


class LspProtocolError(RuntimeError):
    """Protocol-level error from the LSP client."""


class LspRequestError(RuntimeError):
    """Error returned by LSP server for a request."""


class LspLaunchError(RuntimeError):
    """Error launching an LSP server process."""


@dataclass
class LspDocumentState:
    version: int
    text: str


class LspServer:
    """Manage a single LSP server process."""

    def __init__(self, config: LspServerConfig, workspace_root: Path) -> None:
        self.config = config
        self.workspace_root = workspace_root
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._pending: Dict[int, asyncio.Future[Any]] = {}
        self._next_id = 1
        self._send_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        self._open_docs: Dict[str, LspDocumentState] = {}
        self._workspace_folders = [
            {"uri": file_path_to_uri(self.workspace_root), "name": self.workspace_root.name}
        ]

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def _start_process(self) -> None:
        if self._process and self._process.returncode is None:
            return

        env = os.environ.copy()
        env.update(self.config.env)

        if not self.config.command:
            raise ValueError(f"LSP server '{self.config.name}' has no command configured")

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root),
                env=env,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise LspLaunchError(
                f"Failed to launch LSP server '{self.config.name}': {exc}"
            ) from exc

        self._reader_task = asyncio.create_task(self._read_messages())
        if self._process.stderr:
            self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self._process and self._process.stderr
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug(
                    "[lsp] stderr",
                    extra={"server": self.config.name, "line": line.decode(errors="replace").strip()},
                )
        except (asyncio.CancelledError, RuntimeError):
            return

    async def _read_messages(self) -> None:
        assert self._process and self._process.stdout
        try:
            while True:
                message = await self._read_message()
                if message is None:
                    break
                await self._handle_message(message)
        except (asyncio.CancelledError, RuntimeError, OSError):
            pass
        finally:
            self._closed = True
            self._fail_pending("LSP server disconnected")

    async def _read_message(self) -> Optional[Dict[str, Any]]:
        assert self._process and self._process.stdout
        headers: Dict[str, str] = {}
        while True:
            line = await self._process.stdout.readline()
            if not line:
                return None
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        length_str = headers.get("content-length")
        if not length_str:
            return None
        try:
            length = int(length_str)
        except ValueError:
            raise LspProtocolError(f"Invalid Content-Length header: {length_str}")

        body = await self._process.stdout.readexactly(length)
        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise LspProtocolError(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(payload, dict):
            raise LspProtocolError("LSP payload is not an object")
        return payload

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            request_id = message.get("id")
            if not isinstance(request_id, int):
                return
            future = self._pending.pop(request_id, None)
            if future:
                if "error" in message:
                    error = message.get("error")
                    future.set_exception(LspRequestError(str(error)))
                else:
                    future.set_result(message.get("result"))
            return

        if "id" in message and "method" in message:
            response = await self._handle_server_request(message)
            if response is not None:
                await self._send_payload(response)
            return

        # Notifications are ignored.

    async def _handle_server_request(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        result: Any = None
        if method == "workspace/configuration":
            items = params.get("items") if isinstance(params, dict) else None
            if isinstance(items, list):
                result = [self.config.settings or {} for _ in items]
            else:
                result = [self.config.settings or {}]
        elif method == "workspace/workspaceFolders":
            result = self._workspace_folders
        elif method in ("client/registerCapability", "client/unregisterCapability"):
            result = None
        elif method == "workspace/applyEdit":
            result = {"applied": False}
        elif method == "window/showMessageRequest":
            result = None
        else:
            result = None

        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    async def _send_payload(self, payload: Dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise LspProtocolError("LSP server process is not running")

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        async with self._send_lock:
            try:
                self._process.stdin.write(header + body)
                await self._process.stdin.drain()
            except (ConnectionError, BrokenPipeError, OSError) as exc:
                raise LspProtocolError(f"Failed to write to LSP server: {exc}") from exc

    async def _send_request(self, method: str, params: Optional[Dict[str, Any]]) -> Any:
        await self._start_process()
        request_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future

        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        await self._send_payload(payload)

        try:
            return await asyncio.wait_for(future, timeout=DEFAULT_REQUEST_TIMEOUT)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise LspProtocolError(f"LSP request timed out: {method}") from exc

    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        await self._start_process()
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._send_payload(payload)

    def _fail_pending(self, message: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(LspProtocolError(message))
        self._pending.clear()

    async def ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            params = {
                "processId": os.getpid(),
                "rootUri": file_path_to_uri(self.workspace_root),
                "workspaceFolders": self._workspace_folders,
                "capabilities": {
                    "textDocument": {
                        "definition": {"dynamicRegistration": False},
                        "references": {"dynamicRegistration": False},
                        "hover": {"dynamicRegistration": False},
                        "documentSymbol": {"dynamicRegistration": False},
                        "implementation": {"dynamicRegistration": False},
                    },
                    "workspace": {"symbol": {"dynamicRegistration": False}},
                },
                "initializationOptions": self.config.initialization_options or {},
            }

            await self._send_request("initialize", params)
            await self._send_notification("initialized", {})
            if self.config.settings:
                await self._send_notification(
                    "workspace/didChangeConfiguration", {"settings": self.config.settings}
                )
            self._initialized = True

    async def ensure_document_open(self, file_path: Path, text: str, language_id: str) -> None:
        uri = file_path_to_uri(file_path)
        state = self._open_docs.get(uri)

        if state is None:
            self._open_docs[uri] = LspDocumentState(version=1, text=text)
            await self._send_notification(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": language_id,
                        "version": 1,
                        "text": text,
                    }
                },
            )
            return

        if state.text != text:
            state.version += 1
            state.text = text
            await self._send_notification(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": state.version},
                    "contentChanges": [{"text": text}],
                },
            )

    async def request(self, method: str, params: Optional[Dict[str, Any]]) -> Any:
        return await self._send_request(method, params)

    async def notify(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        await self._send_notification(method, params)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process and self._process.returncode is None:
                try:
                    await self._send_request("shutdown", None)
                    await self._send_notification("exit", None)
                except (LspProtocolError, LspRequestError):
                    pass
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    self._process.kill()
        finally:
            self._fail_pending("LSP server shut down")
            if self._reader_task:
                self._reader_task.cancel()
            if self._stderr_task:
                self._stderr_task.cancel()


class LspManager:
    """Track configured LSP servers and route requests."""

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.configs = load_lsp_server_configs(project_path)
        self._servers: Dict[Tuple[str, Path], LspServer] = {}
        self._closed = False

    def _match_config(self, file_path: Path) -> Optional[LspServerConfig]:
        if not self.configs:
            return None
        ext = file_path.suffix.lower()
        language_id = guess_language_id(file_path)

        matches_by_extension: List[LspServerConfig] = []
        matches_by_language: List[LspServerConfig] = []

        for config in self.configs.values():
            if config.extension_language_map and ext in config.extension_language_map:
                matches_by_extension.append(config)
            elif config.extensions and ext in config.extensions:
                matches_by_extension.append(config)
            elif config.languages and language_id and language_id in config.languages:
                matches_by_language.append(config)

        if matches_by_extension:
            return matches_by_extension[0]
        if matches_by_language:
            return matches_by_language[0]
        if len(self.configs) == 1:
            return next(iter(self.configs.values()))
        return None

    def _language_id_for(self, file_path: Path, config: LspServerConfig) -> str:
        ext = file_path.suffix.lower()
        if config.extension_language_map and ext in config.extension_language_map:
            return config.extension_language_map[ext]

        guessed = guess_language_id(file_path)
        if guessed:
            return guessed
        if config.languages:
            return config.languages[0]
        return "plaintext"

    async def server_for_path(
        self, file_path: Path
    ) -> Optional[Tuple[LspServer, LspServerConfig, str]]:
        if self._closed:
            return None

        config = self._match_config(file_path)
        if not config:
            return None

        workspace_root = _resolve_workspace_root(file_path, config.root_patterns)
        key = (config.name, workspace_root)
        server = self._servers.get(key)
        if not server or server.is_closed:
            server = LspServer(config, workspace_root)
            self._servers[key] = server

        language_id = self._language_id_for(file_path, config)
        return server, config, language_id

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        servers = list(self._servers.values())
        self._servers.clear()
        for server in servers:
            try:
                await server.shutdown()
            except (RuntimeError, OSError, LspProtocolError):
                logger.warning(
                    "[lsp] Failed to shutdown LSP server",
                    extra={"server": getattr(server.config, "name", "unknown")},
                )


_runtime_var: contextvars.ContextVar[Optional[LspManager]] = contextvars.ContextVar(
    "ripperdoc_lsp_runtime", default=None
)
_global_runtime: Optional[LspManager] = None


def get_existing_lsp_manager() -> Optional[LspManager]:
    runtime = _runtime_var.get()
    return runtime or _global_runtime


async def ensure_lsp_manager(project_path: Optional[Path] = None) -> LspManager:
    runtime = get_existing_lsp_manager()
    if runtime and not runtime._closed:
        return runtime

    project_path = project_path or Path.cwd()
    runtime = LspManager(project_path)
    _runtime_var.set(runtime)
    global _global_runtime
    _global_runtime = runtime
    return runtime


async def shutdown_lsp_manager() -> None:
    runtime = get_existing_lsp_manager()
    if not runtime:
        return
    try:
        await runtime.shutdown()
    finally:
        _runtime_var.set(None)
        global _global_runtime
        _global_runtime = None


__all__ = [
    "LspManager",
    "LspServer",
    "LspServerConfig",
    "LspLaunchError",
    "LspProtocolError",
    "LspRequestError",
    "ensure_lsp_manager",
    "get_existing_lsp_manager",
    "shutdown_lsp_manager",
    "guess_language_id",
    "file_path_to_uri",
    "uri_to_path",
    "load_lsp_server_configs",
]
