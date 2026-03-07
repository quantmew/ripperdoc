"""Session initialization for stdio protocol handler."""

from __future__ import annotations

import logging
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from ripperdoc import __version__
from ripperdoc.core.agents import load_agent_definitions
from ripperdoc.cli.commands import list_custom_commands, list_slash_commands
from ripperdoc.core import tool_defaults as tool_defaults_module
from ripperdoc.core.config import (
    ProtocolType,
    get_effective_model_profile,
    get_project_config,
    get_project_local_config,
)
from ripperdoc.core.oauth import get_oauth_token
from ripperdoc.core.session_agents import (
    normalize_agent_name,
    parse_session_agents,
    resolve_session_agent_prompt,
)
from ripperdoc.core.output_styles import load_all_output_styles, resolve_output_style
from ripperdoc.core.plugins import discover_plugins, set_runtime_plugin_dirs
from ripperdoc.core.system_prompt_overrides import select_base_system_prompt
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.state import bind_pending_message_queue, bind_hook_scopes
from ripperdoc.core.query import QueryContext
from ripperdoc.protocol.models import (
    DEFAULT_PROTOCOL_VERSION,
    JsonRpcErrorCodes,
    InitializeParams,
    InitializeResult,
    InitializeServerInfo,
    ProtocolCapabilities,
    model_to_dict,
)
from .error_codes import resolve_protocol_request_error_code
from ripperdoc.tools.dynamic_mcp_tool import (
    load_dynamic_mcp_tools_async,
    merge_tools_with_dynamic,
)
from ripperdoc.utils.asyncio_compat import asyncio_timeout
from ripperdoc.utils.mcp import format_mcp_instructions, load_mcp_servers_async
from ripperdoc.utils.mcp import (
    clear_sdk_mcp_request_sender,
    load_mcp_server_configs,
    parse_mcp_config_option,
    set_mcp_runtime_overrides,
    set_sdk_mcp_request_sender,
)
from ripperdoc.utils.sessions.session_history import SessionHistory
from ripperdoc.utils.collaboration.tasks import set_runtime_task_scope
from ripperdoc.utils.filesystem.working_directories import coerce_directory_list, normalize_directory_inputs

from .timeouts import STDIO_HOOK_TIMEOUT_SEC

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")

# Keep these module-level aliases patchable for tests and other callers that
# inject deterministic tool lists during stdio initialization.
get_default_tools = tool_defaults_module.get_default_tools
get_default_tools_async = tool_defaults_module.get_default_tools_async
_DEFAULT_SYNC_TOOL_LOADER = tool_defaults_module.get_default_tools


class StdioSessionMixin:
    _pre_plan_mode: str | None
    _query_context: QueryContext | None
    _session_id: str | None
    _custom_system_prompt: str | None
    _append_system_prompt: str | None
    _skill_instructions: str | None
    _output_style: str
    _output_language: str
    _session_start_time: float | None
    _session_history: SessionHistory | None
    _session_additional_working_dirs: set[str]
    _fallback_model: str | None
    _max_budget_usd: float | None
    _json_schema: dict[str, Any] | None
    _session_agent_name: str | None
    _session_agents: dict[str, dict[str, str]]
    _session_agent_prompt: str | None
    _active_agent_names: list[str]
    _enabled_skill_names: list[str]
    _plugin_payloads: list[dict[str, str]]
    _disable_slash_commands: bool
    _replay_user_messages: bool
    _permission_mode: str
    _init_stream_message_sent: bool
    _sdk_betas: list[str]

    async def _load_mcp_servers_for_initialize(self) -> list[Any]:
        try:
            return await load_mcp_servers_async(self._project_path, wait_for_connections=True)
        except TypeError as exc:
            if "wait_for_connections" not in str(exc):
                raise
            return await load_mcp_servers_async(self._project_path)

    async def _load_dynamic_mcp_tools_for_initialize(self) -> list[Any]:
        try:
            return await load_dynamic_mcp_tools_async(
                self._project_path,
                wait_for_connections=True,
            )
        except TypeError as exc:
            if "wait_for_connections" not in str(exc):
                raise
            return await load_dynamic_mcp_tools_async(self._project_path)

    async def _get_initialize_tools(self) -> list[Any]:
        """Resolve the initialize-time tool list with a patchable sync fallback."""
        if get_default_tools is not _DEFAULT_SYNC_TOOL_LOADER:
            return list(get_default_tools())
        return await get_default_tools_async(project_path=self._project_path)

    async def _send_sdk_mcp_message(self, server_name: str, message: dict[str, Any]) -> dict[str, Any]:
        """Bridge SDK-backed MCP traffic over stdio control requests."""
        response = await self._send_control_request(
            subtype="mcp_message",
            request={
                "server_name": server_name,
                "message": message,
            },
            timeout=STDIO_HOOK_TIMEOUT_SEC,
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid SDK MCP response for server '{server_name}'")
        return response

    def _build_sdk_init_stream_message(
        self,
        *,
        tools: list[Any],
        servers: list[Any],
    ) -> dict[str, Any]:
        slash_commands = [
            str(getattr(cmd, "name", "")).strip()
            for cmd in list_slash_commands()
            if str(getattr(cmd, "name", "")).strip()
        ]
        slash_commands.extend(
            str(getattr(cmd, "name", "")).strip()
            for cmd in list_custom_commands(self._project_path)
            if str(getattr(cmd, "name", "")).strip()
        )

        mcp_servers: list[dict[str, str]] = []
        for server in servers:
            name = str(getattr(server, "name", "")).strip()
            if not name:
                continue
            mcp_servers.append(
                {
                    "name": name,
                    "status": str(getattr(server, "status", "unknown") or "unknown"),
                }
            )

        model_name = "main"
        if self._query_context is not None:
            model_name = str(getattr(self._query_context, "model", model_name) or model_name)

        api_key_source = "none"
        if os.getenv("RIPPERDOC_AUTH_TOKEN"):
            api_key_source = "RIPPERDOC_AUTH_TOKEN"
        elif os.getenv("RIPPERDOC_API_KEY"):
            api_key_source = "RIPPERDOC_API_KEY"
        else:
            model_profile = get_effective_model_profile(model_name)
            if model_profile is not None and getattr(model_profile, "protocol", None) == ProtocolType.OAUTH:
                token_name = str(getattr(model_profile, "oauth_token_name", "") or "").strip()
                if token_name and get_oauth_token(token_name):
                    api_key_source = token_name

        payload = {
            "type": "system",
            "subtype": "init",
            "cwd": str(self._project_path),
            "session_id": self._session_id or "",
            "tools": [
                str(getattr(tool, "name", "")).strip()
                for tool in tools
                if str(getattr(tool, "name", "")).strip()
            ],
            "mcp_servers": mcp_servers,
            "model": model_name,
            "permissionMode": self._permission_mode or "default",
            "slash_commands": slash_commands,
            "apiKeySource": api_key_source,
            "ripperdoc_version": __version__,
            "output_style": self._output_style,
            "agents": list(self._active_agent_names),
            "skills": list(self._enabled_skill_names),
            "plugins": list(self._plugin_payloads),
            "uuid": str(uuid.uuid4()),
            "fast_mode_state": "off",
        }
        if self._sdk_betas:
            payload["betas"] = list(self._sdk_betas)
        return payload

    async def _handle_initialize(self, request: dict[str, Any], request_id: str) -> None:
        """Handle initialize request from SDK.

        Args:
            request: The initialize request data.
            request_id: The request ID.
        """
        if self._initialized:
            await self._fail_initialize_request(
                request_id,
                "Already initialized",
                JsonRpcErrorCodes.InvalidRequest,
            )
            return

        try:
            initialize_request = self._coerce_initialize_request(request)
            try:
                initialize_params = InitializeParams.model_validate(initialize_request)
            except ValidationError as exc:
                await self._fail_initialize_request(
                    request_id,
                    f"Invalid initialize request: {exc}",
                    resolve_protocol_request_error_code(
                        exc,
                        default=JsonRpcErrorCodes.InvalidParams,
                    ),
                )
                return

            requested_protocol_version = initialize_params.protocolVersion
            protocol_version = (
                requested_protocol_version
                if requested_protocol_version == DEFAULT_PROTOCOL_VERSION
                else DEFAULT_PROTOCOL_VERSION
            )
            if requested_protocol_version != protocol_version:
                logger.info(
                    "[stdio] Unsupported protocol version %s, falling back to %s",
                    requested_protocol_version,
                    protocol_version,
                )

            request_options: dict[str, Any] = {}
            request_body = initialize_request
            meta = request_body.get("_meta")
            if isinstance(meta, dict):
                ripperdoc_options = meta.get("ripperdoc_options")
                if isinstance(ripperdoc_options, dict):
                    request_options.update(ripperdoc_options)
            legacy_options = request.get("options")
            if isinstance(legacy_options, dict):
                request_options.update(legacy_options)

            hooks = request_body.get("hooks")
            if isinstance(hooks, dict):
                request_options.setdefault("hooks", hooks)

            agents = request_body.get("agents")
            if isinstance(agents, dict):
                request_options.setdefault("agents", agents)

            options = {**self._default_options, **request_options}
            self._session_id = options.get("session_id") or str(uuid.uuid4())
            self._custom_system_prompt = options.get("system_prompt")
            self._append_system_prompt = options.get("append_system_prompt")
            raw_sdk_url = options.get("sdk_url")
            self._sdk_url = (
                raw_sdk_url.strip()
                if isinstance(raw_sdk_url, str) and raw_sdk_url.strip()
                else None
            )
            if self._sdk_url and (
                self._input_format != "stream-json"
                or self._output_format != "stream-json"
            ):
                await self._fail_initialize_request(
                    request_id,
                    "Error: --sdk-url requires both --input-format=stream-json and --output-format=stream-json.",
                    JsonRpcErrorCodes.InvalidParams,
                )
                return
            self._replay_user_messages = self._coerce_bool_option(
                options.get("replay_user_messages"),
                getattr(self, "_replay_user_messages", False),
            )
            raw_betas = options.get("betas")
            if isinstance(raw_betas, str):
                self._sdk_betas = [item.strip() for item in raw_betas.split(",") if item.strip()]
            elif isinstance(raw_betas, (list, tuple, set)):
                self._sdk_betas = [str(item).strip() for item in raw_betas if str(item).strip()]
            else:
                self._sdk_betas = []
            if self._replay_user_messages and (
                self._input_format != "stream-json"
                or self._output_format != "stream-json"
            ):
                await self._fail_initialize_request(
                    request_id,
                    "Error: --replay-user-messages requires both --input-format=stream-json and --output-format=stream-json.",
                    JsonRpcErrorCodes.InvalidParams,
                )
                return
            raw_max_turns = options.get("max_turns")
            raw_max_thinking_tokens = options.get("max_thinking_tokens")
            max_turns: int | None = None
            max_thinking_tokens = 0
            if raw_max_turns is not None:
                try:
                    max_turns = int(raw_max_turns)
                except (TypeError, ValueError):
                    logger.warning(
                        "[stdio] Invalid max_turns %r; ignoring",
                        raw_max_turns,
                    )
            if raw_max_thinking_tokens is not None:
                try:
                    max_thinking_tokens = max(0, int(raw_max_thinking_tokens))
                except (TypeError, ValueError):
                    logger.warning(
                        "[stdio] Invalid max_thinking_tokens %r; defaulting to 0",
                        raw_max_thinking_tokens,
                    )

            fallback_model = options.get("fallback_model")
            self._fallback_model = (
                str(fallback_model).strip()
                if isinstance(fallback_model, str) and fallback_model.strip()
                else None
            )

            raw_max_budget = options.get("max_budget_usd")
            self._max_budget_usd = None
            if raw_max_budget is not None:
                try:
                    parsed_budget = float(raw_max_budget)
                    if parsed_budget > 0:
                        self._max_budget_usd = parsed_budget
                except (TypeError, ValueError):
                    logger.warning("[stdio] Invalid max_budget_usd %r; ignoring", raw_max_budget)

            raw_json_schema = options.get("json_schema")
            self._json_schema = None
            if isinstance(raw_json_schema, dict):
                self._json_schema = raw_json_schema
            elif isinstance(raw_json_schema, str) and raw_json_schema.strip():
                try:
                    parsed_schema = json.loads(raw_json_schema)
                except json.JSONDecodeError:
                    logger.warning(
                        "[stdio] Invalid json_schema payload; expected valid JSON object"
                    )
                else:
                    if isinstance(parsed_schema, dict):
                        self._json_schema = parsed_schema
                    else:
                        logger.warning("[stdio] json_schema must be a JSON object; ignoring")

            ignored_option_keys = [
                "permission_prompt_tool",
                "include_partial_messages",
                "fork_session",
                "setting_sources",
            ]
            ignored_in_use = [
                key
            for key in ignored_option_keys
            if options.get(key) not in (None, "", [], (), False)
            ]
            if ignored_in_use:
                logger.info(
                    "[stdio] Accepted compatibility options without runtime effect: %s",
                    ", ".join(sorted(ignored_in_use)),
                )

            # Setup working directory
            cwd = options.get("cwd")
            if cwd:
                self._project_path = Path(cwd)
            else:
                self._project_path = Path.cwd()
            plugin_dirs_option = options.get("plugin_dirs")
            if isinstance(plugin_dirs_option, list):
                set_runtime_plugin_dirs(
                    [str(item) for item in plugin_dirs_option if item],
                    base_dir=self._project_path,
                )
            elif isinstance(plugin_dirs_option, tuple):
                set_runtime_plugin_dirs(
                    [str(item) for item in plugin_dirs_option if item],
                    base_dir=self._project_path,
                )
            else:
                set_runtime_plugin_dirs([])
            set_runtime_task_scope(
                session_id=self._session_id,
                project_root=self._project_path,
            )
            set_sdk_mcp_request_sender(self._send_sdk_mcp_message)

            raw_additional_dirs = options.get("additional_directories")
            additional_dirs = coerce_directory_list(raw_additional_dirs)
            normalized_dirs, dir_errors = normalize_directory_inputs(
                additional_dirs,
                base_dir=self._project_path,
                require_exists=True,
            )
            if dir_errors:
                logger.warning(
                    "[stdio] Ignoring invalid additional directories",
                    extra={"errors": dir_errors},
                )
            self._session_additional_working_dirs = set(normalized_dirs)

            # Initialize project config
            get_project_config(self._project_path)
            project_local_config = get_project_local_config(self._project_path)
            configured_output_style = (
                getattr(project_local_config, "output_style", "default") or "default"
            )
            requested_output_style = options.get("output_style") or configured_output_style
            resolved_output_style, _ = resolve_output_style(
                str(requested_output_style),
                project_path=self._project_path,
            )
            self._output_style = resolved_output_style.key
            configured_output_language = (
                getattr(project_local_config, "output_language", "auto") or "auto"
            )
            requested_output_language = options.get("output_language") or configured_output_language
            self._output_language = str(requested_output_language or "auto").strip() or "auto"

            try:
                self._session_agents = parse_session_agents(options.get("agents"), source="agents")
                self._session_agent_name = normalize_agent_name(options.get("agent"), source="agent")
                self._session_agent_prompt = resolve_session_agent_prompt(
                    self._session_agent_name,
                    self._session_agents,
                    source="agent",
                )
                self._custom_system_prompt = select_base_system_prompt(
                    agent_system_prompt=self._session_agent_prompt,
                    custom_system_prompt=self._custom_system_prompt,
                )
            except ValueError as exc:
                await self._fail_initialize_request(
                    request_id,
                    str(exc),
                    JsonRpcErrorCodes.InvalidParams,
                )
                return
            self._disable_slash_commands = self._coerce_bool_option(
                options.get("disable_slash_commands"),
                False,
            )
            self._session_persistence_enabled = options.get("session_persistence", True) is not False

            # Parse tool options
            self._allowed_tools = self._normalize_tool_list(options.get("allowed_tools"))
            self._disallowed_tools = self._normalize_tool_list(options.get("disallowed_tools"))
            self._tools_list = self._normalize_tool_list(options.get("tools"))
            self._tools_preset = options.get("tools_preset")  # "default" or None

            # Get the tool list (apply SDK filters)
            tools = await self._get_initialize_tools()
            if self._disable_slash_commands:
                tools = [tool for tool in tools if getattr(tool, "name", None) != "Skill"]
            tools = self._apply_tool_filters(
                tools,
                allowed_tools=self._allowed_tools,
                disallowed_tools=self._disallowed_tools,
                tools_list=self._tools_list,
            )

            # Parse permission mode
            permission_mode = self._normalize_permission_mode(
                options.get("permission_mode", "default")
            )
            self._permission_mode = permission_mode
            yolo_mode = permission_mode == "bypassPermissions"
            self._pre_plan_mode = None
            self._clear_context_after_turn = False
            self._sdk_can_use_tool_enabled = self._coerce_bool_option(
                options.get("sdk_can_use_tool"),
                False,
            )

            # Setup model
            model = options.get("model") or "main"

            # Validate whether the model configuration is valid.
            model_profile = get_effective_model_profile(model)
            if model_profile is None and self._fallback_model:
                fallback_profile = get_effective_model_profile(self._fallback_model)
                if fallback_profile is not None:
                    logger.warning(
                        "[stdio] Falling back to configured fallback_model '%s' (requested: '%s')",
                        self._fallback_model,
                        model,
                    )
                    model = self._fallback_model
                    model_profile = fallback_profile
            if model_profile is None:
                error_msg = (
                    f"No valid model configuration found for '{model}'. "
                    "Please set RIPPERDOC_MODEL/RIPPERDOC_BASE_URL/RIPPERDOC_PROTOCOL "
                    "environment variables or complete onboarding."
                )
                logger.error(f"[stdio] {error_msg}")
                await self._fail_initialize_request(
                    request_id,
                    error_msg,
                    JsonRpcErrorCodes.InvalidParams,
                )
                return

            # Create query context
            self._query_context = QueryContext(
                tools=tools,
                yolo_mode=yolo_mode,
                verbose=options.get("verbose", False),
                model=model,
                max_thinking_tokens=max_thinking_tokens,
                max_turns=max_turns,
                permission_mode=permission_mode,
                pre_plan_mode=self._pre_plan_mode,
                on_enter_plan_mode=self._enter_plan_mode,
                on_exit_plan_mode=self._on_exit_plan_mode,
                working_directory=str(self._project_path),
            )

            # Initialize hook manager
            hook_manager.set_project_dir(self._project_path)
            hook_manager.set_session_id(self._session_id)
            hook_manager.set_llm_callback(build_hook_llm_callback())
            self._session_history = SessionHistory(
                self._project_path,
                self._session_id or str(uuid.uuid4()),
                session_persistence=self._session_persistence_enabled,
            )
            hook_manager.set_transcript_path(str(self._session_history.path))

            # Configure SDK-provided hooks (if any)
            hooks = request.get("hooks", None)
            if hooks is None:
                hooks = options.get("hooks", {})
            self._configure_sdk_hooks(hooks)

            # Run SessionStart hooks during initialize (once per session)
            queue = self._query_context.pending_message_queue if self._query_context else None
            hook_scopes = self._query_context.hook_scopes if self._query_context else []
            with bind_pending_message_queue(queue), bind_hook_scopes(hook_scopes):
                try:
                    async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                        session_start_result = await hook_manager.run_session_start_async("startup")
                    self._session_hook_messages = self._collect_hook_context_messages(
                        session_start_result, "SessionStart"
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[stdio] Session start hook timed out after {STDIO_HOOK_TIMEOUT_SEC}s"
                    )
                except Exception as e:
                    logger.warning(f"[stdio] Session start hook failed: {e}")
                finally:
                    self._session_started = True
                    if self._session_start_time is None:
                        self._session_start_time = time.time()
                    self._session_end_sent = False

            raw_mcp_config = options.get("mcp_config")
            if raw_mcp_config not in (None, ""):
                base_configs = load_mcp_server_configs(self._project_path)
                cli_configs = parse_mcp_config_option(
                    raw_mcp_config,
                    base_dir=self._project_path,
                )
                merged_configs = {**base_configs, **cli_configs}
                self._mcp_server_overrides = self._clone_mcp_config_map(merged_configs)
                set_mcp_runtime_overrides(
                    self._project_path,
                    servers=self._mcp_server_overrides,
                    disabled=self._mcp_disabled_servers or None,
                )

            # Load MCP servers and dynamic tools
            servers, dynamic_tools = await asyncio.gather(
                self._load_mcp_servers_for_initialize(),
                self._load_dynamic_mcp_tools_for_initialize(),
            )
            if dynamic_tools:
                tools = merge_tools_with_dynamic(tools, dynamic_tools)
                tools = self._apply_tool_filters(
                    tools,
                    allowed_tools=self._allowed_tools,
                    disallowed_tools=self._disallowed_tools,
                    tools_list=self._tools_list,
                )
                self._query_context.tools = tools

            mcp_instructions = format_mcp_instructions(servers)

            # Build system prompt components
            from ripperdoc.core.skills import (
                build_skill_summary,
                filter_enabled_skills,
                load_all_skills,
            )

            enabled_skills = []
            if self._disable_slash_commands:
                self._skill_instructions = ""
            else:
                skill_result = load_all_skills(self._project_path)
                enabled_skills = filter_enabled_skills(
                    skill_result.skills, project_path=self._project_path
                )
                self._skill_instructions = build_skill_summary(enabled_skills)
            self._enabled_skill_names = sorted(
                str(getattr(skill, "name", "")).strip()
                for skill in enabled_skills
                if str(getattr(skill, "name", "")).strip()
            )

            agent_result = load_agent_definitions(project_path=self._project_path)
            plugin_result = discover_plugins(project_path=self._project_path)
            self._active_agent_names = sorted(
                str(getattr(agent, "agent_type", "")).strip()
                for agent in getattr(agent_result, "active_agents", [])
                if str(getattr(agent, "agent_type", "")).strip()
            )
            self._plugin_payloads = sorted(
                [
                    {
                        "name": str(getattr(plugin, "name", "")).strip(),
                        "path": str(getattr(plugin, "root", "")),
                    }
                    for plugin in getattr(plugin_result, "plugins", [])
                    if str(getattr(plugin, "name", "")).strip()
                ],
                key=lambda item: item["name"],
            )

            system_prompt = self._resolve_system_prompt(
                tools,
                "",  # Will be set per query
                mcp_instructions,
                None,
            )

            # Apply permission mode to runtime state (checker + query context)
            self._apply_permission_mode(permission_mode)

            # Mark as initialized
            self._initialized = True
            self._init_stream_message_sent = False

            # Build enhanced response payload aligned with SDK conventions
            slash_commands = list_slash_commands()
            custom_commands = list_custom_commands(self._project_path)

            # Build commands list with name, description, argumentHint
            commands_payload: list[dict[str, str]] = []
            for cmd in slash_commands:
                commands_payload.append({
                    "name": getattr(cmd, "name", ""),
                    "description": getattr(cmd, "description", "") or "",
                    "argumentHint": "",  # SlashCommand doesn't have argumentHint field
                })
            for custom_cmd in custom_commands:
                commands_payload.append({
                    "name": getattr(custom_cmd, "name", ""),
                    "description": getattr(custom_cmd, "description", "") or "",
                    "argumentHint": "",  # CustomCommandDefinition doesn't have argumentHint
                })

            # Load available output styles
            output_style_result = load_all_output_styles(self._project_path)
            available_styles = [style.key for style in output_style_result.styles]

            # Build models info (simplified - just current model)
            models_payload: dict[str, Any] = {}
            if model_profile is not None and hasattr(model_profile, "model"):
                models_payload = {
                    "default": getattr(model_profile, "model", model),
                    "current": model,
                }
            else:
                models_payload = {
                    "default": model,
                    "current": model,
                }

            # Account info placeholder (ripperdoc doesn't have account management)
            account_payload: dict[str, Any] = {
                "email": None,
                "organization": None,
                "subscriptionType": None,
                "tokenSource": None,
                "apiKeySource": None,
            }

            # Build enhanced response
            init_response = InitializeResult(
                protocolVersion=protocol_version,
                capabilities=ProtocolCapabilities(
                    tools={"listChanged": False},
                    sampling={"tools": True},
                ),
                serverInfo=InitializeServerInfo(
                    name="ripperdoc",
                    title="Ripperdoc",
                    version=__version__,
                ),
                instructions=system_prompt if (system_prompt or "").strip() else None,
            )

            # Convert to dict and add compatibility fields used by SDK clients
            response_dict = model_to_dict(init_response)
            response_dict["commands"] = commands_payload
            response_dict["output_style"] = self._output_style
            response_dict["available_output_styles"] = available_styles
            response_dict["models"] = models_payload
            response_dict["account"] = account_payload
            response_dict["pid"] = os.getpid()

            await self._write_control_response(
                request_id,
                response=response_dict,
            )

        except Exception as e:
            logger.error(f"Initialize failed: {e}", exc_info=True)
            error_code = resolve_protocol_request_error_code(
                e,
                default=JsonRpcErrorCodes.InvalidParams,
            )
            await self._write_control_response(
                request_id,
                error={
                    "code": error_code,
                    "message": str(e),
                },
            )
            clear_sdk_mcp_request_sender()

    def _coerce_initialize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Normalize initialize payload into strict `InitializeParams` shape."""
        request_body: dict[str, Any] = {}
        if isinstance(request, dict):
            request_body = dict(request)

        # Control protocol variant may wrap payload in `request`.
        nested_request = request_body.get("request")
        if isinstance(nested_request, dict):
            request_body = dict(nested_request) | request_body

        # JSON-RPC wrapper variant may pass params.
        params_request = request_body.get("params")
        if isinstance(params_request, dict):
            request_body = dict(params_request) | request_body

        # Allow SDK option passthrough for direct handler usage and control-style
        # initialize calls that currently include only a subset of required fields.
        options = request_body.get("options")
        if isinstance(options, dict):
            if "protocolVersion" not in request_body or request_body.get("protocolVersion") is None:
                request_body["protocolVersion"] = DEFAULT_PROTOCOL_VERSION
            if "capabilities" not in request_body or request_body.get("capabilities") is None:
                request_body["capabilities"] = {}
            if "clientInfo" not in request_body or request_body.get("clientInfo") is None:
                request_body["clientInfo"] = {
                    "name": "ripperdoc",
                    "version": __version__,
                }
            elif isinstance(request_body["clientInfo"], dict):
                client_info_payload = cast(dict[str, Any], dict(request_body["clientInfo"]))
                client_info_payload.setdefault("name", "ripperdoc")
                client_info_payload.setdefault("version", __version__)
                request_body["clientInfo"] = client_info_payload

        # Keep only expected initialize payload keys; unknown keys are stripped to
        # preserve strict schema behavior for `extra="forbid"` on pydantic models.
        initialized_request = {
            "protocolVersion": request_body.get("protocolVersion"),
            "capabilities": request_body.get("capabilities"),
            "clientInfo": request_body.get("clientInfo"),
            "_meta": request_body.get("_meta"),
        }

        # Control-side initialize payloads from SDK may include only hooks/agents;
        # fill missing required fields here to remain protocol-compatible while
        # keeping model-level schema strictness for user-provided fields.
        if (
            request_body.get("subtype") == "initialize"
            or "hooks" in request_body
            or "agents" in request_body
            and initialized_request["protocolVersion"] is None
        ):
            initialized_request["protocolVersion"] = DEFAULT_PROTOCOL_VERSION
            initialized_request["capabilities"] = initialized_request.get("capabilities") or {}
            fallback_client_info = initialized_request.get("clientInfo")
            if not isinstance(fallback_client_info, dict):
                client_info_payload = {
                    "name": "ripperdoc",
                    "version": __version__,
                }
            else:
                client_info_payload = cast(dict[str, Any], dict(fallback_client_info))
                client_info_payload.setdefault("name", "ripperdoc")
                client_info_payload.setdefault("version", __version__)
            initialized_request["clientInfo"] = client_info_payload

        return initialized_request

    async def _fail_initialize_request(
        self,
        request_id: str,
        error_msg: str,
        error_code: int = JsonRpcErrorCodes.InvalidParams,
    ) -> None:
        """Send a JSON-RPC error for initialize request."""
        await self._write_control_response(
            request_id,
            error={
                "code": int(error_code),
                "message": error_msg,
            },
        )
