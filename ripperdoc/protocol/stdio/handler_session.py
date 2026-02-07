"""Session initialization for stdio protocol handler."""

from __future__ import annotations

import logging
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from ripperdoc import __version__
from ripperdoc.cli.commands import list_custom_commands, list_slash_commands
from ripperdoc.core.agents import load_agent_definitions
from ripperdoc.core.config import (
    get_effective_model_profile,
    get_project_config,
    get_project_local_config,
)
from ripperdoc.core.output_styles import resolve_output_style
from ripperdoc.core.tool_defaults import get_default_tools
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.state import bind_pending_message_queue, bind_hook_scopes
from ripperdoc.core.query import QueryContext
from ripperdoc.core.message_utils import resolve_model_profile
from ripperdoc.protocol.models import (
    InitializeResponseData,
    MCPServerInfo,
    MCPServerStatusInfo,
    SystemStreamMessage,
    model_to_dict,
)
from ripperdoc.tools.dynamic_mcp_tool import (
    load_dynamic_mcp_tools_async,
    merge_tools_with_dynamic,
)
from ripperdoc.utils.asyncio_compat import asyncio_timeout
from ripperdoc.utils.mcp import format_mcp_instructions, load_mcp_servers_async
from ripperdoc.utils.session_history import SessionHistory
from ripperdoc.utils.working_directories import coerce_directory_list, normalize_directory_inputs

from .timeouts import STDIO_HOOK_TIMEOUT_SEC

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


class StdioSessionMixin:
    _query_context: QueryContext | None
    _session_id: str | None
    _custom_system_prompt: str | None
    _skill_instructions: str | None
    _output_style: str
    _output_language: str
    _session_start_time: float | None
    _session_history: SessionHistory | None
    _session_additional_working_dirs: set[str]
    _fallback_model: str | None
    _max_budget_usd: float | None
    _json_schema: dict[str, Any] | None

    async def _handle_initialize(self, request: dict[str, Any], request_id: str) -> None:
        """Handle initialize request from SDK.

        Args:
            request: The initialize request data.
            request_id: The request ID.
        """
        if self._initialized:
            await self._write_control_response(request_id, error="Already initialized")
            return

        try:
            # Extract options from request
            request_options = request.get("options", {}) or {}
            options = {**self._default_options, **request_options}
            self._session_id = options.get("session_id") or str(uuid.uuid4())
            self._custom_system_prompt = options.get("system_prompt")
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
                    logger.warning("[stdio] Invalid json_schema payload; expected valid JSON object")
                else:
                    if isinstance(parsed_schema, dict):
                        self._json_schema = parsed_schema
                    else:
                        logger.warning("[stdio] json_schema must be a JSON object; ignoring")

            ignored_option_keys = [
                "mcp_config",
                "permission_prompt_tool",
                "include_partial_messages",
                "fork_session",
                "agents",
                "setting_sources",
                "plugin_dirs",
                "betas",
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
            requested_output_style = (
                options.get("output_style")
                or configured_output_style
            )
            resolved_output_style, _ = resolve_output_style(
                str(requested_output_style),
                project_path=self._project_path,
            )
            self._output_style = resolved_output_style.key
            configured_output_language = (
                getattr(project_local_config, "output_language", "auto") or "auto"
            )
            requested_output_language = (
                options.get("output_language")
                or configured_output_language
            )
            self._output_language = str(requested_output_language or "auto").strip() or "auto"

            # Parse tool options
            self._allowed_tools = self._normalize_tool_list(options.get("allowed_tools"))
            self._disallowed_tools = self._normalize_tool_list(options.get("disallowed_tools"))
            self._tools_list = self._normalize_tool_list(options.get("tools"))

            # Get the tool list (apply SDK filters)
            tools = get_default_tools()
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
            yolo_mode = permission_mode == "bypassPermissions"
            self._sdk_can_use_tool_enabled = self._coerce_bool_option(
                options.get("sdk_can_use_tool"),
                self._input_format == "stream-json" and self._output_format == "stream-json",
            )

            # Setup model
            model = options.get("model") or "main"

            # 验证模型配置是否有效
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
                await self._write_control_response(request_id, error=error_msg)
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
            )

            # Initialize hook manager
            hook_manager.set_project_dir(self._project_path)
            hook_manager.set_session_id(self._session_id)
            hook_manager.set_llm_callback(build_hook_llm_callback())
            self._session_history = SessionHistory(
                self._project_path, self._session_id or str(uuid.uuid4())
            )
            hook_manager.set_transcript_path(str(self._session_history.path))

            # Configure SDK-provided hooks (if any)
            hooks = request.get("hooks", None)
            if hooks is None:
                hooks = options.get("hooks", {})
            self._configure_sdk_hooks(hooks)

            session_start_notices: list[str] = []
            # Run SessionStart hooks during initialize (once per session)
            queue = self._query_context.pending_message_queue if self._query_context else None
            hook_scopes = self._query_context.hook_scopes if self._query_context else []
            with bind_pending_message_queue(queue), bind_hook_scopes(hook_scopes):
                try:
                    async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                        session_start_result = await hook_manager.run_session_start_async("startup")
                    if getattr(session_start_result, "system_message", None):
                        session_start_notices.append(str(session_start_result.system_message))
                    self._session_hook_contexts = self._collect_hook_contexts(
                        session_start_result
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

            # Load MCP servers and dynamic tools
            servers = await load_mcp_servers_async(self._project_path)
            dynamic_tools = await load_dynamic_mcp_tools_async(self._project_path)
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

            skill_result = load_all_skills(self._project_path)
            enabled_skills = filter_enabled_skills(
                skill_result.skills, project_path=self._project_path
            )
            self._skill_instructions = build_skill_summary(enabled_skills)

            agent_result = load_agent_definitions()

            system_prompt = self._resolve_system_prompt(
                tools,
                "",  # Will be set per query
                mcp_instructions,
                self._session_hook_contexts,
            )

            # Apply permission mode to runtime state (checker + query context)
            self._apply_permission_mode(permission_mode)

            # Mark as initialized
            self._initialized = True

            # Send success response with available tools
            # Use simple list format for Claude SDK compatibility
            def _display_agent_name(agent_type: str) -> str:
                if agent_type in ("explore", "plan"):
                    return agent_type.title()
                return agent_type

            agent_names = [
                _display_agent_name(agent.agent_type) for agent in agent_result.active_agents
            ]
            skill_names = [skill.name for skill in enabled_skills] if enabled_skills else []

            slash_commands = [cmd.name for cmd in list_slash_commands()]
            for custom_cmd in list_custom_commands(self._project_path):
                if custom_cmd.name not in slash_commands:
                    slash_commands.append(custom_cmd.name)

            resolved_model_profile = resolve_model_profile(model or "main")
            resolved_model = resolved_model_profile.model if resolved_model_profile else (model or "main")

            init_response = InitializeResponseData(
                session_id=self._session_id or "",
                system_prompt=system_prompt,
                tools=[t.name for t in tools],
                mcp_servers=[MCPServerInfo(name=s.name) for s in servers] if servers else [],
                slash_commands=slash_commands,
                ripperdoc_version=__version__,
                output_style=self._output_style,
                output_language=self._output_language,
                agents=agent_names,
                skills=skill_names,
                plugins=[],
            )

            # Emit a system/init stream message first (Claude CLI compatibility)
            try:
                system_message = SystemStreamMessage(
                    uuid=str(uuid.uuid4()),
                    session_id=self._session_id or "",
                    api_key_source=init_response.apiKeySource,
                    cwd=str(self._project_path),
                    tools=[t.name for t in tools],
                    mcp_servers=[
                        MCPServerStatusInfo(name=s.name, status=getattr(s, "status", "unknown"))
                        for s in servers
                    ]
                    if servers
                    else [],
                    model=resolved_model,
                    permission_mode=permission_mode,
                    slash_commands=slash_commands,
                    ripperdoc_version=init_response.ripperdoc_version,
                    output_style=init_response.output_style,
                    output_language=init_response.output_language,
                    agents=agent_names,
                    skills=skill_names,
                    plugins=[],
                )
                await self._write_message_stream(model_to_dict(system_message))
            except Exception as e:
                logger.warning(f"[stdio] Failed to emit system init message: {e}")

            for notice_text in session_start_notices:
                stream_message = self._build_hook_notice_stream_message(
                    notice_text,
                    "SessionStart",
                    tool_name=None,
                    level="info",
                )
                await self._write_message_stream(model_to_dict(stream_message))

            await self._write_control_response(
                request_id,
                response=model_to_dict(init_response),
            )

        except Exception as e:
            logger.error(f"Initialize failed: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))
