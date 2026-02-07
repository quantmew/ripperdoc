"""Hook command executor.

This module handles the actual execution of hook commands,
including environment setup, input passing, and output parsing.

Supports hook types:
- command: Execute a shell command
- prompt: Use LLM to evaluate (requires LLM callback)
- agent: Spawn a subagent to evaluate with tool access
- callback: Invoke an external hook callback
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Awaitable, Any

from ripperdoc.core.hooks.config import HookDefinition
from ripperdoc.core.hooks.events import (
    AnyHookInput,
    HookOutput,
    SessionStartInput,
)
from ripperdoc.core.hooks.state import suspend_hooks
from ripperdoc.core.system_prompt import build_environment_prompt
from ripperdoc.tools.bash_output_tool import BashOutputTool
from ripperdoc.tools.bash_tool import BashTool
from ripperdoc.tools.file_read_tool import FileReadTool
from ripperdoc.tools.glob_tool import GlobTool
from ripperdoc.tools.grep_tool import GrepTool
from ripperdoc.tools.kill_bash_tool import KillBashTool
from ripperdoc.tools.ls_tool import LSTool
from ripperdoc.tools.lsp_tool import LspTool
from ripperdoc.utils.messages import AssistantMessage, create_user_message
from ripperdoc.utils.log import get_logger

logger = get_logger()

HOOK_AGENT_MAX_TURNS = 50
HOOK_AGENT_DEFAULT_MODEL = "quick"


def _extract_message_text(message: AssistantMessage) -> str:
    content = message.message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else None
            )
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return ""


def _build_hook_agent_tools() -> list[Any]:
    """Create a constrained toolset for agent-based hooks."""
    return [
        FileReadTool(),
        GlobTool(),
        GrepTool(),
        LSTool(),
        BashTool(),
        BashOutputTool(),
        KillBashTool(),
        LspTool(),
    ]

# Type for LLM callback used by prompt hooks
# Takes prompt string, returns LLM response string
LLMCallback = Callable[[str], Awaitable[str]]

# Type for external hook callbacks (e.g., SDK hooks)
# Takes (callback_id, input_data, tool_use_id, timeout_sec) and returns HookOutput or raw dict
HookCallback = Callable[
    [str, Dict[str, Any], Optional[str], Optional[float]],
    Awaitable[HookOutput | Dict[str, Any] | None],
]


class HookExecutor:
    """Executes hook commands with proper environment and I/O handling.

    Supports hook types:
    - command: Execute shell commands
    - prompt: Use LLM to evaluate (requires llm_callback to be set)
    - agent: Spawn a subagent to evaluate with tool access
    - callback: Invoke an external hook callback
    """

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        transcript_path: Optional[str] = None,
        llm_callback: Optional[LLMCallback] = None,
        hook_callback: Optional[HookCallback] = None,
    ):
        """Initialize the executor.

        Args:
            project_dir: The project directory for resolving relative paths
                        and setting RIPPERDOC_PROJECT_DIR environment variable.
            session_id: Current session ID.
            transcript_path: Path to the conversation transcript JSON file.
            llm_callback: Async callback for prompt-based hooks. Takes prompt string,
                         returns LLM response string. If not set, prompt hooks will
                         be skipped with a warning.
            hook_callback: Async callback for external hook execution (e.g., SDK hooks).
        """
        self.project_dir = project_dir
        self.session_id = session_id
        self.transcript_path = transcript_path
        self.llm_callback = llm_callback
        self.hook_callback = hook_callback
        self._env_file: Optional[Path] = None

    def _get_env_file(self) -> Path:
        """Get or create the environment file for SessionStart hooks.

        This file can be used by SessionStart hooks to persist environment
        variables that will be loaded into the session.
        """
        if self._env_file is None:
            # Create a temporary file that persists for the session
            fd, path = tempfile.mkstemp(prefix="ripperdoc_env_", suffix=".json")
            os.close(fd)
            self._env_file = Path(path)
            # Initialize with empty JSON object
            self._env_file.write_text("{}")
        return self._env_file

    def cleanup_env_file(self) -> None:
        """Clean up the environment file when session ends."""
        if self._env_file and self._env_file.exists():
            try:
                self._env_file.unlink()
            except OSError:
                pass
            self._env_file = None

    def load_env_from_file(self) -> Dict[str, str]:
        """Load environment variables from the env file.

        This is called after SessionStart hooks run to load any
        environment variables they may have set.
        """
        if self._env_file is None or not self._env_file.exists():
            return {}

        try:
            content = self._env_file.read_text()
            data = json.loads(content) if content.strip() else {}
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items() if isinstance(k, str)}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load env file: {e}")
        return {}

    def _build_env(self, input_data: Optional[AnyHookInput] = None) -> Dict[str, str]:
        """Build the environment variables for hook execution."""
        env = os.environ.copy()

        # Add RIPPERDOC_PROJECT_DIR
        if self.project_dir:
            env["RIPPERDOC_PROJECT_DIR"] = str(self.project_dir)

        # Add session ID if available
        if self.session_id:
            env["RIPPERDOC_SESSION_ID"] = self.session_id

        # Add transcript path if available
        if self.transcript_path:
            env["RIPPERDOC_TRANSCRIPT_PATH"] = self.transcript_path

        # For SessionStart hooks, provide the env file path
        if isinstance(input_data, SessionStartInput):
            env_file = self._get_env_file()
            env["RIPPERDOC_ENV_FILE"] = str(env_file)

        return env

    def _expand_command(self, command: str) -> str:
        """Expand environment variables in the command string."""
        # Expand $RIPPERDOC_PROJECT_DIR
        if self.project_dir:
            project_dir_str = str(self.project_dir)
            command = command.replace("$RIPPERDOC_PROJECT_DIR", project_dir_str)
            command = command.replace("${RIPPERDOC_PROJECT_DIR}", project_dir_str)
        return command

    def _expand_prompt(self, prompt: str, input_data: AnyHookInput) -> str:
        """Expand variables in the prompt string.

        Replaces $ARGUMENTS with the JSON-serialized input data.
        """
        input_json = input_data.model_dump_json()
        prompt = prompt.replace("$ARGUMENTS", input_json)
        prompt = prompt.replace("${ARGUMENTS}", input_json)
        return prompt

    def _parse_prompt_response(
        self, response: str, *, allow_raw_output_context: bool = True
    ) -> HookOutput:
        """Parse LLM response from a prompt hook.

        Expected response format (JSON):
        {
            "decision": "allow|deny|ask|block",
            "reason": "explanation",
            "continue": false,           // optional
            "stopReason": "message",      // optional
            "systemMessage": "warning"    // optional
        }

        Or plain text (treated as raw output and added as additional context).
        """
        response = response.strip()
        if not response:
            return HookOutput()

        # Try to parse as JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return HookOutput._parse_json_output(data, "")
        except json.JSONDecodeError:
            pass

        # Not JSON, treat as raw output (and optional additional context)
        return HookOutput(
            raw_output=response,
            additional_context=response if allow_raw_output_context else None,
        )

    def _allow_raw_output_context(self, input_data: AnyHookInput) -> bool:
        """Return True if non-JSON stdout should be injected as context."""
        # Plain-text hook output should be treated as additional context
        # for all hook events (matches docs/examples).
        return True

    def _build_agent_system_prompt(self, tool_names: str) -> str:
        """Build the system prompt for agent-based hooks."""
        return "\n\n".join(
            [
                (
                    "You are a hook verification subagent. "
                    "Use the allowed tools to inspect the codebase and determine whether the hook should allow or block. "
                    "Do not modify files. Do not ask the user questions. "
                    "When you are done, respond with a single JSON object and nothing else.\n\n"
                    "Response schema:\n"
                    '- {"decision": "allow"}\n'
                    '- {"decision": "deny", "reason": "..."}\n\n'
                    "If you are unsure, return decision=deny with a clear reason."
                ),
                f"Allowed tools: {tool_names}",
                build_environment_prompt(),
            ]
        )

    async def _execute_agent_async(
        self,
        hook: HookDefinition,
        input_data: AnyHookInput,
    ) -> HookOutput:
        """Execute an agent-based hook asynchronously."""
        if not hook.prompt:
            logger.warning("Agent hook has no prompt template")
            return HookOutput(error="Agent hook missing prompt template")

        # Expand the prompt template
        prompt = self._expand_prompt(hook.prompt, input_data)
        model = hook.model or HOOK_AGENT_DEFAULT_MODEL

        logger.debug(
            "Executing agent hook",
            extra={
                "event": input_data.hook_event_name,
                "timeout": hook.timeout,
                "model": model,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            },
        )

        tools = _build_hook_agent_tools()
        tool_names = ", ".join(tool.name for tool in tools if getattr(tool, "name", None))
        system_prompt = self._build_agent_system_prompt(tool_names)

        async def _run_agent() -> str:
            assistant_messages: list[AssistantMessage] = []
            from ripperdoc.core.query import QueryContext, query

            query_context = QueryContext(
                tools=tools,
                yolo_mode=True,
                model=model,
                max_turns=HOOK_AGENT_MAX_TURNS,
                permission_mode=input_data.permission_mode,
            )
            with suspend_hooks():
                async for message in query(
                    [create_user_message(prompt)],
                    system_prompt,
                    {},
                    query_context,
                    None,
                ):
                    if getattr(message, "type", "") == "assistant":
                        if isinstance(message, AssistantMessage):
                            assistant_messages.append(message)
            if not assistant_messages:
                return ""
            return _extract_message_text(assistant_messages[-1]).strip()

        try:
            response = await asyncio.wait_for(_run_agent(), timeout=hook.timeout)
            output = self._parse_prompt_response(
                response, allow_raw_output_context=self._allow_raw_output_context(input_data)
            )
            logger.debug(
                "Agent hook completed",
                extra={
                    "decision": output.decision.value if output.decision else None,
                },
            )
            return output
        except asyncio.TimeoutError:
            logger.warning(f"Agent hook timed out after {hook.timeout}s")
            return HookOutput.from_raw("", "", 1, timed_out=True)
        except Exception as exc:
            logger.error(f"Agent hook execution failed: {exc}")
            return HookOutput(error=str(exc), exit_code=1)

    async def execute_prompt_async(
        self,
        hook: HookDefinition,
        input_data: AnyHookInput,
    ) -> HookOutput:
        """Execute a prompt-based hook asynchronously.

        Uses the LLM callback to evaluate the prompt and parse the response.

        Args:
            hook: The hook definition with prompt template
            input_data: The input data to pass to the hook

        Returns:
            HookOutput containing the LLM's decision
        """
        if not hook.prompt:
            logger.warning("Prompt hook has no prompt template")
            return HookOutput(error="Prompt hook missing prompt template")

        if not self.llm_callback:
            logger.warning(
                "Prompt hook skipped: no LLM callback configured. "
                "Set llm_callback on HookExecutor to enable prompt hooks."
            )
            return HookOutput()

        # Expand the prompt template
        prompt = self._expand_prompt(hook.prompt, input_data)

        logger.debug(
            "Executing prompt hook",
            extra={
                "event": input_data.hook_event_name,
                "timeout": hook.timeout,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            },
        )

        try:
            # Call LLM with timeout
            response = await asyncio.wait_for(
                self.llm_callback(prompt),
                timeout=hook.timeout,
            )

            output = self._parse_prompt_response(
                response, allow_raw_output_context=self._allow_raw_output_context(input_data)
            )

            logger.debug(
                "Prompt hook completed",
                extra={
                    "decision": output.decision.value if output.decision else None,
                },
            )

            return output

        except asyncio.TimeoutError:
            logger.warning(f"Prompt hook timed out after {hook.timeout}s")
            return HookOutput.from_raw("", "", 1, timed_out=True)

        except Exception as e:
            logger.error(f"Prompt hook execution failed: {e}")
            return HookOutput(error=str(e), exit_code=1)

    def execute_sync(
        self,
        hook: HookDefinition,
        input_data: AnyHookInput,
    ) -> HookOutput:
        """Execute a hook synchronously.

        Dispatches to appropriate method based on hook type.
        Note: Prompt hooks are not supported in sync mode and will be skipped.

        Args:
            hook: The hook definition to execute
            input_data: The input data to pass to the hook

        Returns:
            HookOutput containing the result or error
        """
        # Prompt, agent, and callback hooks require async - skip in sync mode
        if hook.is_prompt_hook() or hook.is_agent_hook() or hook.is_callback_hook():
            logger.warning(
                "Prompt/agent/callback hook skipped in sync mode. Use execute_async for prompt/agent/callback hooks."
            )
            return HookOutput()

        return self._execute_command_sync(hook, input_data)

    def _execute_command_sync(
        self,
        hook: HookDefinition,
        input_data: AnyHookInput,
    ) -> HookOutput:
        """Execute a command-based hook synchronously.

        Args:
            hook: The hook definition to execute
            input_data: The input data to pass to the hook (as JSON via stdin)

        Returns:
            HookOutput containing the result or error
        """
        if not hook.command:
            logger.warning("Command hook has no command")
            return HookOutput(error="Command hook missing command")

        command = self._expand_command(hook.command)
        env = self._build_env(input_data)

        # Serialize input data for stdin
        input_json = input_data.model_dump_json()

        logger.debug(
            f"Executing hook: {command}",
            extra={
                "event": input_data.hook_event_name,
                "timeout": hook.timeout,
            },
        )

        try:
            result = subprocess.run(
                command,
                shell=True,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=hook.timeout,
                env=env,
                cwd=str(self.project_dir) if self.project_dir else None,
            )

            output = HookOutput.from_raw(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                allow_raw_output_context=self._allow_raw_output_context(input_data),
            )

            logger.debug(
                f"Hook completed: {command}",
                extra={
                    "exit_code": result.returncode,
                    "decision": output.decision.value if output.decision else None,
                },
            )

            return output

        except subprocess.TimeoutExpired:
            logger.warning(f"Hook timed out after {hook.timeout}s: {command}")
            return HookOutput.from_raw("", "", 1, timed_out=True)

        except Exception as e:
            logger.error(f"Hook execution failed: {command}: {e}")
            return HookOutput(
                error=str(e),
                exit_code=1,
            )

    async def execute_async(
        self,
        hook: HookDefinition,
        input_data: AnyHookInput,
    ) -> HookOutput:
        """Execute a hook asynchronously.

        Dispatches to appropriate method based on hook type.

        Args:
            hook: The hook definition to execute
            input_data: The input data to pass to the hook

        Returns:
            HookOutput containing the result or error
        """
        if hook.is_prompt_hook():
            return await self.execute_prompt_async(hook, input_data)
        if hook.is_agent_hook():
            return await self._execute_agent_async(hook, input_data)
        if hook.is_callback_hook():
            return await self._execute_callback_async(hook, input_data)

        return await self._execute_command_async(hook, input_data)

    async def _execute_callback_async(
        self,
        hook: HookDefinition,
        input_data: AnyHookInput,
    ) -> HookOutput:
        """Execute an external callback-based hook asynchronously."""
        if not hook.callback_id:
            logger.warning("Callback hook missing callback_id")
            return HookOutput(error="Callback hook missing callback_id", exit_code=1)
        if not self.hook_callback:
            logger.warning("Callback hook invoked without a configured hook callback")
            return HookOutput(error="Hook callback not configured", exit_code=1)

        input_payload: Dict[str, Any]
        if hasattr(input_data, "model_dump"):
            input_payload = input_data.model_dump()
        elif isinstance(input_data, dict):
            input_payload = dict(input_data)
        else:
            input_payload = {}

        tool_use_id = getattr(input_data, "tool_use_id", None)

        try:
            result = await self.hook_callback(
                hook.callback_id,
                input_payload,
                tool_use_id,
                float(hook.timeout) if hook.timeout is not None else None,
            )
        except asyncio.TimeoutError:
            logger.warning("Callback hook timed out after %ss", hook.timeout)
            return HookOutput.from_raw("", "", 1, timed_out=True)
        except Exception as exc:
            logger.error("Callback hook execution failed: %s", exc)
            return HookOutput(error=str(exc), exit_code=1)

        if result is None:
            return HookOutput()
        if isinstance(result, HookOutput):
            return result
        if isinstance(result, dict):
            return HookOutput.from_raw(
                stdout=json.dumps(result),
                stderr="",
                exit_code=0,
                allow_raw_output_context=self._allow_raw_output_context(input_data),
            )
        return HookOutput()

    async def _execute_command_async(
        self,
        hook: HookDefinition,
        input_data: AnyHookInput,
    ) -> HookOutput:
        """Execute a command-based hook asynchronously.

        Args:
            hook: The hook definition to execute
            input_data: The input data to pass to the hook (as JSON via stdin)

        Returns:
            HookOutput containing the result or error
        """
        if not hook.command:
            logger.warning("Command hook has no command")
            return HookOutput(error="Command hook missing command")

        command = self._expand_command(hook.command)
        env = self._build_env(input_data)

        # Serialize input data for stdin
        input_json = input_data.model_dump_json()

        logger.debug(
            f"Executing hook (async): {command}",
            extra={
                "event": input_data.hook_event_name,
                "timeout": hook.timeout,
            },
        )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.project_dir) if self.project_dir else None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_json.encode()),
                    timeout=hook.timeout,
                )

                output = HookOutput.from_raw(
                    stdout=stdout.decode(),
                    stderr=stderr.decode(),
                    exit_code=process.returncode or 0,
                    allow_raw_output_context=self._allow_raw_output_context(input_data),
                )

                logger.debug(
                    f"Hook completed (async): {command}",
                    extra={
                        "exit_code": process.returncode,
                        "decision": output.decision.value if output.decision else None,
                    },
                )

                return output

            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                await process.wait()
                logger.warning(f"Hook timed out after {hook.timeout}s: {command}")
                return HookOutput.from_raw("", "", 1, timed_out=True)

        except Exception as e:
            logger.error(f"Hook execution failed (async): {command}: {e}")
            return HookOutput(
                error=str(e),
                exit_code=1,
            )

    async def execute_hooks_async(
        self,
        hooks: list[HookDefinition],
        input_data: AnyHookInput,
    ) -> list[HookOutput]:
        """Execute multiple hooks in sequence.

        Hooks are executed in order. If a hook returns a blocking decision,
        subsequent hooks are still executed but the blocking result is returned.

        Args:
            hooks: List of hook definitions to execute
            input_data: The input data to pass to all hooks

        Returns:
            List of HookOutput objects, one per hook
        """
        results = []
        for hook in hooks:
            result = await self.execute_async(hook, input_data)
            results.append(result)
        return results

    def execute_hooks_sync(
        self,
        hooks: list[HookDefinition],
        input_data: AnyHookInput,
    ) -> list[HookOutput]:
        """Execute multiple hooks synchronously in sequence.

        Args:
            hooks: List of hook definitions to execute
            input_data: The input data to pass to all hooks

        Returns:
            List of HookOutput objects, one per hook
        """
        results = []
        for hook in hooks:
            result = self.execute_sync(hook, input_data)
            results.append(result)
        return results
