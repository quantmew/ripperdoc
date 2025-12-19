"""Hook command executor.

This module handles the actual execution of hook commands,
including environment setup, input passing, and output parsing.

Supports two hook types:
- command: Execute a shell command
- prompt: Use LLM to evaluate (requires LLM callback)
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Awaitable

from ripperdoc.core.hooks.config import HookDefinition
from ripperdoc.core.hooks.events import AnyHookInput, HookOutput, HookDecision, SessionStartInput
from ripperdoc.utils.log import get_logger

logger = get_logger()

# Type for LLM callback used by prompt hooks
# Takes prompt string, returns LLM response string
LLMCallback = Callable[[str], Awaitable[str]]


class HookExecutor:
    """Executes hook commands with proper environment and I/O handling.

    Supports two hook types:
    - command: Execute shell commands
    - prompt: Use LLM to evaluate (requires llm_callback to be set)
    """

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        transcript_path: Optional[str] = None,
        llm_callback: Optional[LLMCallback] = None,
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
        """
        self.project_dir = project_dir
        self.session_id = session_id
        self.transcript_path = transcript_path
        self.llm_callback = llm_callback
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

    def _parse_prompt_response(self, response: str) -> HookOutput:
        """Parse LLM response from a prompt hook.

        Expected response format (JSON):
        {
            "decision": "approve|block",
            "reason": "explanation",
            "continue": false,           // optional
            "stopReason": "message",      // optional
            "systemMessage": "warning"    // optional
        }

        Or plain text (treated as additional context with no decision).
        """
        response = response.strip()
        if not response:
            return HookOutput()

        # Try to parse as JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                output = HookOutput()

                # Parse decision
                decision_str = data.get("decision", "").lower()
                if decision_str == "approve":
                    output.decision = HookDecision.ALLOW
                elif decision_str == "block":
                    output.decision = HookDecision.BLOCK
                elif decision_str == "allow":
                    output.decision = HookDecision.ALLOW
                elif decision_str == "deny":
                    output.decision = HookDecision.DENY
                elif decision_str == "ask":
                    output.decision = HookDecision.ASK

                output.reason = data.get("reason")
                output.continue_execution = data.get("continue", True)
                output.stop_reason = data.get("stopReason")
                output.system_message = data.get("systemMessage")
                output.additional_context = data.get("additionalContext")

                return output
        except json.JSONDecodeError:
            pass

        # Not JSON, treat as additional context
        return HookOutput(raw_output=response, additionalContext=response)

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

            output = self._parse_prompt_response(response)

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
        # Prompt hooks require async - skip in sync mode
        if hook.is_prompt_hook():
            logger.warning("Prompt hook skipped in sync mode. Use execute_async for prompt hooks.")
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

        return await self._execute_command_async(hook, input_data)

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
