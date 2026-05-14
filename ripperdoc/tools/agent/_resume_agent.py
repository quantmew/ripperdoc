"""Extracted resume-agent logic for the Task tool.

This module contains the standalone ``handle_resume_call`` function which
implements the resume path of the Task tool.  It was extracted from
``TaskTool._handle_resume_call`` so the code can be maintained and tested
independently of the large ``TaskTool`` class.

The function accepts generic :class:`Callable` parameters for every
operation that would otherwise require a ``self`` reference, which avoids
circular-import issues and keeps the module self-contained.
"""

from __future__ import annotations

from typing import AsyncGenerator, Callable, List, Optional

from ripperdoc.core.agents import BASH_TOOL_NAME, READ_TOOL_NAME
from ripperdoc.core.hooks.manager import HookResult
from ripperdoc.core.tool import ToolOutput, ToolProgress, ToolResult, ToolUseContext
from ripperdoc.utils.log import get_logger

from ripperdoc.tools.agent._tool import AgentToolInput, AgentToolOutput
from ripperdoc.tools.agent._store import (
    AgentRunRecord,
    _get_agent_run,
    _set_team_member_active_state,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Public async generator
# ---------------------------------------------------------------------------

async def handle_resume_call(
    input_data: AgentToolInput,
    context: ToolUseContext,
    *,
    # -- hook helpers (from TaskTool) --
    run_subagent_start_hook: Callable,
    build_subagent_start_notices: Callable,
    build_subagent_hook_messages: Callable,
    reset_record_for_resume_prompt: Callable,
    build_subagent_query_context: Callable,
    run_subagent_foreground: Callable,
    # -- output helpers (from TaskTool) --
    output_from_record: Callable[..., AgentToolOutput],
    render_tool_result: Callable[[AgentToolOutput], ToolResult],
    subagent_progress_sender: Callable[[AgentRunRecord], str],
    available_tools_provider: Callable[[], List[object]],
) -> AsyncGenerator[ToolOutput, None]:
    """Resume a previously-completed (or idle) subagent run.

    Parameters mirror the original ``TaskTool._handle_resume_call`` method.
    Every ``self.*`` reference has been replaced by an explicitly-passed
    callable so that this function has no dependency on the ``TaskTool``
    class itself.
    """
    if not input_data.resume:
        return

    record: Optional[AgentRunRecord] = _get_agent_run(input_data.resume)
    if not record:
        raise ValueError(
            f"Agent id '{input_data.resume}' not found. "
            "Start a new agent to obtain a valid agent_id."
        )
    should_activate = bool(record.team_name and record.teammate_name)

    if should_activate:
        _set_team_member_active_state(
            record.team_name,
            record.teammate_name,
            True,
            default_agent_type=record.agent_type,
        )

    if record.task and not record.task.done():
        raise ValueError(
            f"Cannot resume agent '{record.agent_id}' while it is still running. "
            "Use TaskOutput to check progress/output."
        )

    hook_result: HookResult = await run_subagent_start_hook(
        context,
        subagent_type=record.agent_type,
        prompt=input_data.prompt,
        resume=input_data.resume,
        run_in_background=False,
    )
    for notice in build_subagent_start_notices(hook_result, agent_type=record.agent_type):
        yield notice
    hook_messages = build_subagent_hook_messages(
        hook_result, parent_tool_use_id=context.message_id
    )
    if hook_messages:
        record.history.extend(hook_messages)

    if input_data.model:
        record.model_used = input_data.model
    if input_data.mode:
        record.permission_mode = input_data.mode
    if input_data.max_turns:
        record.max_turns = input_data.max_turns
    record.task_description = input_data.description

    reset_record_for_resume_prompt(record, input_data.prompt or record.task_prompt or "")
    subagent_context = build_subagent_query_context(
        tools=record.tools,
        yolo_mode=context.yolo_mode,
        verbose=context.verbose,
        model=record.model_used or "main",
        agent_type=record.agent_type,
        team_name=record.team_name,
        teammate_name=record.teammate_name,
        agent_id=record.agent_id,
        hook_scopes=record.hook_scopes,
        max_turns=record.max_turns,
        permission_mode=record.permission_mode,
        working_directory=record.worktree_path,
        task_notification_queue=context.task_notification_queue,
    )
    yield ToolProgress(
        content=f"Resuming subagent '{record.agent_type}'",
        progress_sender=subagent_progress_sender(record),
    )

    async for progress in run_subagent_foreground(
        record=record,
        subagent_context=subagent_context,
        permission_checker=context.permission_checker,
        parent_abort_signal=context.abort_signal,
        notification_queue=context.task_notification_queue,
        parent_tool_use_id=context.message_id,
    ):
        yield progress

    if record.task and not record.task.done():
        output = output_from_record(
            record,
            status_override="async_launched",
            result_text_override="Resumed subagent moved to background after interrupt.",
            is_background=True,
            is_resumed=True,
            can_read_output_file=any(
                getattr(tool, "name", "") in {READ_TOOL_NAME, BASH_TOOL_NAME}
                for tool in available_tools_provider()
            ),
        )
        yield render_tool_result(output)
        return

    output = output_from_record(record, is_resumed=True)
    yield render_tool_result(output)
