"""Agent tool that delegates work to configured subagents."""

from ripperdoc.tools.agent._tool import (
    AgentTool,
    AgentToolInput,
    AgentToolContentBlock,
    AgentToolOutput,
)
from ripperdoc.tools.agent._store import (
    AgentRunRecord,
    cancel_agent_run,
    get_agent_run_snapshot,
    inject_user_message_to_teammate,
    list_agent_runs,
    list_running_agent_worktree_paths,
    list_running_team_members,
    pop_pending_user_message_from_teammate,
    set_agent_idle_state,
    wait_for_agent_run_snapshot,
)

__all__ = [
    "AgentTool",
    "AgentToolInput",
    "AgentToolOutput",
    "AgentToolContentBlock",
    "AgentRunRecord",
    "cancel_agent_run",
    "get_agent_run_snapshot",
    "inject_user_message_to_teammate",
    "list_agent_runs",
    "list_running_agent_worktree_paths",
    "list_running_team_members",
    "pop_pending_user_message_from_teammate",
    "set_agent_idle_state",
    "wait_for_agent_run_snapshot",
]
