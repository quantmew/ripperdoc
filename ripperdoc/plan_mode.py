"""Plan mode helpers aligned with Claude Code style behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ripperdoc.utils.filesystem.config_paths import project_config_dir

PLAN_DIR_NAME = "plans"
TURNS_BETWEEN_ATTACHMENTS = 5
FULL_REMINDER_EVERY_N_ATTACHMENTS = 5


@dataclass(frozen=True)
class PlanModeAttachmentDecision:
    """Resolved plan-mode attachment injection decision for one model turn."""

    should_inject: bool
    reminder_type: str = "full"


def _message_type(message: Any) -> str | None:
    return getattr(message, "type", None)


def _message_metadata(message: Any) -> dict[str, Any]:
    if getattr(message, "type", None) == "attachment":
        metadata = getattr(message, "metadata", None)
        return dict(metadata) if isinstance(metadata, dict) else {}
    payload = getattr(message, "message", None)
    metadata = getattr(payload, "metadata", None) if payload is not None else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _is_plan_mode_attachment(message: Any) -> bool:
    metadata = _message_metadata(message)
    return metadata.get("plan_mode_attachment_type") == "plan_mode"


def _is_plan_mode_reentry_attachment(message: Any) -> bool:
    metadata = _message_metadata(message)
    return metadata.get("plan_mode_attachment_type") == "plan_mode_reentry"


def _is_plan_mode_exit_attachment(message: Any) -> bool:
    metadata = _message_metadata(message)
    return metadata.get("plan_mode_attachment_type") == "plan_mode_exit"


def _assistant_turns_since_last_plan_attachment(messages: Sequence[Any]) -> tuple[int, bool]:
    assistant_count = 0
    found_plan_attachment = False
    for message in reversed(messages):
        message_type = _message_type(message)
        if message_type == "assistant":
            assistant_count += 1
            continue
        if message_type in {"user", "attachment"} and (
            _is_plan_mode_attachment(message) or _is_plan_mode_reentry_attachment(message)
        ):
            found_plan_attachment = True
            break
    return assistant_count, found_plan_attachment


def _plan_mode_attachment_count_since_exit(messages: Sequence[Any]) -> int:
    attachment_count = 0
    for message in reversed(messages):
        if _message_type(message) not in {"user", "attachment"}:
            continue
        if _is_plan_mode_exit_attachment(message):
            break
        if _is_plan_mode_attachment(message):
            attachment_count += 1
    return attachment_count


def resolve_plan_mode_attachment_decision(messages: Sequence[Any]) -> PlanModeAttachmentDecision:
    """Apply the plan attachment cadence used by the runtime.

    Inject a new attachment when:
    - plan mode is active and no prior plan attachment exists, or
    - at least ``TURNS_BETWEEN_ATTACHMENTS`` assistant turns happened since the last
      plan-mode / plan-mode-reentry attachment.

    Use a full reminder for every ``FULL_REMINDER_EVERY_N_ATTACHMENTS``th attachment,
    sparse otherwise.
    """

    turn_count, found_plan_attachment = _assistant_turns_since_last_plan_attachment(messages)
    if found_plan_attachment and turn_count < TURNS_BETWEEN_ATTACHMENTS:
        return PlanModeAttachmentDecision(should_inject=False)

    attachment_count = _plan_mode_attachment_count_since_exit(messages)
    reminder_type = (
        "full"
        if (attachment_count + 1) % FULL_REMINDER_EVERY_N_ATTACHMENTS == 1
        else "sparse"
    )
    return PlanModeAttachmentDecision(should_inject=True, reminder_type=reminder_type)


def resolve_plan_file_path(
    *,
    working_directory: str | Path | None = None,
    agent_id: Optional[str] = None,
) -> Path:
    """Return the canonical plan file path for the current session scope.

    Main sessions store their plan file in ``.ripperdoc/plans/main.md`` under the
    active project/working directory. Subagents get deterministic, isolated plan files.
    """

    base_dir = Path(working_directory).expanduser() if working_directory else Path.cwd()
    try:
        project_dir = project_config_dir(base_dir.resolve())
    except (OSError, RuntimeError, ValueError):
        project_dir = project_config_dir(base_dir)

    safe_name = "main"
    if agent_id:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in agent_id)
        safe_name = f"agent-{cleaned or 'unknown'}"

    plan_dir = project_dir / PLAN_DIR_NAME
    return plan_dir / f"{safe_name}.md"


def ensure_plan_file_directory(plan_file_path: str | Path) -> Path:
    """Ensure the parent directory for the plan file exists and return the path."""

    path = Path(plan_file_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_plan_file_path(path: str | Path | None, plan_file_path: str | Path | None) -> bool:
    """Return whether ``path`` resolves to the active plan file path."""

    if not path or not plan_file_path:
        return False
    try:
        return Path(path).expanduser().resolve() == Path(plan_file_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def build_plan_mode_full_system_prompt(
    *,
    plan_file_path: str | Path,
    available_tool_names: Iterable[str],
) -> str:
    """Build a Claude Code style system instruction block for plan mode."""

    tool_names = set(available_tool_names)
    read_tools = [name for name in ("Read", "Glob", "Grep") if name in tool_names]
    read_tools_text = ", ".join(read_tools) if read_tools else "read-only tools"
    plan_file = str(Path(plan_file_path))
    create_tool = "Write" if "Write" in tool_names else "Edit"
    update_tool = "MultiEdit" if "MultiEdit" in tool_names else "Edit"
    ask_tool = "AskUserQuestion" if "AskUserQuestion" in tool_names else "your question tool"
    approval_tool = "ExitPlanMode" if "ExitPlanMode" in tool_names else "the plan approval tool"

    return (
        "Plan mode is active. The user indicated that they do not want you to execute yet. "
        "You MUST NOT make any edits, run any non-read-only tools, change configs, commit code, "
        "or otherwise modify the system, except for the plan file described below. "
        "This supersedes any other instruction that would cause implementation.\n\n"
        "## Plan File Info\n"
        f"- The plan file path is `{plan_file}`.\n"
        f"- This is the ONLY file you may edit while plan mode is active.\n"
        f"- If the file does not exist yet, create it with `{create_tool}`.\n"
        f"- If it already exists, refine it incrementally with `{update_tool}` or `{create_tool}`.\n\n"
        "## Required Workflow\n"
        f"1. Explore the codebase using {read_tools_text}.\n"
        "2. Capture discoveries in the plan file as you go.\n"
        f"3. Use `{ask_tool}` when requirements or tradeoffs need user input.\n"
        f"4. When the plan is complete, call `{approval_tool}` to request approval.\n\n"
        "## Important Constraints\n"
        "- Do not edit any non-plan files.\n"
        "- Do not run Bash or any other mutating tool.\n"
        "- Do not ask for plan approval in plain text; use the plan approval tool.\n"
        "- End turns in plan mode only by asking clarifying questions or by requesting plan approval."
    )


def build_plan_mode_sparse_system_prompt(
    *,
    plan_file_path: str | Path,
    available_tool_names: Iterable[str],
) -> str:
    """Build the sparse follow-up reminder used between full plan-mode attachments."""

    tool_names = set(available_tool_names)
    ask_tool = "AskUserQuestion" if "AskUserQuestion" in tool_names else "your question tool"
    approval_tool = "ExitPlanMode" if "ExitPlanMode" in tool_names else "the plan approval tool"
    workflow_instructions = "Follow the iterative planning workflow from earlier in the conversation."
    return (
        "Plan mode still active (see full instructions earlier in conversation). "
        f"Read-only except plan file ({Path(plan_file_path)}). "
        f"{workflow_instructions} "
        f"End turns with `{ask_tool}` (for clarifications) or `{approval_tool}` (for plan approval). "
        f"Never ask about plan approval via text or `{ask_tool}`."
    )


def build_plan_mode_reentry_system_prompt(
    *,
    plan_file_path: str | Path,
    available_tool_names: Iterable[str],
) -> str:
    """Build the plan-mode re-entry reminder."""

    tool_names = set(available_tool_names)
    approval_tool = "ExitPlanMode" if "ExitPlanMode" in tool_names else "the plan approval tool"
    return (
        "## Re-entering Plan Mode\n\n"
        f"You are returning to plan mode after having previously exited it. A plan file exists at {Path(plan_file_path)} "
        "from your previous planning session.\n\n"
        "**Before proceeding with any new planning, you should:**\n"
        "1. Read the existing plan file to understand what was previously planned\n"
        "2. Evaluate the user's current request against that plan\n"
        "3. Decide how to proceed:\n"
        "   - **Different task**: If the user's request is for a different task, start fresh by overwriting the existing plan\n"
        "   - **Same task, continuing**: If this is explicitly a continuation of the exact same task, modify the existing plan while cleaning up outdated sections\n"
        f"4. Continue on with the plan process and always edit the plan file one way or the other before calling `{approval_tool}`\n\n"
        "Treat this as a fresh planning session. Do not assume the existing plan is relevant without evaluating it first."
    )


def build_plan_mode_exit_system_prompt(*, plan_file_path: str | Path, plan_exists: bool) -> str:
    """Build the plan-mode exit reminder."""

    suffix = f" The plan file is located at {Path(plan_file_path)} if you need to reference it." if plan_exists else ""
    return (
        "## Exited Plan Mode\n\n"
        "You have exited plan mode. You can now make edits, run tools, and take actions."
        f"{suffix}"
    )


def build_rejected_plan_user_message(plan: str) -> str:
    """Build the synthetic user message Claude Code uses after plan rejection."""

    return (
        "The agent proposed a plan that was rejected by the user. The user chose to stay in plan mode rather than proceed with implementation.\n\n"
        "Rejected plan:\n"
        f"{plan}"
    )


def build_plan_mode_system_prompt(
    *,
    plan_file_path: str | Path,
    available_tool_names: Iterable[str],
) -> str:
    """Backwards-compatible alias for the full plan-mode reminder."""

    return build_plan_mode_full_system_prompt(
        plan_file_path=plan_file_path,
        available_tool_names=available_tool_names,
    )
