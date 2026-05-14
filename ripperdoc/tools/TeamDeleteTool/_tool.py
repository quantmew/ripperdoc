"""TeamDelete tool — removes a team and its task resources."""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from pydantic import BaseModel, ConfigDict

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.tools.TeamDeleteTool._prompt import TEAM_DELETE_PROMPT
from ripperdoc.utils.collaboration.team_context import (
    clear_agent_active_team,
    resolve_active_team_name,
)
from ripperdoc.utils.collaboration.teams import (
    TEAM_LEAD_NAME,
    clear_active_team_name,
    cleanup_team_directories,
    clear_mailbox,
    get_team,
    unregister_team_for_session_cleanup,
)
from ripperdoc.utils.log import get_logger


logger = get_logger()


class TeamDeleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TeamDeleteOutput(BaseModel):
    success: bool
    message: str
    team_name: Optional[str] = None


class TeamDeleteTool(Tool[TeamDeleteInput, TeamDeleteOutput]):
    @property
    def name(self) -> str:
        return "TeamDelete"

    async def description(self) -> str:
        return "Clean up team and task directories when the swarm is complete"

    @property
    def input_schema(self) -> type[TeamDeleteInput]:
        return TeamDeleteInput

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TEAM_DELETE_PROMPT

    def needs_permissions(self, _input_data: Optional[TeamDeleteInput] = None) -> bool:
        return False

    def render_result_for_assistant(self, output: TeamDeleteOutput) -> str:
        return output.message

    def render_tool_use_message(self, _input_data: TeamDeleteInput, _verbose: bool = False) -> str:
        return "cleanup team: current"

    async def call(
        self,
        input_data: TeamDeleteInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        del input_data
        team_name = resolve_active_team_name(context)

        if team_name:
            team = get_team(team_name)

            if team is not None:
                # Filter out the team lead — only count non-lead members
                non_lead_members = [
                    member for member in team.members
                    if (member.name or "").strip() != TEAM_LEAD_NAME
                ]

                # Separate truly active members from idle/dead ones
                active_members = [
                    member for member in non_lead_members
                    if member.active
                ]

                if active_members:
                    member_names = ", ".join(
                        member.name for member in active_members if member.name
                    )
                    output = TeamDeleteOutput(
                        success=False,
                        message=(
                            f"Cannot cleanup team with {len(active_members)} active member(s): "
                            f"{member_names}. Use requestShutdown to gracefully terminate "
                            "teammates first."
                        ),
                        team_name=team_name,
                    )
                    yield ToolResult(
                        data=output,
                        result_for_assistant=self.render_result_for_assistant(output),
                    )
                    return

                # Clear member inboxes before directory removal
                for member in team.members:
                    if member.name:
                        try:
                            clear_mailbox(member.name, team_name)
                        except Exception as exc:
                            logger.debug(
                                "[team_delete] Could not clear inbox for %s: %s",
                                member.name,
                                exc,
                            )

                # Comprehensive cleanup: team dir + task dir + inbox dir
                cleanup_team_directories(team_name, task_list_id=team.task_list_id)

                # Already cleaned — don't try again on session end
                unregister_team_for_session_cleanup(team_name)

                # Clear leader team name so task resolution falls back
                from ripperdoc.utils.collaboration.tasks import clear_leader_team_name

                clear_leader_team_name()

                logger.info("[team_delete] Cleaned up directories for team '%s'", team_name)

            # Clear team context
            clear_active_team_name(team_name)
            clear_agent_active_team(context)

            output = TeamDeleteOutput(
                success=True,
                message=(
                    f"Cleaned up directories and worktrees for team \"{team_name}\""
                    if team
                    else f"Cleaned up context for team \"{team_name}\" (config already absent)"
                ),
                team_name=team_name,
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
            return

        # No team found — graceful return
        output = TeamDeleteOutput(
            success=True,
            message="No team name found, nothing to clean up",
            team_name=None,
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
