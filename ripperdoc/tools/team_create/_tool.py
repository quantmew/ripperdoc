"""TeamCreate tool — creates a team collaboration domain."""

from __future__ import annotations

import os
import random
import string
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseExample,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.tools.team_create._prompt import TEAM_CREATE_PROMPT
from ripperdoc.utils.collaboration.team_context import (
    remember_active_team,
    resolve_active_team_name,
)
from ripperdoc.utils.collaboration.tasks import (
    ensure_task_list_dir,
    reset_task_list,
    set_leader_team_name,
)
from ripperdoc.utils.collaboration.teams import (
    TEAM_LEAD_NAME,
    TeamMember,
    create_team,
    format_agent_id,
    get_team,
    list_teams,
    participant_color,
    register_team_for_session_cleanup,
    set_active_team_name,
    team_config_path,
)
from ripperdoc.utils.log import get_logger


logger = get_logger()


def _random_suffix(length: int = 4) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


_ADJECTIVES = ("swift", "bright", "calm", "bold", "keen")
_NOUNS = ("river", "stone", "grove", "ridge", "forge")


def _generate_unique_team_name(provided_name: str) -> str:
    """Return *provided_name* if available, else generate a unique slug."""
    clean = (provided_name or "").strip()
    if clean and get_team(clean) is None:
        return clean

    existing = {t.name for t in list_teams()}
    for _ in range(50):
        slug = f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}-{_random_suffix()}"
        if slug not in existing:
            return slug
    raise RuntimeError("Could not generate a unique team name")


class TeamCreateInput(BaseModel):
    team_name: str
    description: Optional[str] = None
    team_description: Optional[str] = None
    team_lead: Optional[str] = None
    agent_type: Optional[str] = None
    cwd: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize_compat_fields(self) -> "TeamCreateInput":
        if not self.description and self.team_description:
            self.description = self.team_description
        return self


class TeamCreateOutput(BaseModel):
    team_name: str
    team_file_path: str
    lead_agent_id: str


class TeamCreateTool(Tool[TeamCreateInput, TeamCreateOutput]):
    @property
    def name(self) -> str:
        return "TeamCreate"

    async def description(self) -> str:
        return "Create a new team for coordinating multiple agents"

    @property
    def input_schema(self) -> type[TeamCreateInput]:
        return TeamCreateInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Create a team for auth refactor",
                example={
                    "team_name": "auth-refactor",
                    "description": "Refactor authentication module",
                    "agent_type": "general-purpose",
                },
            )
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TEAM_CREATE_PROMPT

    def needs_permissions(self, _input_data: Optional[TeamCreateInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: TeamCreateInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if not input_data.team_name or not input_data.team_name.strip():
            return ValidationResult(result=False, message="team_name is required for TeamCreate")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TeamCreateOutput) -> str:
        return (
            f"Created team '{output.team_name}' (lead={output.lead_agent_id}) at "
            f"{output.team_file_path}"
        )

    def render_tool_use_message(self, input_data: TeamCreateInput, _verbose: bool = False) -> str:
        return f"create team: {input_data.team_name}"

    async def call(
        self,
        input_data: TeamCreateInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        # Check if already in a team — restrict to one team per leader
        existing_active_team = resolve_active_team_name(context, allow_single_team_fallback=False)
        if existing_active_team and get_team(existing_active_team) is not None:
            raise ValueError(
                f"Already leading team \"{existing_active_team}\". "
                "A leader can only manage one team at a time. "
                "Use TeamDelete to end the current team before creating a new one."
            )

        # If team name already exists, generate a unique name instead of failing
        final_team_name = _generate_unique_team_name(input_data.team_name.strip())
        if final_team_name != input_data.team_name.strip():
            logger.info(
                "[team_create] Team name '%s' already exists, using '%s'",
                input_data.team_name,
                final_team_name,
            )

        lead_name = (input_data.team_lead or "").strip() or TEAM_LEAD_NAME
        lead_agent_type = (input_data.agent_type or "").strip() or "general-purpose"
        lead_agent_id = format_agent_id(lead_name, final_team_name)

        metadata: Dict[str, Any] = {
            "team_lead": lead_name,
            "cwd": input_data.cwd or os.getcwd(),
        }
        if input_data.description is not None:
            metadata["description"] = input_data.description
        metadata["agent_type"] = lead_agent_type

        # Store session ID for team discovery
        session_id = os.getenv("RIPPERDOC_SESSION_ID", "")
        if session_id:
            metadata["lead_session_id"] = session_id

        lead_member = TeamMember(
            name=lead_name,
            agent_id=lead_agent_id,
            agent_type=lead_agent_type,
            backend_type="in-process",
            color=participant_color(lead_agent_id),
            role="lead",
            active=True,
            metadata={"is_team_lead": True},
        )

        try:
            team = create_team(name=final_team_name, metadata=metadata, members=[lead_member])
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            logger.warning("[team_create] TeamCreate failed: %s: %s", type(exc).__name__, exc)
            raise ValueError(f"TeamCreate failed: {exc}") from exc

        # Reset and create the corresponding task list directory
        reset_task_list(team.task_list_id)
        ensure_task_list_dir(task_list_id=team.task_list_id)

        # Register the team name so task resolution works for the leader
        set_leader_team_name(team.name)

        # Track for session-end cleanup
        register_team_for_session_cleanup(team.name, session_id)

        # Set active team context
        set_active_team_name(team.name)
        remember_active_team(context, team.name)

        logger.info(
            "[team_create] Created team '%s' (lead=%s)",
            team.name,
            lead_agent_id,
        )

        output = TeamCreateOutput(
            team_name=team.name,
            team_file_path=str(team_config_path(team.name)),
            lead_agent_id=lead_agent_id,
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
