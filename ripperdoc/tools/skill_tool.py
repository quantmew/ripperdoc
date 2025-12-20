"""Skill loader tool.

Loads SKILL.md content from .ripperdoc/skills or ~/.ripperdoc/skills so the
assistant can pull in specialized instructions only when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.skills import SkillDefinition, find_skill
from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()


class SkillToolInput(BaseModel):
    """Input schema for the Skill tool."""

    skill: str = Field(description='The skill name (e.g. "pdf-processing").')


class SkillToolOutput(BaseModel):
    """Structured output for a loaded skill."""

    success: bool = True
    skill: str
    description: str
    location: str
    base_dir: str
    path: str
    allowed_tools: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    skill_type: str = "prompt"
    disable_model_invocation: bool = False
    content: str


class SkillTool(Tool[SkillToolInput, SkillToolOutput]):
    """Load a skill's instructions by name."""

    def __init__(self, project_path: Optional[Path] = None, home: Optional[Path] = None) -> None:
        self._project_path = project_path
        self._home = home

    @property
    def name(self) -> str:
        return "Skill"

    async def description(self) -> str:
        return (
            "Execute a skill by name to load its SKILL.md instructions. "
            "Use this only when the skill description clearly matches the user's request. "
            "Skill metadata may include allowed-tools, model, or max-thinking-tokens hints."
        )

    @property
    def input_schema(self) -> type[SkillToolInput]:
        return SkillToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Load PDF processing guidance",
                example={"skill": "pdf-processing"},
            ),
            ToolUseExample(
                description="Load commit message helper instructions",
                example={"skill": "generating-commit-messages"},
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Load a skill by name to read its SKILL.md content. "
            "Only call this when the skill description is clearly relevant. "
            "If the skill specifies allowed-tools, model, or max-thinking-tokens in frontmatter, "
            "assume those hints apply for subsequent reasoning. "
            "Skill files may reference additional files under the same directory; "
            "use file tools to read them if needed."
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[SkillToolInput] = None) -> bool:  # noqa: ARG002
        return False

    async def validate_input(
        self,
        input_data: SkillToolInput,
        context: Optional[ToolUseContext] = None,  # noqa: ARG002
    ) -> ValidationResult:
        skill_name = (input_data.skill or "").strip().lstrip("/")
        if not skill_name:
            return ValidationResult(
                result=False, message="Provide a skill name to load.", error_code=1
            )
        skill = find_skill(skill_name, project_path=self._project_path, home=self._home)
        if not skill:
            return ValidationResult(
                result=False, message=f"Unknown skill: {skill_name}", error_code=2
            )
        if skill.disable_model_invocation:
            return ValidationResult(
                result=False,
                message=f"Skill {skill_name} is blocked by disable-model-invocation.",
                error_code=4,
            )
        if skill.skill_type and skill.skill_type != "prompt":
            return ValidationResult(
                result=False,
                message=f"Skill {skill_name} is not a prompt-based skill (type={skill.skill_type}).",
                error_code=5,
                meta={"skill_type": skill.skill_type},
            )
        return ValidationResult(result=True)

    def _render_result(self, skill: SkillDefinition) -> str:
        allowed = ", ".join(skill.allowed_tools) if skill.allowed_tools else "no specific limit"
        model_hint = f"\nModel hint: {skill.model}" if skill.model else ""
        max_tokens = (
            f"\nMax thinking tokens hint: {skill.max_thinking_tokens}"
            if skill.max_thinking_tokens is not None
            else ""
        )
        lines = [
            f"Skill loaded: {skill.name} ({skill.location.value})",
            f"Description: {skill.description}",
            f"Skill directory: {skill.base_dir}",
            f"Allowed tools (if specified): {allowed}{model_hint}{max_tokens}",
            "SKILL.md content:",
            skill.content,
        ]
        return "\n".join(lines)

    def _to_output(self, skill: SkillDefinition) -> SkillToolOutput:
        return SkillToolOutput(
            success=True,
            skill=skill.name,
            description=skill.description,
            location=skill.location.value,
            base_dir=str(skill.base_dir),
            path=str(skill.path),
            allowed_tools=list(skill.allowed_tools),
            model=skill.model,
            max_thinking_tokens=skill.max_thinking_tokens,
            skill_type=skill.skill_type,
            disable_model_invocation=skill.disable_model_invocation,
            content=skill.content,
        )

    async def call(
        self, input_data: SkillToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:  # noqa: ARG002
        skill_name = (input_data.skill or "").strip().lstrip("/")
        skill = find_skill(skill_name, project_path=self._project_path, home=self._home)
        if not skill:
            error_text = (
                f"Skill '{skill_name}' not found. Ensure it exists under "
                "~/.ripperdoc/skills or ./.ripperdoc/skills."
            )
            yield ToolResult(data={"error": error_text}, result_for_assistant=error_text)
            return
        if skill.allowed_tools and context.tool_registry is not None:
            # Ensure preferred tools for this skill are activated in the registry.
            context.tool_registry.activate_tools(skill.allowed_tools)

        output = self._to_output(skill)
        yield ToolResult(data=output, result_for_assistant=self._render_result(skill))

    def render_result_for_assistant(self, output: SkillToolOutput) -> str:
        allowed = ", ".join(output.allowed_tools) if output.allowed_tools else "no specific limit"
        model_hint = f"\nModel hint: {output.model}" if output.model else ""
        max_tokens = (
            f"\nMax thinking tokens hint: {output.max_thinking_tokens}"
            if output.max_thinking_tokens is not None
            else ""
        )
        return (
            f"Skill loaded: {output.skill} ({output.location})\n"
            f"Description: {output.description}\n"
            f"Skill directory: {output.base_dir}\n"
            f"Allowed tools (if specified): {allowed}{model_hint}{max_tokens}\n"
            "SKILL.md content:\n"
            f"{output.content}"
        )

    def render_tool_use_message(self, input_data: SkillToolInput, verbose: bool = False) -> str:  # noqa: ARG002
        return f"Load skill '{input_data.skill}'"
