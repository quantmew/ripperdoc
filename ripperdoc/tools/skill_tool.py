"""Skill loader tool.

Loads SKILL.md content from .ripperdoc/skills or ~/.ripperdoc/skills so the
assistant can pull in specialized instructions only when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.skills import SkillDefinition, find_skill, is_skill_disabled
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
    display_name: Optional[str] = None
    description: str
    location: str
    base_dir: str
    path: str
    allowed_tools: List[str] = Field(default_factory=list)
    argument_hint: Optional[str] = None
    argument_names: List[str] = Field(default_factory=list)
    when_to_use: Optional[str] = None
    version: Optional[str] = None
    user_invocable: bool = True
    model: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    skill_type: str = "prompt"
    disable_model_invocation: bool = False
    context: Optional[str] = None
    agent: Optional[str] = None
    paths: List[str] = Field(default_factory=list)
    hooks: Optional[Dict[str, Any]] = None
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
            "Skill metadata may include allowed-tools, model, context, agent, paths, or max-thinking-tokens hints."
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
            "If the skill specifies allowed-tools, model, context, agent, paths, or max-thinking-tokens in frontmatter, "
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
            if is_skill_disabled(skill_name, project_path=self._project_path, home=self._home):
                return ValidationResult(
                    result=False,
                    message=f"Skill {skill_name} is disabled. Re-enable it with /skills.",
                    error_code=6,
                )
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

    def _list_skill_files(self, base_dir: Path, max_depth: int = 2) -> List[str]:
        """List documentation files in the skill directory (excluding SKILL.md)."""
        files: List[str] = []
        doc_extensions = {".md", ".txt", ".rst", ".json", ".yaml", ".yml"}

        def scan_dir(dir_path: Path, depth: int, prefix: str = "") -> None:
            if depth > max_depth or not dir_path.exists():
                return
            try:
                entries = sorted(dir_path.iterdir())
            except PermissionError:
                return

            for entry in entries:
                # Skip hidden files/directories and SKILL.md
                if entry.name.startswith(".") or entry.name == "SKILL.md":
                    continue
                rel_path = f"{prefix}{entry.name}"
                if entry.is_dir():
                    files.append(f"{rel_path}/")
                    scan_dir(entry, depth + 1, f"{rel_path}/")
                elif entry.suffix.lower() in doc_extensions:
                    files.append(rel_path)

        scan_dir(base_dir, 0)
        return files

    def _render_result(self, skill: SkillDefinition) -> str:
        allowed = ", ".join(skill.allowed_tools) if skill.allowed_tools else "no specific limit"
        display_name = f"\nDisplay name: {skill.display_name}" if skill.display_name else ""
        model_hint = f"\nModel hint: {skill.model}" if skill.model else ""
        max_tokens = (
            f"\nMax thinking tokens hint: {skill.max_thinking_tokens}"
            if skill.max_thinking_tokens is not None
            else ""
        )
        argument_hint = f"\nArgument hint: {skill.argument_hint}" if skill.argument_hint else ""
        argument_names = (
            f"\nArgument names: {', '.join(skill.argument_names)}"
            if skill.argument_names
            else ""
        )
        when_to_use = f"\nWhen to use: {skill.when_to_use}" if skill.when_to_use else ""
        version = f"\nVersion: {skill.version}" if skill.version else ""
        user_invocable = f"\nUser invocable: {'yes' if skill.user_invocable else 'no'}"
        exec_context = (
            f"\nExecution context: {skill.execution_context.value}"
            if skill.execution_context is not None
            else ""
        )
        agent_hint = f"\nPreferred agent: {skill.agent}" if skill.agent else ""
        path_hints = f"\nPath scopes: {', '.join(skill.paths)}" if skill.paths else ""

        # List available documentation files in skill directory
        skill_files = self._list_skill_files(skill.base_dir)
        files_section = ""
        if skill_files:
            files_list = "\n".join(f"  - {f}" for f in skill_files)
            files_section = (
                f"\n\nAvailable documentation files in skill directory (use Read tool to access when needed):\n"
                f"{files_list}"
            )

        lines = [
            f"Skill loaded: {skill.name} ({skill.location.value})",
            f"Description: {skill.description}",
            f"Skill directory: {skill.base_dir}",
            (
                "Allowed tools (if specified): "
                f"{allowed}{display_name}{argument_hint}{argument_names}{when_to_use}"
                f"{version}{user_invocable}{model_hint}{max_tokens}{exec_context}{agent_hint}{path_hints}"
            ),
            "SKILL.md content:",
            skill.content,
        ]
        result = "\n".join(lines)
        return result + files_section

    def _to_output(self, skill: SkillDefinition) -> SkillToolOutput:
        hooks_payload: Optional[Dict[str, Any]] = None
        if skill.hooks and skill.hooks.hooks:
            hooks_payload = (
                skill.hooks.model_dump(by_alias=True, exclude_none=True).get("hooks") or None
            )
        return SkillToolOutput(
            success=True,
            skill=skill.name,
            display_name=skill.display_name,
            description=skill.description,
            location=skill.location.value,
            base_dir=str(skill.base_dir),
            path=str(skill.path),
            allowed_tools=list(skill.allowed_tools),
            argument_hint=skill.argument_hint,
            argument_names=list(skill.argument_names),
            when_to_use=skill.when_to_use,
            version=skill.version,
            user_invocable=skill.user_invocable,
            model=skill.model,
            max_thinking_tokens=skill.max_thinking_tokens,
            skill_type=skill.skill_type,
            disable_model_invocation=skill.disable_model_invocation,
            context=skill.execution_context.value if skill.execution_context else None,
            agent=skill.agent,
            paths=list(skill.paths),
            hooks=hooks_payload,
            content=skill.content,
        )

    async def call(
        self, input_data: SkillToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:  # noqa: ARG002
        skill_name = (input_data.skill or "").strip().lstrip("/")
        skill = find_skill(skill_name, project_path=self._project_path, home=self._home)
        if not skill:
            if is_skill_disabled(skill_name, project_path=self._project_path, home=self._home):
                error_text = f"Skill '{skill_name}' is disabled. Re-enable it with /skills."
            else:
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
        display_name = f"\nDisplay name: {output.display_name}" if output.display_name else ""
        model_hint = f"\nModel hint: {output.model}" if output.model else ""
        max_tokens = (
            f"\nMax thinking tokens hint: {output.max_thinking_tokens}"
            if output.max_thinking_tokens is not None
            else ""
        )
        argument_hint = f"\nArgument hint: {output.argument_hint}" if output.argument_hint else ""
        argument_names = (
            f"\nArgument names: {', '.join(output.argument_names)}"
            if output.argument_names
            else ""
        )
        when_to_use = f"\nWhen to use: {output.when_to_use}" if output.when_to_use else ""
        version = f"\nVersion: {output.version}" if output.version else ""
        user_invocable = f"\nUser invocable: {'yes' if output.user_invocable else 'no'}"
        exec_context = f"\nExecution context: {output.context}" if output.context else ""
        agent_hint = f"\nPreferred agent: {output.agent}" if output.agent else ""
        path_hints = f"\nPath scopes: {', '.join(output.paths)}" if output.paths else ""

        # List available documentation files in skill directory
        skill_files = self._list_skill_files(Path(output.base_dir))
        files_section = ""
        if skill_files:
            files_list = "\n".join(f"  - {f}" for f in skill_files)
            files_section = (
                f"\n\nAvailable documentation files in skill directory (use Read tool to access when needed):\n"
                f"{files_list}"
            )

        return (
            f"Skill loaded: {output.skill} ({output.location})\n"
            f"Description: {output.description}\n"
            f"Skill directory: {output.base_dir}\n"
            "Allowed tools (if specified): "
            f"{allowed}{display_name}{argument_hint}{argument_names}{when_to_use}{version}"
            f"{user_invocable}{model_hint}{max_tokens}{exec_context}{agent_hint}{path_hints}\n"
            "SKILL.md content:\n"
            f"{output.content}"
            f"{files_section}"
        )

    def render_tool_use_message(self, input_data: SkillToolInput, verbose: bool = False) -> str:  # noqa: ARG002
        return f"Load skill '{input_data.skill}'"
