from pathlib import Path
from typing import Optional

import pytest

from ripperdoc.core.skills import SkillLocation, build_skill_summary, load_all_skills
from ripperdoc.core.tool import ToolUseContext
from ripperdoc.tools.skill_tool import SkillTool, SkillToolInput


def _write_skill(
    root: Path,
    name: str,
    description: str,
    body: str,
    *,
    extra_frontmatter: Optional[str] = None,
) -> Path:
    skill_dir = root / ".ripperdoc" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    extra = (extra_frontmatter or "").strip()
    if extra:
        lines.extend(extra.splitlines())
    lines.append("---")
    lines.append(body)
    content = "\n".join(lines) + "\n"
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(content, encoding="utf-8")
    return skill_path


def test_load_all_skills_prefers_project_and_parses_fields(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    home_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    _write_skill(
        home_dir,
        "shared-skill",
        "User version",
        "User content",
    )
    _write_skill(
        project_dir,
        "shared-skill",
        "Project override",
        "Project content",
    )
    _write_skill(
        project_dir,
        "worker",
        "Handles work",
        "Do the work well.",
        extra_frontmatter="allowed-tools: View, Grep\nmodel: gpt-4o\nmax-thinking-tokens: 128",
    )

    result = load_all_skills(project_path=project_dir, home=home_dir)
    assert not result.errors
    skills = {skill.name: skill for skill in result.skills}

    assert "shared-skill" in skills
    assert skills["shared-skill"].description == "Project override"
    assert skills["shared-skill"].location == SkillLocation.PROJECT

    worker = skills["worker"]
    assert worker.allowed_tools == ["View", "Grep"]
    assert worker.model == "gpt-4o"
    assert worker.max_thinking_tokens == 128
    assert worker.content.strip().startswith("Do the work")

    summary = build_skill_summary(result.skills)
    assert "Skills" in summary
    assert "worker" in summary
    assert "(project)" in summary


@pytest.mark.asyncio
async def test_skill_tool_loads_skill_content(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    home_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    _write_skill(
        project_dir,
        "helper",
        "Provides help",
        "Helper skill body.",
        extra_frontmatter="allowed-tools: View",
    )

    tool = SkillTool(project_path=project_dir, home=home_dir)
    input_data = SkillToolInput(skill="helper")

    validation = await tool.validate_input(input_data, ToolUseContext())
    assert validation.result

    context = ToolUseContext()
    results = []
    async for output in tool.call(input_data, context):
        results.append(output)

    assert len(results) == 1
    result = results[0]
    assert result.result_for_assistant
    assert "SKILL.md content" in result.result_for_assistant

    data = result.data
    assert getattr(data, "skill", None) == "helper"
    assert getattr(data, "allowed_tools", []) == ["View"]

    missing_validation = await tool.validate_input(SkillToolInput(skill="missing"), ToolUseContext())
    assert not missing_validation.result


def test_build_skill_summary_handles_empty() -> None:
    summary = build_skill_summary([])
    assert "No skills detected" in summary
    assert ".ripperdoc/skills" in summary
