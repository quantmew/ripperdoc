from pathlib import Path

from rich.console import Console

from ripperdoc.cli.commands.skills_cmd import command as skills_command
from ripperdoc.core.skills import SkillLocation, get_disabled_skill_names, save_disabled_skill_names


class _DummyUI:
    def __init__(self, console: Console, project_path: Path) -> None:
        self.console = console
        self.project_path = project_path


def _write_skill(project_dir: Path, name: str, description: str) -> None:
    skill_dir = project_dir / ".ripperdoc" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n"
        "Skill body.\n"
    )
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def test_skills_list_shows_disabled_state(tmp_path: Path) -> None:
    _write_skill(tmp_path, "helper", "Provides help")
    save_disabled_skill_names(
        ["helper"],
        project_path=tmp_path,
        location=SkillLocation.PROJECT,
    )

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    skills_command.handler(ui, "list")

    output = ui.console.export_text()
    assert "helper" in output
    assert "disabled" in output


def test_skills_enable_disable_updates_skill_state_file(tmp_path: Path) -> None:
    _write_skill(tmp_path, "helper", "Provides help")
    ui = _DummyUI(Console(record=True, width=120), tmp_path)

    skills_command.handler(ui, "disable helper")
    disabled = get_disabled_skill_names(project_path=tmp_path, location=SkillLocation.PROJECT)
    assert "helper" in disabled

    skills_command.handler(ui, "enable helper")
    enabled = get_disabled_skill_names(project_path=tmp_path, location=SkillLocation.PROJECT)
    assert "helper" not in enabled
