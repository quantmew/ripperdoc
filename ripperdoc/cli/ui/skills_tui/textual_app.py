"""Textual app for enabling/disabling skills."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from ripperdoc.core.skills import (
    SkillDefinition,
    SkillLocation,
    get_disabled_skill_names,
    load_all_skills,
    save_disabled_skill_names,
)


def _display_skill_name(skill_name: str) -> str:
    words = [part for part in skill_name.replace("_", "-").split("-") if part]
    if not words:
        return skill_name
    return " ".join(word.capitalize() for word in words)


class SkillsApp(App[None]):
    CSS = """
    #title {
        text-style: bold;
        padding: 1 1 0 1;
    }

    #description {
        color: $text-muted;
        padding: 0 1 0 1;
    }

    #search_header {
        color: $text-muted;
        padding: 1 1 0 1;
    }

    #search_value {
        color: $text;
        padding: 0 1 0 1;
    }

    #skills_list {
        margin: 0 1 1 1;
        height: 1fr;
    }

    #hint {
        color: $text-muted;
        padding: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
    ]

    def __init__(self, project_path: Optional[Path]) -> None:
        super().__init__()
        self._project_path = project_path
        self._skills: list[SkillDefinition] = []
        self._skills_by_name: dict[str, SkillDefinition] = {}
        self._disabled_by_location: dict[SkillLocation, set[str]] = {}
        self._search_query = ""
        self._visible_skill_names: list[str] = []
        self._highlighted_skill_name: Optional[str] = None
        self._updating_list = False

    def compose(self) -> ComposeResult:
        yield Static("Enable/Disable Skills", id="title")
        yield Static("Turn skills on or off. Your changes are saved automatically.", id="description")
        yield Static("Type to search skills", id="search_header")
        yield Static("> ", id="search_value")
        yield OptionList(id="skills_list", markup=False)
        yield Static("Press space or enter to toggle; esc to close", id="hint")

    def on_mount(self) -> None:
        result = load_all_skills(self._project_path)
        location_order = {
            SkillLocation.PROJECT: 0,
            SkillLocation.USER: 1,
            SkillLocation.OTHER: 2,
        }
        self._skills = sorted(
            result.skills,
            key=lambda s: (location_order.get(s.location, 9), s.name),
        )
        self._skills_by_name = {skill.name: skill for skill in self._skills}
        self._disabled_by_location = {
            SkillLocation.USER: get_disabled_skill_names(
                self._project_path, location=SkillLocation.USER
            ),
            SkillLocation.PROJECT: get_disabled_skill_names(
                self._project_path, location=SkillLocation.PROJECT
            ),
        }
        self._refresh_view()
        self.query_one("#skills_list", OptionList).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "space":
            self._toggle_highlighted()
            event.stop()
            return
        if event.key == "backspace":
            if self._search_query:
                self._search_query = self._search_query[:-1]
                self._refresh_view()
                event.stop()
            return
        if event.character and event.character.isprintable():
            if event.character not in ("\t", "\n"):
                self._search_query += event.character
                self._refresh_view()
                event.stop()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if self._updating_list:
            return
        option_id = event.option.id or ""
        if not option_id.startswith("skill:"):
            return
        skill_name = option_id.split(":", 1)[1]
        if skill_name == self._highlighted_skill_name:
            return
        self._highlighted_skill_name = skill_name
        self._refresh_list(preserve_highlight=True)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if not option_id.startswith("skill:"):
            return
        skill_name = option_id.split(":", 1)[1]
        self._toggle_skill(skill_name)

    def action_close(self) -> None:
        self.exit()

    def _refresh_view(self) -> None:
        self.query_one("#search_value", Static).update(f"> {self._search_query}")
        self._refresh_list(preserve_highlight=True)

    def _refresh_list(self, preserve_highlight: bool = False) -> None:
        option_list = self.query_one("#skills_list", OptionList)
        previous = self._highlighted_skill_name if preserve_highlight else None
        if previous is None and self._visible_skill_names:
            idx = option_list.highlighted or 0
            if 0 <= idx < len(self._visible_skill_names):
                previous = self._visible_skill_names[idx]

        self._updating_list = True
        option_list.clear_options()
        self._visible_skill_names = []

        filtered = self._filtered_skills()
        if not filtered:
            option_list.add_option(Option("No skills match your search.", id="empty"))
            option_list.highlighted = 0
            self._highlighted_skill_name = None
            self._updating_list = False
            return

        name_width = max(12, min(24, max(len(_display_skill_name(s.name)) for s in filtered)))
        filtered_names = {skill.name for skill in filtered}
        target_highlight = previous if previous in filtered_names else filtered[0].name
        highlighted_idx = 0
        for skill in filtered:
            self._visible_skill_names.append(skill.name)
            disabled = skill.name in self._disabled_by_location.get(skill.location, set())
            checked = " " if disabled else "x"
            display_name = _display_skill_name(skill.name)
            if skill.name == target_highlight:
                highlighted_idx = len(self._visible_skill_names) - 1
            is_highlighted = skill.name == target_highlight
            cursor = "› " if is_highlighted else "  "
            label = f"{cursor}[{checked}] {display_name:<{name_width}}  {skill.description}"
            option_list.add_option(Option(label, id=f"skill:{skill.name}"))
        option_list.highlighted = highlighted_idx
        self._highlighted_skill_name = self._visible_skill_names[highlighted_idx]
        self._updating_list = False

    def _filtered_skills(self) -> list[SkillDefinition]:
        if not self._search_query:
            return list(self._skills)
        needle = self._search_query.lower()
        filtered: list[SkillDefinition] = []
        for skill in self._skills:
            display_name = _display_skill_name(skill.name).lower()
            if needle in skill.name.lower() or needle in display_name or needle in skill.description.lower():
                filtered.append(skill)
        return filtered

    def _toggle_highlighted(self) -> None:
        option_list = self.query_one("#skills_list", OptionList)
        idx = option_list.highlighted
        if idx is None:
            return
        if idx < 0 or idx >= len(self._visible_skill_names):
            return
        self._toggle_skill(self._visible_skill_names[idx])

    def _toggle_skill(self, skill_name: str) -> None:
        skill = self._skills_by_name.get(skill_name)
        if not skill:
            return
        disabled = self._disabled_by_location.setdefault(skill.location, set())
        if skill.name in disabled:
            disabled.remove(skill.name)
        else:
            disabled.add(skill.name)
        save_disabled_skill_names(disabled, self._project_path, location=skill.location)
        self._highlighted_skill_name = skill_name
        self._refresh_list(preserve_highlight=True)


def run_skills_tui(project_path: Optional[Path]) -> bool:
    """Run the Textual skills UI."""
    app = SkillsApp(project_path)
    app.run()
    return True
