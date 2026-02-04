"""Textual app for managing agent definitions and runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rich import box
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    Select,
    Static,
    TextArea,
)

from ripperdoc.core.agents import (
    AgentDefinition,
    AgentLocation,
    delete_agent_definition,
    load_agent_definitions,
    save_agent_definition,
)
from ripperdoc.core.config import get_global_config
from ripperdoc.core.default_tools import BUILTIN_TOOL_NAMES
from ripperdoc.core.query import query_llm
from ripperdoc.tools.task_tool import (
    cancel_agent_run,
    get_agent_run_snapshot,
    list_agent_runs,
)
from ripperdoc.utils.json_utils import safe_parse_json
from ripperdoc.utils.messages import create_user_message


@dataclass
class AgentFormResult:
    name: str
    description: str
    tools: list[str]
    system_prompt: str
    location: AgentLocation
    model: Optional[str] = None


@dataclass
class AgentDraft:
    name: str
    description: str
    tools: list[str]
    system_prompt: str
    location: AgentLocation
    model: Optional[str] = None


class ConfirmScreen(ModalScreen[bool]):
    """Simple confirmation dialog."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm_dialog"):
            yield Static(self._message, id="confirm_message")
            with Horizontal(id="confirm_buttons"):
                yield Button("Yes", id="confirm_yes", variant="primary")
                yield Button("No", id="confirm_no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm_yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


class InfoScreen(ModalScreen[None]):
    """Modal screen for showing rich details."""

    def __init__(self, title: str, renderable: Any) -> None:
        super().__init__()
        self._title = title
        self._renderable = renderable

    def compose(self) -> ComposeResult:
        with Container(id="info_dialog"):
            yield Static(self._title, id="info_title")
            with VerticalScroll(id="info_body"):
                yield Static(self._renderable, id="info_content")
            with Horizontal(id="info_buttons"):
                yield Button("Close", id="info_close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "info_close":
            self.dismiss(None)


class CreateMethodScreen(ModalScreen[Optional[str]]):
    """Select how to create a new agent."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Container(id="method_dialog"):
            yield Static("Create new agent", id="method_title")
            yield Static("Creation method", id="method_subtitle")
            yield Static(
                "Choose how to create the agent configuration.",
                id="method_hint",
            )
            with Horizontal(id="method_buttons"):
                yield Button("Generate with AI", id="method_generate", variant="primary")
                yield Button("Manual configuration", id="method_manual")
                yield Button("Cancel", id="method_cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "method_generate":
            self.dismiss("generate")
        elif event.button.id == "method_manual":
            self.dismiss("manual")
        else:
            self.dismiss(None)


class AgentGenerateScreen(ModalScreen[Optional[str]]):
    """Collect a description for AI-generated agent configs."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._busy: bool = False

    def on_mount(self) -> None:
        spinner = self.query_one("#generate_spinner", LoadingIndicator)
        spinner.display = False

    def compose(self) -> ComposeResult:
        with Container(id="generate_dialog"):
            yield Static("Create new agent", id="generate_title")
            yield Static(
                "Describe what this agent should do and when it should be used.",
                id="generate_subtitle",
            )
            yield Static("", id="generate_status")
            with VerticalScroll(id="generate_fields"):
                yield TextArea(
                    id="generate_input",
                    placeholder="e.g., Help me write unit tests for my code...",
                )
                yield LoadingIndicator(id="generate_spinner")
            with Horizontal(id="generate_buttons"):
                yield Button("Generate", id="generate_submit", variant="primary")
                yield Button("Back", id="generate_cancel")

    def action_cancel(self) -> None:
        if self._busy:
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate_cancel":
            if self._busy:
                return
            self.dismiss(None)
            return
        if event.button.id != "generate_submit":
            return
        description = (self.query_one("#generate_input", TextArea).text or "").strip()
        if not description:
            self._set_error("Please enter a description.")
            return
        app = getattr(self, "app", None)
        if hasattr(app, "start_agent_generation"):
            app.start_agent_generation(description, self)

    def _set_error(self, message: str) -> None:
        if not message:
            return
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)

    def clear_error(self) -> None:
        return

    def set_error(self, message: str) -> None:
        self._set_error(message)

    def set_status(self, message: str) -> None:
        status_widget = self.query_one("#generate_status", Static)
        status_widget.update(message)

    def set_busy(self, busy: bool, message: str = "") -> None:
        input_area = self.query_one("#generate_input", TextArea)
        submit_button = self.query_one("#generate_submit", Button)
        cancel_button = self.query_one("#generate_cancel", Button)
        spinner = self.query_one("#generate_spinner", LoadingIndicator)
        self._busy = busy
        input_area.disabled = busy
        submit_button.disabled = busy
        cancel_button.disabled = busy
        spinner.display = busy
        if message:
            self.set_status(message)


class AgentFormScreen(ModalScreen[Optional[AgentFormResult]]):
    """Modal form for adding/editing agents."""

    def __init__(
        self,
        mode: str,
        *,
        existing_agent: Optional[AgentDefinition] = None,
        draft: Optional[AgentDraft] = None,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._existing_agent = existing_agent
        self._draft = draft
        config = get_global_config()
        pointer_map = config.model_pointers.model_dump()
        self._model_default = (
            (self._existing_agent.model if self._existing_agent else None)
            or (self._draft.model if self._draft else None)
            or pointer_map.get("main", "main")
        )
        self._location_default = (
            self._existing_agent.location
            if self._existing_agent
            else (self._draft.location if self._draft else AgentLocation.USER)
        )

    def compose(self) -> ComposeResult:
        title = "Add agent" if self._mode == "add" else "Edit agent"
        with Container(id="form_dialog"):
            yield Static(title, id="form_title")
            with VerticalScroll(id="form_fields"):
                if self._mode == "add":
                    name_default = self._draft.name if self._draft else ""
                    yield Static("Agent name", classes="field_label")
                    yield Input(
                        value=name_default,
                        placeholder="Agent name",
                        id="name_input",
                    )
                else:
                    name_display = self._existing_agent.agent_type if self._existing_agent else ""
                    yield Static("Agent name", classes="field_label")
                    yield Static(name_display, id="name_static", classes="field_value")

                description_default = ""
                if self._existing_agent:
                    description_default = self._existing_agent.when_to_use
                elif self._draft:
                    description_default = self._draft.description
                yield Static("Description (when to use)", classes="field_label")
                yield Input(
                    value=description_default,
                    placeholder="When should this agent be used?",
                    id="description_input",
                )

                tools_default = "*"
                if self._existing_agent:
                    tools_default = "*" if "*" in self._existing_agent.tools else ", ".join(
                        self._existing_agent.tools
                    )
                elif self._draft:
                    tools_default = (
                        "*"
                        if "*" in self._draft.tools
                        else ", ".join(self._draft.tools)
                    )
                yield Static("Tools", classes="field_label")
                yield Input(
                    value=tools_default,
                    placeholder="Comma-separated tool names or *",
                    id="tools_input",
                )

                yield Static("Model (profile or pointer)", classes="field_label")
                yield Input(
                    value=self._model_default,
                    placeholder="Model profile or pointer",
                    id="model_input",
                )

                location_default = getattr(self._location_default, "value", AgentLocation.USER.value)
                location_options = [
                    (AgentLocation.USER.value, AgentLocation.USER.value),
                    (AgentLocation.PROJECT.value, AgentLocation.PROJECT.value),
                ]
                yield Static("Location", classes="field_label")
                yield Select(location_options, value=location_default, id="location_select")

                prompt_default = ""
                if self._existing_agent:
                    prompt_default = self._existing_agent.system_prompt
                elif self._draft:
                    prompt_default = self._draft.system_prompt
                yield Static("System prompt", classes="field_label")
                yield TextArea(
                    text=prompt_default,
                    placeholder="System prompt",
                    id="prompt_input",
                )

            with Horizontal(id="form_buttons"):
                yield Button("Save", id="form_save", variant="primary")
                yield Button("Cancel", id="form_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form_cancel":
            self.dismiss(None)
            return
        if event.button.id != "form_save":
            return

        name_input = self.query_one("#name_input", Input) if self._mode == "add" else None
        description_input = self.query_one("#description_input", Input)
        tools_input = self.query_one("#tools_input", Input)
        model_input = self.query_one("#model_input", Input)
        location_select = self.query_one("#location_select", Select)
        prompt_input = self.query_one("#prompt_input", TextArea)

        name = self._existing_agent.agent_type if self._existing_agent else ""
        if self._mode == "add":
            name = (name_input.value or "").strip() if name_input else ""
            if not name:
                self._set_error("Agent name is required.")
                return

        description = (description_input.value or "").strip()
        if not description:
            self._set_error("Description is required.")
            return

        tools_raw = (tools_input.value or "").strip() or "*"
        tools = [item.strip() for item in tools_raw.split(",") if item.strip()]
        if not tools:
            tools = ["*"]
        if "*" in tools:
            tools = ["*"]

        system_prompt = (prompt_input.text or "").strip()
        if not system_prompt:
            self._set_error("System prompt is required.")
            return

        location_raw = (location_select.value or "").strip().lower()
        location = AgentLocation.PROJECT if location_raw == "project" else AgentLocation.USER

        model_value = (model_input.value or "").strip()
        model = model_value or self._model_default or None

        self.dismiss(
            AgentFormResult(
                name=name,
                description=description,
                tools=tools,
                system_prompt=system_prompt,
                location=location,
                model=model,
            )
        )

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)


def _extract_assistant_text(message: Any) -> str:
    content = getattr(message, "message", None)
    if content is not None and hasattr(content, "content"):
        content = content.content
    elif hasattr(message, "content"):
        content = message.content

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if not text and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return ""


def _extract_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"```json\\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_json_from_text(text: str) -> Optional[Any]:
    candidate = _extract_json_candidate(text)
    parsed = safe_parse_json(candidate)
    if parsed is not None:
        return parsed
    if not candidate:
        return None
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return safe_parse_json(candidate[start : end + 1])
    return None


def _slugify(value: str) -> str:
    raw = (value or "").lower()
    raw = re.sub(r"[^a-z0-9_-]+", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw).strip("-")
    return raw or "new-agent"


def _coerce_agent_draft(data: dict[str, Any], fallback_description: str) -> AgentDraft:
    name_raw = data.get("name") or data.get("agent_name") or data.get("agent")
    description_raw = data.get("description") or data.get("when_to_use") or fallback_description
    tools_raw = data.get("tools") or data.get("tooling") or ["*"]
    system_prompt_raw = data.get("system_prompt") or data.get("prompt") or ""
    model_raw = data.get("model")
    location_raw = data.get("location") or data.get("scope") or "user"

    name = _slugify(str(name_raw or fallback_description))
    description = str(description_raw or fallback_description).strip() or fallback_description

    tools: list[str] = []
    if isinstance(tools_raw, str):
        tools = [item.strip() for item in tools_raw.split(",") if item.strip()]
    elif isinstance(tools_raw, list):
        tools = [str(item).strip() for item in tools_raw if str(item).strip()]
    if not tools or "*" in tools:
        tools = ["*"]

    system_prompt = str(system_prompt_raw or "").strip()
    if not system_prompt:
        system_prompt = (
            "You are a specialized subagent for Ripperdoc. "
            "Follow the user's request and complete tasks autonomously. "
            "Provide a concise report when done."
        )

    model = str(model_raw).strip() if isinstance(model_raw, str) and model_raw.strip() else None
    location = (
        AgentLocation.PROJECT
        if str(location_raw).strip().lower() == "project"
        else AgentLocation.USER
    )

    return AgentDraft(
        name=name,
        description=description,
        tools=tools,
        system_prompt=system_prompt,
        location=location,
        model=model,
    )


class AgentsApp(App[None]):
    CSS = """
    #status_bar {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #items_table {
        width: 44%;
        min-width: 40;
    }

    #details_panel {
        width: 56%;
        padding: 0 1;
    }

    #form_dialog, #confirm_dialog, #info_dialog, #method_dialog, #generate_dialog {
        width: 76;
        max-height: 90%;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }

    #form_title, #info_title, #method_title, #generate_title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #form_fields Input, #form_fields Select, #form_fields TextArea {
        margin: 0 0 1 0;
    }

    #generate_fields TextArea {
        margin: 0 0 1 0;
    }

    #form_fields {
        height: 1fr;
        overflow: auto;
    }

    #generate_fields {
        height: 1fr;
        overflow: auto;
    }

    #prompt_input {
        height: 8;
        min-height: 6;
    }

    #generate_input {
        height: 8;
        min-height: 6;
    }

    .field_label {
        color: $text-muted;
        padding: 0 0 0 0;
    }

    .field_value {
        color: $accent;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #form_buttons, #confirm_buttons, #info_buttons, #method_buttons, #generate_buttons {
        align-horizontal: right;
        padding-top: 1;
        height: auto;
    }

    #confirm_message {
        padding: 0 0 1 0;
    }

    #info_body {
        height: 1fr;
    }

    #method_subtitle, #method_hint, #generate_subtitle {
        color: $text-muted;
        padding: 0 0 1 0;
    }

    #generate_status {
        color: $text-muted;
        padding: 0 0 1 0;
    }

    #generate_spinner {
        height: 1;
    }
    """

    BINDINGS = [
        ("escape", "quit", "Quit"),
        ("a", "add", "Add"),
        ("e", "edit", "Edit"),
        ("d", "delete", "Delete"),
        ("t", "toggle_view", "Toggle view"),
        ("s", "show", "Show"),
        ("c", "cancel_run", "Cancel run"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._view_mode: str = "agents"
        self._row_keys: list[str] = []
        self._selected_key: Optional[str] = None
        self._agent_map: dict[str, AgentDefinition] = {}
        self._run_snapshots: dict[str, dict[str, Any]] = {}
        self._failed_files: list[tuple[str, str]] = []
        self._generation_worker: Optional[Worker[Optional[AgentDraft]]] = None
        self._generate_screen: Optional[AgentGenerateScreen] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="status_bar")
        with Container(id="body"):
            yield DataTable(id="items_table")
            yield Static(id="details_panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#items_table", DataTable)
        try:
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        self._switch_view("agents", select_first=True)

    def action_toggle_view(self) -> None:
        next_view = "runs" if self._view_mode == "agents" else "agents"
        self._switch_view(next_view, select_first=True)

    def action_refresh(self) -> None:
        if self._view_mode == "agents":
            self._refresh_agents(select_first=False)
        else:
            self._refresh_runs(select_first=False)
        self._set_status("Refreshed.")

    def action_add(self) -> None:
        if self._view_mode != "agents":
            self._set_status("Switch to agents view to add.")
            return
        screen = CreateMethodScreen()
        self.push_screen(screen, self._handle_create_method)

    def _handle_create_method(self, method: Optional[str]) -> None:
        if method == "manual":
            screen = AgentFormScreen("add")
            self.push_screen(screen, self._handle_add_result)
            return
        if method == "generate":
            screen = AgentGenerateScreen()
            self.push_screen(screen)

    def start_agent_generation(self, description: str, screen: AgentGenerateScreen) -> None:
        if not description:
            return
        if self._generation_worker and self._generation_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            self._set_status("Generation already in progress.")
            return
        self._generate_screen = screen
        screen.clear_error()
        screen.set_busy(True, "Generating agent configuration...")
        self._set_status("Generating agent configuration...")
        self._generation_worker = self.run_worker(
            self._generate_agent_draft(description),
            name="agent_generate",
            group="agents",
            exit_on_error=False,
        )

    async def _generate_agent_draft(self, description: str) -> Optional[AgentDraft]:
        tools_list = ", ".join(BUILTIN_TOOL_NAMES)
        system_prompt = (
            "You are an expert at writing Ripperdoc subagent definitions. "
            "Return ONLY a valid JSON object and no extra prose."
        )
        user_prompt = (
            "Create a subagent configuration based on the description below.\n"
            "Return JSON with keys:\n"
            "- name: short lowercase slug (letters, numbers, hyphens)\n"
            "- description: when to use the agent\n"
            "- tools: array of tool names from the allowed list or [\"*\"] for all\n"
            "- system_prompt: detailed instructions for the agent\n"
            "- model: optional model pointer (main/quick or profile name)\n"
            "- location: optional \"user\" or \"project\"\n\n"
            f"Allowed tools: {tools_list}\n\n"
            f"Description: {description}\n"
        )
        assistant = await query_llm(
            [create_user_message(user_prompt)],
            system_prompt,
            [],
            model="quick",
            stream=False,
        )
        response_text = _extract_assistant_text(assistant)
        parsed = _parse_json_from_text(response_text)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            return None
        return _coerce_agent_draft(parsed, description)

    def action_edit(self) -> None:
        if self._view_mode != "agents":
            self._set_status("Switch to agents view to edit.")
            return
        agent = self._selected_agent()
        if not agent:
            self._set_status("No agent selected.")
            return
        if agent.location == AgentLocation.BUILT_IN:
            self._set_status("Built-in agents cannot be edited.")
            return
        screen = AgentFormScreen("edit", existing_agent=agent)
        self.push_screen(screen, self._handle_edit_result)

    def action_delete(self) -> None:
        if self._view_mode != "agents":
            self._set_status("Switch to agents view to delete.")
            return
        agent = self._selected_agent()
        if not agent:
            self._set_status("No agent selected.")
            return
        if agent.location == AgentLocation.BUILT_IN:
            self._set_status("Built-in agents cannot be deleted.")
            return
        screen = ConfirmScreen(f"Delete agent '{agent.agent_type}'?")
        self.push_screen(screen, self._handle_delete_confirm)

    def action_show(self) -> None:
        if self._view_mode == "agents":
            agent = self._selected_agent()
            if not agent:
                self._set_status("No agent selected.")
                return
            renderable = self._build_agent_details(agent, full_prompt=True)
            self.push_screen(InfoScreen(f"Agent: {agent.agent_type}", renderable))
            return
        run_id = self._selected_key
        if not run_id:
            self._set_status("No run selected.")
            return
        snapshot = self._run_snapshots.get(run_id) or get_agent_run_snapshot(run_id)
        if not snapshot:
            self._set_status("Run not found.")
            return
        renderable = self._build_run_details(run_id, snapshot)
        self.push_screen(InfoScreen(f"Run: {run_id}", renderable))

    async def action_cancel_run(self) -> None:
        if self._view_mode != "runs":
            self._set_status("Switch to runs view to cancel.")
            return
        run_id = self._selected_key
        if not run_id:
            self._set_status("No run selected.")
            return
        try:
            cancelled = await cancel_agent_run(run_id)
        except (OSError, RuntimeError, ValueError) as exc:
            self._set_status(f"Failed to cancel '{run_id}': {exc}")
            return
        if cancelled:
            self._set_status(f"Cancelled subagent {run_id}.")
        else:
            self._set_status(f"No running subagent found for '{run_id}'.")
        self._refresh_runs(select_first=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        worker = event.worker
        if worker.name != "agent_generate":
            return
        screen = self._generate_screen
        if event.state == WorkerState.ERROR:
            error = worker.error or "Generation failed."
            self._set_status(f"Generation failed: {error}")
            if screen and screen.is_attached:
                screen.set_busy(False)
                screen.set_status("Generation failed.")
                screen.set_error(str(error))
            return
        if event.state != WorkerState.SUCCESS:
            return
        draft = worker.result
        if not draft:
            self._set_status("Failed to parse generated agent config.")
            if screen and screen.is_attached:
                screen.set_busy(False)
                screen.set_status("Failed to parse generated agent config.")
                screen.set_error("Failed to parse generated agent config.")
            return
        self._set_status("Draft generated. Review and save.")
        if screen and screen.is_attached:
            screen.dismiss(None)
        form_screen = AgentFormScreen("add", draft=draft)
        self.push_screen(form_screen, self._handle_add_result)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._select_by_index(int(event.cursor_row))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_by_index(int(event.cursor_row))
        if self._view_mode == "agents":
            self.action_edit()
        else:
            self.action_show()

    def _switch_view(self, mode: str, select_first: bool) -> None:
        if mode not in ("agents", "runs"):
            return
        self._view_mode = mode
        table = self.query_one("#items_table", DataTable)
        try:
            table.clear(columns=True)
        except TypeError:
            table.clear()
        if mode == "agents":
            table.add_columns("#", "Name", "Location", "Model", "Tools")
            self._refresh_agents(select_first=select_first)
            self._set_status("Agents view.")
        else:
            table.add_columns("#", "ID", "Status", "Agent", "Duration", "BG", "Result")
            self._refresh_runs(select_first=select_first)
            self._set_status("Runs view.")

    def _handle_add_result(self, result: Optional[AgentFormResult]) -> None:
        if not result:
            return
        try:
            save_agent_definition(
                agent_type=result.name,
                description=result.description,
                tools=result.tools,
                system_prompt=result.system_prompt,
                location=result.location,
                model=result.model,
                overwrite=False,
            )
        except FileExistsError as exc:
            self._set_status(str(exc))
            return
        except (OSError, IOError, PermissionError, ValueError) as exc:
            self._set_status(f"Failed to create agent: {exc}")
            return

        config = get_global_config()
        pointer_map = config.model_pointers.model_dump()
        if result.model and result.model not in config.model_profiles and result.model not in pointer_map:
            self._set_status("Saved agent. Model not found; will fall back to main.")
        else:
            self._set_status(f"Saved agent '{result.name}'.")
        self._selected_key = result.name
        self._refresh_agents(select_first=False)

    def _handle_edit_result(self, result: Optional[AgentFormResult]) -> None:
        if not result:
            return
        try:
            save_agent_definition(
                agent_type=result.name,
                description=result.description,
                tools=result.tools,
                system_prompt=result.system_prompt,
                location=result.location,
                model=result.model,
                overwrite=True,
            )
        except (OSError, IOError, PermissionError, ValueError) as exc:
            self._set_status(f"Failed to update agent: {exc}")
            return

        config = get_global_config()
        pointer_map = config.model_pointers.model_dump()
        if result.model and result.model not in config.model_profiles and result.model not in pointer_map:
            self._set_status("Updated agent. Model not found; will fall back to main.")
        else:
            self._set_status(f"Updated agent '{result.name}'.")
        self._selected_key = result.name
        self._refresh_agents(select_first=False)

    def _handle_delete_confirm(self, confirmed: bool) -> None:
        if not confirmed:
            return
        agent = self._selected_agent()
        if not agent:
            return
        try:
            delete_agent_definition(agent.agent_type, agent.location)
        except FileNotFoundError as exc:
            self._set_status(str(exc))
            return
        except (OSError, IOError, PermissionError, ValueError) as exc:
            self._set_status(f"Failed to delete agent: {exc}")
            return
        self._set_status(f"Deleted agent '{agent.agent_type}'.")
        self._selected_key = None
        self._refresh_agents(select_first=True)

    def _refresh_agents(self, select_first: bool) -> None:
        table = self.query_one("#items_table", DataTable)
        table.clear(columns=False)
        result = load_agent_definitions()
        self._failed_files = [(str(path), str(error)) for path, error in result.failed_files]
        self._agent_map = {agent.agent_type: agent for agent in result.active_agents}
        self._row_keys = []

        for idx, agent in enumerate(result.active_agents, start=1):
            self._row_keys.append(agent.agent_type)
            tools_label = "all" if "*" in agent.tools else ", ".join(agent.tools)
            model_label = agent.model or "main (default)"
            table.add_row(
                str(idx),
                agent.agent_type,
                agent.location.value,
                model_label,
                self._shorten(tools_label, 28),
            )

        if not result.active_agents:
            self._selected_key = None
            self._update_details()
            return

        if self._selected_key and self._selected_key in self._agent_map:
            try:
                row_index = self._row_keys.index(self._selected_key)
            except ValueError:
                row_index = 0
            self._move_cursor(table, row_index)
            self._update_details()
            return

        if select_first:
            self._selected_key = self._row_keys[0]
            self._move_cursor(table, 0)
            self._update_details()

    def _refresh_runs(self, select_first: bool) -> None:
        table = self.query_one("#items_table", DataTable)
        table.clear(columns=False)
        self._row_keys = []
        self._run_snapshots = {}

        run_ids = list_agent_runs()
        for idx, run_id in enumerate(sorted(run_ids), start=1):
            snapshot = get_agent_run_snapshot(run_id) or {}
            self._run_snapshots[run_id] = snapshot
            result_text = snapshot.get("result_text") or snapshot.get("error") or ""
            result_preview = self._shorten(result_text, 60)
            table.add_row(
                str(idx),
                str(run_id),
                str(snapshot.get("status") or "unknown"),
                str(snapshot.get("agent_type") or "unknown"),
                self._format_duration(snapshot.get("duration_ms")),
                "yes" if snapshot.get("is_background") else "no",
                result_preview,
            )
            self._row_keys.append(run_id)

        if not run_ids:
            self._selected_key = None
            self._update_details()
            return

        if self._selected_key and self._selected_key in self._run_snapshots:
            try:
                row_index = self._row_keys.index(self._selected_key)
            except ValueError:
                row_index = 0
            self._move_cursor(table, row_index)
            self._update_details()
            return

        if select_first:
            self._selected_key = self._row_keys[0]
            self._move_cursor(table, 0)
            self._update_details()

    def _select_by_index(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._row_keys):
            return
        self._selected_key = self._row_keys[row_index]
        self._update_details()

    def _move_cursor(self, table: DataTable, row_index: int) -> None:
        try:
            table.move_cursor(row=row_index)
        except TypeError:
            try:
                table.move_cursor(row_index, 0)
            except Exception:
                pass
        except Exception:
            pass

    def _selected_agent(self) -> Optional[AgentDefinition]:
        if not self._selected_key:
            return None
        return self._agent_map.get(self._selected_key)

    def _update_details(self) -> None:
        details = self.query_one("#details_panel", Static)
        if self._view_mode == "agents":
            agent = self._selected_agent()
            if not agent:
                if not self._agent_map:
                    details.update("No agents configured.")
                else:
                    details.update("No agent selected.")
                return
            details.update(self._build_agent_details(agent, full_prompt=False))
            return

        if not self._selected_key:
            if not self._run_snapshots:
                details.update("No subagent runs recorded.")
            else:
                details.update("No run selected.")
            return
        snapshot = self._run_snapshots.get(self._selected_key)
        if not snapshot:
            details.update("No run selected.")
            return
        details.update(self._build_run_details(self._selected_key, snapshot, preview=True))

    def _build_agent_details(self, agent: AgentDefinition, *, full_prompt: bool) -> Group:
        tools_label = "all tools" if "*" in agent.tools else ", ".join(agent.tools)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        table.add_row("Agent", escape(agent.agent_type))
        table.add_row("Location", escape(agent.location.value))
        table.add_row("Model", escape(agent.model or "main (default)"))
        table.add_row("Tools", escape(tools_label))
        table.add_row("Fork context", "yes" if agent.fork_context else "no")
        if agent.color:
            table.add_row("Color", escape(agent.color))
        table.add_row("Description", escape(agent.when_to_use))

        prompt_text = agent.system_prompt or "(empty)"
        if not full_prompt:
            prompt_text = self._shorten(prompt_text, 260)
        prompt_panel = Panel(
            Text(prompt_text, overflow="fold"),
            title="System prompt",
            box=box.SIMPLE,
            padding=(1, 2),
        )

        return Group(
            Panel(table, title=f"Agent: {escape(agent.agent_type)}", box=box.ROUNDED),
            prompt_panel,
        )

    def _build_run_details(self, run_id: str, snapshot: dict[str, Any], preview: bool = False) -> Group:
        details = Table(box=box.SIMPLE_HEAVY, show_header=False)
        details.add_row("ID", escape(run_id))
        details.add_row("Status", escape(str(snapshot.get("status") or "unknown")))
        details.add_row("Agent", escape(str(snapshot.get("agent_type") or "unknown")))
        details.add_row("Duration", self._format_duration(snapshot.get("duration_ms")))
        details.add_row("Background", "yes" if snapshot.get("is_background") else "no")
        if snapshot.get("model_used"):
            details.add_row("Model", escape(str(snapshot.get("model_used"))))
        if snapshot.get("tool_use_count"):
            details.add_row("Tool uses", str(snapshot.get("tool_use_count")))
        if snapshot.get("missing_tools"):
            details.add_row("Missing tools", escape(", ".join(snapshot["missing_tools"])))
        if snapshot.get("error"):
            details.add_row("Error", escape(str(snapshot.get("error"))))

        result_text = snapshot.get("result_text") or snapshot.get("error") or ""
        if preview:
            result_text = self._shorten(result_text, 320)
        result_panel = Panel(
            Text(result_text or "(no result)", overflow="fold"),
            title="Result",
            box=box.SIMPLE,
            padding=(1, 2),
        )

        return Group(
            Panel(details, title=f"Run: {escape(run_id)}", box=box.ROUNDED, padding=(1, 2)),
            result_panel,
        )

    def _set_status(self, message: str) -> None:
        status = self.query_one("#status_bar", Static)
        status.update(message)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    @staticmethod
    def _format_duration(duration_ms: float | None) -> str:
        if duration_ms is None:
            return "-"
        try:
            duration = float(duration_ms)
        except (TypeError, ValueError):
            return "-"
        if duration < 1000:
            return f"{int(duration)} ms"
        seconds = duration / 1000.0
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(int(seconds), 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, mins = divmod(minutes, 60)
        return f"{hours}h {mins}m"


def run_agents_tui(on_exit: Optional[Callable[[], Any]] = None) -> bool:
    """Run the Textual agents TUI."""
    app = AgentsApp()
    app.run()
    if on_exit:
        on_exit()
    return True
