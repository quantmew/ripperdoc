from types import SimpleNamespace
from pathlib import Path

from rich.console import Console

from ripperdoc.cli.ui.rich_ui.commands import (
    ForkedCustomCommandRequest,
    handle_slash_command,
)
from ripperdoc.core.custom_commands import CommandExecutionContext


class _DummyUI:
    def __init__(self, project_path: Path) -> None:
        self.console = Console(record=True, width=120)
        self.project_path = project_path
        self.disable_slash_commands = False


def test_handle_slash_command_returns_fork_request_for_forked_custom_command(
    tmp_path: Path, monkeypatch
) -> None:
    ui = _DummyUI(tmp_path)

    command = SimpleNamespace(
        argument_hint=None,
        user_invocable=True,
        disable_model_invocation=False,
        execution_context=CommandExecutionContext.FORK,
        agent="general-purpose",
        model="quick",
    )

    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.commands.get_slash_command",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.commands.get_custom_command",
        lambda _name, _project_path: command,
    )
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.commands.expand_command_content",
        lambda _cmd, _arg, _project_path: "expanded content",
    )

    handled = handle_slash_command(ui, "/mycmd arg1", lambda _name, _path: [])
    assert isinstance(handled, ForkedCustomCommandRequest)
    assert handled.command_name == "mycmd"
    assert handled.subagent_type == "general-purpose"
    assert handled.expanded_content == "expanded content"
    assert handled.model == "quick"
