"""Tests for piping input into the CLI."""

from click.testing import CliRunner

from ripperdoc.cli import cli as cli_module


def test_cli_consumes_piped_stdin_as_initial_query(monkeypatch, tmp_path):
    """Piped stdin should launch interactive mode with the content as initial query."""
    recorded: dict = {}

    def fake_main_rich(
        yolo_mode=False,
        verbose=False,
        show_full_thinking=None,
        session_id=None,
        log_file_path=None,
        allowed_tools=None,
        custom_system_prompt=None,
        append_system_prompt=None,
        model=None,
        resume_messages=None,
        initial_query=None,
    ):
        """Mock main_rich to record the initial_query parameter."""
        recorded["initial_query"] = initial_query
        recorded["yolo_mode"] = yolo_mode
        recorded["session_id"] = session_id

    # Import the ui module to mock main_rich
    from ripperdoc.cli.ui import rich_ui

    monkeypatch.setattr(rich_ui, "main_rich", fake_main_rich)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["--yolo"], input="你好\n", env={"HOME": str(tmp_path)})

    assert result.exit_code == 0
    assert recorded["initial_query"] == "你好"
    assert recorded["yolo_mode"] is True
    assert recorded["session_id"]
