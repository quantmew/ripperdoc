"""Status calculations for background shell tasks."""

from ripperdoc.tools import background_shell


def test_get_background_status_returns_runtime_and_age(monkeypatch):
    """Duration should stop at end_time while age keeps growing."""
    background_shell._get_tasks().clear()

    start = 10.0
    end = 15.0
    now = 25.0
    monkeypatch.setattr(background_shell, "_loop_time", lambda: now)

    dummy_process = type("Proc", (), {})()
    task = background_shell.BackgroundTask(
        id="bash_static",
        command="echo done",
        process=dummy_process,
        start_time=start,
    )
    task.exit_code = 0
    task.end_time = end

    with background_shell._get_tasks_lock():
        background_shell._get_tasks()[task.id] = task

    status = background_shell.get_background_status(task.id, consume=False)

    assert status["duration_ms"] == (end - start) * 1000.0
    assert status["age_ms"] == (now - start) * 1000.0

    background_shell._get_tasks().clear()
