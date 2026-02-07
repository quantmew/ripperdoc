"""Tests for task-list scope resolution behavior."""

from __future__ import annotations

import pytest

from ripperdoc.utils.tasks import resolve_task_list_id, set_runtime_task_scope


@pytest.fixture(autouse=True)
def _reset_runtime_task_scope() -> None:
    set_runtime_task_scope(session_id=None)
    yield
    set_runtime_task_scope(session_id=None)


def test_resolve_task_list_id_defaults_to_runtime_session_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RIPPERDOC_TASK_LIST_ID", raising=False)
    monkeypatch.delenv("RIPPERDOC_SESSION_ID", raising=False)

    set_runtime_task_scope(session_id="session-123", project_root=tmp_path)
    resolved = resolve_task_list_id(project_root=tmp_path)

    assert resolved.startswith("session-")
    assert "session-123" in resolved


def test_explicit_task_list_env_overrides_runtime_session_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RIPPERDOC_TASK_LIST_ID", "shared-list")

    set_runtime_task_scope(session_id="session-123", project_root=tmp_path)
    resolved = resolve_task_list_id(project_root=tmp_path)

    assert resolved == "shared-list"


def test_resolve_task_list_id_uses_env_session_when_runtime_scope_not_bound(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RIPPERDOC_TASK_LIST_ID", raising=False)
    monkeypatch.setenv("RIPPERDOC_SESSION_ID", "sdk-session")

    resolved = resolve_task_list_id(project_root=tmp_path)

    assert resolved.startswith("session-")
    assert "sdk-session" in resolved


def test_claude_code_task_list_env_is_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CODE_TASK_LIST_ID", "legacy-shared-list")
    monkeypatch.delenv("RIPPERDOC_TASK_LIST_ID", raising=False)
    monkeypatch.delenv("RIPPERDOC_SESSION_ID", raising=False)

    resolved = resolve_task_list_id(project_root=tmp_path)

    assert resolved != "legacy-shared-list"
