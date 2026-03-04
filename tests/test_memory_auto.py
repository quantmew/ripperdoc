"""Tests for auto-memory behavior in ripperdoc.utils.memory."""

from __future__ import annotations

from types import SimpleNamespace

from ripperdoc.utils import memory
from ripperdoc.utils.filesystem.path_utils import sanitize_project_path


def _clear_memory_env(monkeypatch) -> None:
    for key in (
        "RIPPERDOC_DISABLE_AUTO_MEMORY",
        "RIPPERDOC_REMOTE",
        "RIPPERDOC_REMOTE_MEMORY_DIR",
        "RIPPERDOC_MEMORY_INCLUDE_SESSION_SEARCH",
    ):
        monkeypatch.delenv(key, raising=False)


def _mock_effective_config(monkeypatch, enabled):
    monkeypatch.setattr(
        memory,
        "get_effective_config",
        lambda project_path=None: SimpleNamespace(auto_memory_enabled=enabled),
    )


def test_auto_memory_disabled_by_default(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, None)
    monkeypatch.chdir(tmp_path)
    assert memory.is_auto_memory_enabled() is False


def test_auto_memory_enabled_via_false_disable_env(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, None)
    monkeypatch.setenv("RIPPERDOC_DISABLE_AUTO_MEMORY", "false")
    monkeypatch.chdir(tmp_path)
    assert memory.is_auto_memory_enabled() is True


def test_auto_memory_truthy_disable_env_takes_precedence(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, True)
    monkeypatch.setenv("RIPPERDOC_DISABLE_AUTO_MEMORY", "1")
    monkeypatch.chdir(tmp_path)
    assert memory.is_auto_memory_enabled() is False


def test_auto_memory_remote_without_memory_dir_disables(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, True)
    monkeypatch.setenv("RIPPERDOC_REMOTE", "1")
    monkeypatch.chdir(tmp_path)
    assert memory.is_auto_memory_enabled() is False


def test_auto_memory_respects_config_toggle(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, True)
    monkeypatch.chdir(tmp_path)
    assert memory.is_auto_memory_enabled() is True


def test_auto_memory_directory_path_uses_git_root_and_remote_base(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, True)

    repo_root = tmp_path / "repo"
    nested = repo_root / "src"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    remote_base = tmp_path / "remote-memory"
    monkeypatch.setenv("RIPPERDOC_REMOTE_MEMORY_DIR", str(remote_base))
    monkeypatch.setattr(memory, "get_git_root", lambda _path: repo_root)

    path = memory.auto_memory_directory_path()
    expected = remote_base / "projects" / sanitize_project_path(repo_root) / "memory"
    assert path == expected


def test_build_memory_instructions_auto_memory_only(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, None)

    monkeypatch.setenv("RIPPERDOC_DISABLE_AUTO_MEMORY", "false")
    monkeypatch.setenv("RIPPERDOC_REMOTE_MEMORY_DIR", str(tmp_path / "remote-memory"))
    monkeypatch.setattr(memory, "get_git_root", lambda _path: None)
    monkeypatch.setattr(memory.Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    instructions = memory.build_memory_instructions()
    assert instructions.startswith("# auto memory")
    assert "## MEMORY.md" in instructions
    assert memory.MEMORY_INSTRUCTIONS not in instructions


def test_build_memory_instructions_auto_memory_plus_agents(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, None)

    monkeypatch.setenv("RIPPERDOC_DISABLE_AUTO_MEMORY", "false")
    monkeypatch.setenv("RIPPERDOC_REMOTE_MEMORY_DIR", str(tmp_path / "remote-memory"))
    monkeypatch.setattr(memory, "get_git_root", lambda _path: None)
    monkeypatch.setattr(memory.Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    agents = tmp_path / memory.MEMORY_FILE_NAME
    agents.write_text("project instructions", encoding="utf-8")

    instructions = memory.build_memory_instructions()
    assert "# auto memory" in instructions
    assert "## MEMORY.md" in instructions
    assert memory.MEMORY_INSTRUCTIONS in instructions
    assert "project instructions" in instructions


def test_auto_memory_content_is_loaded_and_truncated(monkeypatch, tmp_path):
    _clear_memory_env(monkeypatch)
    _mock_effective_config(monkeypatch, None)

    monkeypatch.setenv("RIPPERDOC_DISABLE_AUTO_MEMORY", "false")
    monkeypatch.setenv("RIPPERDOC_REMOTE_MEMORY_DIR", str(tmp_path / "remote-memory"))
    monkeypatch.setattr(memory, "get_git_root", lambda _path: None)
    monkeypatch.chdir(tmp_path)

    auto_memory_file = memory.auto_memory_file_path()
    auto_memory_file.parent.mkdir(parents=True, exist_ok=True)
    content_lines = [f"line-{idx}" for idx in range(1, 206)]
    auto_memory_file.write_text("\n".join(content_lines), encoding="utf-8")

    instructions = memory.build_memory_instructions()
    assert "line-1" in instructions
    assert "line-200" in instructions
    assert "line-205" not in instructions
    assert "> WARNING: MEMORY.md is 205 lines (limit: 200)." in instructions
