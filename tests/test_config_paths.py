from pathlib import Path

from ripperdoc.utils.filesystem.config_paths import (
    config_dir_for_scope,
    config_file_for_scope,
    local_config_dir,
    managed_config_dir,
    project_config_dir,
    user_config_dir,
)


def test_scope_dirs_default_layout(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"

    assert user_config_dir(home=home) == home / ".ripperdoc"
    assert project_config_dir(project) == project.resolve() / ".ripperdoc"
    assert local_config_dir(project) == project.resolve() / ".ripperdoc"
    managed = managed_config_dir(project, home=home)
    assert managed.name in {"ripperdoc", "Ripperdoc"}


def test_scope_file_helper(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"

    assert config_file_for_scope("user", "config.json", home=home) == home / ".ripperdoc" / "config.json"
    assert (
        config_file_for_scope("project", "config.json", project_path=project)
        == project.resolve() / ".ripperdoc" / "config.json"
    )
    assert (
        config_file_for_scope("local", "config.local.json", project_path=project)
        == project.resolve() / ".ripperdoc" / "config.local.json"
    )
    assert (
        config_file_for_scope("managed", "managed-settings.json", project_path=project, home=home)
        == managed_config_dir(project, home=home) / "managed-settings.json"
    )


def test_user_config_dir_honors_env_override(monkeypatch, tmp_path: Path) -> None:
    user_override_dir = tmp_path / "user-override"
    managed_override_dir = tmp_path / "managed-override"
    monkeypatch.setenv("RIPPERDOC_CONFIG_DIR", str(user_override_dir))
    monkeypatch.setenv("RIPPERDOC_MANAGED_CONFIG_DIR", str(managed_override_dir))

    assert user_config_dir() == user_override_dir
    assert config_dir_for_scope("managed") == managed_override_dir
