from __future__ import annotations

import json
from pathlib import Path

from ripperdoc.core.config import ConfigManager
from ripperdoc.core.managed_settings import (
    load_managed_settings_snapshot,
    reset_managed_settings_cache,
)


def test_effective_config_managed_has_highest_precedence(monkeypatch, tmp_path: Path) -> None:
    managed_dir = tmp_path / "managed"
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    project_cfg_dir = project_dir / ".ripperdoc"

    monkeypatch.setenv("RIPPERDOC_MANAGED_CONFIG_DIR", str(managed_dir))
    monkeypatch.setenv("RIPPERDOC_CONFIG_DIR", str(user_dir))
    reset_managed_settings_cache()

    (managed_dir).mkdir(parents=True, exist_ok=True)
    (user_dir).mkdir(parents=True, exist_ok=True)
    project_cfg_dir.mkdir(parents=True, exist_ok=True)

    (user_dir / "config.json").write_text(
        json.dumps({"theme": "dark", "verbose": False}),
        encoding="utf-8",
    )
    (project_cfg_dir / "config.json").write_text(
        json.dumps({"theme": "dracula", "verbose": True}),
        encoding="utf-8",
    )
    (project_cfg_dir / "config.local.json").write_text(
        json.dumps({"theme": "local-theme", "verbose": False}),
        encoding="utf-8",
    )
    (managed_dir / "managed-settings.json").write_text(
        json.dumps({"theme": "managed-theme", "verbose": True}),
        encoding="utf-8",
    )

    manager = ConfigManager()
    effective = manager.get_effective_config(project_path=project_dir)
    assert effective.theme == "managed-theme"
    assert effective.verbose is True

    reset_managed_settings_cache()


def test_windows_hkcu_policy_ignored_when_admin_source_exists(monkeypatch) -> None:
    reset_managed_settings_cache()
    import ripperdoc.core.managed_settings as managed_settings

    monkeypatch.setattr(managed_settings, "_load_file_managed_settings", lambda: {"x": "file"})
    monkeypatch.setattr(managed_settings, "_load_macos_mdm_settings", lambda: None)
    monkeypatch.setattr(managed_settings, "_load_windows_registry_settings", lambda: (None, {"x": "hkcu"}))
    monkeypatch.setattr(managed_settings, "_load_server_managed_settings", lambda: None)

    snapshot = load_managed_settings_snapshot()
    assert snapshot.data["x"] == "file"
    assert "windows-user-policy" not in snapshot.sources

    reset_managed_settings_cache()
