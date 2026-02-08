from __future__ import annotations

import json
from pathlib import Path

from ripperdoc.core.plugins import (
    PluginSettingsScope,
    add_enabled_plugin_for_scope,
    discover_plugins,
    get_installed_plugins_path,
    get_plugin_storage_root,
    plugin_scopes_for_path,
)


def _write_plugin_manifest(plugin_dir: Path, name: str) -> None:
    manifest = plugin_dir / ".ripperdoc-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"name": name, "description": f"{name} plugin"}),
        encoding="utf-8",
    )


def test_add_enabled_plugin_writes_installed_registry_by_scope(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)

    project_plugin = project / ".ripperdoc" / "plugins" / "demo-project"
    local_plugin = project / ".ripperdoc" / "plugins" / "demo-local"
    user_plugin = home / ".ripperdoc" / "plugins" / "demo-user"
    _write_plugin_manifest(project_plugin, "demo-project")
    _write_plugin_manifest(local_plugin, "demo-local")
    _write_plugin_manifest(user_plugin, "demo-user")

    add_enabled_plugin_for_scope(
        project_plugin,
        scope=PluginSettingsScope.PROJECT,
        project_path=project,
        home=home,
    )
    add_enabled_plugin_for_scope(
        local_plugin,
        scope=PluginSettingsScope.LOCAL,
        project_path=project,
        home=home,
    )
    add_enabled_plugin_for_scope(
        user_plugin,
        scope=PluginSettingsScope.USER,
        project_path=project,
        home=home,
    )

    project_registry = get_installed_plugins_path(
        PluginSettingsScope.PROJECT,
        project_path=project,
        home=home,
    )
    user_registry = get_installed_plugins_path(
        PluginSettingsScope.USER,
        project_path=project,
        home=home,
    )
    project_payload = json.loads(project_registry.read_text(encoding="utf-8"))
    user_payload = json.loads(user_registry.read_text(encoding="utf-8"))

    project_scopes = {item.get("scope") for item in project_payload["installedPlugins"]}
    user_scopes = {item.get("scope") for item in user_payload["installedPlugins"]}
    assert "project" in project_scopes
    assert "local" in project_scopes
    assert user_scopes == {"user"}


def test_discover_plugins_reads_installed_registry_entries(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)

    plugin_dir = project / ".ripperdoc" / "plugins" / "demo"
    _write_plugin_manifest(plugin_dir, "demo")

    registry_path = get_installed_plugins_path(
        PluginSettingsScope.PROJECT,
        project_path=project,
        home=home,
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "installedPlugins": [
                    {"name": "demo", "path": "./demo", "scope": "project"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = discover_plugins(project_path=project, home=home)
    names = {plugin.name for plugin in result.plugins}
    assert "demo" in names


def test_plugin_scopes_for_path_reads_registry_when_settings_absent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)

    plugin_root = get_plugin_storage_root(
        PluginSettingsScope.PROJECT,
        project_path=project,
        home=home,
    ) / "demo-local"
    _write_plugin_manifest(plugin_root, "demo-local")

    registry_path = get_installed_plugins_path(
        PluginSettingsScope.LOCAL,
        project_path=project,
        home=home,
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "installedPlugins": [
                    {"name": "demo-local", "path": "./demo-local", "scope": "local"},
                ]
            }
        ),
        encoding="utf-8",
    )

    scopes = plugin_scopes_for_path(plugin_root, project_path=project, home=home)
    assert scopes == [PluginSettingsScope.LOCAL]
