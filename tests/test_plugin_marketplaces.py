from __future__ import annotations

import json
from pathlib import Path

import pytest

from ripperdoc.core.plugin_marketplaces import (
    MarketplaceSource,
    PluginCatalogEntry,
    add_marketplace,
    discover_marketplace_plugins,
    install_marketplace_plugin,
    load_marketplaces,
    normalize_marketplace_source,
    remove_marketplace,
)
from ripperdoc.core.plugins import PluginSettingsScope


def test_normalize_marketplace_source_github_forms() -> None:
    assert (
        normalize_marketplace_source("anthropics/claude-plugins-official")
        == "github:anthropics/claude-plugins-official"
    )
    assert (
        normalize_marketplace_source("https://github.com/anthropics/claude-plugins-official.git")
        == "github:anthropics/claude-plugins-official"
    )
    assert (
        normalize_marketplace_source("git@github.com:anthropics/claude-plugins-official.git")
        == "github:anthropics/claude-plugins-official"
    )


def test_marketplace_add_and_remove(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)

    source_repo = tmp_path / "my-market"
    source_repo.mkdir(parents=True, exist_ok=True)
    (source_repo / ".git").mkdir(parents=True, exist_ok=True)
    (source_repo / ".ripperdoc-plugin").mkdir(parents=True, exist_ok=True)
    (source_repo / ".ripperdoc-plugin" / "marketplace.json").write_text(
        json.dumps({"plugins": []}),
        encoding="utf-8",
    )

    created, config_path = add_marketplace(str(source_repo), home=home)
    assert created.marketplace_id == "my-market"
    assert config_path.exists()

    loaded = load_marketplaces(home=home)
    assert any(item.marketplace_id == "my-market" for item in loaded)

    removed, _ = remove_marketplace("my-market", home=home)
    assert removed is True
    loaded_after = load_marketplaces(home=home)
    assert not any(item.marketplace_id == "my-market" for item in loaded_after)


def test_discover_marketplace_plugins_from_local_path(tmp_path: Path) -> None:
    marketplace_root = tmp_path / "marketplace"
    plugin_root = marketplace_root / "plugins" / "demo"
    manifest_path = plugin_root / ".ripperdoc-plugin" / "plugin.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "description": "Demo plugin",
                "author": {"name": "Ripperdoc"},
            }
        ),
        encoding="utf-8",
    )

    source = MarketplaceSource(
        marketplace_id="local-demo",
        source=str(marketplace_root),
        enabled=True,
    )
    result = discover_marketplace_plugins([source], force_refresh=True, home=tmp_path / "home")
    assert not result.errors
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.name == "demo"
    assert entry.description == "Demo plugin"
    assert entry.local_path == plugin_root


def test_add_marketplace_copies_git_source_into_marketplaces_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    source_repo = tmp_path / "source-marketplace"
    source_repo.mkdir(parents=True, exist_ok=True)
    (source_repo / ".git").mkdir(parents=True, exist_ok=True)
    (source_repo / ".ripperdoc-plugin").mkdir(parents=True, exist_ok=True)
    (source_repo / ".ripperdoc-plugin" / "marketplace.json").write_text(
        json.dumps({"plugins": []}),
        encoding="utf-8",
    )

    created, config_path = add_marketplace(
        str(source_repo),
        home=home,
    )
    assert config_path == home / ".ripperdoc" / "plugins" / "known_marketplaces.json"
    assert created.local_path is not None
    assert created.local_path.exists()
    assert created.local_path.parent == home / ".ripperdoc" / "plugins" / "marketplaces"


def test_add_marketplace_rejects_git_source_without_marketplace_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    source_repo = tmp_path / "bad-marketplace"
    source_repo.mkdir(parents=True, exist_ok=True)
    (source_repo / ".git").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError) as exc:
        add_marketplace(
            str(source_repo),
            home=home,
        )
    assert "marketplace.json" in str(exc.value)


def test_install_marketplace_plugin_user_scope_uses_user_plugin_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    plugin_root = tmp_path / "marketplace" / "plugins" / "demo"
    (plugin_root / ".ripperdoc-plugin").mkdir(parents=True, exist_ok=True)
    (plugin_root / ".ripperdoc-plugin" / "plugin.json").write_text(
        json.dumps({"name": "demo", "description": "demo plugin"}),
        encoding="utf-8",
    )

    entry = PluginCatalogEntry(
        name="demo",
        description="demo plugin",
        author="Ripperdoc",
        marketplace_id="local-demo",
        marketplace_source=str(tmp_path / "marketplace"),
        source_label="local",
        plugin_source=str(plugin_root),
        local_path=plugin_root,
    )

    ok, _, settings_path = install_marketplace_plugin(
        entry,
        project_path=project,
        scope=PluginSettingsScope.USER,
        home=home,
    )
    assert ok
    assert settings_path == home / ".ripperdoc" / "plugins.json"
    assert (home / ".ripperdoc" / "plugins" / "demo").exists()
    assert (home / ".ripperdoc" / "plugins" / "installed_plugins.json").exists()
