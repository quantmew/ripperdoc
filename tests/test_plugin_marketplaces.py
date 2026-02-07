from __future__ import annotations

import json
from pathlib import Path

from ripperdoc.core.plugin_marketplaces import (
    MarketplaceSource,
    add_marketplace,
    discover_marketplace_plugins,
    load_marketplaces,
    normalize_marketplace_source,
    remove_marketplace,
)


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

    created, config_path = add_marketplace("./my-market", home=home)
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
