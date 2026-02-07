"""Marketplace discovery and install helpers for plugins."""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ripperdoc.core.plugins import (
    PluginSettingsScope,
    add_enabled_plugin_for_scope,
    get_plugin_settings_path,
    remove_enabled_plugin_for_scope,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()

DEFAULT_MARKETPLACE_ID = "claude-plugins-official"
DEFAULT_MARKETPLACE_SOURCE = "anthropics/claude-plugins-official"
MARKETPLACE_CONFIG_FILE = "plugin_marketplaces.json"
MARKETPLACE_CACHE_DIR = "plugin_marketplaces"
MARKETPLACE_CACHE_TTL_SECONDS = 12 * 60 * 60
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class MarketplaceError:
    marketplace_id: str
    reason: str


@dataclass(frozen=True)
class MarketplaceSource:
    marketplace_id: str
    source: str
    enabled: bool = True
    title: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def source_type(self) -> str:
        normalized = normalize_marketplace_source(self.source)
        if normalized.startswith("github:"):
            return "github"
        if normalized.startswith("http://") or normalized.startswith("https://"):
            return "url"
        return "path"


@dataclass(frozen=True)
class PluginCatalogEntry:
    name: str
    description: str
    author: str
    marketplace_id: str
    marketplace_source: str
    source_label: str
    plugin_source: str
    installs: Optional[str] = None
    community_managed: bool = False
    updated_at: Optional[str] = None
    local_path: Optional[Path] = None
    github_repo: Optional[str] = None
    github_ref: Optional[str] = None
    github_subdir: Optional[str] = None


@dataclass(frozen=True)
class MarketplaceDiscoverResult:
    entries: List[PluginCatalogEntry]
    errors: List[MarketplaceError]


def _default_marketplace() -> MarketplaceSource:
    return MarketplaceSource(
        marketplace_id=DEFAULT_MARKETPLACE_ID,
        source=DEFAULT_MARKETPLACE_SOURCE,
        enabled=True,
        title="claude-plugins-official",
    )


def _marketplace_config_path(home: Optional[Path] = None) -> Path:
    home_dir = (home or Path.home()).expanduser()
    return home_dir / ".ripperdoc" / MARKETPLACE_CONFIG_FILE


def _cache_dir(home: Optional[Path] = None) -> Path:
    home_dir = (home or Path.home()).expanduser()
    return home_dir / ".ripperdoc" / "cache" / MARKETPLACE_CACHE_DIR


def normalize_marketplace_source(raw: str) -> str:
    source = raw.strip()
    if not source:
        return source
    if _OWNER_REPO_RE.match(source):
        return f"github:{source}"
    if source.startswith("https://github.com/"):
        remainder = source.removeprefix("https://github.com/").strip("/")
        remainder = remainder.removesuffix(".git")
        if _OWNER_REPO_RE.match(remainder):
            return f"github:{remainder}"
    if source.startswith("git@github.com:"):
        remainder = source.removeprefix("git@github.com:").strip("/")
        remainder = remainder.removesuffix(".git")
        if _OWNER_REPO_RE.match(remainder):
            return f"github:{remainder}"
    if source.startswith(("http://", "https://")):
        return source
    try:
        path = Path(source).expanduser().resolve()
        return str(path)
    except (OSError, RuntimeError, ValueError):
        return source


def _canonical_marketplace_id(source: str) -> str:
    normalized = normalize_marketplace_source(source)
    if normalized.startswith("github:"):
        repo = normalized.removeprefix("github:")
        base = repo.split("/")[-1]
    else:
        base = Path(normalized).name or "marketplace"
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-").lower()
    return base or "marketplace"


def load_marketplaces(home: Optional[Path] = None) -> List[MarketplaceSource]:
    path = _marketplace_config_path(home)
    marketplaces: List[MarketplaceSource] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            raw_items = raw.get("marketplaces", [])
            if isinstance(raw_items, list):
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    source = str(item.get("source") or "").strip()
                    if not source:
                        continue
                    marketplace_id = str(item.get("id") or "").strip() or _canonical_marketplace_id(
                        source
                    )
                    marketplaces.append(
                        MarketplaceSource(
                            marketplace_id=marketplace_id,
                            source=source,
                            enabled=bool(item.get("enabled", True)),
                            title=str(item.get("title") or "").strip() or None,
                            updated_at=str(item.get("updated_at") or "").strip() or None,
                        )
                    )

    if not any(item.marketplace_id == DEFAULT_MARKETPLACE_ID for item in marketplaces):
        marketplaces.insert(0, _default_marketplace())

    deduped: List[MarketplaceSource] = []
    seen: set[str] = set()
    for item in marketplaces:
        key = item.marketplace_id.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def save_marketplaces(
    marketplaces: Iterable[MarketplaceSource], home: Optional[Path] = None
) -> Path:
    path = _marketplace_config_path(home)
    serialized = {
        "marketplaces": [
            {
                "id": item.marketplace_id,
                "source": item.source,
                "enabled": item.enabled,
                "title": item.title,
                "updated_at": item.updated_at,
            }
            for item in marketplaces
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def add_marketplace(source: str, home: Optional[Path] = None) -> Tuple[MarketplaceSource, Path]:
    normalized = normalize_marketplace_source(source)
    marketplaces = load_marketplaces(home)
    existing = next(
        (
            item
            for item in marketplaces
            if normalize_marketplace_source(item.source) == normalized
            or item.marketplace_id == _canonical_marketplace_id(normalized)
        ),
        None,
    )
    if existing:
        return existing, _marketplace_config_path(home)
    created = MarketplaceSource(
        marketplace_id=_canonical_marketplace_id(normalized),
        source=normalized,
        enabled=True,
    )
    marketplaces.append(created)
    config_path = save_marketplaces(marketplaces, home)
    return created, config_path


def remove_marketplace(identifier: str, home: Optional[Path] = None) -> Tuple[bool, Path]:
    marketplaces = load_marketplaces(home)
    normalized_id = identifier.strip().lower()
    normalized_source = normalize_marketplace_source(identifier)
    kept: List[MarketplaceSource] = []
    removed = False
    for item in marketplaces:
        if item.marketplace_id == DEFAULT_MARKETPLACE_ID:
            kept.append(item)
            continue
        if (
            item.marketplace_id.lower() == normalized_id
            or normalize_marketplace_source(item.source) == normalized_source
        ):
            removed = True
            continue
        kept.append(item)
    config_path = save_marketplaces(kept, home)
    return removed, config_path


def _cache_file(marketplace_id: str, home: Optional[Path] = None) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", marketplace_id).strip("-") or "marketplace"
    return _cache_dir(home) / f"{safe_id}.json"


def _load_cache(
    marketplace: MarketplaceSource, home: Optional[Path] = None
) -> Optional[List[PluginCatalogEntry]]:
    path = _cache_file(marketplace.marketplace_id, home)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    fetched_at = raw.get("fetched_at")
    if not isinstance(fetched_at, (float, int)):
        return None
    if (time.time() - float(fetched_at)) > MARKETPLACE_CACHE_TTL_SECONDS:
        return None
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list):
        return None
    parsed: List[PluginCatalogEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        parsed.append(
            PluginCatalogEntry(
                name=str(item.get("name") or ""),
                description=str(item.get("description") or ""),
                author=str(item.get("author") or ""),
                marketplace_id=marketplace.marketplace_id,
                marketplace_source=marketplace.source,
                source_label=str(item.get("source_label") or ""),
                plugin_source=str(item.get("plugin_source") or ""),
                installs=str(item.get("installs")) if item.get("installs") is not None else None,
                community_managed=bool(item.get("community_managed", False)),
                updated_at=(
                    str(item.get("updated_at")) if item.get("updated_at") is not None else None
                ),
                local_path=Path(item["local_path"]) if item.get("local_path") else None,
                github_repo=str(item.get("github_repo")) if item.get("github_repo") else None,
                github_ref=str(item.get("github_ref")) if item.get("github_ref") else None,
                github_subdir=str(item.get("github_subdir")) if item.get("github_subdir") else None,
            )
        )
    return [item for item in parsed if item.name]


def _save_cache(
    marketplace: MarketplaceSource,
    entries: List[PluginCatalogEntry],
    home: Optional[Path] = None,
) -> None:
    path = _cache_file(marketplace.marketplace_id, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = {
        "fetched_at": time.time(),
        "entries": [
            {
                "name": item.name,
                "description": item.description,
                "author": item.author,
                "source_label": item.source_label,
                "plugin_source": item.plugin_source,
                "installs": item.installs,
                "community_managed": item.community_managed,
                "updated_at": item.updated_at,
                "local_path": str(item.local_path) if item.local_path else None,
                "github_repo": item.github_repo,
                "github_ref": item.github_ref,
                "github_subdir": item.github_subdir,
            }
            for item in entries
        ],
    }
    path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _http_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Ripperdoc/1.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _github_entries(marketplace: MarketplaceSource) -> List[PluginCatalogEntry]:
    normalized = normalize_marketplace_source(marketplace.source)
    owner_repo = normalized.removeprefix("github:")
    repo_meta = _http_json(f"https://api.github.com/repos/{owner_repo}")
    default_branch = str(repo_meta.get("default_branch") or "main")
    archive_request = urllib.request.Request(
        f"https://codeload.github.com/{owner_repo}/zip/refs/heads/{default_branch}",
        headers={"User-Agent": "Ripperdoc/1.0"},
    )
    with urllib.request.urlopen(archive_request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        manifest_paths = [
            file_name
            for file_name in archive.namelist()
            if file_name.endswith("/.claude-plugin/plugin.json")
            or file_name.endswith("/.ripperdoc-plugin/plugin.json")
        ]
        manifests: List[Tuple[str, dict[str, Any]]] = []
        for file_name in manifest_paths:
            relative_path = file_name.split("/", 1)[1] if "/" in file_name else file_name
            try:
                manifest = json.loads(archive.read(file_name).decode("utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(manifest, dict):
                continue
            manifests.append((relative_path, manifest))

    catalog_entries: List[PluginCatalogEntry] = []
    for manifest_path, manifest in manifests:
        if "/.claude-plugin/plugin.json" in manifest_path:
            plugin_subdir = manifest_path.rsplit("/.claude-plugin/plugin.json", 1)[0]
        else:
            plugin_subdir = manifest_path.rsplit("/.ripperdoc-plugin/plugin.json", 1)[0]
        plugin_name = str(manifest.get("name") or Path(plugin_subdir).name).strip()
        if not plugin_name:
            continue
        description = str(manifest.get("description") or "").strip()
        author_value = manifest.get("author")
        if isinstance(author_value, dict):
            author = str(author_value.get("name") or "").strip()
        else:
            author = str(author_value or "").strip()
        community = plugin_subdir.startswith("external_plugins/")

        catalog_entries.append(
            PluginCatalogEntry(
                name=plugin_name,
                description=description,
                author=author,
                marketplace_id=marketplace.marketplace_id,
                marketplace_source=marketplace.source,
                source_label=owner_repo + (" [Community Managed]" if community else ""),
                plugin_source=f"https://github.com/{owner_repo}.git",
                community_managed=community,
                updated_at=str(repo_meta.get("updated_at") or ""),
                github_repo=owner_repo,
                github_ref=default_branch,
                github_subdir=plugin_subdir,
            )
        )

    return sorted(catalog_entries, key=lambda item: item.name.lower())


def _parse_plugin_entries_from_json(
    payload: Any,
    marketplace: MarketplaceSource,
) -> List[PluginCatalogEntry]:
    if isinstance(payload, dict):
        plugins = payload.get("plugins")
    elif isinstance(payload, list):
        plugins = payload
    else:
        plugins = None
    if not isinstance(plugins, list):
        return []
    entries: List[PluginCatalogEntry] = []
    for item in plugins:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        entries.append(
            PluginCatalogEntry(
                name=name,
                description=str(item.get("description") or "").strip(),
                author=str(item.get("author") or "").strip(),
                marketplace_id=marketplace.marketplace_id,
                marketplace_source=marketplace.source,
                source_label=str(item.get("source_label") or marketplace.source),
                plugin_source=str(item.get("source") or ""),
                installs=str(item.get("installs")) if item.get("installs") is not None else None,
                community_managed=bool(item.get("community_managed", False)),
                updated_at=(
                    str(item.get("updated_at")) if item.get("updated_at") is not None else None
                ),
            )
        )
    return entries


def _url_entries(marketplace: MarketplaceSource) -> List[PluginCatalogEntry]:
    payload = _http_json(marketplace.source)
    return _parse_plugin_entries_from_json(payload, marketplace)


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _path_entries(marketplace: MarketplaceSource) -> List[PluginCatalogEntry]:
    source_path = Path(normalize_marketplace_source(marketplace.source))
    if source_path.is_file():
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        return _parse_plugin_entries_from_json(payload, marketplace)
    if not source_path.exists() or not source_path.is_dir():
        return []

    manifests: List[Path] = []
    manifests.extend(source_path.rglob(".claude-plugin/plugin.json"))
    manifests.extend(source_path.rglob(".ripperdoc-plugin/plugin.json"))
    entries: List[PluginCatalogEntry] = []
    for manifest_path in manifests:
        manifest = _read_manifest(manifest_path)
        if not manifest:
            continue
        plugin_root = manifest_path.parent.parent
        plugin_name = str(manifest.get("name") or plugin_root.name).strip()
        if not plugin_name:
            continue
        description = str(manifest.get("description") or "").strip()
        author_value = manifest.get("author")
        if isinstance(author_value, dict):
            author = str(author_value.get("name") or "").strip()
        else:
            author = str(author_value or "").strip()
        entries.append(
            PluginCatalogEntry(
                name=plugin_name,
                description=description,
                author=author,
                marketplace_id=marketplace.marketplace_id,
                marketplace_source=marketplace.source,
                source_label=str(source_path),
                plugin_source=str(plugin_root),
                local_path=plugin_root,
            )
        )
    return sorted(entries, key=lambda item: item.name.lower())


def _fetch_entries(marketplace: MarketplaceSource) -> List[PluginCatalogEntry]:
    source_type = marketplace.source_type
    if source_type == "github":
        return _github_entries(marketplace)
    if source_type == "url":
        return _url_entries(marketplace)
    return _path_entries(marketplace)


def discover_marketplace_plugins(
    marketplaces: Optional[Iterable[MarketplaceSource]] = None,
    *,
    force_refresh: bool = False,
    home: Optional[Path] = None,
) -> MarketplaceDiscoverResult:
    sources = list(marketplaces or load_marketplaces(home))
    entries: List[PluginCatalogEntry] = []
    errors: List[MarketplaceError] = []

    for marketplace in sources:
        if not marketplace.enabled:
            continue
        cached = None if force_refresh else _load_cache(marketplace, home)
        if cached is not None:
            entries.extend(cached)
            continue
        try:
            fetched = _fetch_entries(marketplace)
            _save_cache(marketplace, fetched, home)
            entries.extend(fetched)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                MarketplaceError(
                    marketplace_id=marketplace.marketplace_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    deduped: Dict[Tuple[str, str], PluginCatalogEntry] = {}
    for item in entries:
        key = (item.marketplace_id, item.name)
        deduped[key] = item
    merged = sorted(deduped.values(), key=lambda item: (item.marketplace_id, item.name.lower()))
    return MarketplaceDiscoverResult(entries=merged, errors=errors)


def install_marketplace_plugin(
    entry: PluginCatalogEntry,
    *,
    project_path: Path,
    scope: PluginSettingsScope = PluginSettingsScope.PROJECT,
    home: Optional[Path] = None,
) -> Tuple[bool, str, Optional[Path]]:
    project_path = project_path.resolve()
    if entry.local_path and entry.local_path.exists():
        settings_path = add_enabled_plugin_for_scope(
            entry.local_path.resolve(),
            scope=scope,
            project_path=project_path,
            home=home,
        )
        return True, f"Enabled local plugin from {entry.local_path}", settings_path

    if entry.github_repo and entry.github_subdir:
        plugins_root = project_path / ".ripperdoc" / "plugins"
        plugins_root.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", entry.name).strip("-").lower() or "plugin"
        target = plugins_root / slug
        if target.exists():
            shutil.rmtree(target)

        clone_url = f"https://github.com/{entry.github_repo}.git"
        with tempfile.TemporaryDirectory(prefix="ripperdoc-plugin-") as temp_dir:
            repo_dir = Path(temp_dir) / "repo"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    entry.github_ref or "main",
                    clone_url,
                    str(repo_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            source_dir = (repo_dir / entry.github_subdir).resolve()
            if not source_dir.exists() or not source_dir.is_dir():
                return False, f"Plugin subdirectory not found: {entry.github_subdir}", None
            shutil.copytree(source_dir, target)

        settings_path = add_enabled_plugin_for_scope(
            target.resolve(),
            scope=scope,
            project_path=project_path,
            home=home,
        )
        return True, f"Installed {entry.name} into {target}", settings_path

    return False, "This marketplace entry does not provide an installable source.", None


def uninstall_plugin_by_path(
    plugin_dir: Path,
    *,
    project_path: Path,
    scope: PluginSettingsScope = PluginSettingsScope.PROJECT,
    home: Optional[Path] = None,
) -> Tuple[bool, str]:
    _, removed = remove_enabled_plugin_for_scope(
        plugin_dir.resolve(),
        scope=scope,
        project_path=project_path.resolve(),
        home=home,
    )
    if removed:
        return True, f"Removed plugin {plugin_dir}"
    return False, f"Plugin {plugin_dir} was not enabled in {scope.value} scope."


def marketplace_scope_settings_path(
    *,
    scope: PluginSettingsScope,
    project_path: Path,
    home: Optional[Path] = None,
) -> Path:
    return get_plugin_settings_path(scope, project_path=project_path, home=home)


__all__ = [
    "MarketplaceSource",
    "MarketplaceError",
    "PluginCatalogEntry",
    "MarketplaceDiscoverResult",
    "DEFAULT_MARKETPLACE_ID",
    "DEFAULT_MARKETPLACE_SOURCE",
    "normalize_marketplace_source",
    "load_marketplaces",
    "save_marketplaces",
    "add_marketplace",
    "remove_marketplace",
    "discover_marketplace_plugins",
    "install_marketplace_plugin",
    "uninstall_plugin_by_path",
    "marketplace_scope_settings_path",
]
