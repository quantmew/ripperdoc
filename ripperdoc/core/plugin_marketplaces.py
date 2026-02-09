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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ripperdoc.core.plugins import (
    INSTALLED_PLUGINS_FILE,
    PluginSettingsScope,
    add_enabled_plugin_for_scope,
    get_plugin_storage_root,
    remove_enabled_plugin_for_scope,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()

DEFAULT_MARKETPLACE_ID = "claude-plugins-official"
DEFAULT_MARKETPLACE_SOURCE = "anthropics/claude-plugins-official"
MARKETPLACE_CONFIG_FILE = "plugin_marketplaces.json"
KNOWN_MARKETPLACES_FILE = "known_marketplaces.json"
MARKETPLACES_DIR = "marketplaces"
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
    local_path: Optional[Path] = None

    @property
    def source_type(self) -> str:
        if self.local_path is not None:
            return "path"
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


def _marketplace_root(
    *,
    home: Optional[Path] = None,
) -> Path:
    return get_plugin_storage_root(PluginSettingsScope.USER, home=home)


def _known_marketplaces_path(
    *,
    home: Optional[Path] = None,
) -> Path:
    return _marketplace_root(home=home) / KNOWN_MARKETPLACES_FILE


def _legacy_marketplace_config_path(home: Optional[Path] = None) -> Path:
    home_dir = (home or Path.home()).expanduser()
    return home_dir / ".ripperdoc" / MARKETPLACE_CONFIG_FILE


def _marketplaces_dir(
    *,
    home: Optional[Path] = None,
) -> Path:
    return _marketplace_root(home=home) / MARKETPLACES_DIR


def _cache_dir(home: Optional[Path] = None) -> Path:
    home_dir = (home or Path.home()).expanduser()
    return home_dir / ".ripperdoc" / "cache" / MARKETPLACE_CACHE_DIR


def normalize_marketplace_source(raw: str) -> str:
    source = raw.strip()
    if not source:
        return source
    if _OWNER_REPO_RE.match(source) and not source.startswith((".", "/", "~")):
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


def _normalize_source_record(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("source") or "").strip().lower()
    if not kind:
        return None
    if kind == "github":
        repo = str(raw.get("repo") or "").strip()
        return f"github:{repo}" if repo else None
    if kind == "url":
        url = str(raw.get("url") or "").strip()
        return url or None
    if kind == "path":
        path = str(raw.get("path") or "").strip()
        return path or None
    if kind == "git":
        url = str(raw.get("url") or "").strip()
        return url or None
    return None


def _serialize_source_record(normalized: str) -> Dict[str, str]:
    value = normalized.strip()
    if value.startswith("github:"):
        return {"source": "github", "repo": value.removeprefix("github:")}
    if value.startswith(("http://", "https://")):
        return {"source": "url", "url": value}
    if value.startswith("git@") or value.endswith(".git"):
        return {"source": "git", "url": value}
    return {"source": "path", "path": value}


def _load_known_marketplaces(home: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    path = _known_marketplaces_path(home=home)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_known_marketplaces(payload: Dict[str, Dict[str, Any]], home: Optional[Path] = None) -> Path:
    path = _known_marketplaces_path(home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _marketplace_from_known_record(
    marketplace_id: str, record: dict[str, Any]
) -> Optional[MarketplaceSource]:
    source = _normalize_source_record(record.get("source")) or ""
    if not source:
        return None
    local_path_raw = str(record.get("installLocation") or "").strip()
    local_path = Path(local_path_raw).expanduser().resolve() if local_path_raw else None
    return MarketplaceSource(
        marketplace_id=str(marketplace_id),
        source=source,
        enabled=True,
        title=None,
        updated_at=str(record.get("lastUpdated") or "").strip() or None,
        local_path=local_path,
    )


def _load_marketplaces_from_known(home: Optional[Path] = None) -> List[MarketplaceSource]:
    marketplaces: List[MarketplaceSource] = []
    for marketplace_id, record in _load_known_marketplaces(home).items():
        if not isinstance(record, dict):
            continue
        parsed = _marketplace_from_known_record(str(marketplace_id), record)
        if parsed is not None:
            marketplaces.append(parsed)
    return marketplaces


def _marketplace_from_legacy_record(record: dict[str, Any]) -> Optional[MarketplaceSource]:
    source = str(record.get("source") or "").strip()
    if not source:
        return None
    marketplace_id = str(record.get("id") or "").strip() or _canonical_marketplace_id(source)
    local_path_raw = str(record.get("local_path") or "").strip()
    local_path = Path(local_path_raw).expanduser().resolve() if local_path_raw else None
    return MarketplaceSource(
        marketplace_id=marketplace_id,
        source=source,
        enabled=bool(record.get("enabled", True)),
        title=str(record.get("title") or "").strip() or None,
        updated_at=str(record.get("updated_at") or "").strip() or None,
        local_path=local_path,
    )


def _load_marketplaces_from_legacy(home: Optional[Path] = None) -> List[MarketplaceSource]:
    legacy = _legacy_marketplace_config_path(home)
    if not legacy.exists():
        return []
    try:
        legacy_raw: Any = json.loads(legacy.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(legacy_raw, dict):
        return []
    raw_items = legacy_raw.get("marketplaces")
    if not isinstance(raw_items, list):
        return []
    marketplaces: List[MarketplaceSource] = []
    for record in raw_items:
        if not isinstance(record, dict):
            continue
        parsed = _marketplace_from_legacy_record(record)
        if parsed is not None:
            marketplaces.append(parsed)
    return marketplaces


def _dedupe_marketplaces(marketplaces: Iterable[MarketplaceSource]) -> List[MarketplaceSource]:
    deduped: List[MarketplaceSource] = []
    seen: set[str] = set()
    for marketplace in marketplaces:
        key = marketplace.marketplace_id.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(marketplace)
    return deduped


def load_marketplaces(home: Optional[Path] = None) -> List[MarketplaceSource]:
    path = _known_marketplaces_path(home=home)
    marketplaces = _load_marketplaces_from_known(home) if path.exists() else _load_marketplaces_from_legacy(home)

    if not any(item.marketplace_id == DEFAULT_MARKETPLACE_ID for item in marketplaces):
        marketplaces.insert(0, _default_marketplace())

    return _dedupe_marketplaces(marketplaces)


def save_marketplaces(
    marketplaces: Iterable[MarketplaceSource], home: Optional[Path] = None
) -> Path:
    payload: Dict[str, Dict[str, Any]] = {}
    for item in marketplaces:
        install_location = (
            str(item.local_path.resolve())
            if item.local_path
            else str(_marketplaces_dir(home=home) / item.marketplace_id)
        )
        payload[item.marketplace_id] = {
            "source": _serialize_source_record(item.source),
            "installLocation": install_location,
            "lastUpdated": item.updated_at,
        }
    return _save_known_marketplaces(payload, home=home)


def _candidate_marketplace_manifest(root: Path) -> Optional[Path]:
    for candidate in (
        root / ".ripperdoc-plugin" / "marketplace.json",
        root / ".claude-plugin" / "marketplace.json",
    ):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _validate_marketplace_dir(path: Path) -> Tuple[bool, Optional[str]]:
    manifest = _candidate_marketplace_manifest(path)
    if manifest is None:
        return False, (
            "Marketplace is missing .ripperdoc-plugin/marketplace.json "
            "or .claude-plugin/marketplace.json"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"Invalid marketplace manifest JSON: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return False, "Marketplace manifest must be a JSON object"
    return True, None


def _is_git_source(raw_source: str, normalized: str) -> bool:
    source = raw_source.strip()
    if normalized.startswith("github:"):
        return True
    if source.startswith("git@"):
        return True
    if source.endswith(".git"):
        return True
    if source.startswith("https://github.com/") and not source.endswith("marketplace.json"):
        return True
    try:
        source_path = Path(source).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return source_path.is_dir() and (source_path / ".git").exists()


def _clone_or_copy_marketplace(
    source: str,
    normalized: str,
    destination: Path,
) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if normalized.startswith("github:"):
        clone_url = f"https://github.com/{normalized.removeprefix('github:')}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )
        return destination

    try:
        source_path = Path(source).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        source_path = None

    if source_path and source_path.exists() and source_path.is_dir():
        shutil.copytree(source_path, destination)
        return destination

    subprocess.run(
        ["git", "clone", "--depth", "1", source, str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination


def add_marketplace(
    source: str,
    home: Optional[Path] = None,
) -> Tuple[MarketplaceSource, Path]:
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
        return existing, _known_marketplaces_path(home=home)

    local_path: Optional[Path] = None
    if _is_git_source(source, normalized):
        destination = (
            _marketplaces_dir(home=home)
            / _canonical_marketplace_id(normalized)
        )
        checked_path = _clone_or_copy_marketplace(source, normalized, destination)
        ok, reason = _validate_marketplace_dir(checked_path)
        if not ok:
            try:
                shutil.rmtree(checked_path)
            except OSError:
                logger.debug("failed to cleanup invalid marketplace copy: %s", checked_path)
            raise ValueError(reason or "Invalid marketplace format")
        local_path = checked_path.resolve()
    else:
        try:
            source_path = Path(normalized).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            source_path = None
        if source_path and source_path.exists() and source_path.is_dir():
            ok, reason = _validate_marketplace_dir(source_path)
            if not ok:
                raise ValueError(reason or "Invalid marketplace format")
            destination = _marketplaces_dir(home=home) / _canonical_marketplace_id(normalized)
            checked_path = _clone_or_copy_marketplace(str(source_path), normalized, destination)
            local_path = checked_path.resolve()

    if local_path is None:
        raise ValueError("Marketplace source must be a git repository or local directory.")

    updated_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    created = MarketplaceSource(
        marketplace_id=_canonical_marketplace_id(normalized),
        source=normalized,
        enabled=True,
        updated_at=updated_at,
        local_path=local_path,
    )
    marketplaces.append(created)
    config_path = save_marketplaces(marketplaces, home)
    return created, config_path


def remove_marketplace(
    identifier: str,
    home: Optional[Path] = None,
) -> Tuple[bool, Path]:
    marketplaces = load_marketplaces(home)
    normalized_id = identifier.strip().lower()
    normalized_source = normalize_marketplace_source(identifier)
    kept: List[MarketplaceSource] = []
    removed = False
    managed_marketplaces_root = _marketplaces_dir(
        home=home,
    ).resolve()
    for item in marketplaces:
        if item.marketplace_id == DEFAULT_MARKETPLACE_ID:
            kept.append(item)
            continue
        if (
            item.marketplace_id.lower() == normalized_id
            or normalize_marketplace_source(item.source) == normalized_source
        ):
            removed = True
            if item.local_path and item.local_path.exists():
                try:
                    local_resolved = item.local_path.resolve()
                    local_resolved.relative_to(managed_marketplaces_root)
                    shutil.rmtree(local_resolved)
                except ValueError:
                    logger.debug(
                        "skip deleting non-managed marketplace path: %s",
                        item.local_path,
                    )
                except OSError:
                    logger.debug("failed to remove marketplace directory: %s", item.local_path)
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
    if marketplace.local_path is not None:
        source_path = marketplace.local_path
    else:
        source_path = Path(normalize_marketplace_source(marketplace.source))
    if source_path.is_file():
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        return _parse_plugin_entries_from_json(payload, marketplace)
    if not source_path.exists() or not source_path.is_dir():
        return []

    marketplace_manifest = _candidate_marketplace_manifest(source_path)
    if marketplace_manifest is not None:
        payload = _read_manifest(marketplace_manifest)
        if payload:
            manifest_entries = _parse_plugin_entries_from_json(payload, marketplace)
            resolved_entries: List[PluginCatalogEntry] = []
            for item in manifest_entries:
                raw_source = item.plugin_source.strip()
                local_path: Optional[Path] = None
                if raw_source:
                    try:
                        candidate = Path(raw_source).expanduser()
                        if not candidate.is_absolute():
                            candidate = (source_path / candidate).resolve()
                        else:
                            candidate = candidate.resolve()
                        if candidate.exists() and candidate.is_dir():
                            local_path = candidate
                    except (OSError, RuntimeError, ValueError):
                        local_path = None
                resolved_entries.append(
                    PluginCatalogEntry(
                        name=item.name,
                        description=item.description,
                        author=item.author,
                        marketplace_id=item.marketplace_id,
                        marketplace_source=item.marketplace_source,
                        source_label=item.source_label,
                        plugin_source=item.plugin_source,
                        installs=item.installs,
                        community_managed=item.community_managed,
                        updated_at=item.updated_at,
                        local_path=local_path,
                        github_repo=item.github_repo,
                        github_ref=item.github_ref,
                        github_subdir=item.github_subdir,
                    )
                )
            if resolved_entries:
                return sorted(resolved_entries, key=lambda item: item.name.lower())

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
    sources = list(
        marketplaces
        or load_marketplaces(home)
    )
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


def update_marketplace(
    marketplace_id: str,
    *,
    home: Optional[Path] = None,
) -> Tuple[bool, str, Optional[MarketplaceSource]]:
    marketplaces = load_marketplaces(home)
    target = next((item for item in marketplaces if item.marketplace_id == marketplace_id), None)
    if target is None:
        return False, f"Marketplace '{marketplace_id}' not found.", None

    normalized = normalize_marketplace_source(target.source)
    destination = _marketplaces_dir(home=home) / marketplace_id
    checked_path = _clone_or_copy_marketplace(target.source, normalized, destination)
    ok, reason = _validate_marketplace_dir(checked_path)
    if not ok:
        try:
            shutil.rmtree(checked_path)
        except OSError:
            logger.debug("failed to cleanup invalid marketplace copy: %s", checked_path)
        return False, reason or "Invalid marketplace format", None

    updated_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    updated = MarketplaceSource(
        marketplace_id=target.marketplace_id,
        source=target.source,
        enabled=target.enabled,
        title=target.title,
        updated_at=updated_at,
        local_path=checked_path.resolve(),
    )
    new_list = [updated if item.marketplace_id == marketplace_id else item for item in marketplaces]
    save_marketplaces(new_list, home)
    return True, f"Updated marketplace: {marketplace_id}", updated


def resolve_marketplace_entry(
    marketplace_id: str,
    plugin_name: str,
    *,
    home: Optional[Path] = None,
) -> Tuple[bool, str, Optional[PluginCatalogEntry]]:
    ok, message, marketplace = update_marketplace(marketplace_id, home=home)
    if not ok or marketplace is None:
        return False, message, None
    entries = _path_entries(marketplace)
    match = next((item for item in entries if item.name == plugin_name), None)
    if match is None:
        return False, f"Plugin '{plugin_name}' not found in marketplace '{marketplace_id}'.", None
    return True, "ok", match


def install_marketplace_plugin(
    entry: PluginCatalogEntry,
    *,
    project_path: Path,
    scope: PluginSettingsScope = PluginSettingsScope.PROJECT,
    home: Optional[Path] = None,
) -> Tuple[bool, str, Optional[Path]]:
    project_path = project_path.resolve()
    plugins_root = get_plugin_storage_root(scope, project_path=project_path, home=home)
    plugins_root.mkdir(parents=True, exist_ok=True)

    if entry.local_path and entry.local_path.exists():
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", entry.name).strip("-").lower() or "plugin"
        if slug in {INSTALLED_PLUGINS_FILE, KNOWN_MARKETPLACES_FILE, MARKETPLACES_DIR}:
            slug = f"{slug}-plugin"
        target = plugins_root / slug
        source = entry.local_path.resolve()
        if target.exists():
            shutil.rmtree(target)
        if source != target:
            shutil.copytree(source, target)
        settings_path = add_enabled_plugin_for_scope(
            target.resolve(),
            scope=scope,
            project_path=project_path,
            home=home,
        )
        return True, f"Installed {entry.name} into {target}", settings_path

    if entry.github_repo and entry.github_subdir:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", entry.name).strip("-").lower() or "plugin"
        if slug in {INSTALLED_PLUGINS_FILE, KNOWN_MARKETPLACES_FILE, MARKETPLACES_DIR}:
            slug = f"{slug}-plugin"
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
    return _known_marketplaces_path(home=home)


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
    "update_marketplace",
    "resolve_marketplace_entry",
    "install_marketplace_plugin",
    "uninstall_plugin_by_path",
    "marketplace_scope_settings_path",
]
