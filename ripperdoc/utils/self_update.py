"""Self-update helpers for the ripperdoc CLI."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, cast

from ripperdoc.utils.user_agent import build_user_agent

PACKAGE_NAME = "ripperdoc"
REPO = "quantmew/ripperdoc"
PYPI_RELEASE_API = "https://pypi.org/pypi/ripperdoc/json"
GITHUB_RELEASE_API = f"https://api.github.com/repos/{REPO}/releases/latest"
GITHUB_INSTALL_SH = f"https://raw.githubusercontent.com/{REPO}/main/install.sh"
GITHUB_INSTALL_PS1 = f"https://raw.githubusercontent.com/{REPO}/main/install.ps1"

VersionParser = Callable[[str], Any]


def _get_loose_version() -> VersionParser:
    """Get LooseVersion class with fallback chain for Python 3.12+ compatibility."""
    try:
        from distutils.version import LooseVersion as DistutilsLooseVersion

        return cast(VersionParser, DistutilsLooseVersion)
    except ModuleNotFoundError:
        pass

    try:
        from setuptools._distutils.version import (  # type: ignore[import-untyped]
            LooseVersion as SetuptoolsLooseVersion,
        )

        return cast(VersionParser, SetuptoolsLooseVersion)
    except ModuleNotFoundError:
        pass

    try:
        from packaging.version import parse as parse_version

        return cast(VersionParser, parse_version)
    except ModuleNotFoundError:
        pass

    def _noop_loose_version(version: str) -> str:
        """Fallback comparator if semantic parsing libraries are unavailable."""
        return version.strip()

    return _noop_loose_version


LooseVersion = _get_loose_version()

INSTALL_METHOD_PIP = "pip"
INSTALL_METHOD_BINARY = "binary"
INSTALL_METHOD_SOURCE = "source"
INSTALL_METHOD_UNKNOWN = "unknown"


@dataclass(frozen=True)
class InstallMetadata:
    method: str
    method_label: str
    location: str
    upgradable: bool
    upgrade_hint: str


def get_install_metadata() -> InstallMetadata:
    """Detect how the current running CLI instance is installed."""
    if getattr(sys, "frozen", False):
        location = Path(sys.executable).resolve() if sys.executable else "unknown executable"
        return InstallMetadata(
            method=INSTALL_METHOD_BINARY,
            method_label="Binary",
            location=str(location),
            upgradable=True,
            upgrade_hint="Will run the official installer script (install.sh/install.ps1).",
        )

    pip_dist_path = _get_pip_dist_info_path()
    if pip_dist_path is not None:
        return InstallMetadata(
            method=INSTALL_METHOD_PIP,
            method_label="pip",
            location=str(pip_dist_path),
            upgradable=True,
            upgrade_hint="Will run `python -m pip install --upgrade ripperdoc`.",
        )

    repo_root = Path(__file__).resolve().parents[2]
    return InstallMetadata(
        method=INSTALL_METHOD_SOURCE,
        method_label="Source",
        location=str(repo_root),
        upgradable=False,
        upgrade_hint=(
            "Please update manually with `git pull` and `pip install -e .` "
            "(or reinstall from your package source)."
        ),
    )


def get_latest_version(method: str, *, timeout_seconds: float = 8.0) -> Optional[str]:
    """Return latest available version for the detected installation path."""
    fetch_order: Sequence[str]
    if method == INSTALL_METHOD_BINARY:
        fetch_order = (GITHUB_RELEASE_API, PYPI_RELEASE_API)
    else:
        fetch_order = (PYPI_RELEASE_API, GITHUB_RELEASE_API)

    for source in fetch_order:
        version = _fetch_latest_version_from_source(source, timeout_seconds=timeout_seconds)
        if version is not None:
            return version
    return None


def is_update_available(current_version: str, latest_version: str) -> bool:
    """Return whether latest_version is newer than current_version."""
    normalized_current = _normalize_version(current_version)
    normalized_latest = _normalize_version(latest_version)
    try:
        parsed_latest = LooseVersion(normalized_latest)
        parsed_current = LooseVersion(normalized_current)
        return bool(parsed_latest > parsed_current)
    except Exception:
        return normalized_latest > normalized_current


def run_upgrade() -> Tuple[bool, str, str]:
    """Run the auto-upgrade path for the detected install method."""
    metadata = get_install_metadata()
    if not metadata.upgradable:
        return (
            False,
            "update-skipped",
            metadata.upgrade_hint,
        )

    if metadata.method == INSTALL_METHOD_PIP:
        return _run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME],
            metadata.upgrade_hint,
            fail_message="pip upgrade failed.",
        )

    if metadata.method == INSTALL_METHOD_BINARY:
        return _run_binary_upgrade()

    return (
        False,
        "update-skipped",
        "No automatic upgrade path for unknown install method.",
    )


def _get_pip_dist_info_path() -> Optional[Path]:
    """Return package dist-info/egg-info path for pip installs, if available."""
    try:
        from importlib.metadata import distribution

        dist = distribution(PACKAGE_NAME)
    except Exception:
        return None

    dist_path = getattr(dist, "_path", None)
    if dist_path is None:
        return None

    resolved = Path(dist_path).resolve()
    return resolved if resolved.exists() else None


def _fetch_latest_version_from_source(
    source_url: str,
    *,
    timeout_seconds: float,
) -> Optional[str]:
    payload = _fetch_json(source_url, timeout_seconds=timeout_seconds)
    if payload is None:
        return None

    if source_url == PYPI_RELEASE_API:
        version = str(payload.get("info", {}).get("version", "")).strip()
        return _normalize_version(version) if version else None

    if source_url == GITHUB_RELEASE_API:
        tag = str(payload.get("tag_name", "")).strip()
        return _normalize_version(tag) if tag else None

    return None


def _fetch_json(url: str, timeout_seconds: float = 8.0) -> Optional[dict[str, Any]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": build_user_agent()},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
        data = json.loads(payload)
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ):
        return None
    return data if isinstance(data, dict) else None


def _run_binary_upgrade() -> Tuple[bool, str, str]:
    """Run installer script fetched from GitHub to upgrade binary install."""
    is_windows = platform.system().lower().startswith("win")
    script_url = GITHUB_INSTALL_PS1 if is_windows else GITHUB_INSTALL_SH
    installer_path = Path(tempfile.gettempdir()) / ("ripperdoc-install.ps1" if is_windows else "ripperdoc-install.sh")

    script_text = _download_text(script_url)
    if not script_text:
        return (
            False,
            "update-failed",
            "Failed to download binary installer script.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        installer_path = Path(temp_dir) / installer_path.name
        installer_path.write_text(script_text, encoding="utf-8")

        if is_windows:
            return _run_command(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(installer_path),
                    "latest",
                ],
                "Ran PowerShell installer script.",
                fail_message="Binary upgrade failed (PowerShell installer).",
            )

        sh_bin = shutil.which("sh")
        if sh_bin is None:
            return (
                False,
                "update-failed",
                "No shell available to execute binary installer.",
            )
        installer_path.chmod(0o755)
        return _run_command(
            [sh_bin, str(installer_path), "latest"],
            "Ran binary installer script.",
            fail_message="Binary upgrade failed (install.sh).",
        )


def _download_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": build_user_agent()},
    )
    try:
        with urllib.request.urlopen(request, timeout=12.0) as response:
            payload = response.read()
            if isinstance(payload, bytes):
                return payload.decode("utf-8", errors="replace")
            return str(payload)
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return ""


def _run_command(
    command: Sequence[str],
    success_message: str,
    *,
    fail_message: str,
) -> Tuple[bool, str, str]:
    """Run a command and return (success, code, message)."""
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return (
            False,
            "command-missing",
            fail_message,
        )
    except Exception:
        return (
            False,
            "run-failed",
            fail_message,
        )

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        details = message if message else fail_message
        return False, f"exit:{result.returncode}", details
    return True, "ok", success_message


def _normalize_version(version: str) -> str:
    """Normalize user-facing versions into plain numeric-like form."""
    normalized = version.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized


__all__ = [
    "INSTALL_METHOD_BINARY",
    "INSTALL_METHOD_PIP",
    "INSTALL_METHOD_SOURCE",
    "INSTALL_METHOD_UNKNOWN",
    "InstallMetadata",
    "get_install_metadata",
    "get_latest_version",
    "is_update_available",
    "run_upgrade",
]
