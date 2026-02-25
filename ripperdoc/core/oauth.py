"""OAuth token storage and Codex OAuth model presets."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from ripperdoc.utils.log import get_logger

logger = get_logger()

USER_CONFIG_DIR_NAME = ".ripperdoc"
USER_OAUTH_FILE_NAME = "oauth_tokens.json"


class OAuthTokenType(str, Enum):
    """Supported OAuth token provider types."""

    CODEX = "codex"


class OAuthToken(BaseModel):
    """Stored OAuth credentials for a provider type."""

    type: OAuthTokenType = OAuthTokenType.CODEX
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    account_id: Optional[str] = None


class OAuthTokenStore(BaseModel):
    """On-disk OAuth token document."""

    tokens: Dict[str, OAuthToken] = Field(default_factory=dict)


class OAuthModelOption(BaseModel):
    """Selectable model option for OAuth-backed providers."""

    model: str
    label: str
    description: str


_CODEX_MODEL_OPTIONS: tuple[OAuthModelOption, ...] = (
    OAuthModelOption(
        model="gpt-5.3-codex",
        label="gpt-5.3-codex (current)",
        description="Latest frontier agentic coding model.",
    ),
    OAuthModelOption(
        model="gpt-5.3-codex-spark",
        label="gpt-5.3-codex-spark",
        description="Ultra-fast coding model.",
    ),
    OAuthModelOption(
        model="gpt-5.2-codex",
        label="gpt-5.2-codex",
        description="Frontier agentic coding model.",
    ),
    OAuthModelOption(
        model="gpt-5.1-codex-max",
        label="gpt-5.1-codex-max",
        description="Codex-optimized flagship for deep and fast reasoning.",
    ),
    OAuthModelOption(
        model="gpt-5.2",
        label="gpt-5.2",
        description="Latest frontier model with improvements across knowledge, reasoning and coding.",
    ),
    OAuthModelOption(
        model="gpt-5.1-codex-mini",
        label="gpt-5.1-codex-mini",
        description="Optimized for codex. Cheaper, faster, but less capable.",
    ),
)

OAUTH_MODEL_OPTIONS: Dict[OAuthTokenType, tuple[OAuthModelOption, ...]] = {
    OAuthTokenType.CODEX: _CODEX_MODEL_OPTIONS,
}


def get_oauth_tokens_path() -> Path:
    """Return the OAuth token storage path."""
    return Path.home() / USER_CONFIG_DIR_NAME / USER_OAUTH_FILE_NAME


def _safe_read_store(path: Path) -> OAuthTokenStore:
    if not path.exists():
        return OAuthTokenStore()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "tokens" not in payload:
            # Backward-compatible shape: { "<name>": { ... token ... } }
            payload = {"tokens": payload}
        return OAuthTokenStore(**payload)
    except (json.JSONDecodeError, OSError, IOError, UnicodeDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "[oauth] Failed to load OAuth token file: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return OAuthTokenStore()


def list_oauth_tokens() -> Dict[str, OAuthToken]:
    """Load and return all configured OAuth tokens."""
    return _safe_read_store(get_oauth_tokens_path()).tokens


def save_oauth_tokens(tokens: Dict[str, OAuthToken]) -> None:
    """Persist OAuth tokens to disk with restrictive file permissions."""
    path = get_oauth_tokens_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    store = OAuthTokenStore(tokens=tokens)
    path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        logger.debug("[oauth] Failed to set strict permissions", extra={"path": str(path)})


def get_oauth_token(name: str) -> Optional[OAuthToken]:
    """Return a named OAuth token, if present."""
    if not name:
        return None
    return list_oauth_tokens().get(name)


def add_oauth_token(name: str, token: OAuthToken, *, overwrite: bool = False) -> Dict[str, OAuthToken]:
    """Add or replace a named OAuth token."""
    normalized = (name or "").strip()
    if not normalized:
        raise ValueError("OAuth token name is required.")
    tokens = list_oauth_tokens()
    if not overwrite and normalized in tokens:
        raise ValueError(f"OAuth token '{normalized}' already exists.")
    tokens[normalized] = token
    save_oauth_tokens(tokens)
    return tokens


def delete_oauth_token(name: str) -> Dict[str, OAuthToken]:
    """Delete a named OAuth token."""
    normalized = (name or "").strip()
    tokens = list_oauth_tokens()
    if normalized not in tokens:
        raise KeyError(f"OAuth token '{normalized}' does not exist.")
    del tokens[normalized]
    save_oauth_tokens(tokens)
    return tokens


def oauth_models_for_type(token_type: OAuthTokenType) -> tuple[OAuthModelOption, ...]:
    """Return selectable model options for an OAuth token type."""
    return OAUTH_MODEL_OPTIONS.get(token_type, ())

