"""Configuration management for Ripperdoc.

This module handles global and project-specific configuration,
including API keys, model settings, and user preferences.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Literal, cast
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from ripperdoc.utils.log import get_logger


logger = get_logger()

USER_CONFIG_DIR_NAME = ".ripperdoc"
USER_CONFIG_FILE_NAME = "config.json"


class ProtocolType(str, Enum):
    """Supported model protocols (not individual model vendors)."""

    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"


def provider_protocol(protocol: ProtocolType) -> str:
    """Return the message formatting protocol for a model protocol type."""
    if protocol == ProtocolType.ANTHROPIC:
        return "anthropic"
    if protocol == ProtocolType.OPENAI_COMPATIBLE:
        return "openai"
    if protocol == ProtocolType.GEMINI:
        # Gemini support is planned; default to OpenAI-style formatting for now.
        return "openai"
    return "openai"


def _default_model_for_protocol(protocol: ProtocolType) -> str:
    """Reasonable default model per protocol family."""
    if protocol == ProtocolType.ANTHROPIC:
        # Keep existing Anthropic default for RIPPERDOC_BASE_URL-only setups.
        return "claude-sonnet-4-5-20250929"
    if protocol == ProtocolType.GEMINI:
        return "gemini-1.5-pro"
    return "gpt-4o-mini"


def _lookup_model_metadata_safely(model_name: str, protocol: Optional[ProtocolType] = None) -> Any:
    """Best-effort model catalog lookup without importing model_catalog at module import time."""
    try:
        from ripperdoc.core.model_catalog import lookup_model_metadata

        return lookup_model_metadata(model_name, protocol)
    except (ModuleNotFoundError, ImportError, ValueError, TypeError):
        return None


class ModelPrice(BaseModel):
    """Token pricing for a model, per 1M tokens."""

    input: float = 0.0
    output: float = 0.0


class ModelProfile(BaseModel):
    """Configuration for a specific AI model."""

    protocol: ProtocolType
    model: str
    api_key: Optional[str] = None
    # Anthropic supports either api_key or auth_token; api_key takes precedence when both are set.
    auth_token: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # Total context window in tokens (if known). Falls back to heuristics when unset.
    context_window: Optional[int] = None
    # Optional split limits if provider reports separate input/output budgets.
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # Model mode from catalog (e.g. chat, completion, responses, image_generation).
    mode: Optional[str] = None
    # Tool handling for OpenAI-compatible providers. "native" uses tool_calls, "text" flattens tool
    # interactions into plain text to support providers that reject tool roles.
    openai_tool_mode: Literal["native", "text"] = "native"
    # Optional override for thinking protocol handling (e.g., "deepseek", "openrouter",
    # "qwen", "gemini_openai", "openai"). When unset, provider heuristics are used.
    thinking_mode: Optional[str] = None
    # Optional reasoning capability flag from model catalog.
    supports_reasoning: Optional[bool] = None
    # Vision support flag. None = infer from packaged model catalog when available.
    supports_vision: Optional[bool] = None
    # Pricing (per 1M tokens). Leave as 0 to skip cost calculation.
    price: ModelPrice = Field(default_factory=ModelPrice)
    currency: str = "USD"

    @model_validator(mode="after")
    def _apply_catalog_defaults(self) -> "ModelProfile":
        """Fill optional capability/pricing fields from packaged model catalog."""
        metadata = _lookup_model_metadata_safely(self.model, self.protocol)
        if metadata is None:
            return self

        fields_set = set(getattr(self, "model_fields_set", set()))
        if self.max_input_tokens is None and metadata.max_input_tokens is not None:
            self.max_input_tokens = metadata.max_input_tokens
        if self.max_output_tokens is None and metadata.max_output_tokens is not None:
            self.max_output_tokens = metadata.max_output_tokens
        if self.mode is None and metadata.mode:
            self.mode = metadata.mode
        if self.supports_reasoning is None and metadata.supports_reasoning is not None:
            self.supports_reasoning = metadata.supports_reasoning
        if self.supports_vision is None and metadata.supports_vision is not None:
            self.supports_vision = metadata.supports_vision

        inferred_max_tokens = (
            metadata.max_tokens or metadata.max_output_tokens or metadata.max_input_tokens
        )
        if inferred_max_tokens is not None and "max_tokens" not in fields_set:
            self.max_tokens = inferred_max_tokens

        if (
            "price" not in fields_set
            and self.price.input == 0.0
            and self.price.output == 0.0
            and metadata.input_cost_per_token is not None
            and metadata.output_cost_per_token is not None
        ):
            self.price = ModelPrice(
                input=metadata.input_cost_per_token * 1_000_000,
                output=metadata.output_cost_per_token * 1_000_000,
            )

        if "currency" not in fields_set and metadata.currency:
            self.currency = metadata.currency.upper()

        return self


def model_supports_vision(model_profile: ModelProfile) -> bool:
    """Detect whether a model supports vision/image input.

    Args:
        model_profile: The model profile to check

    Returns:
        True if the model supports vision capabilities, False otherwise
    """
    # If explicitly configured, use the config value
    if model_profile.supports_vision is not None:
        return model_profile.supports_vision

    metadata = _lookup_model_metadata_safely(model_profile.model, model_profile.protocol)
    if metadata and metadata.supports_vision is not None:
        return metadata.supports_vision
    return False


class ModelPointers(BaseModel):
    """Pointers to different model profiles for different purposes."""

    main: str = "default"
    quick: str = "default"


class UserConfig(BaseModel):
    """User configuration stored in ~/.ripperdoc/config.json."""

    model_config = {"protected_namespaces": (), "populate_by_name": True}

    # Model configuration
    model_profiles: Dict[str, ModelProfile] = Field(default_factory=dict)
    model_pointers: ModelPointers = Field(default_factory=ModelPointers)

    # User preferences
    theme: str = "dark"
    verbose: bool = False
    yolo_mode: bool = Field(default=False)
    show_full_thinking: bool = Field(default=False)
    auto_compact_enabled: bool = True
    context_token_limit: Optional[int] = None
    # Default thinking tokens budget when thinking mode is enabled (0 = disabled by default)
    default_thinking_tokens: int = Field(default=10240)

    # User-level permission rules (applied globally)
    user_allow_rules: list[str] = Field(default_factory=list)
    user_deny_rules: list[str] = Field(default_factory=list)
    user_ask_rules: list[str] = Field(default_factory=list)

    # Onboarding
    has_completed_onboarding: bool = False
    last_onboarding_version: Optional[str] = None

    # Statistics
    num_startups: int = 0

class ProjectConfig(BaseModel):
    """Project-specific configuration stored in .ripperdoc/config.json"""

    # Tool permissions (project level - checked into git)
    allowed_tools: list[str] = Field(default_factory=list)
    bash_allow_rules: list[str] = Field(default_factory=list)
    bash_deny_rules: list[str] = Field(default_factory=list)
    bash_ask_rules: list[str] = Field(default_factory=list)
    working_directories: list[str] = Field(default_factory=list)

    # Path ignore patterns (gitignore-style)
    ignore_patterns: list[str] = Field(
        default_factory=list,
        description="Gitignore-style patterns for paths to ignore in file operations",
    )

    # Context
    context: Dict[str, str] = Field(default_factory=dict)
    context_files: list[str] = Field(default_factory=list)

    # History
    history: list[str] = Field(default_factory=list)

    # Project settings
    dont_crawl_directory: bool = False

    # Trust
    has_trust_dialog_accepted: bool = False

    # Session tracking
    last_cost: Optional[float] = None
    last_duration: Optional[float] = None
    last_session_id: Optional[str] = None

class ProjectLocalConfig(BaseModel):
    """Project-local configuration stored in .ripperdoc/config.local.json (not checked into git)"""

    # Local permission rules (project-specific but not shared)
    local_allow_rules: list[str] = Field(default_factory=list)
    local_deny_rules: list[str] = Field(default_factory=list)
    local_ask_rules: list[str] = Field(default_factory=list)

    # Output style for this project/session context
    output_style: str = Field(default="default")


class ConfigManager:
    """Manages user and project-specific configuration."""

    def __init__(self) -> None:
        home = Path.home()
        self.global_config_path = home / USER_CONFIG_DIR_NAME / USER_CONFIG_FILE_NAME
        self.current_project_path: Optional[Path] = None
        self._global_config: Optional[UserConfig] = None
        self._project_config: Optional[ProjectConfig] = None
        self._project_local_config: Optional[ProjectLocalConfig] = None

    @staticmethod
    def _merge_dict_layers(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dict values while replacing scalars/lists."""
        merged: Dict[str, Any] = dict(base)
        for key, value in override.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = ConfigManager._merge_dict_layers(
                    cast(Dict[str, Any], existing),
                    cast(Dict[str, Any], value),
                )
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
        """Read a JSON object from disk."""
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
            logger.warning(
                "[config] Ignoring non-object JSON config",
                extra={"path": str(path), "type": type(data).__name__},
            )
            return None
        except (
            json.JSONDecodeError,
            OSError,
            IOError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
        ) as e:
            logger.warning(
                "Error loading config file %s: %s: %s",
                path,
                type(e).__name__,
                e,
                extra={"error": str(e), "path": str(path)},
            )
            return None

    @staticmethod
    def _filter_user_config_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only keys that belong to UserConfig."""
        allowed = set(UserConfig.model_fields.keys())
        return {key: value for key, value in data.items() if key in allowed}

    def get_global_config_path(self) -> Path:
        """Return the active user config path."""
        return self.global_config_path

    def get_global_config(self) -> UserConfig:
        """Load and return raw user configuration from ~/.ripperdoc/config.json."""
        if self._global_config is None:
            source_path: Optional[Path] = None
            data: Dict[str, Any] = {}

            if self.global_config_path.exists():
                source_path = self.global_config_path
                loaded = self._read_json_file(self.global_config_path)
                if loaded is not None:
                    data = loaded

            try:
                self._global_config = UserConfig(**data)
                if source_path is not None:
                    logger.debug(
                        "[config] Loaded user configuration",
                        extra={
                            "path": str(source_path),
                            "profile_count": len(self._global_config.model_profiles),
                        },
                    )
                else:
                    logger.debug(
                        "[config] User config not found; using defaults",
                        extra={"path": str(self.global_config_path)},
                    )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Error parsing user config: %s: %s",
                    type(e).__name__,
                    e,
                    extra={"error": str(e), "path": str(source_path or self.global_config_path)},
                )
                self._global_config = UserConfig()

        return self._global_config

    def get_effective_config(self, project_path: Optional[Path] = None) -> UserConfig:
        """Return effective config merged as local > project > user."""
        merged: Dict[str, Any] = self.get_global_config().model_dump()
        resolved_project_path = project_path or self.current_project_path or Path.cwd()

        project_config_path = resolved_project_path / ".ripperdoc" / "config.json"
        if project_config_path.exists():
            project_data = self._read_json_file(project_config_path)
            if project_data is not None:
                merged = self._merge_dict_layers(
                    merged, self._filter_user_config_fields(project_data)
                )

        local_config_path = resolved_project_path / ".ripperdoc" / "config.local.json"
        if local_config_path.exists():
            local_data = self._read_json_file(local_config_path)
            if local_data is not None:
                merged = self._merge_dict_layers(
                    merged, self._filter_user_config_fields(local_data)
                )

        try:
            return UserConfig(**merged)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Error building effective config; falling back to user config: %s: %s",
                type(e).__name__,
                e,
                extra={"error": str(e), "project_path": str(resolved_project_path)},
            )
            return self.get_global_config()

    def save_global_config(self, config: UserConfig) -> None:
        """Save user configuration to ~/.ripperdoc/config.json."""
        self._global_config = config
        self.global_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_config_path.write_text(config.model_dump_json(indent=2))
        logger.debug(
            "[config] Saved user configuration",
            extra={
                "path": str(self.global_config_path),
                "profile_count": len(config.model_profiles),
                "pointers": config.model_pointers.model_dump(),
            },
        )

    def get_project_config(self, project_path: Optional[Path] = None) -> ProjectConfig:
        """Load and return project configuration."""
        if project_path is not None:
            # Reset cached project config when switching projects
            if self.current_project_path != project_path:
                self._project_config = None
            self.current_project_path = project_path

        if self.current_project_path is None:
            return ProjectConfig()

        config_path = self.current_project_path / ".ripperdoc" / "config.json"

        if self._project_config is None:
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text())
                    self._project_config = ProjectConfig(**data)
                    logger.debug(
                        "[config] Loaded project config",
                        extra={
                            "path": str(config_path),
                            "project_path": str(self.current_project_path),
                            "allowed_tools": len(self._project_config.allowed_tools),
                        },
                    )
                except (
                    json.JSONDecodeError,
                    OSError,
                    IOError,
                    UnicodeDecodeError,
                    ValueError,
                    TypeError,
                ) as e:
                    logger.warning(
                        "Error loading project config: %s: %s",
                        type(e).__name__,
                        e,
                        extra={"error": str(e), "path": str(config_path)},
                    )
                    self._project_config = ProjectConfig()
            else:
                self._project_config = ProjectConfig()
                logger.debug(
                    "[config] Project config not found; using defaults",
                    extra={
                        "path": str(config_path),
                        "project_path": str(self.current_project_path),
                    },
                )

        return self._project_config

    def save_project_config(
        self, config: ProjectConfig, project_path: Optional[Path] = None
    ) -> None:
        """Save project configuration."""
        if project_path is not None:
            self.current_project_path = project_path

        if self.current_project_path is None:
            return

        config_dir = self.current_project_path / ".ripperdoc"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "config.json"
        self._project_config = config
        config_path.write_text(config.model_dump_json(indent=2))
        logger.debug(
            "[config] Saved project config",
            extra={
                "path": str(config_path),
                "project_path": str(self.current_project_path),
                "allowed_tools": len(config.allowed_tools),
            },
        )

    def get_project_local_config(self, project_path: Optional[Path] = None) -> ProjectLocalConfig:
        """Load and return project-local configuration (not checked into git)."""
        if project_path is not None:
            if self.current_project_path != project_path:
                self._project_local_config = None
            self.current_project_path = project_path

        if self.current_project_path is None:
            return ProjectLocalConfig()

        config_path = self.current_project_path / ".ripperdoc" / "config.local.json"

        if self._project_local_config is None:
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text())
                    self._project_local_config = ProjectLocalConfig(**data)
                    logger.debug(
                        "[config] Loaded project-local config",
                        extra={
                            "path": str(config_path),
                            "project_path": str(self.current_project_path),
                        },
                    )
                except (
                    json.JSONDecodeError,
                    OSError,
                    IOError,
                    UnicodeDecodeError,
                    ValueError,
                    TypeError,
                ) as e:
                    logger.warning(
                        "Error loading project-local config: %s: %s",
                        type(e).__name__,
                        e,
                        extra={"error": str(e), "path": str(config_path)},
                    )
                    self._project_local_config = ProjectLocalConfig()
            else:
                self._project_local_config = ProjectLocalConfig()

        return self._project_local_config

    def save_project_local_config(
        self, config: ProjectLocalConfig, project_path: Optional[Path] = None
    ) -> None:
        """Save project-local configuration."""
        if project_path is not None:
            self.current_project_path = project_path

        if self.current_project_path is None:
            return

        config_dir = self.current_project_path / ".ripperdoc"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "config.local.json"
        self._project_local_config = config
        config_path.write_text(config.model_dump_json(indent=2))

        # Ensure config.local.json is in .gitignore
        self._ensure_gitignore_entry("config.local.json")

        logger.debug(
            "[config] Saved project-local config",
            extra={
                "path": str(config_path),
                "project_path": str(self.current_project_path),
            },
        )

    def _ensure_gitignore_entry(self, entry: str) -> bool:
        """Ensure an entry exists in .ripperdoc/.gitignore. Returns True if added."""
        if self.current_project_path is None:
            return False

        gitignore_path = self.current_project_path / ".ripperdoc" / ".gitignore"
        try:
            text = ""
            if gitignore_path.exists():
                text = gitignore_path.read_text(encoding="utf-8", errors="ignore")
                existing_lines = text.splitlines()
                if entry in existing_lines:
                    return False
            with gitignore_path.open("a", encoding="utf-8") as f:
                if text and not text.endswith("\n"):
                    f.write("\n")
                f.write(f"{entry}\n")
            return True
        except (OSError, IOError):
            return False

    def get_api_key(self, protocol: ProtocolType, project_path: Optional[Path] = None) -> Optional[str]:
        """Get API key for a protocol."""
        # Only honor RIPPERDOC_* environment overrides
        if ripperdoc_api_key := os.getenv(RIPPERDOC_API_KEY):
            return ripperdoc_api_key

        # Then check effective layered config
        effective_config = self.get_effective_config(project_path=project_path)
        for profile in effective_config.model_profiles.values():
            if profile.protocol == protocol and profile.api_key:
                return profile.api_key

        return None

    def get_current_model_profile(
        self, pointer: str = "main", project_path: Optional[Path] = None
    ) -> Optional[ModelProfile]:
        """Get the current model profile for a given pointer."""
        effective_config = self.get_effective_config(project_path=project_path)

        # Get the profile name from the pointer
        profile_name = getattr(effective_config.model_pointers, pointer, "default")

        # Return the profile
        return effective_config.model_profiles.get(profile_name)

    def _fallback_profile_name(self, preferred: str = "default") -> str:
        """Pick a valid profile name to use as a fallback for pointers."""
        config = self.get_global_config()
        if preferred in config.model_profiles:
            return preferred
        if config.model_profiles:
            return next(iter(config.model_profiles.keys()))
        return ""

    def add_model_profile(
        self,
        name: str,
        profile: ModelProfile,
        overwrite: bool = False,
        set_as_main: bool = False,
    ) -> UserConfig:
        """Add or replace a model profile and optionally set it as the main pointer."""
        config = self.get_global_config()
        if not overwrite and name in config.model_profiles:
            raise ValueError(f"Model profile '{name}' already exists.")

        config.model_profiles[name] = profile
        current_main = getattr(config.model_pointers, "main", "")
        if set_as_main or not current_main or current_main not in config.model_profiles:
            config.model_pointers.main = name
        self.save_global_config(config)
        return config

    def delete_model_profile(self, name: str) -> UserConfig:
        """Delete a model profile and repair pointers that referenced it."""
        config = self.get_global_config()
        if name not in config.model_profiles:
            raise KeyError(f"Model profile '{name}' does not exist.")

        del config.model_profiles[name]

        fallback = self._fallback_profile_name()
        for pointer_field in ModelPointers.model_fields:
            current = getattr(config.model_pointers, pointer_field, "")
            if current == name:
                setattr(config.model_pointers, pointer_field, fallback)

        self.save_global_config(config)
        return config

    def set_model_pointer(self, pointer: str, profile_name: str) -> UserConfig:
        """Point a logical model slot (e.g., main/quick) to a profile name."""
        if pointer not in ModelPointers.model_fields:
            raise ValueError(f"Unknown model pointer '{pointer}'.")

        config = self.get_global_config()
        if profile_name not in config.model_profiles:
            raise ValueError(f"Model profile '{profile_name}' does not exist.")

        setattr(config.model_pointers, pointer, profile_name)
        self.save_global_config(config)
        return config


# Global instance
config_manager = ConfigManager()


def get_global_config() -> UserConfig:
    """Get raw user-level configuration."""
    return config_manager.get_global_config()


def get_global_config_path() -> Path:
    """Get the active user config file path."""
    return config_manager.get_global_config_path()


def get_effective_config(project_path: Optional[Path] = None) -> UserConfig:
    """Get effective config merged as local > project > user."""
    return config_manager.get_effective_config(project_path)


def save_global_config(config: UserConfig) -> None:
    """Save user configuration."""
    config_manager.save_global_config(config)


def get_project_config(project_path: Optional[Path] = None) -> ProjectConfig:
    """Get project configuration."""
    return config_manager.get_project_config(project_path)


def save_project_config(config: ProjectConfig, project_path: Optional[Path] = None) -> None:
    """Save project configuration."""
    config_manager.save_project_config(config, project_path)


def add_model_profile(
    name: str,
    profile: ModelProfile,
    overwrite: bool = False,
    set_as_main: bool = False,
) -> UserConfig:
    """Add or replace a model profile and persist the update."""
    return config_manager.add_model_profile(
        name,
        profile,
        overwrite=overwrite,
        set_as_main=set_as_main,
    )


def delete_model_profile(name: str) -> UserConfig:
    """Remove a model profile and update pointers as needed."""
    return config_manager.delete_model_profile(name)


def set_model_pointer(pointer: str, profile_name: str) -> UserConfig:
    """Update a model pointer (e.g., main/quick) to target a profile."""
    return config_manager.set_model_pointer(pointer, profile_name)


def get_current_model_profile(
    pointer: str = "main", project_path: Optional[Path] = None
) -> Optional[ModelProfile]:
    """Convenience wrapper to fetch the active profile for a pointer."""
    return config_manager.get_current_model_profile(pointer, project_path=project_path)


def get_project_local_config(project_path: Optional[Path] = None) -> ProjectLocalConfig:
    """Get project-local configuration (not checked into git)."""
    return config_manager.get_project_local_config(project_path)


def save_project_local_config(
    config: ProjectLocalConfig, project_path: Optional[Path] = None
) -> None:
    """Save project-local configuration."""
    config_manager.save_project_local_config(config, project_path)


# ==============================================================================
# RIPPERDOC_* global environment variable support
# ==============================================================================

# Environment variable name constants
RIPPERDOC_BASE_URL = "RIPPERDOC_BASE_URL"
RIPPERDOC_AUTH_TOKEN = "RIPPERDOC_AUTH_TOKEN"
RIPPERDOC_MODEL = "RIPPERDOC_MODEL"
RIPPERDOC_SMALL_FAST_MODEL = "RIPPERDOC_SMALL_FAST_MODEL"
RIPPERDOC_API_KEY = "RIPPERDOC_API_KEY"
RIPPERDOC_PROTOCOL = "RIPPERDOC_PROTOCOL"


def _infer_protocol_from_url_and_model(base_url: str, model_name: str = "") -> ProtocolType:
    """Infer protocol type from BASE_URL and model name.

    Args:
        base_url: API base URL
        model_name: Model name

    Returns:
        Inferred ProtocolType
    """
    base_lower = base_url.lower()
    model_lower = model_name.lower()

    # Explicit domain detection
    if "anthropic.com" in base_lower:
        return ProtocolType.ANTHROPIC
    if "generativelanguage.googleapis.com" in base_lower:
        return ProtocolType.GEMINI

    # URL path detection - check if path contains protocol identifier
    if "/anthropic" in base_lower or base_lower.endswith("/anthropic"):
        return ProtocolType.ANTHROPIC
    if "/v1/" in base_lower or "/v1" in base_lower:
        # Most /v1/ paths are OpenAI compatible format
        return ProtocolType.OPENAI_COMPATIBLE

    # Fall back to packaged model catalog when model name is known.
    if model_lower:
        metadata = _lookup_model_metadata_safely(model_lower)
        provider = str(getattr(metadata, "provider", "") or "").lower()
        if "anthropic" in provider:
            return ProtocolType.ANTHROPIC
        if any(token in provider for token in ("gemini", "vertex", "google")):
            return ProtocolType.GEMINI

    # Default to OpenAI compatible protocol
    return ProtocolType.OPENAI_COMPATIBLE


def _get_ripperdoc_env_overrides() -> Dict[str, Any]:
    """Get values of all RIPPERDOC_* environment variables.

    Returns:
        Dictionary containing all set environment variables
    """
    overrides: Dict[str, Any] = {}
    if base_url := os.getenv(RIPPERDOC_BASE_URL):
        overrides["base_url"] = base_url
    if api_key := os.getenv(RIPPERDOC_API_KEY):
        overrides["api_key"] = api_key
    if auth_token := os.getenv(RIPPERDOC_AUTH_TOKEN):
        overrides["auth_token"] = auth_token
    if model := os.getenv(RIPPERDOC_MODEL):
        overrides["model"] = model
    if small_fast_model := os.getenv(RIPPERDOC_SMALL_FAST_MODEL):
        overrides["small_fast_model"] = small_fast_model
    if protocol_str := os.getenv(RIPPERDOC_PROTOCOL):
        try:
            overrides["protocol"] = ProtocolType(protocol_str.lower())
        except ValueError:
            logger.warning(
                "[config] Invalid RIPPERDOC_PROTOCOL value: %s (must be anthropic, openai_compatible, or gemini)",
                protocol_str,
            )
    return overrides


def has_ripperdoc_env_overrides() -> bool:
    """Check if any RIPPERDOC_* environment variables are set."""
    return bool(_get_ripperdoc_env_overrides())


def get_effective_model_profile(pointer: str = "main") -> Optional[ModelProfile]:
    """Get model profile with RIPPERDOC_* environment variable overrides applied.

    Only uses RIPPERDOC_* environment variables for overrides; no longer reads
    provider-specific environment variables. When any RIPPERDOC_* variable is set,
    it overrides the corresponding field in the config file (if present).

    Args:
        pointer: Model pointer name ("main" or "quick")

    Returns:
        ModelProfile with environment variable overrides applied, or None if not found
    """
    env_overrides = _get_ripperdoc_env_overrides()
    profile = get_current_model_profile(pointer)

    if not env_overrides:
        return profile

    base_override = env_overrides.get("base_url")
    api_key_override = env_overrides.get("api_key")
    auth_token_override = env_overrides.get("auth_token")
    protocol_override = env_overrides.get("protocol")
    if pointer == "quick":
        model_override = env_overrides.get("small_fast_model") or env_overrides.get("model")
    else:
        model_override = env_overrides.get("model")

    if profile:
        updates: Dict[str, Any] = {}
        if base_override:
            updates["api_base"] = base_override
        if api_key_override:
            updates["api_key"] = api_key_override
        if auth_token_override:
            updates["auth_token"] = auth_token_override
        if model_override:
            updates["model"] = model_override
        if protocol_override:
            updates["protocol"] = protocol_override
        elif base_override or model_override:
            inferred = _infer_protocol_from_url_and_model(
                base_override or profile.api_base or "", model_override or profile.model
            )
            updates["protocol"] = inferred
        return profile.model_copy(update=updates)

    # No profile exists; only synthesize a profile if env provides enough context.
    if not (base_override or model_override or protocol_override):
        return None

    protocol = protocol_override or _infer_protocol_from_url_and_model(
        base_override or "", model_override or ""
    )
    model_name = model_override or _default_model_for_protocol(protocol)

    return ModelProfile(
        protocol=protocol,
        model=model_name,
        api_base=base_override,
        api_key=api_key_override,
        auth_token=auth_token_override,
    )


def get_ripperdoc_env_status() -> Dict[str, str]:
    """Get RIPPERDOC_* environment variable status information for diagnostic display.

    Returns:
        Dictionary with environment variable names as keys and formatted display strings as values
    """
    status: Dict[str, str] = {}
    if base_url := os.getenv(RIPPERDOC_BASE_URL):
        status["BASE_URL"] = base_url
    if protocol := os.getenv(RIPPERDOC_PROTOCOL):
        status["PROTOCOL"] = protocol
    if model := os.getenv(RIPPERDOC_MODEL):
        status["MODEL"] = model
    if small_fast_model := os.getenv(RIPPERDOC_SMALL_FAST_MODEL):
        status["SMALL_FAST_MODEL"] = small_fast_model
    if api_key := os.getenv(RIPPERDOC_API_KEY):
        masked = api_key[:4] + "…" if len(api_key) > 4 else "set"
        status["API_KEY"] = f"{masked} (${RIPPERDOC_API_KEY})"
    if auth_token := os.getenv(RIPPERDOC_AUTH_TOKEN):
        masked = auth_token[:4] + "…" if len(auth_token) > 4 else "set"
        status["AUTH_TOKEN"] = f"{masked} (${RIPPERDOC_AUTH_TOKEN})"
    return status
