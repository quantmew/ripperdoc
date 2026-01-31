"""Configuration management for Ripperdoc.

This module handles global and project-specific configuration,
including API keys, model settings, and user preferences.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from ripperdoc.utils.log import get_logger


logger = get_logger()


class ProviderType(str, Enum):
    """Supported model protocols (not individual model vendors)."""

    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"

    @classmethod
    def _legacy_aliases(cls) -> Dict[str, "ProviderType"]:
        """Map legacy provider labels to protocol families."""
        return {
            "openai": cls.OPENAI_COMPATIBLE,
            "openai-compatible": cls.OPENAI_COMPATIBLE,
            "openai compatible": cls.OPENAI_COMPATIBLE,
            "mistral": cls.OPENAI_COMPATIBLE,
            "deepseek": cls.OPENAI_COMPATIBLE,
            "kimi": cls.OPENAI_COMPATIBLE,
            "qwen": cls.OPENAI_COMPATIBLE,
            "glm": cls.OPENAI_COMPATIBLE,
            "google": cls.GEMINI,
        }

    @classmethod
    def _missing_(cls, value: object) -> Optional["ProviderType"]:
        """Support legacy provider strings by mapping to their protocol."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapped = cls._legacy_aliases().get(normalized)
            if mapped:
                return mapped
        return None


def provider_protocol(provider: ProviderType) -> str:
    """Return the message formatting protocol for a provider."""
    if provider == ProviderType.ANTHROPIC:
        return "anthropic"
    if provider == ProviderType.OPENAI_COMPATIBLE:
        return "openai"
    if provider == ProviderType.GEMINI:
        # Gemini support is planned; default to OpenAI-style formatting for now.
        return "openai"
    return "openai"


def api_key_env_candidates(provider: ProviderType) -> list[str]:
    """Environment variables to check for an API key based on protocol."""
    if provider == ProviderType.ANTHROPIC:
        return ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]
    if provider == ProviderType.GEMINI:
        return ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    return [
        "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "KIMI_API_KEY",
        "QWEN_API_KEY",
        "GLM_API_KEY",
    ]


def api_base_env_candidates(provider: ProviderType) -> list[str]:
    """Environment variables to check for API base overrides."""
    if provider == ProviderType.ANTHROPIC:
        return ["ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"]
    if provider == ProviderType.GEMINI:
        return ["GEMINI_API_BASE", "GEMINI_BASE_URL", "GOOGLE_API_BASE_URL"]
    return [
        "OPENAI_COMPATIBLE_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_BASE_URL",
    ]


# Known vision-enabled model patterns for auto-detection
VISION_ENABLED_MODELS = {
    # Anthropic Claude models
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-3-5-sonnet",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku",
    "claude-3-5-haiku-20241022",
    "claude-3-opus",
    "claude-3-opus-20240229",
    "claude-3-sonnet",
    "claude-3-sonnet-20240229",
    "claude-3-haiku",
    "claude-3-haiku-20240307",
    # OpenAI models
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-vision-preview",
    "chatgpt-4o-latest",
    # Google Gemini models
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-thinking-exp",
    "gemini-exp-1206",
    "gemini-pro-vision",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    # Alibaba Qwen models (vision)
    "qwen-vl-max",
    "qwen-vl-plus",
    "qwen-vl-plus-latest",
    "qwen2-vl-72b-instruct",
    "qwen-vl-chat",
    "qwen-vl-7b-chat",
    # DeepSeek models (some support vision)
    "deepseek-vl",
    "deepseek-vl-chat",
    # Other vision models
    "glm-4v",
    "glm-4v-plus",
    "minivision-3b",
    "internvl2",
}


class ModelProfile(BaseModel):
    """Configuration for a specific AI model."""

    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    # Anthropic supports either api_key or auth_token; api_key takes precedence when both are set.
    auth_token: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # Total context window in tokens (if known). Falls back to heuristics when unset.
    context_window: Optional[int] = None
    # Tool handling for OpenAI-compatible providers. "native" uses tool_calls, "text" flattens tool
    # interactions into plain text to support providers that reject tool roles.
    openai_tool_mode: Literal["native", "text"] = "native"
    # Optional override for thinking protocol handling (e.g., "deepseek", "openrouter",
    # "qwen", "gemini_openai", "openai"). When unset, provider heuristics are used.
    thinking_mode: Optional[str] = None
    # Vision support flag. None = auto-detect based on model name, True/False = override.
    supports_vision: Optional[bool] = None
    # Pricing (USD per 1M tokens). Leave as 0 to skip cost calculation.
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0


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

    # Auto-detect based on model name
    model_name = model_profile.model.lower()
    return any(pattern in model_name for pattern in VISION_ENABLED_MODELS)


class ModelPointers(BaseModel):
    """Pointers to different model profiles for different purposes."""

    main: str = "default"
    quick: str = "default"


class GlobalConfig(BaseModel):
    """Global configuration stored in ~/.ripperdoc.json"""

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

    # Onboarding
    has_completed_onboarding: bool = False
    last_onboarding_version: Optional[str] = None

    # Statistics
    num_startups: int = 0

    @model_validator(mode="before")
    @classmethod
    def _migrate_safe_mode(cls, data: Any) -> Any:
        """Translate legacy safe_mode to the new yolo_mode flag."""
        if isinstance(data, dict) and "safe_mode" in data and "yolo_mode" not in data:
            data = dict(data)
            try:
                data["yolo_mode"] = not bool(data.pop("safe_mode"))
            except Exception:
                data["yolo_mode"] = False
        return data


class ProjectConfig(BaseModel):
    """Project-specific configuration stored in .ripperdoc/config.json"""

    # Tool permissions (project level - checked into git)
    allowed_tools: list[str] = Field(default_factory=list)
    bash_allow_rules: list[str] = Field(default_factory=list)
    bash_deny_rules: list[str] = Field(default_factory=list)
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


class ConfigManager:
    """Manages global and project-specific configuration."""

    def __init__(self) -> None:
        self.global_config_path = Path.home() / ".ripperdoc.json"
        self.current_project_path: Optional[Path] = None
        self._global_config: Optional[GlobalConfig] = None
        self._project_config: Optional[ProjectConfig] = None
        self._project_local_config: Optional[ProjectLocalConfig] = None

    def get_global_config(self) -> GlobalConfig:
        """Load and return global configuration."""
        if self._global_config is None:
            if self.global_config_path.exists():
                try:
                    data = json.loads(self.global_config_path.read_text())
                    self._global_config = GlobalConfig(**data)
                    logger.debug(
                        "[config] Loaded global configuration",
                        extra={
                            "path": str(self.global_config_path),
                            "profile_count": len(self._global_config.model_profiles),
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
                        "Error loading global config: %s: %s",
                        type(e).__name__,
                        e,
                        extra={"error": str(e)},
                    )
                    self._global_config = GlobalConfig()
            else:
                self._global_config = GlobalConfig()
                logger.debug(
                    "[config] Global config not found; using defaults",
                    extra={"path": str(self.global_config_path)},
                )
        return self._global_config

    def save_global_config(self, config: GlobalConfig) -> None:
        """Save global configuration."""
        self._global_config = config
        self.global_config_path.write_text(config.model_dump_json(indent=2))
        logger.debug(
            "[config] Saved global configuration",
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

    def get_api_key(self, provider: ProviderType) -> Optional[str]:
        """Get API key for a provider."""
        # First check environment variables
        env_candidates = api_key_env_candidates(provider)
        provider_env = f"{provider.value.upper()}_API_KEY"
        if provider_env not in env_candidates:
            env_candidates.insert(0, provider_env)
        for env_var in env_candidates:
            if env_var in os.environ:
                return os.environ[env_var]

        # Then check global config
        global_config = self.get_global_config()
        for profile in global_config.model_profiles.values():
            if profile.provider == provider and profile.api_key:
                return profile.api_key

        return None

    def get_current_model_profile(self, pointer: str = "main") -> Optional[ModelProfile]:
        """Get the current model profile for a given pointer."""
        global_config = self.get_global_config()

        # Get the profile name from the pointer
        profile_name = getattr(global_config.model_pointers, pointer, "default")

        # Return the profile
        return global_config.model_profiles.get(profile_name)

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
    ) -> GlobalConfig:
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

    def delete_model_profile(self, name: str) -> GlobalConfig:
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

    def set_model_pointer(self, pointer: str, profile_name: str) -> GlobalConfig:
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


def get_global_config() -> GlobalConfig:
    """Get global configuration."""
    return config_manager.get_global_config()


def save_global_config(config: GlobalConfig) -> None:
    """Save global configuration."""
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
) -> GlobalConfig:
    """Add or replace a model profile and persist the update."""
    return config_manager.add_model_profile(
        name,
        profile,
        overwrite=overwrite,
        set_as_main=set_as_main,
    )


def delete_model_profile(name: str) -> GlobalConfig:
    """Remove a model profile and update pointers as needed."""
    return config_manager.delete_model_profile(name)


def set_model_pointer(pointer: str, profile_name: str) -> GlobalConfig:
    """Update a model pointer (e.g., main/quick) to target a profile."""
    return config_manager.set_model_pointer(pointer, profile_name)


def get_current_model_profile(pointer: str = "main") -> Optional[ModelProfile]:
    """Convenience wrapper to fetch the active profile for a pointer."""
    return config_manager.get_current_model_profile(pointer)


def get_project_local_config(project_path: Optional[Path] = None) -> ProjectLocalConfig:
    """Get project-local configuration (not checked into git)."""
    return config_manager.get_project_local_config(project_path)


def save_project_local_config(
    config: ProjectLocalConfig, project_path: Optional[Path] = None
) -> None:
    """Save project-local configuration."""
    config_manager.save_project_local_config(config, project_path)


# ==============================================================================
# RIPPERDOC_* 全局环境变量支持
# ==============================================================================

# 环境变量名称常量
RIPPERDOC_BASE_URL = "RIPPERDOC_BASE_URL"
RIPPERDOC_AUTH_TOKEN = "RIPPERDOC_AUTH_TOKEN"
RIPPERDOC_MODEL = "RIPPERDOC_MODEL"
RIPPERDOC_SMALL_FAST_MODEL = "RIPPERDOC_SMALL_FAST_MODEL"
RIPPERDOC_API_KEY = "RIPPERDOC_API_KEY"
RIPPERDOC_PROTOCOL = "RIPPERDOC_PROTOCOL"


def _infer_protocol_from_url_and_model(base_url: str, model_name: str = "") -> ProviderType:
    """根据 BASE_URL 和模型名称推断协议类型.

    Args:
        base_url: API基础URL
        model_name: 模型名称

    Returns:
        推断的 ProviderType
    """
    base_lower = base_url.lower()
    model_lower = model_name.lower()

    # 显式域名检测
    if "anthropic.com" in base_lower:
        return ProviderType.ANTHROPIC
    if "generativelanguage.googleapis.com" in base_lower or "gemini" in model_lower:
        return ProviderType.GEMINI

    # URL 路径检测 - 检查路径中是否包含协议标识
    if "/anthropic" in base_lower or base_lower.endswith("/anthropic"):
        return ProviderType.ANTHROPIC
    if "/v1/" in base_lower or "/v1" in base_lower:
        # 大多数 /v1/ 路径是 OpenAI 兼容格式
        return ProviderType.OPENAI_COMPATIBLE

    # 模型名称前缀检测
    if model_lower.startswith("claude-"):
        return ProviderType.ANTHROPIC
    if model_lower.startswith("gemini-"):
        return ProviderType.GEMINI

    # 默认使用 OpenAI 兼容协议
    return ProviderType.OPENAI_COMPATIBLE


def _get_ripperdoc_env_overrides() -> Dict[str, Any]:
    """获取所有 RIPPERDOC_* 环境变量的值.

    Returns:
        包含所有已设置环境变量的字典
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
            overrides["protocol"] = ProviderType(protocol_str.lower())
        except ValueError:
            logger.warning(
                "[config] Invalid RIPPERDOC_PROTOCOL value: %s (must be anthropic, openai_compatible, or gemini)",
                protocol_str,
            )
    return overrides


def has_ripperdoc_env_overrides() -> bool:
    """检查是否设置了任何 RIPPERDOC_* 环境变量."""
    return bool(_get_ripperdoc_env_overrides())


def get_effective_model_profile(pointer: str = "main") -> Optional[ModelProfile]:
    """获取模型配置，应用 RIPPERDOC_* 环境变量覆盖.

    当设置了 RIPPERDOC_BASE_URL 环境变量时，完全在内存中创建 ModelProfile，
    不依赖也不写入配置文件。这是获取模型配置的新入口点，替代 get_current_model_profile()。

    Args:
        pointer: 模型指针名称 ("main" 或 "quick")

    Returns:
        应用环境变量覆盖后的 ModelProfile，如果没有则返回 None
    """
    env_overrides = _get_ripperdoc_env_overrides()
    base_url = env_overrides.get("base_url")

    # 如果设置了 RIPPERDOC_BASE_URL，完全在内存中创建 profile
    if base_url:
        # 确定模型名称
        if pointer == "quick":
            model_name = env_overrides.get("small_fast_model") or env_overrides.get("model")
        else:
            model_name = env_overrides.get("model")

        if not model_name:
            model_name = "claude-sonnet-4-5-20250929"

        # 确定协议类型
        protocol = env_overrides.get("protocol")
        if not protocol:
            protocol = _infer_protocol_from_url_and_model(base_url, model_name)

        # 在内存中创建新的 profile，不写入配置文件
        return ModelProfile(
            provider=protocol,
            model=model_name,
            api_base=base_url,
            api_key=env_overrides.get("api_key"),
            auth_token=env_overrides.get("auth_token"),
        )

    # 没有设置 RIPPERDOC_BASE_URL，返回配置文件中的 profile
    return get_current_model_profile(pointer)


def get_ripperdoc_env_status() -> Dict[str, str]:
    """获取 RIPPERDOC_* 环境变量状态信息用于诊断显示.

    Returns:
        字典，键为环境变量名称，值为格式化的显示字符串
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
