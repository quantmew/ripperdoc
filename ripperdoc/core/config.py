"""Configuration management for Ripperdoc.

This module handles global and project-specific configuration,
including API keys, model settings, and user preferences.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum

from ripperdoc.utils.log import get_logger


logger = get_logger()


class ProviderType(str, Enum):
    """Supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    KIMI = "kimi"
    QWEN = "qwen"
    GLM = "glm"


class ModelProfile(BaseModel):
    """Configuration for a specific AI model."""

    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # Total context window in tokens (if known). Falls back to heuristics when unset.
    context_window: Optional[int] = None


class ModelPointers(BaseModel):
    """Pointers to different model profiles for different purposes."""

    main: str = "default"
    task: str = "default"
    reasoning: str = "default"
    quick: str = "default"


class GlobalConfig(BaseModel):
    """Global configuration stored in ~/.ripperdoc.json"""

    model_config = {"protected_namespaces": ()}

    # Model configuration
    model_profiles: Dict[str, ModelProfile] = Field(default_factory=dict)
    model_pointers: ModelPointers = Field(default_factory=ModelPointers)

    # User preferences
    theme: str = "dark"
    verbose: bool = False
    safe_mode: bool = True
    auto_compact_enabled: bool = True
    context_token_limit: Optional[int] = None

    # Onboarding
    has_completed_onboarding: bool = False
    last_onboarding_version: Optional[str] = None

    # Statistics
    num_startups: int = 0


class ProjectConfig(BaseModel):
    """Project-specific configuration stored in .ripperdoc/config.json"""

    # Tool permissions
    allowed_tools: list[str] = Field(default_factory=list)
    bash_allow_rules: list[str] = Field(default_factory=list)
    bash_deny_rules: list[str] = Field(default_factory=list)
    working_directories: list[str] = Field(default_factory=list)

    # Context
    context: Dict[str, str] = Field(default_factory=dict)
    context_files: list[str] = Field(default_factory=list)

    # History
    history: list[str] = Field(default_factory=list)

    # Project settings
    dont_crawl_directory: bool = False
    enable_architect_tool: bool = False

    # Trust
    has_trust_dialog_accepted: bool = False

    # Session tracking
    last_cost: Optional[float] = None
    last_duration: Optional[float] = None
    last_session_id: Optional[str] = None


class ConfigManager:
    """Manages global and project-specific configuration."""

    def __init__(self) -> None:
        self.global_config_path = Path.home() / ".ripperdoc.json"
        self.current_project_path: Optional[Path] = None
        self._global_config: Optional[GlobalConfig] = None
        self._project_config: Optional[ProjectConfig] = None

    def get_global_config(self) -> GlobalConfig:
        """Load and return global configuration."""
        if self._global_config is None:
            if self.global_config_path.exists():
                try:
                    data = json.loads(self.global_config_path.read_text())
                    self._global_config = GlobalConfig(**data)
                except Exception as e:
                    logger.error(f"Error loading global config: {e}")
                    self._global_config = GlobalConfig()
            else:
                self._global_config = GlobalConfig()
        return self._global_config

    def save_global_config(self, config: GlobalConfig) -> None:
        """Save global configuration."""
        self._global_config = config
        self.global_config_path.write_text(config.model_dump_json(indent=2))

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
                except Exception as e:
                    logger.error(f"Error loading project config: {e}")
                    self._project_config = ProjectConfig()
            else:
                self._project_config = ProjectConfig()

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

    def get_api_key(self, provider: ProviderType) -> Optional[str]:
        """Get API key for a provider."""
        # First check environment variables
        env_var = f"{provider.value.upper()}_API_KEY"
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
        """Point a logical model slot (e.g., main/task) to a profile name."""
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
    """Update a model pointer (e.g., main/task) to target a profile."""
    return config_manager.set_model_pointer(pointer, profile_name)


def get_current_model_profile(pointer: str = "main") -> Optional[ModelProfile]:
    """Convenience wrapper to fetch the active profile for a pointer."""
    return config_manager.get_current_model_profile(pointer)
