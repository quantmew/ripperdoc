"""Test configuration management."""

import pytest
from pathlib import Path
import json
import tempfile

from ripperdoc.core.config import (
    GlobalConfig,
    ProjectConfig,
    ModelProfile,
    ProviderType,
    ConfigManager
)


def test_global_config_creation():
    """Test creating a global config."""
    config = GlobalConfig()
    assert config.has_completed_onboarding == False
    assert config.theme == "dark"
    assert config.verbose == False


def test_model_profile_creation():
    """Test creating a model profile."""
    profile = ModelProfile(
        provider=ProviderType.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        api_key="test_key"
    )
    assert profile.provider == ProviderType.ANTHROPIC
    assert profile.model == "claude-3-5-sonnet-20241022"
    assert profile.api_key == "test_key"


def test_config_manager():
    """Test config manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager()
        manager.global_config_path = Path(tmpdir) / "config.json"

        # Get default config
        config = manager.get_global_config()
        assert isinstance(config, GlobalConfig)

        # Modify and save
        config.has_completed_onboarding = True
        config.model_profiles["test"] = ModelProfile(
            provider=ProviderType.ANTHROPIC,
            model="test-model"
        )
        manager.save_global_config(config)

        # Load again and verify
        manager._global_config = None
        loaded_config = manager.get_global_config()
        assert loaded_config.has_completed_onboarding == True
        assert "test" in loaded_config.model_profiles


def test_project_config():
    """Test project configuration."""
    config = ProjectConfig()
    assert config.dont_crawl_directory == False
    assert config.enable_architect_tool == False
    assert isinstance(config.allowed_tools, list)
