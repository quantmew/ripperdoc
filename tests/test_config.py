"""Test configuration management."""

from pathlib import Path
import tempfile

from ripperdoc.core.config import (
    GlobalConfig,
    ProjectConfig,
    ModelProfile,
    ProviderType,
    ConfigManager,
)


def test_global_config_creation():
    """Test creating a global config."""
    config = GlobalConfig()
    assert not config.has_completed_onboarding
    assert config.theme == "dark"
    assert not config.verbose


def test_model_profile_creation():
    """Test creating a model profile."""
    profile = ModelProfile(
        provider=ProviderType.ANTHROPIC, model="claude-3-5-sonnet-20241022", api_key="test_key"
    )
    assert profile.provider == ProviderType.ANTHROPIC
    assert profile.model == "claude-3-5-sonnet-20241022"
    assert profile.api_key == "test_key"


def test_model_profile_auth_token():
    """Anthropic profiles should allow setting auth_token."""
    profile = ModelProfile(
        provider=ProviderType.ANTHROPIC, model="claude-3-5-sonnet-20241022", auth_token="tok"
    )
    assert profile.auth_token == "tok"


def test_provider_type_normalizes_aliases():
    """ProviderType should treat legacy provider names as protocol families."""
    assert ProviderType("openai-compatible") == ProviderType.OPENAI_COMPATIBLE
    assert ProviderType("deepseek") == ProviderType.OPENAI_COMPATIBLE
    assert ProviderType("google") == ProviderType.GEMINI


def test_model_profile_accepts_legacy_provider_label():
    """ModelProfile should coerce legacy provider labels to the right protocol."""
    profile = ModelProfile(provider="deepseek", model="deepseek-chat")
    assert profile.provider == ProviderType.OPENAI_COMPATIBLE


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
            provider=ProviderType.ANTHROPIC, model="test-model", auth_token="tok"
        )
        manager.save_global_config(config)

        # Load again and verify
        manager._global_config = None
        loaded_config = manager.get_global_config()
        assert loaded_config.has_completed_onboarding
        assert "test" in loaded_config.model_profiles
        assert loaded_config.model_profiles["test"].auth_token == "tok"


def test_project_config():
    """Test project configuration."""
    config = ProjectConfig()
    assert not config.dont_crawl_directory
    assert not config.enable_architect_tool
    assert isinstance(config.allowed_tools, list)
    assert isinstance(config.bash_allow_rules, list)
    assert isinstance(config.bash_deny_rules, list)
    assert isinstance(config.working_directories, list)
