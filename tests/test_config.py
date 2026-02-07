"""Test configuration management."""

import json
from pathlib import Path
import tempfile

import pytest

from ripperdoc.cli.ui.provider_options import KNOWN_PROVIDERS
from ripperdoc.core.config import (
    UserConfig,
    ProjectConfig,
    ProjectLocalConfig,
    ModelProfile,
    ProtocolType,
    ConfigManager,
)


def test_user_config_creation():
    """Test creating a user config."""
    config = UserConfig()
    assert not config.has_completed_onboarding
    assert config.theme == "dark"
    assert not config.verbose


def test_model_profile_creation():
    """Test creating a model profile."""
    profile = ModelProfile(
         protocol=ProtocolType.ANTHROPIC, model="claude-3-5-sonnet-20241022", api_key="test_key"
    )
    assert profile.protocol == ProtocolType.ANTHROPIC
    assert profile.model == "claude-3-5-sonnet-20241022"
    assert profile.api_key == "test_key"


def test_model_profile_auth_token():
    """Anthropic profiles should allow setting auth_token."""
    profile = ModelProfile(
         protocol=ProtocolType.ANTHROPIC, model="claude-3-5-sonnet-20241022", auth_token="tok"
    )
    assert profile.auth_token == "tok"


def test_protocol_type_rejects_legacy_aliases():
    """ProtocolType should no longer accept legacy provider aliases."""
    for legacy_value in ("openai-compatible", "deepseek", "google"):
        with pytest.raises(ValueError):
            ProtocolType(legacy_value)


def test_config_manager():
    """Test config manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager()
        manager.global_config_path = Path(tmpdir) / "config.json"

        # Get default config
        config = manager.get_global_config()
        assert isinstance(config, UserConfig)

        # Modify and save
        config.has_completed_onboarding = True
        config.model_profiles["test"] = ModelProfile(
             protocol=ProtocolType.ANTHROPIC, model="test-model", auth_token="tok"
        )
        manager.save_global_config(config)

        # Load again and verify
        manager._global_config = None
        loaded_config = manager.get_global_config()
        assert loaded_config.has_completed_onboarding
        assert "test" in loaded_config.model_profiles
        assert loaded_config.model_profiles["test"].auth_token == "tok"


def test_config_manager_uses_directory_user_config_path():
    """Config manager should default to ~/.ripperdoc/config.json."""
    manager = ConfigManager()
    assert manager.global_config_path.name == "config.json"
    assert manager.global_config_path.parent.name == ".ripperdoc"


def test_effective_config_respects_local_project_user_precedence(tmp_path: Path):
    """Effective config should merge local > project > user."""
    manager = ConfigManager()
    manager.global_config_path = tmp_path / ".ripperdoc" / "config.json"

    manager.global_config_path.parent.mkdir(parents=True, exist_ok=True)
    manager.global_config_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "verbose": False,
                "model_profiles": {
                    "user_model": {
                        "protocol": "anthropic",
                        "model": "u-model",
                    }
                },
                "model_pointers": {"main": "user_model"},
            }
        )
    )

    project_dir = tmp_path / "project"
    project_config_dir = project_dir / ".ripperdoc"
    project_config_dir.mkdir(parents=True, exist_ok=True)
    (project_config_dir / "config.json").write_text(
        json.dumps(
            {
                "theme": "dracula",
                "verbose": True,
                "model_profiles": {
                    "project_model": {
                        "protocol": "openai_compatible",
                        "model": "p-model",
                    }
                },
                "model_pointers": {"main": "project_model"},
            }
        )
    )
    (project_config_dir / "config.local.json").write_text(
        json.dumps(
            {
                "verbose": False,
                "model_profiles": {
                    "local_model": {
                        "protocol": "anthropic",
                        "model": "l-model",
                    }
                },
                "model_pointers": {"main": "local_model"},
            }
        )
    )

    effective = manager.get_effective_config(project_path=project_dir)
    assert effective.theme == "dracula"
    assert effective.verbose is False
    assert "user_model" in effective.model_profiles
    assert "project_model" in effective.model_profiles
    assert "local_model" in effective.model_profiles
    assert effective.model_pointers.main == "local_model"


def test_project_config():
    """Test project configuration."""
    config = ProjectConfig()
    assert not config.dont_crawl_directory
    assert isinstance(config.allowed_tools, list)
    assert isinstance(config.bash_allow_rules, list)
    assert isinstance(config.bash_deny_rules, list)
    assert isinstance(config.bash_ask_rules, list)
    assert isinstance(config.working_directories, list)


def test_project_local_config_ignores_output_style_alias():
    """ProjectLocalConfig should ignore legacy outputStyle alias."""
    config = ProjectLocalConfig(**{"outputStyle": "learning"})
    assert config.output_style == "default"


def test_known_providers_use_model_profile_objects():
    """Provider model_list entries should be ModelProfile objects."""
    provider = KNOWN_PROVIDERS.default_choice.provider
    assert provider.model_list
    assert isinstance(provider.model_list[0], ModelProfile)
