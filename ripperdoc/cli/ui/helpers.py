"""Shared helper functions for the Rich UI."""

from typing import Optional

from ripperdoc.core.config import get_current_model_profile, get_global_config, ModelProfile


def get_profile_for_pointer(pointer: str = "main") -> Optional[ModelProfile]:
    """Return the configured ModelProfile for a logical pointer or default."""
    profile = get_current_model_profile(pointer)
    if profile:
        return profile
    config = get_global_config()
    if "default" in config.model_profiles:
        return config.model_profiles.get("default")
    if config.model_profiles:
        first_name = next(iter(config.model_profiles))
        return config.model_profiles.get(first_name)
    return None


__all__ = ["get_profile_for_pointer"]
