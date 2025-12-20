from typing import Any
import os
from pathlib import Path
from typing import List, Optional, Tuple

from ripperdoc import __version__
from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.config import (
    ModelProfile,
    ProviderType,
    api_base_env_candidates,
    api_key_env_candidates,
    get_global_config,
)
from ripperdoc.utils.memory import MAX_CONTENT_LENGTH, MemoryFile, collect_all_memory_files

from .base import SlashCommand


def _auth_token_display(profile: Optional[ModelProfile]) -> Tuple[str, Optional[str]]:
    """Return a safe auth token summary and the env var used, if any."""
    if not profile:
        return ("Not configured", None)

    provider_value = (
        profile.provider.value if hasattr(profile.provider, "value") else str(profile.provider)
    )

    env_candidates = api_key_env_candidates(profile.provider)
    provider_env = f"{provider_value.upper()}_API_KEY" if provider_value else None
    if provider_env and provider_env not in env_candidates:
        env_candidates = [provider_env, *env_candidates]

    env_var = next((name for name in env_candidates if os.environ.get(name)), None)
    if env_var:
        return (f"{env_var} (env)", env_var)
    if profile.api_key or getattr(profile, "auth_token", None):
        return ("Configured in profile", None)
    return ("Missing", None)


def _api_base_display(profile: Optional[ModelProfile]) -> str:
    """Return a human-readable API base URL line."""
    if not profile:
        return "API base URL: Not configured"

    label_map = {
        ProviderType.ANTHROPIC: "Anthropic base URL",
        ProviderType.OPENAI_COMPATIBLE: "OpenAI-compatible base URL",
        ProviderType.GEMINI: "Gemini base URL",
    }
    label = label_map.get(profile.provider, "API base URL")
    provider_value = (
        profile.provider.value if hasattr(profile.provider, "value") else str(profile.provider)
    )

    env_candidates = api_base_env_candidates(profile.provider)
    provider_env = f"{provider_value.upper()}_BASE_URL" if provider_value else None
    if provider_env and provider_env not in env_candidates:
        env_candidates = [provider_env, *env_candidates]

    base_url = profile.api_base
    if not base_url:
        base_url = next(
            (os.environ.get(name) for name in env_candidates if os.environ.get(name)), None
        )

    return f"{label}: {base_url or 'default'}"


def _memory_status_lines(memory_files: List[MemoryFile]) -> List[str]:
    """Summarize AGENTS memory files and any issues."""
    if not memory_files:
        return ["None detected"]

    lines = [f"{len(memory_files)} file(s) loaded"]
    oversized = [memory for memory in memory_files if len(memory.content) > MAX_CONTENT_LENGTH]
    for memory in oversized:
        lines.append(f"! {memory.path} ({len(memory.content)} chars > {MAX_CONTENT_LENGTH})")
    return lines


def _setting_sources_summary(
    config: Any,
    profile: Optional[ModelProfile],
    memory_files: List[MemoryFile],
    auth_env_var: Optional[str],
    yolo_mode: bool,
    verbose: bool,
    project_path: Path,
) -> str:
    """Describe where settings for this session were sourced from."""
    sources: list[str] = []
    if (Path.home() / ".ripperdoc.json").exists():
        sources.append("User settings")

    project_config_path = project_path / ".ripperdoc" / "config.json"
    if project_config_path.exists():
        sources.append("Project settings")

    if any(memory.type == "Local" for memory in memory_files):
        sources.append("Local memory")
    if any(memory.type == "Project" for memory in memory_files):
        sources.append("Project memory")
    if auth_env_var:
        sources.append("Environment variables")

    config_yolo_mode = getattr(config, "yolo_mode", False)
    config_verbose = getattr(config, "verbose", False)
    if yolo_mode != config_yolo_mode or verbose != config_verbose:
        sources.append("Command line arguments")

    if profile and profile.api_key and not auth_env_var:
        sources.append("Profile configuration")

    if not sources:
        sources.append("Defaults")

    unique_sources = list(dict.fromkeys(sources))
    return ", ".join(unique_sources)


def _handle(ui: Any, _: str) -> bool:
    config = get_global_config()
    profile = get_profile_for_pointer("main")
    memory_files = collect_all_memory_files()

    auth_summary, auth_env_var = _auth_token_display(profile)
    api_base_summary = _api_base_display(profile)
    memory_lines = _memory_status_lines(memory_files)
    setting_sources = _setting_sources_summary(
        config,
        profile,
        memory_files,
        auth_env_var,
        ui.yolo_mode,
        ui.verbose,
        ui.project_path,
    )

    model_label = profile.model if profile else "Not configured"

    ui.console.print()
    ui.console.rule()
    ui.console.print(" Status:\n")
    ui.console.print(f" Version: {__version__}")
    ui.console.print(f" Session ID: {ui.session_id}")
    ui.console.print(f" cwd: {Path.cwd()}")
    ui.console.print(f" Auth token: {auth_summary}")
    ui.console.print(f" {api_base_summary}")
    ui.console.print()
    ui.console.print(f" Model: {model_label}")
    ui.console.print(" Memory:")
    for line in memory_lines:
        ui.console.print(f"  {line}")
    ui.console.print(f" Setting sources: {setting_sources}")
    return True


command = SlashCommand(
    name="status",
    description="Show session status",
    handler=_handle,
)


__all__ = ["command"]
