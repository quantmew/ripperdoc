"""
Interactive onboarding wizard for Ripperdoc.
"""

from typing import List, Optional, Tuple

import click
from rich.console import Console

from ripperdoc.cli.ui.provider_options import (
    KNOWN_PROVIDERS,
    ProviderOption,
    default_model_for_protocol,
)
from ripperdoc.core.config import (
    GlobalConfig,
    ModelProfile,
    ProviderType,
    get_global_config,
    save_global_config,
)
from ripperdoc.utils.prompt import prompt_secret


console = Console()


def resolve_provider_choice(raw_choice: str, provider_keys: List[str]) -> Optional[str]:
    """Normalize user input into a provider key."""
    normalized = raw_choice.strip().lower()
    if normalized in provider_keys:
        return normalized
    try:
        idx = int(normalized)
        if 1 <= idx <= len(provider_keys):
            return provider_keys[idx - 1]
    except ValueError:
        return None
    return None


def check_onboarding() -> bool:
    """Check if onboarding is complete and run if needed."""
    config = get_global_config()

    if config.has_completed_onboarding:
        return True

    console.print("[bold cyan]Welcome to Ripperdoc![/bold cyan]\n")
    console.print("Let's set up your AI model configuration.\n")

    return run_onboarding_wizard(config)


def run_onboarding_wizard(config: GlobalConfig) -> bool:
    """Run interactive onboarding wizard."""
    provider_keys = KNOWN_PROVIDERS.keys() + ["custom"]
    default_choice_key = KNOWN_PROVIDERS.default_choice.key

    # Display provider options vertically
    console.print("[bold]Available providers:[/bold]")
    for i, provider_key in enumerate(provider_keys, 1):
        marker = "[cyan]→[/cyan]" if provider_key == default_choice_key else " "
        console.print(f"  {marker} {i}. {provider_key}")
    console.print("")

    # Prompt for provider choice with validation
    provider_choice: Optional[str] = None
    while provider_choice is None:
        raw_choice = click.prompt(
            "Choose your model provider",
            default=default_choice_key,
        )
        provider_choice = resolve_provider_choice(raw_choice, provider_keys)
        if provider_choice is None:
            console.print(
                f"[red]Invalid choice. Please enter a provider name or number (1-{len(provider_keys)}).[/red]"
            )

    api_base_override: Optional[str] = None
    if provider_choice == "custom":
        protocol_input = click.prompt(
            "Protocol family (for API compatibility)",
            type=click.Choice([p.value for p in ProviderType]),
            default=ProviderType.OPENAI_COMPATIBLE.value,
        )
        protocol = ProviderType(protocol_input)
        api_base_override = click.prompt("API Base URL")
        provider_option = ProviderOption(
            key="custom",
            protocol=protocol,
            default_model=default_model_for_protocol(protocol),
            model_suggestions=(),
        )
    else:
        provider_option = KNOWN_PROVIDERS.get(provider_choice)
        if provider_option is None:
            provider_option = ProviderOption(
                key=provider_choice,
                protocol=ProviderType.OPENAI_COMPATIBLE,
                default_model=default_model_for_protocol(ProviderType.OPENAI_COMPATIBLE),
                model_suggestions=(),
            )

    api_key = ""
    while not api_key:
        api_key = prompt_secret("Enter your API key").strip()
        if not api_key:
            console.print("[red]API key is required.[/red]")

    # Get model name with provider-specific suggestions
    model, api_base = get_model_name_with_suggestions(provider_option, api_base_override)

    # Get context window
    context_window = get_context_window()

    # Create model profile
    config.model_profiles["default"] = ModelProfile(
        provider=provider_option.protocol,
        model=model,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,
    )

    config.has_completed_onboarding = True
    config.last_onboarding_version = get_version()

    save_global_config(config)

    console.print("\n[green]✓ Configuration saved![/green]\n")
    return True


def get_model_name_with_suggestions(
    provider: ProviderOption,
    api_base_override: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Get model name with provider-specific suggestions and default API base.
    
    Returns:
        Tuple of (model_name, api_base)
    """
    # Set default API base based on provider choice
    api_base = api_base_override
    if api_base is None and provider.default_api_base:
        api_base = provider.default_api_base
        console.print(f"[dim]Using default API base: {api_base}[/dim]")

    default_model = provider.default_model or default_model_for_protocol(provider.protocol)
    suggestions = list(provider.model_suggestions)

    # Show suggestions if available
    if suggestions:
        console.print("\n[dim]Available models for this provider:[/dim]")
        for i, model_name in enumerate(suggestions[:5]):  # Show top 5
            console.print(f"  [dim]{i+1}. {model_name}[/dim]")
        console.print("")

    # Prompt for model name
    if provider.protocol == ProviderType.ANTHROPIC:
        model = click.prompt("Model name", default=default_model)
    elif provider.protocol == ProviderType.OPENAI_COMPATIBLE:
        model = click.prompt("Model name", default=default_model)
        # Prompt for API base if still not set
        if api_base is None:
            api_base_input = click.prompt(
                "API base URL (optional)", default="", show_default=False
            )
            api_base = api_base_input or None
    elif provider.protocol == ProviderType.GEMINI:
        model = click.prompt("Model name", default=default_model)
        if api_base is None:
            api_base_input = click.prompt(
                "API base URL (optional)", default="", show_default=False
            )
            api_base = api_base_input or None
    else:
        model = click.prompt("Model name", default=default_model)

    return model, api_base


def get_context_window() -> Optional[int]:
    """Get context window size from user."""
    context_window_input = click.prompt(
        "Context window in tokens (optional, press Enter to skip)",
        default="",
        show_default=False,
    )
    context_window = None
    if context_window_input.strip():
        try:
            context_window = int(context_window_input.strip())
        except ValueError:
            console.print(
                "[yellow]Invalid context window, using auto-detected defaults.[/yellow]"
            )
    return context_window


def get_version() -> str:
    """Get current version of Ripperdoc."""
    try:
        from ripperdoc import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    # For testing
    config = get_global_config()
    config.has_completed_onboarding = False
    run_onboarding_wizard(config)
