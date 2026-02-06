"""
Interactive onboarding wizard for Ripperdoc.
"""

from typing import List, Optional, Tuple

import click
from rich.console import Console

from ripperdoc.cli.ui.choice import ChoiceOption, onboarding_style, prompt_choice
from ripperdoc.cli.ui.provider_options import (
    KNOWN_PROVIDERS,
    ProviderOption,
    default_model_for_protocol,
)
from ripperdoc.core.config import (
    UserConfig,
    ModelProfile,
    ProviderType,
    get_global_config,
    has_ripperdoc_env_overrides,
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

    # Check if there are valid RIPPERDOC_* environment variable configurations
    # If any RIPPERDOC_* is set, can skip onboarding
    # Don't write to config file, only handle in memory
    if has_ripperdoc_env_overrides():
        # Mark onboarding as completed in memory, but don't save to config file
        # This way it will still work next time if environment variables exist
        config.has_completed_onboarding = True
        config.last_onboarding_version = get_version()
        save_global_config(config)
        return True

    console.print("[bold cyan]Welcome to Ripperdoc![/bold cyan]\n")
    console.print("Let's set up your AI model configuration.\n")

    return run_onboarding_wizard(config)


def run_onboarding_wizard(config: UserConfig) -> bool:
    """Run interactive onboarding wizard."""
    provider_keys = list(KNOWN_PROVIDERS.keys()) + ["custom"]
    default_choice_key = KNOWN_PROVIDERS.default_choice.key

    # Build provider choice options with rich styling
    provider_options = [
        ChoiceOption(
            key,
            key,  # Plain text, not colored
            is_default=(key == default_choice_key),
        )
        for key in provider_keys
    ]

    # Prompt for provider choice using unified choice component
    provider_choice = prompt_choice(
        message="Choose your model provider",
        options=provider_options,
        title="Available providers",
        allow_esc=True,
        esc_value=default_choice_key,
        style=onboarding_style(),
    )

    # Validate the choice (in case user typed an invalid value)
    validated_choice = resolve_provider_choice(provider_choice, provider_keys)
    if validated_choice is None:
        console.print(
            f"[red]Invalid choice. Please enter a provider name or number (1-{len(provider_keys)}).[/red]"
        )
        return run_onboarding_wizard(config)
    provider_choice = validated_choice

    api_base_override: Optional[str] = None
    if provider_choice == "custom":
        # Build protocol choice options
        protocol_options = [
            ChoiceOption(
                p.value,
                p.value,  # Plain text
                is_default=(p == ProviderType.OPENAI_COMPATIBLE),
            )
            for p in ProviderType
        ]

        protocol_input = prompt_choice(
            message="Choose the protocol family (for API compatibility)",
            options=protocol_options,
            title="Custom Provider - Protocol Selection",
            allow_esc=True,
            esc_value=ProviderType.OPENAI_COMPATIBLE.value,
            style=onboarding_style(),
        )
        protocol = ProviderType(protocol_input)

        # Get API base URL - use click.prompt for free-form text input
        console.print("\n[dim]Now we need your API Base URL.[/dim]")
        api_base_override = click.prompt("API Base URL").strip()

        provider_option = ProviderOption(
            key="custom",
            protocol=protocol,
            default_model=default_model_for_protocol(protocol),
            model_suggestions=(),
        )
    else:
        provider_option = KNOWN_PROVIDERS.get(provider_choice) or ProviderOption(
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

    # Prompt for model name using unified choice component if suggestions exist
    if suggestions:
        # Create choice options for models
        model_options = [
            ChoiceOption(model, model)  # Plain text
            for model in suggestions[:5]  # Show top 5
        ]
        # Add custom option
        model_options.append(ChoiceOption("custom", "<dim>Custom model...</dim>"))

        model = prompt_choice(
            message="Select a model or choose 'Custom' to enter manually",
            options=model_options,
            title="Available models for this provider",
            description=f"Default: {default_model}",
            allow_esc=True,
            esc_value=default_model,
            style=onboarding_style(),
        )

        # If user chose custom, prompt for manual input using click.prompt
        if model == "custom":
            console.print("\n[dim]Enter your custom model name:[/dim]")
            model = click.prompt("Model name", default=default_model).strip()
    else:
        # No suggestions, use default
        model = default_model

    # Prompt for API base if still not set (for OpenAI-compatible providers)
    if provider.protocol == ProviderType.OPENAI_COMPATIBLE and api_base is None:
        api_base_input = prompt_choice(
            message="Enter API base URL (or press Enter to skip)",
            options=[ChoiceOption("", "<dim>Skip</dim>")],
            title="API base URL (optional)",
            allow_esc=True,
            esc_value="",
            style=onboarding_style(),
        )
        api_base = api_base_input or None
    elif provider.protocol == ProviderType.GEMINI and api_base is None:
        api_base_input = prompt_choice(
            message="Enter API base URL (or press Enter to skip)",
            options=[ChoiceOption("", "<dim>Skip</dim>")],
            title="API base URL (optional)",
            allow_esc=True,
            esc_value="",
            style=onboarding_style(),
        )
        api_base = api_base_input or None

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
            console.print("[yellow]Invalid context window, using auto-detected defaults.[/yellow]")
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
