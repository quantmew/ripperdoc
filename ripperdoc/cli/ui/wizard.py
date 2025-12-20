"""
Interactive onboarding wizard for Ripperdoc.
"""

from typing import Dict, List, Optional, Tuple

import click
from rich.console import Console

from ripperdoc.core.config import (
    GlobalConfig,
    ModelProfile,
    ProviderType,
    get_global_config,
    save_global_config,
)
from ripperdoc.utils.prompt import prompt_secret


console = Console()


# Mapping from provider choice to default API base URLs
PROVIDER_API_BASES: Dict[str, Optional[str]] = {
    # OpenAI compatible providers
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "minimax": "https://api.minimax.chat/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    # Gemini
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    # Anthropic already has its own handling
    "anthropic": None,  # Uses default Anthropic endpoint
}


# Recommended models for different providers
PROVIDER_MODELS: Dict[str, List[str]] = {
    "openai": [
        "gpt-5.1",
        "gpt-5.1-chat",
        "gpt-5.1-codex",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-small-latest",
        "codestral-latest",
        "pixtral-large-latest",
    ],
    "kimi": [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ],
    "qwen": [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen2.5-32b",
        "qwen2.5-coder-32b",
    ],
    "glm": [
        "glm-4-flash",
        "glm-4-plus",
        "glm-4-air",
        "glm-4-long",
    ],
    "minimax": [
        "abab6.5s-chat",
        "abab6.5-chat",
    ],
    "siliconflow": [
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-14B-Instruct",
        "DeepSeek-V2.5",
    ],
    "gemini": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
}


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
    # Simple onboarding
    provider_choices = [
        *[p.value for p in ProviderType],
        "openai",
        "deepseek",
        "mistral",
        "gemini",
        "kimi",
        "qwen",
        "glm",
        "minimax",
        "siliconflow",
        "custom",
    ]
    provider_choice = click.prompt(
        "Choose your model protocol",
        type=click.Choice(provider_choices),
        default=ProviderType.ANTHROPIC.value,
    )

    api_base = None
    if provider_choice == "custom":
        provider_choice = click.prompt(
            "Protocol family (for API compatibility)",
            type=click.Choice([p.value for p in ProviderType]),
            default=ProviderType.OPENAI_COMPATIBLE.value,
        )
        api_base = click.prompt("API Base URL")

    api_key = ""
    while not api_key:
        api_key = prompt_secret("Enter your API key").strip()
        if not api_key:
            console.print("[red]API key is required.[/red]")

    provider = ProviderType(provider_choice)

    # Get model name with provider-specific suggestions
    model, api_base = get_model_name_with_suggestions(provider, provider_choice, api_base)

    # Get context window
    context_window = get_context_window()

    # Create model profile
    config.model_profiles["default"] = ModelProfile(
        provider=provider,
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
    provider: ProviderType,
    provider_choice: str,
    api_base: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Get model name with provider-specific suggestions and default API base.
    
    Returns:
        Tuple of (model_name, api_base)
    """
    # Set default API base based on provider choice
    if provider == ProviderType.OPENAI_COMPATIBLE:
        if api_base is None and provider_choice in PROVIDER_API_BASES:
            api_base = PROVIDER_API_BASES[provider_choice]
            if api_base:
                console.print(f"[dim]Using default API base: {api_base}[/dim]")
    
    # Get default model and suggestions
    default_model = get_default_model(provider, provider_choice)
    suggestions = get_model_suggestions(provider_choice)
    
    # Show suggestions if available
    if suggestions:
        console.print("\n[dim]Available models for this provider:[/dim]")
        for i, model_name in enumerate(suggestions[:5]):  # Show top 5
            console.print(f"  [dim]{i+1}. {model_name}[/dim]")
        console.print("")
    
    # Prompt for model name
    if provider == ProviderType.ANTHROPIC:
        model = click.prompt("Model name", default=default_model)
    elif provider == ProviderType.OPENAI_COMPATIBLE:
        model = click.prompt("Model name", default=default_model)
        # Prompt for API base if still not set
        if api_base is None:
            api_base_input = click.prompt(
                "API base URL (optional)", default="", show_default=False
            )
            api_base = api_base_input or None
    elif provider == ProviderType.GEMINI:
        console.print(
            "[yellow]Gemini protocol support is not yet available; configuration is saved for "
            "future support.[/yellow]"
        )
        model = click.prompt("Model name", default=default_model)
        if api_base is None:
            api_base_input = click.prompt(
                "API base URL (optional)", default="", show_default=False
            )
            api_base = api_base_input or None
    else:
        model = click.prompt("Model name", default=default_model)
    
    return model, api_base


def get_default_model(provider: ProviderType, provider_choice: str) -> str:
    """Get default model name based on provider."""
    if provider == ProviderType.ANTHROPIC:
        return "claude-3-5-sonnet-20241022"
    elif provider == ProviderType.OPENAI_COMPATIBLE:
        if provider_choice == "deepseek":
            return "deepseek-chat"
        elif provider_choice == "mistral":
            return "mistral-large-latest"
        elif provider_choice == "kimi":
            return "moonshot-v1-8k"
        elif provider_choice == "qwen":
            return "qwen-turbo"
        elif provider_choice == "glm":
            return "glm-4-flash"
        elif provider_choice == "minimax":
            return "abab6.5s-chat"
        elif provider_choice == "siliconflow":
            return "Qwen2.5-32B-Instruct"
        else:
            return "gpt-4o-mini"
    elif provider == ProviderType.GEMINI:
        return "gemini-1.5-pro"
    else:
        return ""


def get_model_suggestions(provider_choice: str) -> List[str]:
    """Get model suggestions for a provider."""
    return PROVIDER_MODELS.get(provider_choice, [])


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