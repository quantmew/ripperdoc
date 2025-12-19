from typing import Any, Optional

from rich.markup import escape

from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.config import (
    ModelProfile,
    ProviderType,
    add_model_profile,
    delete_model_profile,
    get_global_config,
    set_model_pointer,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.prompt import prompt_secret

from .base import SlashCommand

logger = get_logger()


def _handle(ui: Any, trimmed_arg: str) -> bool:
    console = ui.console
    tokens = trimmed_arg.split()
    subcmd = tokens[0].lower() if tokens else ""
    config = get_global_config()
    logger.info(
        "[models_cmd] Handling /models command",
        extra={"subcommand": subcmd or "list", "session_id": getattr(ui, "session_id", None)},
    )

    def print_models_usage() -> None:
        console.print("[bold]/models[/bold] — list configured models")
        console.print("[bold]/models add <name>[/bold] — add or update a model profile")
        console.print("[bold]/models edit <name>[/bold] — edit an existing model profile")
        console.print("[bold]/models delete <name>[/bold] — delete a model profile")
        console.print("[bold]/models use <name>[/bold] — set the main model pointer")
        console.print(
            "[bold]/models use <pointer> <name>[/bold] — set a specific pointer (main/task/reasoning/quick)"
        )

    def parse_int(prompt_text: str, default_value: Optional[int]) -> Optional[int]:
        raw = console.input(prompt_text).strip()
        if not raw:
            return default_value
        try:
            return int(raw)
        except ValueError:
            console.print("[yellow]Invalid number, keeping previous value.[/yellow]")
            return default_value

    def parse_float(prompt_text: str, default_value: float) -> float:
        raw = console.input(prompt_text).strip()
        if not raw:
            return default_value
        try:
            return float(raw)
        except ValueError:
            console.print("[yellow]Invalid number, keeping previous value.[/yellow]")
            return default_value

    if subcmd in ("help", "-h", "--help"):
        print_models_usage()
        return True

    if subcmd in ("add", "create"):
        profile_name = tokens[1] if len(tokens) > 1 else console.input("Profile name: ").strip()
        if not profile_name:
            console.print("[red]Model profile name is required.[/red]")
            print_models_usage()
            return True

        overwrite = False
        existing_profile = config.model_profiles.get(profile_name)
        if existing_profile:
            confirm = (
                console.input(f"Profile '{profile_name}' exists. Overwrite? [y/N]: ")
                .strip()
                .lower()
            )
            if confirm not in ("y", "yes"):
                return True
            overwrite = True

        current_profile = get_profile_for_pointer("main")
        default_provider = (
            (current_profile.provider.value) if current_profile else ProviderType.ANTHROPIC.value
        )
        provider_input = (
            console.input(
                f"Protocol ({', '.join(p.value for p in ProviderType)}) [{default_provider}]: "
            )
            .strip()
            .lower()
            or default_provider
        )
        try:
            provider = ProviderType(provider_input)
        except ValueError:
            console.print(f"[red]Invalid provider: {escape(provider_input)}[/red]")
            print_models_usage()
            return True

        default_model = (
            existing_profile.model
            if existing_profile
            else (current_profile.model if current_profile else "")
        )
        model_prompt = f"Model name to send{f' [{default_model}]' if default_model else ''}: "
        model_name = console.input(model_prompt).strip() or default_model
        if not model_name:
            console.print("[red]Model name is required.[/red]")
            return True

        api_key_input = prompt_secret("API key (leave blank to keep unset)").strip()
        api_key = api_key_input or (existing_profile.api_key if existing_profile else None)

        auth_token = existing_profile.auth_token if existing_profile else None
        if provider == ProviderType.ANTHROPIC:
            auth_token_input = prompt_secret(
                "Auth token (Anthropic only, leave blank to keep unset)"
            ).strip()
            auth_token = auth_token_input or auth_token
        else:
            auth_token = None

        api_base_default = existing_profile.api_base if existing_profile else ""
        api_base = (
            console.input(
                f"API base (optional){f' [{api_base_default}]' if api_base_default else ''}: "
            ).strip()
            or api_base_default
            or None
        )

        max_tokens_default = existing_profile.max_tokens if existing_profile else 4096
        max_tokens = (
            parse_int(
                f"Max output tokens [{max_tokens_default}]: ",
                max_tokens_default,
            )
            or max_tokens_default
        )

        temp_default = existing_profile.temperature if existing_profile else 0.7
        temperature = parse_float(
            f"Temperature [{temp_default}]: ",
            temp_default,
        )

        context_window_default = existing_profile.context_window if existing_profile else None
        context_prompt = "Context window tokens (optional"
        if context_window_default:
            context_prompt += f", current {context_window_default}"
        context_prompt += "): "
        context_window = parse_int(context_prompt, context_window_default)

        default_set_main = (
            not config.model_profiles
            or getattr(config.model_pointers, "main", "") not in config.model_profiles
        )
        set_main_input = (
            console.input(f"Set as main model? [{'Y' if default_set_main else 'y'}/N]: ")
            .strip()
            .lower()
        )
        set_as_main = set_main_input in ("y", "yes") if set_main_input else default_set_main

        profile = ModelProfile(
            provider=provider,
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
            auth_token=auth_token,
        )

        try:
            add_model_profile(
                profile_name,
                profile,
                overwrite=overwrite,
                set_as_main=set_as_main,
            )
        except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
            console.print(f"[red]Failed to save model: {escape(str(exc))}[/red]")
            logger.warning(
                "[models_cmd] Failed to save model profile: %s: %s",
                type(exc).__name__,
                exc,
                extra={"profile": profile_name, "session_id": getattr(ui, "session_id", None)},
            )
            return True

        marker = " (main)" if set_as_main else ""
        console.print(f"[green]✓ Model '{escape(profile_name)}' saved{marker}[/green]")
        return True

    if subcmd in ("edit", "update"):
        profile_name = tokens[1] if len(tokens) > 1 else console.input("Profile to edit: ").strip()
        existing_profile = config.model_profiles.get(profile_name or "")
        if not profile_name or not existing_profile:
            console.print("[red]Model profile not found.[/red]")
            print_models_usage()
            return True

        provider_default = existing_profile.provider.value
        provider_input = (
            console.input(
                f"Protocol ({', '.join(p.value for p in ProviderType)}) [{provider_default}]: "
            )
            .strip()
            .lower()
            or provider_default
        )
        try:
            provider = ProviderType(provider_input)
        except ValueError:
            console.print(f"[red]Invalid provider: {escape(provider_input)}[/red]")
            return True

        model_name = (
            console.input(f"Model name to send [{existing_profile.model}]: ").strip()
            or existing_profile.model
        )

        api_key_label = "[set]" if existing_profile.api_key else "[not set]"
        api_key_prompt = f"API key {api_key_label} (Enter=keep, '-'=clear)"
        api_key_input = prompt_secret(api_key_prompt).strip()
        if api_key_input == "-":
            api_key = None
        elif api_key_input:
            api_key = api_key_input
        else:
            api_key = existing_profile.api_key

        auth_token = existing_profile.auth_token
        if (
            provider == ProviderType.ANTHROPIC
            or existing_profile.provider == ProviderType.ANTHROPIC
        ):
            auth_label = "[set]" if auth_token else "[not set]"
            auth_prompt = f"Auth token (Anthropic only) {auth_label} (Enter=keep, '-'=clear)"
            auth_token_input = prompt_secret(auth_prompt).strip()
            if auth_token_input == "-":
                auth_token = None
            elif auth_token_input:
                auth_token = auth_token_input
        else:
            auth_token = None

        api_base = (
            console.input(f"API base (optional) [{existing_profile.api_base or ''}]: ").strip()
            or existing_profile.api_base
        )
        if api_base == "":
            api_base = None

        max_tokens = (
            parse_int(
                f"Max output tokens [{existing_profile.max_tokens}]: ",
                existing_profile.max_tokens,
            )
            or existing_profile.max_tokens
        )

        temperature = parse_float(
            f"Temperature [{existing_profile.temperature}]: ",
            existing_profile.temperature,
        )

        context_window = parse_int(
            f"Context window tokens [{existing_profile.context_window or 'unset'}]: ",
            existing_profile.context_window,
        )

        updated_profile = ModelProfile(
            provider=provider,
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
            auth_token=auth_token,
        )

        try:
            add_model_profile(
                profile_name,
                updated_profile,
                overwrite=True,
                set_as_main=False,
            )
        except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
            console.print(f"[red]Failed to update model: {escape(str(exc))}[/red]")
            logger.warning(
                "[models_cmd] Failed to update model profile: %s: %s",
                type(exc).__name__,
                exc,
                extra={"profile": profile_name, "session_id": getattr(ui, "session_id", None)},
            )
            return True

        console.print(f"[green]✓ Model '{escape(profile_name)}' updated[/green]")
        return True

    if subcmd in ("delete", "del", "remove"):
        target = tokens[1] if len(tokens) > 1 else console.input("Model to delete: ").strip()
        if not target:
            console.print("[red]Model name is required.[/red]")
            print_models_usage()
            return True
        try:
            delete_model_profile(target)
            console.print(f"[green]✓ Deleted model '{escape(target)}'[/green]")
        except KeyError as exc:
            console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        except (OSError, IOError, PermissionError) as exc:
            console.print(f"[red]Failed to delete model: {escape(str(exc))}[/red]")
            print_models_usage()
            logger.warning(
                "[models_cmd] Failed to delete model profile: %s: %s",
                type(exc).__name__,
                exc,
                extra={"profile": target, "session_id": getattr(ui, "session_id", None)},
            )
        return True

    if subcmd in ("use", "main", "set-main"):
        # Support both "/models use <profile>" and "/models use <pointer> <profile>"
        valid_pointers = {"main", "task", "reasoning", "quick"}

        if len(tokens) >= 3:
            # /models use <pointer> <profile>
            pointer = tokens[1].lower()
            target = tokens[2]
            if pointer not in valid_pointers:
                console.print(
                    f"[red]Invalid pointer '{escape(pointer)}'. Valid pointers: {', '.join(valid_pointers)}[/red]"
                )
                print_models_usage()
                return True
        elif len(tokens) >= 2:
            # Check if second token is a pointer or a profile
            if tokens[1].lower() in valid_pointers:
                pointer = tokens[1].lower()
                target = console.input(f"Model to use for '{pointer}': ").strip()
            else:
                # /models use <profile> (defaults to main)
                pointer = "main"
                target = tokens[1]
        else:
            pointer = (
                console.input("Pointer (main/task/reasoning/quick) [main]: ").strip().lower()
                or "main"
            )
            if pointer not in valid_pointers:
                console.print(
                    f"[red]Invalid pointer '{escape(pointer)}'. Valid pointers: {', '.join(valid_pointers)}[/red]"
                )
                return True
            target = console.input(f"Model to use for '{pointer}': ").strip()

        if not target:
            console.print("[red]Model name is required.[/red]")
            print_models_usage()
            return True
        try:
            set_model_pointer(pointer, target)
            console.print(f"[green]✓ Pointer '{escape(pointer)}' set to '{escape(target)}'[/green]")
        except (ValueError, KeyError, OSError, IOError, PermissionError) as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            print_models_usage()
            logger.warning(
                "[models_cmd] Failed to set model pointer: %s: %s",
                type(exc).__name__,
                exc,
                extra={
                    "pointer": pointer,
                    "profile": target,
                    "session_id": getattr(ui, "session_id", None),
                },
            )
        return True

    print_models_usage()
    pointer_map = config.model_pointers.model_dump()
    if not config.model_profiles:
        console.print("  • No models configured")
        return True

    console.print("\n[bold]Configured Models:[/bold]")
    for name, profile in config.model_profiles.items():
        markers = [ptr for ptr, value in pointer_map.items() if value == name]
        marker_text = f" ({', '.join(markers)})" if markers else ""
        console.print(f"  • {escape(name)}{marker_text}", markup=False)
        console.print(f"      protocol: {profile.provider.value}", markup=False)
        console.print(f"      model: {profile.model}", markup=False)
        if profile.api_base:
            console.print(f"      api_base: {profile.api_base}", markup=False)
        if profile.context_window:
            console.print(f"      context: {profile.context_window} tokens", markup=False)
        console.print(
            f"      max_tokens: {profile.max_tokens}, temperature: {profile.temperature}",
            markup=False,
        )
        console.print(f"      api_key: {'***' if profile.api_key else 'Not set'}", markup=False)
        if profile.provider == ProviderType.ANTHROPIC:
            console.print(
                f"      auth_token: {'***' if getattr(profile, 'auth_token', None) else 'Not set'}",
                markup=False,
            )
        if profile.openai_tool_mode:
            console.print(f"      openai_tool_mode: {profile.openai_tool_mode}", markup=False)
    pointer_labels = ", ".join(f"{p}->{v or '-'}" for p, v in pointer_map.items())
    console.print(f"[dim]Pointers: {escape(pointer_labels)}[/dim]")
    return True


command = SlashCommand(
    name="models",
    description="Manage models: list/create/delete/use",
    handler=_handle,
)


__all__ = ["command"]
