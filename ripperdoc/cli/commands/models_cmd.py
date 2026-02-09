import sys
import textwrap
from typing import Any, Callable, Optional

from rich import box
from rich.layout import Layout
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.config import (
    ModelProfile,
    ProtocolType,
    add_model_profile,
    delete_model_profile,
    get_global_config,
    model_supports_vision,
    set_model_pointer,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.prompt import prompt_secret

from .base import SlashCommand

logger = get_logger()

_KNOWN_THINKING_MODES = {
    "deepseek",
    "openrouter",
    "qwen",
    "gemini_openai",
    "openai",
    "off",
    "disabled",
}


def _resolve_thinking_mode_input(
    *,
    raw_input: str,
    current_value: Optional[str],
    allow_clear: bool,
) -> Optional[str]:
    """Parse a thinking-mode override from interactive input.

    Rules:
    - Empty input keeps the current value.
    - "auto" always means no explicit override (None).
    - When allow_clear=True, "-", "clear", and "none" also clear the override.
    - Other values are normalized to lowercase for consistency.
    """
    value = (raw_input or "").strip()
    if not value:
        return current_value

    lowered = value.lower()
    if lowered == "auto":
        return None
    if allow_clear and lowered in {"-", "clear", "none"}:
        return None
    return lowered


def _prompt_thinking_mode_add(console: Any, default_value: Optional[str]) -> Optional[str]:
    default_display = default_value or "auto"
    raw = console.input(
        "Thinking mode override "
        f"[{default_display}] (auto/deepseek/openrouter/qwen/gemini_openai/openai/off): "
    )
    thinking_mode = _resolve_thinking_mode_input(
        raw_input=raw,
        current_value=default_value,
        allow_clear=False,
    )
    if thinking_mode and thinking_mode not in _KNOWN_THINKING_MODES:
        console.print(
            f"[yellow]Using custom thinking_mode '{escape(thinking_mode)}'.[/yellow]"
        )
    return thinking_mode


def _prompt_thinking_mode_edit(console: Any, current_value: Optional[str]) -> Optional[str]:
    default_display = current_value or "auto"
    raw = console.input(
        "Thinking mode override "
        f"[{default_display}] (Enter=keep, auto=unset, '-'=clear): "
    )
    thinking_mode = _resolve_thinking_mode_input(
        raw_input=raw,
        current_value=current_value,
        allow_clear=True,
    )
    if thinking_mode and thinking_mode not in _KNOWN_THINKING_MODES:
        console.print(
            f"[yellow]Using custom thinking_mode '{escape(thinking_mode)}'.[/yellow]"
        )
    return thinking_mode


def _parse_int(console: Any, prompt_text: str, default_value: Optional[int]) -> Optional[int]:
    raw = console.input(prompt_text).strip()
    if not raw:
        return default_value
    try:
        return int(raw)
    except ValueError:
        console.print("[yellow]Invalid number, keeping previous value.[/yellow]")
        return default_value


def _parse_float(console: Any, prompt_text: str, default_value: float) -> float:
    raw = console.input(prompt_text).strip()
    if not raw:
        return default_value
    try:
        return float(raw)
    except ValueError:
        console.print("[yellow]Invalid number, keeping previous value.[/yellow]")
        return default_value


def _prompt_protocol(console: Any, default_protocol: str) -> Optional[ProtocolType]:
    protocol_input = (
        console.input(
            f"Protocol ({', '.join(p.value for p in ProtocolType)}) [{default_protocol}]: "
        )
        .strip()
        .lower()
        or default_protocol
    )
    try:
        return ProtocolType(protocol_input)
    except ValueError:
        console.print(f"[red]Invalid protocol: {escape(protocol_input)}[/red]")
        return None


def _prompt_supports_vision_add(console: Any, default_value: Optional[bool]) -> Optional[bool]:
    vision_default_display = (
        "auto" if default_value is None else ("yes" if default_value else "no")
    )
    supports_vision_input = (
        console.input(f"Supports vision (images)? [{vision_default_display}] (Y/n/auto): ")
        .strip()
        .lower()
    )
    if supports_vision_input in ("y", "yes"):
        return True
    if supports_vision_input in ("n", "no"):
        return False
    if supports_vision_input in ("auto", ""):
        return None
    return default_value


def _prompt_supports_vision_edit(console: Any, current_value: Optional[bool]) -> Optional[bool]:
    vision_default_display = "auto" if current_value is None else ("yes" if current_value else "no")
    supports_vision_input = (
        console.input(
            f"Supports vision (images)? [{vision_default_display}] (Y/n/auto/C=clear): "
        )
        .strip()
        .lower()
    )
    if supports_vision_input in ("y", "yes"):
        return True
    if supports_vision_input in ("n", "no"):
        return False
    if supports_vision_input in ("c", "clear", "-"):
        return None
    if supports_vision_input in ("auto", ""):
        return current_value
    return current_value


def _collect_add_profile_input(
    console: Any,
    config: Any,
    existing_profile: Optional[ModelProfile],
    current_profile: Optional[ModelProfile],
) -> tuple[Optional[ModelProfile], bool]:
    default_protocol = (
        (current_profile.protocol.value) if current_profile else ProtocolType.ANTHROPIC.value
    )
    protocol = _prompt_protocol(console, default_protocol)
    if protocol is None:
        return None, False

    default_model = (
        existing_profile.model
        if existing_profile
        else (current_profile.model if current_profile else "")
    )
    model_prompt = f"Model name to send{f' [{default_model}]' if default_model else ''}: "
    model_name = console.input(model_prompt).strip() or default_model
    if not model_name:
        console.print("[red]Model name is required.[/red]")
        return None, False

    inferred_profile = ModelProfile(protocol=protocol, model=model_name)

    api_key_input = prompt_secret("API key (leave blank to keep unset)").strip()
    api_key = api_key_input or (existing_profile.api_key if existing_profile else None)

    auth_token = existing_profile.auth_token if existing_profile else None
    if protocol == ProtocolType.ANTHROPIC:
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

    max_input_default = (
        existing_profile.max_input_tokens if existing_profile else inferred_profile.max_input_tokens
    )
    max_input_label = str(max_input_default) if max_input_default is not None else "auto"
    max_input_tokens = _parse_int(
        console,
        f"Max input tokens [{max_input_label}]: ",
        max_input_default,
    )

    max_output_default = (
        existing_profile.max_output_tokens if existing_profile else inferred_profile.max_output_tokens
    )
    max_output_label = str(max_output_default) if max_output_default is not None else "auto"
    max_output_tokens = _parse_int(
        console,
        f"Max output tokens [{max_output_label}]: ",
        max_output_default,
    )

    max_tokens_default = (
        existing_profile.max_tokens if existing_profile else inferred_profile.max_tokens
    )
    max_tokens_input = _parse_int(
        console,
        f"Max tokens [{max_tokens_default}]: ",
        max_tokens_default,
    )
    max_tokens = max_tokens_input if max_tokens_input is not None else max_tokens_default
    if max_output_tokens is not None and max_tokens > max_output_tokens:
        console.print(
            f"[yellow]Max tokens {max_tokens} exceeds max_output_tokens {max_output_tokens}; "
            "clamping to max_output_tokens.[/yellow]"
        )
        max_tokens = max_output_tokens

    temp_default = existing_profile.temperature if existing_profile else 1.0
    temperature = _parse_float(
        console,
        f"Temperature [{temp_default}]: ",
        temp_default,
    )
    thinking_mode_default = (
        existing_profile.thinking_mode if existing_profile else inferred_profile.thinking_mode
    )
    thinking_mode = _prompt_thinking_mode_add(console, thinking_mode_default)

    supports_vision_default = (
        existing_profile.supports_vision if existing_profile else inferred_profile.supports_vision
    )
    supports_vision = _prompt_supports_vision_add(console, supports_vision_default)

    currency_default = (
        existing_profile.currency if existing_profile else inferred_profile.currency
    ) or "USD"
    currency = (
        console.input(f"Currency [{currency_default}]: ").strip().upper() or currency_default
    )
    input_price_default = (
        existing_profile.price.input if existing_profile else inferred_profile.price.input
    )
    output_price_default = (
        existing_profile.price.output if existing_profile else inferred_profile.price.output
    )
    input_price = _parse_float(
        console,
        f"Input price per 1M tokens [{input_price_default}]: ",
        input_price_default,
    )
    output_price = _parse_float(
        console,
        f"Output price per 1M tokens [{output_price_default}]: ",
        output_price_default,
    )

    default_set_main = (
        not config.model_profiles
        or getattr(config.model_pointers, "main", "") not in config.model_profiles
    )
    set_main_input = (
        console.input(f"Set as main model? ({'Y' if default_set_main else 'y'}/N): ")
        .strip()
        .lower()
    )
    set_as_main = set_main_input in ("y", "yes") if set_main_input else default_set_main

    profile = ModelProfile(
        protocol=protocol,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_mode=thinking_mode,
        auth_token=auth_token,
        supports_vision=supports_vision,
        price={"input": input_price, "output": output_price},
        currency=currency,
    )

    return profile, set_as_main


def _collect_edit_profile_input(
    console: Any,
    existing_profile: ModelProfile,
) -> Optional[ModelProfile]:
    protocol_default = existing_profile.protocol.value
    protocol = _prompt_protocol(console, protocol_default)
    if protocol is None:
        return None

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
    if protocol == ProtocolType.ANTHROPIC or existing_profile.protocol == ProtocolType.ANTHROPIC:
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

    max_input_label = (
        str(existing_profile.max_input_tokens)
        if existing_profile.max_input_tokens is not None
        else "auto"
    )
    max_input_tokens = _parse_int(
        console,
        f"Max input tokens [{max_input_label}]: ",
        existing_profile.max_input_tokens,
    )

    max_output_label = (
        str(existing_profile.max_output_tokens)
        if existing_profile.max_output_tokens is not None
        else "auto"
    )
    max_output_tokens = _parse_int(
        console,
        f"Max output tokens [{max_output_label}]: ",
        existing_profile.max_output_tokens,
    )

    max_tokens_input = _parse_int(
        console,
        f"Max tokens [{existing_profile.max_tokens}]: ",
        existing_profile.max_tokens,
    )
    max_tokens = (
        max_tokens_input if max_tokens_input is not None else existing_profile.max_tokens
    )
    if max_output_tokens is not None and max_tokens > max_output_tokens:
        console.print(
            f"[yellow]Max tokens {max_tokens} exceeds max_output_tokens {max_output_tokens}; "
            "clamping to max_output_tokens.[/yellow]"
        )
        max_tokens = max_output_tokens

    temperature = _parse_float(
        console,
        f"Temperature [{existing_profile.temperature}]: ",
        existing_profile.temperature,
    )
    thinking_mode = _prompt_thinking_mode_edit(console, existing_profile.thinking_mode)

    supports_vision = _prompt_supports_vision_edit(console, existing_profile.supports_vision)

    currency = (
        console.input(f"Currency [{existing_profile.currency or 'USD'}]: ").strip().upper()
        or existing_profile.currency
        or "USD"
    )
    input_price = _parse_float(
        console,
        f"Input price per 1M tokens [{existing_profile.price.input}]: ",
        existing_profile.price.input,
    )
    output_price = _parse_float(
        console,
        f"Output price per 1M tokens [{existing_profile.price.output}]: ",
        existing_profile.price.output,
    )

    updated_profile = ModelProfile(
        protocol=protocol,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_mode=thinking_mode,
        auth_token=auth_token,
        supports_vision=supports_vision,
        price={"input": input_price, "output": output_price},
        currency=currency,
    )
    return updated_profile


def _pointer_markers(pointer_map: dict[str, str], name: str) -> list[str]:
    return [ptr for ptr, value in pointer_map.items() if value == name]


def _vision_labels(profile: ModelProfile) -> tuple[str, str]:
    if profile.supports_vision is None:
        detected = model_supports_vision(profile)
        return "auto", f"auto (detected {'yes' if detected else 'no'})"
    if profile.supports_vision:
        return "yes", "yes"
    return "no", "no"


def _render_models_plain(console: Any, config: Any) -> None:
    pointer_map = config.model_pointers.model_dump()
    if not config.model_profiles:
        console.print("  • No models configured")
        return

    console.print("\n[bold]Configured Models:[/bold]")
    for name, profile in config.model_profiles.items():
        markers = [ptr for ptr, value in pointer_map.items() if value == name]
        marker_text = f" ({', '.join(markers)})" if markers else ""
        console.print(f"  • {escape(name)}{marker_text}", markup=False)
        console.print(f"      protocol: {profile.protocol.value}", markup=False)
        console.print(f"      model: {profile.model}", markup=False)
        if profile.api_base:
            console.print(f"      api_base: {profile.api_base}", markup=False)
        if profile.max_input_tokens:
            console.print(f"      max_input_tokens: {profile.max_input_tokens}", markup=False)
        if profile.max_output_tokens:
            console.print(f"      max_output_tokens: {profile.max_output_tokens}", markup=False)
        console.print(
            f"      max_tokens: {profile.max_tokens}, temperature: {profile.temperature}",
            markup=False,
        )
        console.print(
            f"      price: in={profile.price.input}/1M, out={profile.price.output}/1M ({profile.currency})",
            markup=False,
        )
        console.print(f"      api_key: {'***' if profile.api_key else 'Not set'}", markup=False)
        if profile.protocol == ProtocolType.ANTHROPIC:
            console.print(
                f"      auth_token: {'***' if getattr(profile, 'auth_token', None) else 'Not set'}",
                markup=False,
            )
        if profile.openai_tool_mode:
            console.print(f"      openai_tool_mode: {profile.openai_tool_mode}", markup=False)
        if profile.thinking_mode:
            console.print(f"      thinking_mode: {profile.thinking_mode}", markup=False)
        if profile.supports_vision is None:
            vision_display = "auto-detect"
        elif profile.supports_vision:
            vision_display = "yes"
        else:
            vision_display = "no"
        console.print(f"      supports_vision: {vision_display}", markup=False)
    pointer_labels = ", ".join(f"{p}->{v or '-'}" for p, v in pointer_map.items())
    console.print(f"[dim]Pointers: {escape(pointer_labels)}[/dim]")


def _render_models_table(console: Any, config: Any) -> None:
    pointer_map = config.model_pointers.model_dump()
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Ptr", style="magenta", no_wrap=True)
    table.add_column("Protocol", style="green", no_wrap=True)
    table.add_column("Model", style="white", overflow="fold")
    table.add_column("In", style="dim", justify="right", no_wrap=True)
    table.add_column("Max", style="dim", justify="right", no_wrap=True)
    table.add_column("Temp", style="dim", justify="right", no_wrap=True)
    table.add_column("Think", style="blue", no_wrap=True)
    table.add_column("Vision", style="yellow", no_wrap=True)
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("API Base", style="dim", overflow="fold", max_width=28)

    for name, profile in config.model_profiles.items():
        markers = _pointer_markers(pointer_map, name)
        pointer_label = ",".join(markers) if markers else "-"
        context_display = str(profile.max_input_tokens) if profile.max_input_tokens else "-"
        vision_display = _vision_labels(profile)[0]
        thinking_display = profile.thinking_mode or "auto"
        api_base = profile.api_base or "-"
        if profile.api_base:
            api_base = textwrap.shorten(profile.api_base, width=28, placeholder="...")
        key_display = "set" if profile.api_key else "-"
        table.add_row(
            escape(name),
            pointer_label,
            profile.protocol.value,
            escape(profile.model),
            context_display,
            str(profile.max_tokens),
            f"{profile.temperature:.2f}",
            thinking_display,
            vision_display,
            key_display,
            escape(api_base),
        )

    title = f"Models ({len(config.model_profiles)})"
    console.print(Panel(table, title=title, box=box.ROUNDED, padding=(1, 2)))
    pointer_labels = ", ".join(f"{p}->{v or '-'}" for p, v in pointer_map.items())
    console.print(f"[dim]Pointers: {escape(pointer_labels)}[/dim]")


def _build_model_details_panel(
    name: str, profile: ModelProfile, pointer_map: dict[str, str]
) -> Panel:
    markers = _pointer_markers(pointer_map, name)
    marker_text = ", ".join(markers) if markers else "-"
    vision_short, vision_detail = _vision_labels(profile)

    details = Table.grid(padding=(0, 2))
    details.add_column(style="cyan", no_wrap=True)
    details.add_column(style="white")
    details.add_row("Profile", escape(name))
    details.add_row("Pointers", escape(marker_text))
    details.add_row("Protocol", escape(profile.protocol.value))
    details.add_row("Model", escape(profile.model))
    details.add_row("API base", escape(profile.api_base or "-"))
    details.add_row(
        "Max input tokens",
        escape(str(profile.max_input_tokens) if profile.max_input_tokens else "auto"),
    )
    details.add_row(
        "Max output tokens",
        escape(str(profile.max_output_tokens) if profile.max_output_tokens else "auto"),
    )
    details.add_row("Max tokens", escape(str(profile.max_tokens)))
    details.add_row("Temperature", escape(str(profile.temperature)))
    details.add_row(
        "Price",
        escape(
            f"in={profile.price.input}/1M, out={profile.price.output}/1M ({profile.currency})"
        ),
    )
    details.add_row("Vision", escape(vision_detail if vision_short == "auto" else vision_short))
    details.add_row("API key", "set" if profile.api_key else "unset")
    if profile.protocol == ProtocolType.ANTHROPIC:
        details.add_row("Auth token", "set" if getattr(profile, "auth_token", None) else "unset")
    if profile.openai_tool_mode:
        details.add_row("OpenAI tool mode", escape(profile.openai_tool_mode))
    if profile.thinking_mode:
        details.add_row("Thinking mode", escape(profile.thinking_mode))

    return Panel(
        details,
        title=f"Model: {escape(name)}",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _render_model_details(
    console: Any, name: str, profile: ModelProfile, pointer_map: dict[str, str]
) -> None:
    console.print(_build_model_details_panel(name, profile, pointer_map))


def _default_selected_model(config: Any, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in config.model_profiles:
        return preferred
    main_pointer = getattr(config.model_pointers, "main", "")
    if main_pointer in config.model_profiles:
        return main_pointer
    if config.model_profiles:
        return str(next(iter(config.model_profiles)))
    return None


def _build_models_list_panel(
    config: Any, selected_name: Optional[str], pointer_map: dict[str, str]
) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("#", width=3, no_wrap=True, style="dim")
    table.add_column("Sel", width=2, no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Ptr", style="magenta", no_wrap=True)
    table.add_column("Think", style="blue", no_wrap=True)
    table.add_column("Model", style="dim")

    for idx, (name, profile) in enumerate(config.model_profiles.items(), start=1):
        selected = name == selected_name
        marker_text = Text(">", style="bold yellow") if selected else Text(" ", style="dim")
        index_text = Text(str(idx), style="dim")
        name_text = Text(name, style="bold cyan" if selected else "cyan")
        markers = _pointer_markers(pointer_map, name)
        pointer_label = ",".join(markers) if markers else "-"
        pointer_text = Text(pointer_label, style="magenta" if markers else "dim")
        thinking_text = Text(profile.thinking_mode or "auto", style="blue")
        model_label = f"{profile.protocol.value} • {profile.model}"
        model_text = Text(model_label, style="dim")
        table.add_row(index_text, marker_text, name_text, pointer_text, thinking_text, model_text)

    if not config.model_profiles:
        table.add_row(
            Text(" ", style="dim"),
            Text(" ", style="dim"),
            Text("No models configured", style="dim"),
            "",
            "",
            "",
        )

    return Panel(table, title="Models", box=box.ROUNDED, padding=(1, 2))


def _build_models_header_panel(config: Any, selected_name: Optional[str]) -> Panel:
    pointer_map = config.model_pointers.model_dump()
    pointer_labels = ", ".join(f"{p}->{v or '-'}" for p, v in pointer_map.items())
    selected_label = selected_name or "-"

    header = Text()
    header.append("Models ", style="bold")
    header.append(str(len(config.model_profiles)), style="cyan")
    header.append("  Selected: ", style="dim")
    header.append(selected_label, style="bold cyan")
    header.append("  Pointers: ", style="dim")
    header.append(pointer_labels, style="magenta")

    return Panel(header, box=box.ROUNDED, padding=(0, 1))


def _build_models_footer_panel() -> Panel:
    footer = Text(
        "↑/↓ move  A add  E edit  D delete  M set main  K set quick  Q exit  R refresh",
        style="dim",
    )
    return Panel(footer, box=box.ROUNDED, padding=(0, 1))


def _render_models_dashboard(console: Any, config: Any, selected_name: Optional[str]) -> None:
    pointer_map = config.model_pointers.model_dump()
    layout = Layout()
    layout.split_column(
        Layout(_build_models_header_panel(config, selected_name), name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(_build_models_footer_panel(), name="footer", size=3),
    )

    list_panel = _build_models_list_panel(config, selected_name, pointer_map)
    if selected_name and selected_name in config.model_profiles:
        details_panel = _build_model_details_panel(
            selected_name, config.model_profiles[selected_name], pointer_map
        )
    else:
        details_panel = Panel("Select a model to view details.", box=box.ROUNDED, padding=(1, 2))

    layout["body"].split_row(
        Layout(list_panel, name="left", ratio=2),
        Layout(details_panel, name="right", ratio=3),
    )

    console.print(layout)


def _confirm_action(console: Any, prompt_text: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = console.input(f"{prompt_text} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _prompt_models_command(console: Any) -> str:
    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.key_binding import KeyBindings
    except (ImportError, OSError, RuntimeError):
        return str(console.input("Command (a/e/d/m/k/q/r or model #/name): ")).strip()

    key_bindings = KeyBindings()

    def _exit_with(value: str) -> Callable[[Any], None]:
        def _handler(event: Any) -> None:  # noqa: ANN001
            event.app.exit(result=value)

        return _handler

    for key in ("a", "A"):
        key_bindings.add(key, eager=True)(_exit_with("a"))
    for key in ("e", "E"):
        key_bindings.add(key, eager=True)(_exit_with("e"))
    for key in ("d", "D"):
        key_bindings.add(key, eager=True)(_exit_with("d"))
    for key in ("m", "M"):
        key_bindings.add(key, eager=True)(_exit_with("m"))
    for key in ("k", "K"):
        key_bindings.add(key, eager=True)(_exit_with("k"))
    for key in ("q", "Q", "escape"):
        key_bindings.add(key, eager=True)(_exit_with("q"))
    for key in ("r", "R"):
        key_bindings.add(key, eager=True)(_exit_with("r"))

    key_bindings.add("up", eager=True)(_exit_with("__up"))
    key_bindings.add("down", eager=True)(_exit_with("__down"))
    key_bindings.add("pageup", eager=True)(_exit_with("__page_up"))
    key_bindings.add("pagedown", eager=True)(_exit_with("__page_down"))

    @key_bindings.add("enter")
    def _enter(event: Any) -> None:  # noqa: ANN001
        text = event.current_buffer.text
        event.app.exit(result=text)

    @key_bindings.add("c-c", eager=True)
    def _ctrl_c(event: Any) -> None:  # noqa: ANN001
        event.app.exit(result="q")

    return pt_prompt("Command: ", key_bindings=key_bindings)


def _resolve_model_selection(
    raw: str, config: Any, selected_name: Optional[str]
) -> Optional[str]:
    if not raw:
        return selected_name
    raw = raw.strip()
    if raw.isdigit():
        idx = int(raw)
        if idx <= 0:
            return selected_name
        names = list(config.model_profiles.keys())
        if 1 <= idx <= len(names):
            return str(names[idx - 1])
        return selected_name
    if raw in config.model_profiles:
        return raw
    # Case-insensitive match
    lower_map = {name.lower(): name for name in config.model_profiles}
    return lower_map.get(raw.lower(), selected_name)


def _handle_models_rich_tui(ui: Any) -> bool:
    console = ui.console
    if not sys.stdin.isatty():
        console.print("[yellow]Interactive UI requires a TTY. Showing plain list instead.[/yellow]")
        _render_models_plain(console, get_global_config())
        return True

    selected_name: Optional[str] = None

    while True:
        config = get_global_config()
        selected_name = _default_selected_model(config, selected_name)
        console.print()
        _render_models_dashboard(console, config, selected_name)

        if not config.model_profiles:
            if _confirm_action(console, "No models configured. Add one now?", default=True):
                profile_name = console.input("Profile name: ").strip()
                if not profile_name:
                    console.print("[red]Model profile name is required.[/red]")
                    continue
                existing_profile = config.model_profiles.get(profile_name)
                if existing_profile:
                    if not _confirm_action(
                        console, f"Profile '{profile_name}' exists. Overwrite?"
                    ):
                        continue
                profile, set_as_main = _collect_add_profile_input(
                    console, config, existing_profile, get_profile_for_pointer("main")
                )
                if not profile:
                    continue
                try:
                    add_model_profile(
                        profile_name,
                        profile,
                        overwrite=bool(existing_profile),
                        set_as_main=set_as_main,
                    )
                except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
                    console.print(f"[red]Failed to save model: {escape(str(exc))}[/red]")
                    continue
                marker = " (main)" if set_as_main else ""
                console.print(f"[green]✓ Model '{escape(profile_name)}' saved{marker}[/green]")
                continue
            return True

        command = _prompt_models_command(console).strip().lower()
        if command in ("__up", "__down", "__page_up", "__page_down"):
            names = list(config.model_profiles.keys())
            if not names:
                continue
            current_index = names.index(selected_name) if selected_name in names else 0
            if command == "__up":
                current_index = max(0, current_index - 1)
            elif command == "__down":
                current_index = min(len(names) - 1, current_index + 1)
            elif command == "__page_up":
                current_index = max(0, current_index - 5)
            elif command == "__page_down":
                current_index = min(len(names) - 1, current_index + 5)
            selected_name = names[current_index]
            continue
        if command in ("", "r", "refresh"):
            continue
        if command in ("q", "quit", "exit"):
            return True
        if command in ("a", "add"):
            profile_name = console.input("Profile name: ").strip()
            if not profile_name:
                console.print("[red]Model profile name is required.[/red]")
                continue
            existing_profile = config.model_profiles.get(profile_name)
            if existing_profile:
                if not _confirm_action(console, f"Profile '{profile_name}' exists. Overwrite?"):
                    continue
            profile, set_as_main = _collect_add_profile_input(
                console, config, existing_profile, get_profile_for_pointer("main")
            )
            if not profile:
                continue
            try:
                add_model_profile(
                    profile_name,
                    profile,
                    overwrite=bool(existing_profile),
                    set_as_main=set_as_main,
                )
            except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
                console.print(f"[red]Failed to save model: {escape(str(exc))}[/red]")
                continue
            marker = " (main)" if set_as_main else ""
            console.print(f"[green]✓ Model '{escape(profile_name)}' saved{marker}[/green]")
            selected_name = profile_name
            continue

        model_name: Optional[str]
        if command in ("e", "edit", "d", "delete", "del", "remove", "m", "main", "k", "quick"):
            if not selected_name:
                console.print("[yellow]No model selected.[/yellow]")
                continue
            model_name = selected_name
        else:
            model_name = _resolve_model_selection(command, config, selected_name)
            if not model_name:
                console.print("[yellow]Unknown model or command.[/yellow]")
                continue
            if model_name != selected_name:
                selected_name = model_name
                continue

        profile = config.model_profiles.get(model_name or "")
        if not profile:
            console.print(f"[yellow]Model '{escape(model_name or '')}' not found.[/yellow]")
            continue

        if command in ("e", "edit"):
            updated_profile = _collect_edit_profile_input(console, profile)
            if not updated_profile:
                continue
            try:
                add_model_profile(
                    model_name,
                    updated_profile,
                    overwrite=True,
                    set_as_main=False,
                )
            except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
                console.print(f"[red]Failed to update model: {escape(str(exc))}[/red]")
                continue
            console.print(f"[green]✓ Model '{escape(model_name)}' updated[/green]")
            continue

        if command in ("m", "main", "k", "quick"):
            pointer = "main" if command in ("m", "main") else "quick"
            try:
                set_model_pointer(pointer, model_name)
                console.print(
                    f"[green]✓ Pointer '{escape(pointer)}' set to '{escape(model_name)}'[/green]"
                )
            except (ValueError, KeyError, OSError, IOError, PermissionError) as exc:
                console.print(f"[red]{escape(str(exc))}[/red]")
            continue

        if command in ("d", "delete", "del", "remove"):
            if not _confirm_action(console, f"Delete model '{model_name}'?"):
                continue
            try:
                delete_model_profile(model_name)
                console.print(f"[green]✓ Deleted model '{escape(model_name)}'[/green]")
                selected_name = None
            except KeyError as exc:
                console.print(f"[yellow]{escape(str(exc))}[/yellow]")
            except (OSError, IOError, PermissionError) as exc:
                console.print(f"[red]Failed to delete model: {escape(str(exc))}[/red]")
            continue

    return True


def _handle_models_tui(ui: Any) -> bool:
    console = ui.console
    if not sys.stdin.isatty():
        console.print("[yellow]Interactive UI requires a TTY. Showing plain list instead.[/yellow]")
        _render_models_plain(console, get_global_config())
        return True

    try:
        from ripperdoc.cli.ui.models_tui import run_models_tui
    except (ImportError, ModuleNotFoundError) as exc:
        console.print(
            f"[yellow]Textual UI not available ({escape(str(exc))}). Falling back to Rich UI.[/yellow]"
        )
        return _handle_models_rich_tui(ui)

    try:
        return bool(run_models_tui())
    except Exception as exc:  # noqa: BLE001 - fail safe in interactive UI
        console.print(f"[red]Textual UI failed: {escape(str(exc))}[/red]")
        return _handle_models_rich_tui(ui)


def _handle(ui: Any, trimmed_arg: str) -> bool:
    console = ui.console
    tokens = trimmed_arg.split()
    subcmd = tokens[0].lower() if tokens else ""
    config = get_global_config()
    logger.info(
        "[models_cmd] Handling /models command",
        extra={"subcommand": subcmd or "tui", "session_id": getattr(ui, "session_id", None)},
    )

    def print_models_usage() -> None:
        console.print("[bold]/models[/bold] — open interactive models UI")
        console.print("[bold]/models tui[/bold] — open interactive models UI")
        console.print("[bold]/models list[/bold] — list configured models (plain)")
        console.print("[bold]/models add <name>[/bold] — add or update a model profile")
        console.print("[bold]/models edit <name>[/bold] — edit an existing model profile")
        console.print("[bold]/models delete <name>[/bold] — delete a model profile")
        console.print("[bold]/models use <name>[/bold] — set the main model pointer")
        console.print(
            "[bold]/models use <pointer> <name>[/bold] — set a specific pointer (main/quick)"
        )

    if subcmd in ("help", "-h", "--help"):
        print_models_usage()
        return True

    if subcmd in ("", "tui", "ui"):
        return _handle_models_tui(ui)

    if subcmd in ("list", "ls"):
        _render_models_plain(console, config)
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

        profile, set_as_main = _collect_add_profile_input(
            console, config, existing_profile, get_profile_for_pointer("main")
        )
        if not profile:
            return True

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
        updated_profile = _collect_edit_profile_input(console, existing_profile)
        if not updated_profile:
            return True

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
        valid_pointers = {"main", "quick"}

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
            pointer = console.input("Pointer (main/quick) [main]: ").strip().lower() or "main"
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
    _render_models_plain(console, config)
    return True


command = SlashCommand(
    name="models",
    description="Manage models: list/create/delete/use",
    handler=_handle,
)


__all__ = ["command"]
