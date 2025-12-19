from rich.markup import escape

from ripperdoc.core.agents import (
    AGENT_DIR_NAME,
    AgentLocation,
    delete_agent_definition,
    load_agent_definitions,
    save_agent_definition,
)
from ripperdoc.core.config import get_global_config
from ripperdoc.utils.log import get_logger

from typing import Any
from .base import SlashCommand

logger = get_logger()


def _handle(ui: Any, trimmed_arg: str) -> bool:
    console = ui.console
    tokens = trimmed_arg.split()
    subcmd = tokens[0].lower() if tokens else ""
    logger.info(
        "[agents_cmd] Handling /agents command",
        extra={
            "subcommand": subcmd or "list",
            "session_id": getattr(ui, "session_id", None),
        },
    )

    def print_agents_usage() -> None:
        console.print("[bold]/agents[/bold] — list configured agents")
        console.print(
            "[bold]/agents create <name> [location] [model][/bold] — "
            "create agent (location: user|project, default user)"
        )
        console.print("[bold]/agents edit <name> [location][/bold] — edit an existing agent")
        console.print(
            "[bold]/agents delete <name> [location][/bold] — "
            "delete agent (location: user|project, default user)"
        )
        console.print(
            f"[dim]Agent files live in ~/.ripperdoc/{AGENT_DIR_NAME} "
            f"or ./.ripperdoc/{AGENT_DIR_NAME}[/dim]"
        )
        console.print(
            "[dim]Model can be a profile name or pointer (task/main/etc). Defaults to 'task'.[/dim]"
        )

    if subcmd in ("help", "-h", "--help"):
        print_agents_usage()
        return True

    if subcmd in ("create", "add"):
        agent_name = tokens[1] if len(tokens) > 1 else console.input("Agent name: ").strip()
        if not agent_name:
            console.print("[red]Agent name is required.[/red]")
            print_agents_usage()
            return True

        description = console.input("Description (when to use this agent): ").strip()
        if not description:
            console.print("[red]Description is required.[/red]")
            return True

        tools_input = console.input("Tools (comma-separated, * for all) [*]: ").strip() or "*"
        tools = [t.strip() for t in tools_input.split(",") if t.strip()] or ["*"]

        system_prompt = console.input("System prompt (single line, use \\n for newlines): ").strip()
        if not system_prompt:
            console.print("[red]System prompt is required.[/red]")
            print_agents_usage()
            return True

        location_arg = tokens[2] if len(tokens) > 2 else ""
        model_arg = tokens[3] if len(tokens) > 3 else ""
        if location_arg and location_arg.lower() not in ("user", "project"):
            model_arg, location_arg = location_arg, ""

        location_raw = (
            location_arg or console.input("Location [user/project, default user]: ").strip()
        ).lower()
        location = AgentLocation.PROJECT if location_raw == "project" else AgentLocation.USER

        config = get_global_config()
        pointer_map = config.model_pointers.model_dump()
        default_model_value = model_arg or pointer_map.get("task", "task")
        model_input = (
            console.input(f"Model profile or pointer [{default_model_value}]: ").strip()
            or default_model_value
        )
        if (
            model_input
            and model_input not in config.model_profiles
            and model_input not in pointer_map
        ):
            console.print(
                "[yellow]Model not found in profiles or pointers; "
                "will fall back to main if unavailable.[/yellow]"
            )

        try:
            path = save_agent_definition(
                agent_type=agent_name,
                description=description,
                tools=tools,
                system_prompt=system_prompt,
                location=location,
                model=model_input,
            )
            console.print(
                f"[green]✓ Agent '{escape(agent_name)}' created at {escape(str(path))}[/green]"
            )
        except (OSError, IOError, PermissionError, ValueError) as exc:
            console.print(f"[red]Failed to create agent: {escape(str(exc))}[/red]")
            print_agents_usage()
            logger.warning(
                "[agents_cmd] Failed to create agent: %s: %s",
                type(exc).__name__,
                exc,
                extra={"agent": agent_name, "session_id": getattr(ui, "session_id", None)},
            )
        return True

    if subcmd in ("delete", "del", "remove"):
        agent_name = (
            tokens[1] if len(tokens) > 1 else console.input("Agent name to delete: ").strip()
        )
        if not agent_name:
            console.print("[red]Agent name is required.[/red]")
            print_agents_usage()
            return True

        location_raw = (
            tokens[2]
            if len(tokens) > 2
            else console.input("Location to delete from [user/project, default user]: ").strip()
        ).lower()
        location = AgentLocation.PROJECT if location_raw == "project" else AgentLocation.USER
        try:
            path = delete_agent_definition(agent_name, location)
            console.print(
                f"[green]✓ Deleted agent '{escape(agent_name)}' at {escape(str(path))}[/green]"
            )
        except FileNotFoundError as exc:
            console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        except (OSError, IOError, PermissionError, ValueError) as exc:
            console.print(f"[red]Failed to delete agent: {escape(str(exc))}[/red]")
            print_agents_usage()
            logger.warning(
                "[agents_cmd] Failed to delete agent: %s: %s",
                type(exc).__name__,
                exc,
                extra={"agent": agent_name, "session_id": getattr(ui, "session_id", None)},
            )
        return True

    if subcmd in ("edit", "update"):
        agent_name = tokens[1] if len(tokens) > 1 else console.input("Agent to edit: ").strip()
        if not agent_name:
            console.print("[red]Agent name is required.[/red]")
            print_agents_usage()
            return True

        agents = load_agent_definitions()
        target_agent = next((a for a in agents.active_agents if a.agent_type == agent_name), None)
        if not target_agent:
            console.print(f"[red]Agent '{escape(agent_name)}' not found.[/red]")
            print_agents_usage()
            return True
        if target_agent.location == AgentLocation.BUILT_IN:
            console.print("[yellow]Built-in agents cannot be edited.[/yellow]")
            return True

        default_location = target_agent.location
        location_raw = (
            tokens[2]
            if len(tokens) > 2
            else console.input(
                f"Location to save [user/project, default {default_location.value}]: "
            ).strip()
        ).lower()
        location = AgentLocation.PROJECT if location_raw == "project" else AgentLocation.USER

        description = (
            console.input(f"Description (when to use) [{target_agent.when_to_use}]: ").strip()
            or target_agent.when_to_use
        )

        tools_default = "*" if "*" in target_agent.tools else ", ".join(target_agent.tools)
        tools_input = (
            console.input(f"Tools (comma-separated, * for all) [{tools_default}]: ").strip()
            or tools_default
        )
        tools = [t.strip() for t in tools_input.split(",") if t.strip()] or ["*"]

        console.print("[dim]Current system prompt:[/dim]")
        console.print(escape(target_agent.system_prompt or "(empty)"), markup=False)
        system_prompt = (
            console.input(
                "System prompt (single line, use \\n for newlines) [Enter to keep current]: "
            ).strip()
            or target_agent.system_prompt
        )

        config = get_global_config()
        pointer_map = config.model_pointers.model_dump()
        model_default = target_agent.model or pointer_map.get("task", "task")
        model_input = (
            console.input(f"Model profile or pointer [{model_default}]: ").strip() or model_default
        )

        try:
            path = save_agent_definition(
                agent_type=agent_name,
                description=description,
                tools=tools,
                system_prompt=system_prompt,
                location=location,
                model=model_input,
                overwrite=True,
            )
            console.print(
                f"[green]✓ Agent '{escape(agent_name)}' updated at {escape(str(path))}[/green]"
            )
        except (OSError, IOError, PermissionError, ValueError) as exc:
            console.print(f"[red]Failed to update agent: {escape(str(exc))}[/red]")
            print_agents_usage()
            logger.warning(
                "[agents_cmd] Failed to update agent: %s: %s",
                type(exc).__name__,
                exc,
                extra={"agent": agent_name, "session_id": getattr(ui, "session_id", None)},
            )
        return True

    agents = load_agent_definitions()
    console.print("\n[bold]Agents:[/bold]")
    print_agents_usage()
    if not agents.active_agents:
        console.print("  • None configured")
    for agent in agents.active_agents:
        location = getattr(agent.location, "value", agent.location)
        tools_str = "all tools" if "*" in agent.tools else ", ".join(agent.tools)
        console.print(f"  • {escape(agent.agent_type)} ({escape(str(location))})", markup=False)
        console.print(f"      {escape(agent.when_to_use)}", markup=False)
        console.print(f"      tools: {escape(tools_str)}", markup=False)
        console.print(f"      model: {escape(agent.model or 'task (default)')}", markup=False)
    if agents.failed_files:
        console.print("[yellow]Some agent files could not be loaded:[/yellow]")
        for path, error in agents.failed_files:
            console.print(f"  - {escape(str(path))}: {escape(str(error))}", markup=False)
    console.print(
        f"[dim]Add agents in ~/.ripperdoc/{AGENT_DIR_NAME} or ./.ripperdoc/{AGENT_DIR_NAME}[/dim]"
    )
    return True


command = SlashCommand(
    name="agents",
    description="Manage subagents: list/create/delete",
    handler=_handle,
)


__all__ = ["command"]
