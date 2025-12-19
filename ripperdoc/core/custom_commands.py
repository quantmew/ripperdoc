"""Custom Command loading and execution for Ripperdoc.

Custom commands are defined in .md files under:
- `~/.ripperdoc/commands/` (user-level commands)
- `.ripperdoc/commands/` (project-level commands)

Features:
- Frontmatter support (YAML) for metadata: allowed-tools, description, argument-hint, model, thinking-mode
- Parameter substitution: $ARGUMENTS, $1, $2, etc.
- File references: @filename (resolved relative to project root)
- Bash command execution: !`command` syntax
- Nested commands via subdirectories (e.g., git/commit.md -> /git:commit)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from ripperdoc.utils.log import get_logger

logger = get_logger()

COMMAND_DIR_NAME = "commands"
COMMAND_FILE_SUFFIX = ".md"
_COMMAND_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")

# Pattern for bash command execution: !`command`
_BASH_COMMAND_PATTERN = re.compile(r"!\`([^`]+)\`")

# Pattern for file references: @filename
_FILE_REFERENCE_PATTERN = re.compile(r"@([^\s\]]+)")

# Pattern for positional arguments: $1, $2, etc.
_POSITIONAL_ARG_PATTERN = re.compile(r"\$(\d+)")

# Pattern for all arguments: $ARGUMENTS
_ALL_ARGS_PATTERN = re.compile(r"\$ARGUMENTS", re.IGNORECASE)

# Thinking mode keywords
THINKING_MODES = ["think", "think hard", "think harder", "ultrathink"]


class CommandLocation(str, Enum):
    """Where a custom command is sourced from."""

    USER = "user"
    PROJECT = "project"
    OTHER = "other"


@dataclass
class CustomCommandDefinition:
    """Parsed representation of a custom command."""

    name: str
    description: str
    content: str
    path: Path
    base_dir: Path
    location: CommandLocation
    allowed_tools: List[str] = field(default_factory=list)
    argument_hint: Optional[str] = None
    model: Optional[str] = None
    thinking_mode: Optional[str] = None


@dataclass
class CustomCommandLoadError:
    """Error encountered while loading a custom command file."""

    path: Path
    reason: str


@dataclass
class CustomCommandLoadResult:
    """Aggregated result of loading custom commands."""

    commands: List[CustomCommandDefinition]
    errors: List[CustomCommandLoadError]


def _split_frontmatter(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and body content from a markdown file."""
    lines = raw_text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                frontmatter_text = "\n".join(lines[1:idx])
                body = "\n".join(lines[idx + 1 :])
                try:
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                except (yaml.YAMLError, ValueError, TypeError) as exc:
                    logger.warning(
                        "[custom_commands] Invalid frontmatter: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    return {"__error__": f"Invalid frontmatter: {exc}"}, body
                return frontmatter, body
    return {}, raw_text


def _normalize_allowed_tools(value: object) -> List[str]:
    """Normalize allowed-tools values to a clean list of tool names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        tools: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                tools.append(item.strip())
        return tools
    return []


def _extract_thinking_mode(content: str) -> Optional[str]:
    """Extract thinking mode from content if present."""
    content_lower = content.lower()
    for mode in reversed(THINKING_MODES):  # Check longer modes first
        if mode in content_lower:
            return mode
    return None


def _derive_command_name(path: Path, base_commands_dir: Path) -> str:
    """Derive command name from file path, supporting nested commands.

    Examples:
        commands/project-info.md -> project-info
        commands/git/commit.md -> git:commit
        commands/git/git-commit.md -> git:git-commit
    """
    relative = path.relative_to(base_commands_dir)
    parts = list(relative.parts)

    # Remove the .md suffix from the last part
    if parts:
        parts[-1] = parts[-1].removesuffix(COMMAND_FILE_SUFFIX)

    # Join with : for nested commands
    if len(parts) > 1:
        return ":".join(parts)
    return parts[0] if parts else path.stem


def _load_command_file(
    path: Path, location: CommandLocation, base_commands_dir: Path
) -> Tuple[Optional[CustomCommandDefinition], Optional[CustomCommandLoadError]]:
    """Parse a single command .md file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, IOError, UnicodeDecodeError) as exc:
        logger.warning(
            "[custom_commands] Failed to read command file: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return None, CustomCommandLoadError(path=path, reason=f"Failed to read file: {exc}")

    frontmatter, body = _split_frontmatter(text)
    if "__error__" in frontmatter:
        return None, CustomCommandLoadError(path=path, reason=str(frontmatter["__error__"]))

    # Derive command name from path (supports nested commands)
    command_name = _derive_command_name(path, base_commands_dir)

    # Get description from frontmatter or use first line of content
    raw_description = frontmatter.get("description")
    if not isinstance(raw_description, str) or not raw_description.strip():
        # Use first line of content as description (truncated)
        first_line = body.strip().split("\n")[0] if body.strip() else ""
        raw_description = first_line[:60] + "..." if len(first_line) > 60 else first_line
        if not raw_description:
            raw_description = f"Custom command: {command_name}"

    allowed_tools = _normalize_allowed_tools(
        frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    )

    argument_hint = frontmatter.get("argument-hint") or frontmatter.get("argument_hint")
    if argument_hint and not isinstance(argument_hint, str):
        argument_hint = str(argument_hint)

    model_value = frontmatter.get("model")
    model = model_value.strip() if isinstance(model_value, str) and model_value.strip() else None

    # Get thinking mode from frontmatter or content
    thinking_mode = frontmatter.get("thinking-mode") or frontmatter.get("thinking_mode")
    if not thinking_mode:
        thinking_mode = _extract_thinking_mode(body)

    command = CustomCommandDefinition(
        name=command_name,
        description=raw_description.strip(),
        content=body.strip(),
        path=path,
        base_dir=path.parent,
        location=location,
        allowed_tools=allowed_tools,
        argument_hint=argument_hint.strip() if argument_hint else None,
        model=model,
        thinking_mode=thinking_mode.strip() if isinstance(thinking_mode, str) else None,
    )
    return command, None


def _load_commands_from_dir(
    commands_dir: Path, location: CommandLocation
) -> Tuple[List[CustomCommandDefinition], List[CustomCommandLoadError]]:
    """Load all custom commands from a directory, including nested subdirectories."""
    commands: List[CustomCommandDefinition] = []
    errors: List[CustomCommandLoadError] = []

    if not commands_dir.exists() or not commands_dir.is_dir():
        return commands, errors

    # Recursively find all .md files
    try:
        for md_file in commands_dir.rglob(f"*{COMMAND_FILE_SUFFIX}"):
            if not md_file.is_file():
                continue

            command, error = _load_command_file(md_file, location, commands_dir)
            if command:
                commands.append(command)
            elif error:
                errors.append(error)
    except OSError as exc:
        logger.warning(
            "[custom_commands] Failed to scan command directory: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(commands_dir)},
        )

    return commands, errors


def command_directories(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> List[Tuple[Path, CommandLocation]]:
    """Return the standard command directories for user and project scopes."""
    home_dir = (home or Path.home()).expanduser()
    project_dir = (project_path or Path.cwd()).resolve()
    return [
        (home_dir / ".ripperdoc" / COMMAND_DIR_NAME, CommandLocation.USER),
        (project_dir / ".ripperdoc" / COMMAND_DIR_NAME, CommandLocation.PROJECT),
    ]


def load_all_custom_commands(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> CustomCommandLoadResult:
    """Load custom commands from user and project directories.

    Project commands override user commands with the same name.
    """
    commands_by_name: Dict[str, CustomCommandDefinition] = {}
    errors: List[CustomCommandLoadError] = []

    # Load user commands first so project commands take precedence
    for directory, location in command_directories(project_path=project_path, home=home):
        loaded, dir_errors = _load_commands_from_dir(directory, location)
        errors.extend(dir_errors)
        for cmd in loaded:
            if cmd.name in commands_by_name:
                logger.debug(
                    "[custom_commands] Overriding command",
                    extra={
                        "command_name": cmd.name,
                        "previous_location": str(commands_by_name[cmd.name].location),
                        "new_location": str(location),
                    },
                )
            commands_by_name[cmd.name] = cmd

    return CustomCommandLoadResult(commands=list(commands_by_name.values()), errors=errors)


def find_custom_command(
    command_name: str, project_path: Optional[Path] = None, home: Optional[Path] = None
) -> Optional[CustomCommandDefinition]:
    """Find a custom command by name (case-sensitive match)."""
    normalized = command_name.strip().lstrip("/")
    if not normalized:
        return None
    result = load_all_custom_commands(project_path=project_path, home=home)
    return next((cmd for cmd in result.commands if cmd.name == normalized), None)


def _execute_bash_command(command: str, cwd: Optional[Path] = None) -> str:
    """Execute a bash command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output = f"{output}\n{result.stderr}".strip()
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "(command timed out)"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(error: {exc})"


def _resolve_file_reference(filename: str, project_path: Path) -> str:
    """Resolve a file reference and return its content."""
    try:
        file_path = project_path / filename
        if file_path.exists() and file_path.is_file():
            content = file_path.read_text(encoding="utf-8")
            return content.strip()
        return f"(file not found: {filename})"
    except (OSError, IOError) as exc:
        return f"(error reading {filename}: {exc})"


def expand_command_content(
    command: CustomCommandDefinition,
    arguments: str,
    project_path: Optional[Path] = None,
) -> str:
    """Expand a custom command's content with arguments, bash commands, and file references.

    Supports:
    - $ARGUMENTS: All arguments as a single string
    - $1, $2, etc.: Positional arguments
    - !`command`: Execute bash command and include output
    - @filename: Include file content
    """
    project_dir = (project_path or Path.cwd()).resolve()
    content = command.content

    # Split arguments for positional access
    arg_parts = arguments.split() if arguments else []

    # Replace $ARGUMENTS with all arguments
    content = _ALL_ARGS_PATTERN.sub(arguments, content)

    # Replace positional arguments $1, $2, etc.
    def replace_positional(match: re.Match) -> str:
        idx = int(match.group(1)) - 1  # $1 -> index 0
        if 0 <= idx < len(arg_parts):
            return arg_parts[idx]
        return ""

    content = _POSITIONAL_ARG_PATTERN.sub(replace_positional, content)

    # Execute bash commands: !`command`
    def replace_bash(match: re.Match) -> str:
        bash_cmd = match.group(1)
        return _execute_bash_command(bash_cmd, cwd=project_dir)

    content = _BASH_COMMAND_PATTERN.sub(replace_bash, content)

    # Resolve file references: @filename
    def replace_file_ref(match: re.Match) -> str:
        filename = match.group(1)
        return _resolve_file_reference(filename, project_dir)

    content = _FILE_REFERENCE_PATTERN.sub(replace_file_ref, content)

    return content.strip()


def build_custom_command_summary(commands: Sequence[CustomCommandDefinition]) -> str:
    """Render a concise instruction block listing available custom commands."""
    if not commands:
        return ""

    lines = [
        "# Custom Commands",
        "The following custom commands are available:",
    ]

    for cmd in sorted(commands, key=lambda c: c.name):
        location = f" ({cmd.location.value})" if cmd.location else ""
        hint = f" {cmd.argument_hint}" if cmd.argument_hint else ""
        lines.append(f"- /{cmd.name}{hint}{location}: {cmd.description}")

    return "\n".join(lines)


__all__ = [
    "CustomCommandDefinition",
    "CustomCommandLoadError",
    "CustomCommandLoadResult",
    "CommandLocation",
    "COMMAND_DIR_NAME",
    "load_all_custom_commands",
    "find_custom_command",
    "expand_command_content",
    "build_custom_command_summary",
    "command_directories",
]
