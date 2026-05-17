"""AST-based bash command security analysis using tree-sitter.


FAIL-CLOSED design: any tree-sitter node type not in the allowlist causes the
entire command to be classified as 'too-complex', meaning it goes through the
normal permission prompt flow. We never interpret structure we don't understand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ripperdoc.utils.bash.node import Node, PARSE_ABORTED, parse_command_raw
from ripperdoc.utils.bash.bash_parser import get_bash_parser, SHELL_KEYWORDS
from ripperdoc.utils.bash.shell_quote import try_parse_shell_command
from ripperdoc.utils.bash.tree_sitter_analysis import TreeSitterAnalysis, analyze_command
from ripperdoc.utils.log import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class Redirect:
    """A shell redirection."""
    op: str  # '>' | '>>' | '<' | '<<' | '>&' | '>|' | '<&' | '&>' | '&>>' | '<<<'
    target: str
    fd: Optional[int] = None


@dataclass
class SimpleCommand:
    """A parsed simple command with argv, env vars, and redirects."""
    argv: List[str] = field(default_factory=list)
    env_vars: List[Dict[str, str]] = field(default_factory=list)
    redirects: List[Redirect] = field(default_factory=list)
    text: str = ""


ParseForSecurityResult = Union[
    dict,  # {'kind': 'simple', 'commands': [...]}
    dict,  # {'kind': 'too-complex', 'reason': ..., 'node_type': ...}
    dict,  # {'kind': 'parse-unavailable'}
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Placeholder for $() command substitution output.
CMDSUB_PLACEHOLDER = "__CMDSUB_OUTPUT__"

# Placeholder for tracked variable references.
VAR_PLACEHOLDER = "__TRACKED_VAR__"

# Structural node types that compose commands.
STRUCTURAL_TYPES = frozenset({
    "program",
    "list",
    "pipeline",
    "redirected_statement",
})

# Operator tokens that separate commands.
SEPARATOR_TYPES = frozenset({"&&", "||", "|", ";", "&", "|&", "\n"})

# Known-safe env vars that bash sets automatically.
SAFE_ENV_VARS = frozenset({
    "HOME", "PWD", "OLDPWD", "USER", "LOGNAME", "SHELL", "PATH",
    "HOSTNAME", "UID", "EUID", "PPID", "RANDOM", "SECONDS", "LINENO",
    "TMPDIR", "BASH_VERSION", "BASHPID", "SHLVL", "HISTFILE", "IFS",
})

# Wrapper patterns for stripSafeWrappers equivalent.
# Commands that exec their arguments (safe to strip for permission checking).
SAFE_WRAPPER_COMMANDS = frozenset({
    "timeout", "time", "nice", "stdbuf", "nohup",
})

# stdbuf flag forms
_STDBUF_SHORT_SEP_RE = re.compile(r"^-[ioe]$")
_STDBUF_SHORT_FUSED_RE = re.compile(r"^-[ioe].")
_STDBUF_LONG_RE = re.compile(r"^--(input|output|error)=")

# Bare $VAR in unquoted position undergoes word-splitting and globbing.
_BARE_VAR_UNSAFE_RE = re.compile(r"[ \t\n*?[]")


# ---------------------------------------------------------------------------
# Node type allowlist (fail-closed)
# ---------------------------------------------------------------------------

# All node types we understand and can extract argv from.
ALLOWED_NODE_TYPES = STRUCTURAL_TYPES | SEPARATOR_TYPES | frozenset({
    # Command and word nodes
    "command",
    "command_name",
    "simple_command",
    "word",
    "string",
    "string_content",
    "concatenation",
    "variable_name",
    "expansion",
    "simple_expansion",
    "command_substitution",
    "process_substitution",
    "file_redirect",
    "file_descriptor",
    # Redirect operator tokens
    ">", ">>", "<", "<<", ">&", "<&", ">|", "|&",
    "heredoc_redirect",
    "heredoc_body",
    "heredoc_start",
    "case_command",
    "case_item",
    "for_command",
    "while_command",
    "if_command",
    "elif_command",
    "else_command",
    "function_definition",
    "assignment_word",
    "variable_assignment",
    "declaration_command",
    # Arithmetic
    "arithmetic_expansion",
})

# ---------------------------------------------------------------------------
# AST walking and extraction
# ---------------------------------------------------------------------------


def parse_for_security_from_ast(
    command: str,
    parser: Optional[Any] = None,
) -> ParseForSecurityResult:
    """Parse a command using tree-sitter and extract security-relevant structure.

    This is the main entry point for AST-based command analysis.
    Returns a ParseForSecurityResult with 'simple', 'too-complex', or
    'parse-unavailable' kind.

    Args:
        command: The bash command string to analyze.
        parser: Optional pre-configured tree-sitter parser.

    Returns:
        ParseForSecurityResult dict.
    """
    if not command or not command.strip():
        return {"kind": "simple", "commands": []}

    # Get or create parser
    try:
        if parser is None:
            parser = get_bash_parser()
    except ImportError:
        return {"kind": "parse-unavailable"}

    # Parse
    root = parse_command_raw(command, parser)
    if root is PARSE_ABORTED:
        return {"kind": "parse-unavailable"}

    # Walk the tree with fail-closed allowlist
    try:
        commands = _extract_commands(root)
        if commands is None:
            return {"kind": "too-complex", "reason": "Unknown node type encountered"}

        return {"kind": "simple", "commands": commands}
    except Exception as exc:
        logger.debug("[ast] Failed to analyze command: %s", exc)
        return {"kind": "too-complex", "reason": str(exc)}


def _extract_commands(node: Node) -> Optional[List[SimpleCommand]]:
    """Recursively extract simple commands from an AST node.

    Returns None if an unknown node type is encountered (fail-closed).
    """
    if not isinstance(node, Node):
        return None

    node_type = node.type

    # Structural nodes: recurse into children
    if node_type in STRUCTURAL_TYPES:
        all_commands: List[SimpleCommand] = []
        for child in node.children:
            if child.type in SEPARATOR_TYPES:
                continue
            child_commands = _extract_commands(child)
            if child_commands is None:
                return None
            all_commands.extend(child_commands)
        return all_commands

    # Command node: extract the simple command
    if node_type in ("command", "simple_command"):
        cmd = _extract_simple_command(node)
        if cmd is not None:
            return [cmd]
        return None

    # Redirected statement: the child is the command
    if node_type == "redirected_statement":
        return _extract_commands_from_redirected(node)

    # Case/for/while/if: block commands
    if node_type in ("case_command", "for_command", "while_command", "if_command"):
        return _extract_commands_from_compound(node)

    # Function definition: extract the body
    if node_type == "function_definition":
        return _extract_commands_from_function(node)

    # Unknown node type — fail-closed
    if node_type not in ALLOWED_NODE_TYPES:
        logger.debug("[ast] Unknown node type: %s", node_type)
        return None

    # For other allowed types, recurse into children
    all_cmds = []
    for child in node.children:
        child_commands = _extract_commands(child)
        if child_commands is None:
            return None
        all_cmds.extend(child_commands)
    return all_cmds


def _extract_simple_command(node: Node) -> Optional[SimpleCommand]:
    """Extract a SimpleCommand from a command node."""
    cmd = SimpleCommand(text=node.text)
    redirects: List[Redirect] = []
    in_env_vars = True

    for child in node.children:
        child_type = child.type

        if child_type == "command_name":
            # The command name itself (may have a word child)
            cmd_name = _extract_command_name(child)
            if cmd_name:
                cmd.argv.append(cmd_name)
            in_env_vars = False

        elif child_type == "word":
            # Check for env var assignment (VAR=value) at the start
            if in_env_vars and "=" in child.text and not child.text.startswith("-"):
                parts = child.text.split("=", 1)
                cmd.env_vars.append({"name": parts[0], "value": parts[1]})
            else:
                in_env_vars = False
                cmd.argv.append(child.text)

        elif child_type == "string":
            in_env_vars = False
            # Extract string content
            string_content = _extract_string_content(child)
            cmd.argv.append(string_content)

        elif child_type == "expansion":
            in_env_vars = False
            cmd.argv.append(CMDSUB_PLACEHOLDER)

        elif child_type == "simple_expansion":
            in_env_vars = False
            var_name = _extract_var_name(child)
            if var_name and var_name in SAFE_ENV_VARS:
                cmd.argv.append(VAR_PLACEHOLDER)
            else:
                cmd.argv.append(f"${var_name or '?'}")

        elif child_type == "command_substitution":
            in_env_vars = False
            cmd.argv.append(CMDSUB_PLACEHOLDER)

        elif child_type == "assignment_word":
            parts = child.text.split("=", 1)
            cmd.env_vars.append({"name": parts[0], "value": parts[1] if len(parts) > 1 else ""})
            in_env_vars = True

        elif child_type == "variable_assignment":
            # FOO=bar form: extract name and value from children
            var_name = ""
            var_value = ""
            for sub in child.children:
                if sub.type == "variable_name":
                    var_name = sub.text
                elif sub.type == "word" and not sub.text == "=":
                    var_value = sub.text
            if var_name:
                cmd.env_vars.append({"name": var_name, "value": var_value})
                in_env_vars = True

        elif child_type == "file_redirect":
            redirect = _extract_redirect(child)
            if redirect:
                redirects.append(redirect)

        elif child_type in (
            "heredoc_redirect", "heredoc_body", "heredoc_start",
        ):
            pass  # Handle heredocs separately

        elif child_type in (
            "concatenation", "variable_name",
        ):
            in_env_vars = False
            cmd.argv.append(child.text)

        else:
            # Unknown — fail-closed
            logger.debug("[ast] Unknown child type in command: %s", child_type)
            return None

    cmd.redirects = redirects
    return cmd


def _extract_string_content(node: Node) -> str:
    """Extract the content of a string node (quoted string)."""
    for child in node.children:
        if child.type == "string_content":
            return child.text
    return node.text


def _extract_var_name(node: Node) -> Optional[str]:
    """Extract a variable name from a simple_expansion node."""
    for child in node.children:
        if child.type == "variable_name":
            return child.text
    return None


def _extract_command_name(node: Node) -> Optional[str]:
    """Extract the command name from a command_name node.

    The command_name node typically contains a single 'word' child.
    """
    for child in node.children:
        if child.type == "word":
            return child.text
    return node.text


def _extract_redirect(node: Node) -> Optional[Redirect]:
    """Extract a Redirect from a file_redirect node."""
    op = ""
    target = ""
    fd = None

    for child in node.children:
        if child.type == "file_descriptor":
            try:
                fd = int(child.text)
            except ValueError:
                pass
        elif child.type == "word":
            target = child.text
        elif child.type == "string":
            target = _extract_string_content(child)
        else:
            # The operator is in the node text minus the fd and target
            pass

    # Determine operator from the node text
    node_text = node.text
    for op_str in (">>", "<<", ">&", "<&", ">|", "&>>", "&>", "<<<", ">", "<", "|&"):
        if op_str in node_text:
            op = op_str
            break

    if not op:
        return None

    return Redirect(op=op, target=target, fd=fd)


def _extract_commands_from_redirected(node: Node) -> Optional[List[SimpleCommand]]:
    """Extract commands from a redirected_statement node."""
    for child in node.children:
        if child.type in ("command", "simple_command", "list", "pipeline"):
            return _extract_commands(child)
    return []


def _extract_commands_from_compound(node: Node) -> Optional[List[SimpleCommand]]:
    """Extract commands from compound command nodes (case/for/while/if)."""
    all_commands: List[SimpleCommand] = []
    for child in node.children:
        child_commands = _extract_commands(child)
        if child_commands is None:
            return None
        all_commands.extend(child_commands)
    return all_commands


def _extract_commands_from_function(node: Node) -> Optional[List[SimpleCommand]]:
    """Extract commands from a function definition node."""
    for child in node.children:
        if child.type in ("command", "simple_command", "list", "pipeline"):
            return _extract_commands(child)
    return []


# ---------------------------------------------------------------------------
# Wrapper stripping (checkSemantics equivalent)
# ---------------------------------------------------------------------------

def check_semantics(command: str, parsed: ParseForSecurityResult) -> ParseForSecurityResult:
    """Apply semantic checks to parsed commands.

    Strips safe wrappers (timeout, nice, nohup, stdbuf) and env var
    assignments to reveal the underlying command for permission checking.

    Args:
        command: The original command string.
        parsed: The ParseForSecurityResult from AST parsing.

    Returns:
        Updated ParseForSecurityResult with semantics applied.
    """
    if parsed.get("kind") != "simple":
        return parsed

    commands = parsed.get("commands", [])
    if not commands:
        return parsed

    stripped_commands = []
    for cmd in commands:
        stripped = _strip_wrappers(cmd)
        if stripped is not None:
            stripped_commands.append(stripped)

    return {"kind": "simple", "commands": stripped_commands or commands}


def _strip_wrappers(cmd: SimpleCommand) -> Optional[SimpleCommand]:
    """Strip safe wrapper commands from a SimpleCommand's argv.

    Strips leading: timeout, time, nice, stdbuf, nohup, env
    Also strips VAR=val assignments.

    Args:
        cmd: The SimpleCommand to process.

    Returns:
        Stripped SimpleCommand, or None if the command is unrecognized.
    """
    if not cmd.argv:
        return cmd

    # Strip leading env var assignments
    argv = list(cmd.argv)
    while argv and "=" in argv[0] and not argv[0].startswith("-"):
        argv.pop(0)

    if not argv:
        return cmd

    # Strip known wrappers
    changed = True
    while changed and argv:
        changed = False
        first = argv[0]

        if first == "timeout" and len(argv) >= 3:
            # timeout [options] duration command ...
            # Skip -k/--kill-after, -s/--signal, -v, --preserve-status, --foreground, etc.
            idx = 1
            while idx < len(argv) and argv[idx].startswith("-"):
                if argv[idx] in ("-k", "--kill-after", "-s", "--signal"):
                    idx += 2  # Skip flag and value
                elif argv[idx] in ("-v", "--verbose", "--preserve-status", "--foreground"):
                    idx += 1
                elif "--" in argv[idx]:
                    idx += 1  # --flag=value form
                else:
                    idx += 1
            # Skip duration (number with optional unit)
            if idx < len(argv) and re.match(r"^\d+(\.\d+)?[smhd]?$", argv[idx]):
                idx += 1
                argv = argv[idx:]
                changed = True

        elif first == "time" and len(argv) >= 2:
            argv = argv[2:] if argv[1] == "--" else argv[1:]
            changed = True

        elif first == "nice" and len(argv) >= 2:
            if argv[1] == "--":
                argv = argv[2:]
                changed = True
            elif argv[1].startswith("-") and re.match(r"^-n?\d*$", argv[1]):
                argv = argv[2:] if len(argv) > 2 else argv[1:]
                changed = True
            else:
                argv = argv[1:]
                changed = True

        elif first == "nohup" and len(argv) >= 2:
            argv = argv[2:] if argv[1] == "--" else argv[1:]
            changed = True

        elif first == "stdbuf" and len(argv) >= 2:
            # stdbuf -i0 -oL -eL command
            idx = 1
            while idx < len(argv) and (
                _STDBUF_SHORT_SEP_RE.match(argv[idx])
                or _STDBUF_SHORT_FUSED_RE.match(argv[idx])
                or _STDBUF_LONG_RE.match(argv[idx])
            ):
                idx += 1
            if idx < len(argv):
                argv = argv[idx:]
                changed = True

        elif first == "env" and len(argv) >= 2:
            # Strip env and its VAR=val arguments
            idx = 1
            while idx < len(argv) and "=" in argv[idx] and not argv[idx].startswith("-"):
                idx += 1
            if idx < len(argv):
                argv = argv[idx:]
                changed = True

    # Update the command with stripped argv
    result = SimpleCommand(
        argv=argv,
        env_vars=cmd.env_vars,
        redirects=cmd.redirects,
        text=cmd.text,
    )
    return result


__all__ = [
    "Redirect",
    "SimpleCommand",
    "ParseForSecurityResult",
    "parse_for_security_from_ast",
    "check_semantics",
    "CMDSUB_PLACEHOLDER",
    "VAR_PLACEHOLDER",
    "STRUCTURAL_TYPES",
    "SEPARATOR_TYPES",
    "SAFE_ENV_VARS",
]
