"""Compound command permission helpers for bash tool.


Provides pipe/subshell permission checking, cd+git cross-segment detection,
and segmented command permission evaluation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ripperdoc.utils.bash.parsed_command import (
    ParsedCommand,
    build_parsed_command_from_root,
)
from ripperdoc.utils.bash.node import PARSE_ABORTED
from ripperdoc.utils.bash.commands import split_command, split_command_with_operators
from ripperdoc.utils.bash.shell_quote import try_parse_shell_command
from ripperdoc.security import PermissionResult


# ============================================================================
# Command identity checkers
# ============================================================================


def _normalized_tokens(command: str) -> List[str]:
    from ripperdoc.tools.bash.permissions import _normalized_for_rule_matching

    parsed = try_parse_shell_command(_normalized_for_rule_matching(command))
    if parsed.success:
        return [str(token) for token in parsed.tokens]

    parsed = try_parse_shell_command(command)
    if parsed.success:
        return [str(token) for token in parsed.tokens]

    return command.strip().split()


def is_normalized_cd_command(command: str) -> bool:
    """Check if a command changes the current directory after safe normalization."""
    tokens = _normalized_tokens(command)
    if not tokens:
        return False
    return tokens[0] in {"cd", "pushd", "popd"}


def is_normalized_git_command(command: str) -> bool:
    """Check if a command invokes git after safe normalization."""
    tokens = _normalized_tokens(command)
    if not tokens:
        return False
    if tokens[0] == "git":
        return True
    return tokens[0] == "xargs" and "git" in tokens


# ============================================================================
# Segmented command permission result
# ============================================================================


def _operators_in(parts: List[str], operators: set[str]) -> bool:
    return any(part in operators for part in parts)


async def segmented_command_permission_result(
    input_data: Any,
    segments: List[str],
    single_command_checker: Callable[[str], PermissionResult],
    checkers: Tuple[Callable, Callable],
) -> PermissionResult:
    """Evaluate permissions for each segment of a piped command.

    Args:
        input_data: The original tool input.
        segments: Pipe segments.
        single_command_checker: A function that checks a single command string
            and returns a PermissionResult. Must NOT recurse into compound
            command checking.
        checkers: (is_cd_fn, is_git_fn) tuple.

    Returns:
        PermissionResult for the compound command.
    """
    is_cd, is_git = checkers

    # Check for multiple cd commands across segments
    cd_count = sum(1 for s in segments if is_cd(s.strip()))
    if cd_count > 1:
        return PermissionResult.ask(
            "Multiple directory changes in one command require approval for clarity",
            reason={"type": "other", "reason": "Multiple cd commands"},
        )

    # Check for cd+git across pipe segments (bare repo fsmonitor bypass)
    has_cd = any(is_cd(sub.strip()) for s in segments for sub in split_command(s))
    has_git = any(is_git(sub.strip()) for s in segments for sub in split_command(s))
    if has_cd and has_git:
        return PermissionResult.ask(
            "Compound commands with cd and git require approval to prevent bare repository attacks",
            reason={"type": "other", "reason": "cd+git across pipe segments"},
        )

    # Check each segment via the single-command checker (no recursion)
    segment_results: Dict[str, PermissionResult] = {}

    for segment in segments:
        trimmed = segment.strip()
        if not trimmed:
            continue
        result = single_command_checker(trimmed)
        segment_results[trimmed] = result

    # Check for any denied segments
    for seg_cmd, result in segment_results.items():
        if result.behavior == "deny":
            return result

    # All allowed
    if all(r.behavior == "allow" for r in segment_results.values()):
        return PermissionResult.allow(
            updated_input=input_data,
            reason={"type": "subcommandResults", "reasons": {k: v.behavior for k, v in segment_results.items()}},
        )

    # Mixed: ask
    return PermissionResult.ask(
        "Permission required for compound command",
        reason={"type": "subcommandResults", "reasons": {k: v.behavior for k, v in segment_results.items()}},
    )


async def build_segment_without_redirections(segment: str) -> str:
    """Build a command segment with output redirections stripped.

    Uses ParsedCommand to preserve original quoting.

    Args:
        segment: The command segment.

    Returns:
        Segment with redirections stripped.
    """
    if ">" not in segment:
        return segment

    parsed = await ParsedCommand.parse(segment)
    if parsed:
        return parsed.without_output_redirections()

    return segment


# ============================================================================
# Main entry point
# ============================================================================


async def check_command_operator_permissions(
    input_data: Any,
    checkers: Tuple[Callable, Callable],
    single_command_checker: Optional[Callable[[str], PermissionResult]] = None,
    ast_root: Any = None,
) -> PermissionResult:
    """Check if a command has special operators requiring segmented permission checking.

    Args:
        input_data: The tool input.
        checkers: (is_cd_fn, is_git_fn) tuple.
        single_command_checker: A function that checks a single command string.
            Required when pipe segments need individual permission evaluation.
        ast_root: Optional AST root node.

    Returns:
        PermissionResult.
    """
    # Parse the command
    if ast_root and ast_root is not PARSE_ABORTED:
        parsed = build_parsed_command_from_root(input_data.command, ast_root)
    else:
        parsed = await ParsedCommand.parse(input_data.command)

    if single_command_checker is None:
        return PermissionResult.passthrough("No single command checker provided")

    parts = split_command_with_operators(input_data.command)
    if parsed is None:
        if _operators_in(parts, {"&&", "||", ";", "&", "|"}):
            if _operators_in(parts, {"&"}):
                return PermissionResult.ask(
                    "Backgrounded compound commands require approval for safety",
                    reason={"type": "other", "reason": "Background compound command"},
                )
            segments = [part for part in parts if part not in {"&&", "||", ";", "|", "&"}]
            if len(segments) > 1:
                return await segmented_command_permission_result(
                    input_data,
                    await _strip_redirections_from_segments(segments),
                    single_command_checker,
                    checkers,
                )
        return PermissionResult.passthrough("Failed to parse command")

    ts_analysis = parsed.get_tree_sitter_analysis()

    # 2. Extract pipe segments
    pipe_segments = parsed.get_pipe_segments()
    if len(pipe_segments) > 1:
        segments = await _strip_redirections_from_segments(pipe_segments)
        return await segmented_command_permission_result(
            input_data,
            segments,
            single_command_checker,
            checkers,
        )

    # 3. Evaluate sequential compound commands conservatively
    if _operators_in(parts, {"&&", "||", ";", "&"}):
        if _operators_in(parts, {"&"}):
            return PermissionResult.ask(
                "Backgrounded compound commands require approval for safety",
                reason={"type": "other", "reason": "Background compound command"},
            )
        segments = [part for part in parts if part not in {"&&", "||", ";", "|", "&"}]
        if len(segments) <= 1:
            return PermissionResult.ask(
                "Compound command requires approval for safety",
                reason={"type": "other", "reason": "Unclear compound command"},
            )
        return await segmented_command_permission_result(
            input_data,
            segments,
            single_command_checker,
            checkers,
        )

    if ts_analysis and (
        ts_analysis.compound_structure.has_subshell
        or ts_analysis.compound_structure.has_command_group
    ):
        return PermissionResult.ask(
            "This command uses shell operators that require approval for safety",
            reason={"type": "other", "reason": "Unsafe compound command structure"},
        )

    return PermissionResult.passthrough("No compound operators found in command")


async def _strip_redirections_from_segments(segments: List[str]) -> List[str]:
    """Strip output redirections from all segments.

    Args:
        segments: List of command segments.

    Returns:
        Segments with redirections stripped.
    """
    result = []
    for seg in segments:
        stripped = await build_segment_without_redirections(seg)
        result.append(stripped)
    return result


__all__ = [
    "check_command_operator_permissions",
    "segmented_command_permission_result",
    "is_normalized_cd_command",
    "is_normalized_git_command",
]
