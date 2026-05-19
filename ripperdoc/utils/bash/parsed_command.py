"""ParsedCommand — parsed command interface and implementations.


Provides both tree-sitter based and regex fallback implementations
of the IParsedCommand interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, List, Optional

from ripperdoc.utils.bash.node import Node, PARSE_ABORTED, parse_command_raw
from ripperdoc.utils.bash.bash_parser import get_bash_parser
from ripperdoc.utils.bash.ast import (
    parse_for_security_from_ast,
    SimpleCommand,
)
from ripperdoc.utils.bash.commands import (
    split_command_with_operators,
    extract_output_redirections,
)
from ripperdoc.utils.bash.tree_sitter_analysis import TreeSitterAnalysis, analyze_command
from ripperdoc.utils.log import get_logger

logger = get_logger()


@dataclass
class OutputRedirection:
    """An output redirection (>, >>)."""
    target: str
    operator: str  # '>' | '>>'


class IParsedCommand:
    """Interface for parsed command implementations.

    Both tree-sitter and regex fallback implementations conform to this.
    """

    @property
    def original_command(self) -> str:
        raise NotImplementedError

    def get_pipe_segments(self) -> List[str]:
        """Split the command into pipe segments."""
        raise NotImplementedError

    def without_output_redirections(self) -> str:
        """Return the command with output redirections stripped."""
        raise NotImplementedError

    def get_output_redirections(self) -> List[OutputRedirection]:
        """Return all output redirections."""
        raise NotImplementedError

    def get_tree_sitter_analysis(self) -> Optional[TreeSitterAnalysis]:
        """Return tree-sitter analysis data if available."""
        return None


@dataclass
class RegexParsedCommand_DEPRECATED(IParsedCommand):
    """Legacy regex/shell-quote fallback implementation.

    Only used when tree-sitter is unavailable.
    """

    original_command: str

    def get_pipe_segments(self) -> List[str]:
        try:
            parts = split_command_with_operators(self.original_command)
            segments: List[str] = []
            current: List[str] = []

            for part in parts:
                if part == "|":
                    if current:
                        segments.append(" ".join(current))
                        current = []
                else:
                    current.append(part)

            if current:
                segments.append(" ".join(current))

            return segments if segments else [self.original_command]
        except Exception:
            return [self.original_command]

    def without_output_redirections(self) -> str:
        if ">" not in self.original_command:
            return self.original_command
        result = extract_output_redirections(self.original_command)
        if result.redirections:
            return result.command_without_redirections
        return self.original_command

    def get_output_redirections(self) -> List[OutputRedirection]:
        result = extract_output_redirections(self.original_command)
        return [
            OutputRedirection(target=r.target, operator=r.operator)
            for r in result.redirections
            if r.operator in (">", ">>")
        ]

    def get_tree_sitter_analysis(self) -> None:
        return None


class ParsedCommand(IParsedCommand):
    """Tree-sitter based parsed command implementation.

    Provides structured access to pipe segments, output redirections,
    and AST analysis data.
    """

    def __init__(
        self,
        original_command: str,
        ast_root: Optional[Node] = None,
    ):
        self._original_command = original_command
        self._ast_root = ast_root
        self._pipe_segments: Optional[List[str]] = None
        self._without_redirections: Optional[str] = None
        self._output_redirections: Optional[List[OutputRedirection]] = None
        self._ts_analysis: Optional[TreeSitterAnalysis] = None

    @property
    def original_command(self) -> str:
        return self._original_command

    @classmethod
    async def parse(cls, command: str) -> Optional["ParsedCommand"]:
        """Factory method: parse a command and return a ParsedCommand.

        Returns None if tree-sitter parsing is unavailable or fails.

        Args:
            command: The bash command string to parse.

        Returns:
            A ParsedCommand instance, or None.
        """
        try:
            parser = get_bash_parser()
        except ImportError:
            logger.debug("[ParsedCommand] tree-sitter unavailable, falling back to regex")
            return None

        root = parse_command_raw(command, parser)
        if root is PARSE_ABORTED:
            return None

        assert isinstance(root, Node)
        return cls(original_command=command, ast_root=root)

    def get_pipe_segments(self) -> List[str]:
        if self._pipe_segments is not None:
            return self._pipe_segments

        # Try AST first
        ast_result = self._get_ast_commands()
        if ast_result:
            # If we have multiple commands separated by pipes in the AST, extract them
            segments = []
            for cmd in ast_result:
                segments.append(cmd.text)
            self._pipe_segments = segments if segments else [self.original_command]
        else:
            # Fall back to regex-based
            parts = split_command_with_operators(self.original_command)
            segments = []
            current: List[str] = []
            for part in parts:
                if part == "|":
                    if current:
                        segments.append(" ".join(current))
                        current = []
                else:
                    current.append(part)
            if current:
                segments.append(" ".join(current))
            self._pipe_segments = segments if segments else [self.original_command]

        return self._pipe_segments

    def without_output_redirections(self) -> str:
        if self._without_redirections is not None:
            return self._without_redirections

        if ">" not in self.original_command:
            self._without_redirections = self.original_command
            return self._without_redirections

        result = extract_output_redirections(self.original_command)
        if result.redirections:
            self._without_redirections = result.command_without_redirections
        else:
            self._without_redirections = self.original_command

        return self._without_redirections

    def get_output_redirections(self) -> List[OutputRedirection]:
        if self._output_redirections is not None:
            return self._output_redirections

        result = extract_output_redirections(self.original_command)
        self._output_redirections = [
            OutputRedirection(target=r.target, operator=r.operator)
            for r in result.redirections
            if r.operator in (">", ">>")
        ]
        return self._output_redirections

    def get_tree_sitter_analysis(self) -> Optional[TreeSitterAnalysis]:
        if self._ts_analysis is not None:
            return self._ts_analysis

        if self._ast_root:
            self._ts_analysis = analyze_command(self._ast_root)
        else:
            self._ts_analysis = TreeSitterAnalysis()

        return self._ts_analysis

    def _get_ast_commands(self) -> Optional[List[SimpleCommand]]:
        """Extract simple commands from the AST, if available."""
        if not self._ast_root:
            return None

        result = parse_for_security_from_ast(self.original_command)
        if result.get("kind") == "simple":
            return cast(list[SimpleCommand], result.get("commands", []))
        return None


def build_parsed_command_from_root(
    command: str,
    root: Node,
) -> Optional[IParsedCommand]:
    """Build a ParsedCommand from a pre-parsed AST root node.

    Args:
        command: The original command string.
        root: The parsed AST root node (must not be PARSE_ABORTED).

    Returns:
        An IParsedCommand implementation, or None if the AST is unavailable.
    """
    if root is None or root is PARSE_ABORTED:
        return None
    return ParsedCommand(original_command=command, ast_root=root)


__all__ = [
    "IParsedCommand",
    "ParsedCommand",
    "RegexParsedCommand_DEPRECATED",
    "build_parsed_command_from_root",
    "OutputRedirection",
]
