"""Tree-sitter based bash command parser and AST analysis.

Provides fail-closed AST parsing with tree-sitter for bash security analysis.
"""

from .node import Node, PARSE_ABORTED, parse_command_raw
from .bash_parser import SHELL_KEYWORDS, get_bash_language
from .ast import (
    ParseForSecurityResult,
    SimpleCommand,
    Redirect,
    parse_for_security_from_ast,
    CMDSUB_PLACEHOLDER,
    VAR_PLACEHOLDER,
)
from .parsed_command import ParsedCommand, IParsedCommand, RegexParsedCommand_DEPRECATED
from .shell_quote import (
    try_parse_shell_command,
    has_malformed_tokens,
    has_shell_quote_single_quote_bug,
    quote,
)
from .commands import (
    split_command,
    split_command_with_operators,
    extract_output_redirections,
)
from .tree_sitter_analysis import TreeSitterAnalysis, CompoundStructure, analyze_command

__all__ = [
    "Node",
    "PARSE_ABORTED",
    "parse_command_raw",
    "SHELL_KEYWORDS",
    "get_bash_language",
    "ParseForSecurityResult",
    "SimpleCommand",
    "Redirect",
    "parse_for_security_from_ast",
    "CMDSUB_PLACEHOLDER",
    "VAR_PLACEHOLDER",
    "ParsedCommand",
    "IParsedCommand",
    "RegexParsedCommand_DEPRECATED",
    "try_parse_shell_command",
    "has_malformed_tokens",
    "has_shell_quote_single_quote_bug",
    "quote",
    "split_command",
    "split_command_with_operators",
    "extract_output_redirections",
    "TreeSitterAnalysis",
    "CompoundStructure",
    "analyze_command",
]
