"""Tree-sitter AST analysis utilities.

Provides structural analysis of parsed bash commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ripperdoc.utils.bash.node import Node


@dataclass
class CompoundStructure:
    """Information about compound command structures."""
    has_subshell: bool = False
    has_command_group: bool = False


@dataclass
class TreeSitterAnalysis:
    """Analysis data derived from a tree-sitter parse."""
    compound_structure: CompoundStructure = field(default_factory=CompoundStructure)


def analyze_command(root: Node) -> TreeSitterAnalysis:
    """Analyze a parsed command node for structural properties.

    Walks the AST to determine:
    - Whether the command contains subshells ($(...) or (...))
    - Whether it contains command groups ({ ...; })

    Args:
        root: The root node of the parsed command.

    Returns:
        TreeSitterAnalysis with the analysis results.
    """
    result = TreeSitterAnalysis()

    _walk_for_structure(root, result)

    return result


def _walk_for_structure(node: Node, result: TreeSitterAnalysis) -> None:
    """Walk the AST recursively to detect structural patterns."""
    node_type = node.type

    # Detect subshells: ( ... ) or $( ... )
    if node_type in ("subshell", "command_substitution", "process_substitution"):
        result.compound_structure.has_subshell = True
        return  # Don't recurse into subshells

    # Detect command groups: { ...; }
    if node_type in (
        "braced_group",      # { ...; } (internal)
        "do_group",          # do ... done
        "then_group",        # then ... fi
        "else_group",        # else ... fi
    ):
        result.compound_structure.has_command_group = True
        return  # Don't recurse into groups

    # Recurse into children
    for child in node.children:
        _walk_for_structure(child, result)
