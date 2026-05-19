"""Tree node types and tree-sitter parsing wrapper.

Provides a Pythonic wrapper around the tree-sitter library for bash parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Union


# Sentinel value indicating that parsing was aborted (e.g., timeout).
PARSE_ABORTED = object()


@dataclass
class Node:
    """Wrapper around a tree-sitter SyntaxNode.

    Attributes:
        type: The tree-sitter node type (e.g., 'command', 'word', 'pipeline').
        start_byte: Byte offset of the start of this node in the source.
        end_byte: Byte offset of the end of this node in the source.
        text: The source text that this node spans.
        children: Child nodes.
        named_children: Named child nodes (tree-sitter named vs anonymous distinction).
    """

    type: str
    start_byte: int
    end_byte: int
    text: str = ""
    children: List[Node] = field(default_factory=list)
    named_children: List[Node] = field(default_factory=list)


def _ts_node_to_node(ts_node: Any, source_bytes: bytes) -> Node:
    """Convert a tree-sitter SyntaxNode to our Node wrapper."""
    start = ts_node.start_byte
    end = ts_node.end_byte
    text = source_bytes[start:end].decode("utf-8", errors="replace")
    node = Node(
        type=ts_node.type,
        start_byte=start,
        end_byte=end,
        text=text,
    )
    # Recursively convert children
    try:
        n_children = ts_node.child_count
        for i in range(n_children):
            child = ts_node.child(i)
            if child is not None:
                node.children.append(_ts_node_to_node(child, source_bytes))
    except Exception:
        pass

    try:
        n_named = ts_node.named_child_count
        for i in range(n_named):
            child = ts_node.named_child(i)
            if child is not None:
                node.named_children.append(_ts_node_to_node(child, source_bytes))
    except Exception:
        pass

    return node


def parse_command_raw(command: str, parser: Any, timeout_ms: int = 5000) -> Union[Node, object]:
    """Parse a bash command string into a Node tree.

    Args:
        command: The bash command string to parse.
        parser: A tree-sitter Parser instance configured with the bash language.
        timeout_ms: Maximum time to allow for parsing.

    Returns:
        A Node if parsing succeeded, or PARSE_ABORTED if it timed out or failed.
    """
    import time

    try:
        source_bytes = command.encode("utf-8")
        start = time.monotonic()
        tree = parser.parse(source_bytes)
        elapsed = (time.monotonic() - start) * 1000
        if elapsed > timeout_ms * 2:  # Allow some grace
            return PARSE_ABORTED
        if tree is None:
            return PARSE_ABORTED
        root = tree.root_node
        if root is None:
            return PARSE_ABORTED
        return _ts_node_to_node(root, source_bytes)
    except Exception:
        return PARSE_ABORTED
