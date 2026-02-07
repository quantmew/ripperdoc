"""Static import-graph regression tests for known cycle-prone modules."""

from __future__ import annotations

import ast
from pathlib import Path


TARGET_MODULES = [
    "ripperdoc.tools.mcp_tools",
    "ripperdoc.tools.dynamic_mcp_tool",
    "ripperdoc.tools.task_tool",
    "ripperdoc.core.system_prompt",
    "ripperdoc.core.agents",
]


def _module_path(module_name: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    rel = module_name.replace(".", "/") + ".py"
    return root / rel


def _collect_edges(module_name: str, all_modules: set[str]) -> set[tuple[str, str]]:
    path = _module_path(module_name)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    edges: set[tuple[str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in all_modules:
                    edges.add((module_name, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            from_module = node.module
            if from_module in all_modules:
                edges.add((module_name, from_module))
                continue
            for alias in node.names:
                candidate = f"{from_module}.{alias.name}"
                if candidate in all_modules:
                    edges.add((module_name, candidate))

    return edges


def _has_cycle(nodes: set[str], edges: set[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, set()).add(dst)

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in adjacency.get(node, ()):
            if dfs(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(node) for node in nodes)


def test_cycle_prone_module_group_has_no_directed_cycle() -> None:
    module_set = set(TARGET_MODULES)
    edges: set[tuple[str, str]] = set()
    for module_name in TARGET_MODULES:
        edges.update(_collect_edges(module_name, module_set))

    assert not _has_cycle(module_set, edges)
