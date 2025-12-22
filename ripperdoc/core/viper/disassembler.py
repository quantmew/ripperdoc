"""Disassembler for Viper bytecode."""

from __future__ import annotations

from typing import Iterable, List, Set

from ripperdoc.core.viper.bytecode import CodeObject, Instruction


_JUMP_OPS = {
    "JUMP",
    "JUMP_IF_FALSE",
    "JUMP_IF_TRUE",
    "JUMP_IF_FALSE_KEEP",
    "JUMP_IF_TRUE_KEEP",
    "FOR_ITER",
    "SETUP_EXCEPT",
    "SETUP_FINALLY",
}


def disassemble(code: CodeObject, *, recursive: bool = True) -> str:
    """Return a formatted disassembly string for a CodeObject."""
    lines: List[str] = []
    visited: Set[int] = set()

    def _emit_code(obj: CodeObject, indent: str) -> None:
        if id(obj) in visited:
            return
        visited.add(id(obj))
        header = f"{indent}== {obj.name} =="
        if obj.params:
            header = f"{header} params={obj.params}"
        lines.append(header)
        labels = _collect_labels(obj.instructions)
        for idx, instr in enumerate(obj.instructions):
            label = labels.get(idx)
            if label:
                lines.append(f"{indent}{label}:")
            line = f"{indent}{idx:04d} {instr.op}"
            arg_text = _format_arg(instr, labels)
            if arg_text:
                line = f"{line} {arg_text}"
            lines.append(line)
        if recursive:
            for instr in obj.instructions:
                if isinstance(instr.arg, CodeObject):
                    lines.append("")
                    _emit_code(instr.arg, indent + "  ")

    _emit_code(code, "")
    return "\n".join(lines)


def _collect_labels(instructions: Iterable[Instruction]) -> dict[int, str]:
    targets = {
        instr.arg
        for instr in instructions
        if instr.op in _JUMP_OPS and isinstance(instr.arg, int)
    }
    return {target: f"L{index}" for index, target in enumerate(sorted(targets))}


def _format_arg(instr: Instruction, labels: dict[int, str]) -> str:
    if instr.arg is None:
        return ""
    if instr.op in _JUMP_OPS and isinstance(instr.arg, int):
        return labels.get(instr.arg, str(instr.arg))
    if instr.op == "CALL_FUNCTION":
        argc, kwarg_names = instr.arg
        return f"argc={argc} kwargs={list(kwarg_names)}"
    if instr.op in {
        "BUILD_LIST",
        "BUILD_TUPLE",
        "BUILD_DICT",
        "BUILD_STRING",
        "BUILD_BYTES",
        "BUILD_TEMPLATE",
        "BUILD_TEMPLATE_BYTES",
        "UNPACK_SEQUENCE",
    }:
        return str(instr.arg)
    if instr.op == "LOAD_CONST" and isinstance(instr.arg, CodeObject):
        return f"<code {instr.arg.name}>"
    return repr(instr.arg)
