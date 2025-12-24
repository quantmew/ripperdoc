"""Bytecode definitions for Viper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

OpCode = Literal[
    "LOAD_CONST",
    "LOAD_NAME",
    "LOAD_ENV",
    "STORE_NAME",
    "DELETE_NAME",
    "SET_RESULT",
    "LOAD_RESULT",
    "POP_TOP",
    "BUILD_STRING",
    "BUILD_BYTES",
    "ENCODE_UTF8",
    "CONVERT_VALUE",
    "FORMAT_VALUE",
    "BUILD_TEMPLATE_FIELD",
    "BUILD_TEMPLATE",
    "BUILD_TEMPLATE_BYTES",
    "BUILD_LIST",
    "BUILD_LIST_UNPACK",
    "BUILD_TUPLE",
    "BUILD_TUPLE_UNPACK",
    "BUILD_DICT",
    "BUILD_DICT_UNPACK",
    "BUILD_SET",
    "BUILD_SLICE",
    "UNPACK_SEQUENCE",
    "UNPACK_EX",
    "UNARY_OP",
    "BINARY_OP",
    "COMPARE_CHAIN",
    "JUMP",
    "JUMP_IF_FALSE",
    "JUMP_IF_TRUE",
    "JUMP_IF_FALSE_KEEP",
    "JUMP_IF_TRUE_KEEP",
    "GET_ATTR",
    "SET_ATTR",
    "DELETE_ATTR",
    "GET_SUBSCRIPT",
    "SET_SUBSCRIPT",
    "DELETE_SUBSCRIPT",
    "IMPORT_NAME",
    "IMPORT_FROM",
    "IMPORT_STAR",
    "DECLARE_GLOBAL",
    "DECLARE_NONLOCAL",
    "BUILD_ITER",
    "BUILD_AITER",
    "FOR_ITER",
    "FOR_AITER",
    "WITH_ENTER",
    "WITH_EXIT",
    "ASYNC_WITH_ENTER",
    "ASYNC_WITH_EXIT",
    "SETUP_EXCEPT",
    "POP_EXCEPT",
    "SETUP_FINALLY",
    "POP_FINALLY",
    "END_FINALLY",
    "LOAD_EXCEPTION",
    "CLEAR_EXCEPTION",
    "EXC_MATCH",
    "RAISE",
    "ASSERT",
    "MAKE_CLASS",
    "SWAP",
    "ROT_THREE",
    "CALL_FUNCTION",
    "CALL_FUNCTION_EX",
    "MAKE_FUNCTION",
    "AWAIT",
    "YIELD_VALUE",
    "RETURN_VALUE",
]


@dataclass
class Label:
    position: Optional[int] = None


@dataclass
class Instruction:
    op: OpCode
    arg: Any
    line: int
    column: int


@dataclass
class ParamSpec:
    posonly: List[str]
    pos_or_kw: List[str]
    kwonly: List[str]
    vararg: Optional[str]
    varkw: Optional[str]


@dataclass
class CodeObject:
    name: str
    instructions: List[Instruction]
    params: List[str]
    param_spec: Optional[ParamSpec] = None
    is_module: bool = False
    is_generator: bool = False
    is_coroutine: bool = False
