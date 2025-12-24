"""AST node definitions for Viper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union


@dataclass
class Module:
    statements: List[Statement]


@dataclass
class ExpressionStmt:
    expression: Expression
    line: int
    column: int


@dataclass
class AssignStmt:
    targets: List[Expression]
    value: Expression
    type_comment: Optional[str]
    line: int
    column: int


@dataclass
class AugAssignStmt:
    target: Expression
    op: str
    value: Expression
    line: int
    column: int


@dataclass
class AnnAssignStmt:
    target: Expression
    annotation: Expression
    value: Optional[Expression]
    type_comment: Optional[str]
    line: int
    column: int


@dataclass
class IfStmt:
    test: Expression
    body: List[Statement]
    elif_blocks: List[Tuple[Expression, List[Statement]]]
    else_body: Optional[List[Statement]]
    line: int
    column: int


@dataclass
class WhileStmt:
    test: Expression
    body: List[Statement]
    else_body: Optional[List[Statement]]
    line: int
    column: int


@dataclass
class ForStmt:
    target: Expression
    iterable: Expression
    body: List[Statement]
    else_body: Optional[List[Statement]]
    is_async: bool
    line: int
    column: int


@dataclass
class TypeParam:
    name: str
    kind: str
    bound: Optional["Expression"]
    default: Optional["Expression"]
    line: int
    column: int


@dataclass
class Parameter:
    name: str
    kind: str
    annotation: Optional["Expression"]
    default: Optional["Expression"]
    line: int
    column: int


@dataclass
class Comprehension:
    target: "Expression"
    iterable: "Expression"
    ifs: List["Expression"]
    is_async: bool
    line: int
    column: int


@dataclass
class ClassDef:
    name: str
    bases: List[Expression]
    keywords: List[Tuple[str, Expression]]
    type_params: List[TypeParam]
    body: List[Statement]
    decorators: List[Expression]
    line: int
    column: int


@dataclass
class WithItem:
    context_expr: Expression
    target: Optional[Expression]
    line: int
    column: int


@dataclass
class WithStmt:
    items: List[WithItem]
    body: List[Statement]
    is_async: bool
    line: int
    column: int


@dataclass
class FunctionDef:
    name: str
    params: List[Parameter]
    type_params: List["TypeParam"]
    body: List[Statement]
    is_async: bool
    line: int
    column: int
    decorators: List[Expression]
    returns: Optional[Expression]


@dataclass
class ReturnStmt:
    value: Optional[Expression]
    line: int
    column: int


@dataclass
class PassStmt:
    line: int
    column: int


@dataclass
class BreakStmt:
    line: int
    column: int


@dataclass
class ContinueStmt:
    line: int
    column: int


@dataclass
class ImportAlias:
    name: str
    asname: Optional[str]
    line: int
    column: int


@dataclass
class ImportStmt:
    names: List[ImportAlias]
    line: int
    column: int


@dataclass
class ImportFromStmt:
    module: Optional[str]
    names: List[ImportAlias]
    level: int
    is_star: bool
    line: int
    column: int


@dataclass
class GlobalStmt:
    names: List[str]
    line: int
    column: int


@dataclass
class NonlocalStmt:
    names: List[str]
    line: int
    column: int


@dataclass
class DelStmt:
    targets: List[Expression]
    line: int
    column: int


@dataclass
class TypeAliasStmt:
    name: str
    type_params: List["TypeParam"]
    value: Expression
    line: int
    column: int


@dataclass
class ExceptHandler:
    type: Optional["Expression"]
    name: Optional[str]
    body: List["Statement"]
    is_star: bool
    line: int
    column: int


@dataclass
class TryStmt:
    body: List["Statement"]
    handlers: List[ExceptHandler]
    else_body: Optional[List["Statement"]]
    finally_body: Optional[List["Statement"]]
    line: int
    column: int


@dataclass
class RaiseStmt:
    exception: Optional["Expression"]
    cause: Optional["Expression"]
    line: int
    column: int


@dataclass
class AssertStmt:
    test: "Expression"
    message: Optional["Expression"]
    line: int
    column: int


@dataclass
class Name:
    identifier: str
    line: int
    column: int


@dataclass
class AssignmentExpr:
    target: Name
    value: Expression
    line: int
    column: int


@dataclass
class Starred:
    target: Expression
    line: int
    column: int


@dataclass
class Literal:
    value: Any
    line: int
    column: int


@dataclass
class ListLiteral:
    elements: List[Expression]
    line: int
    column: int


@dataclass
class DictLiteral:
    items: List[Tuple[Optional[Expression], Expression]]
    line: int
    column: int


@dataclass
class TupleLiteral:
    elements: List[Expression]
    line: int
    column: int


@dataclass
class SetLiteral:
    elements: List[Expression]
    line: int
    column: int


@dataclass
class StringText:
    value: str | bytes
    line: int
    column: int


@dataclass
class FormatField:
    expression: Expression
    conversion: Optional[str]
    format_spec: Optional[List["FStringPart"]]
    debug: bool
    expr_text: Optional[str]
    line: int
    column: int


@dataclass
class FormattedString:
    parts: List["FStringPart"]
    is_template: bool
    is_bytes: bool
    line: int
    column: int


@dataclass
class ListComp:
    element: Expression
    generators: List[Comprehension]
    line: int
    column: int


@dataclass
class SetComp:
    element: Expression
    generators: List[Comprehension]
    line: int
    column: int


@dataclass
class DictComp:
    key: Expression
    value: Expression
    generators: List[Comprehension]
    line: int
    column: int


@dataclass
class GeneratorExp:
    element: Expression
    generators: List[Comprehension]
    line: int
    column: int


@dataclass
class UnaryOp:
    op: str
    operand: Expression
    line: int
    column: int


@dataclass
class AwaitExpr:
    value: Expression
    line: int
    column: int


@dataclass
class LambdaExpr:
    params: List[Parameter]
    body: Expression
    line: int
    column: int


@dataclass
class ConditionalExpr:
    test: Expression
    body: Expression
    orelse: Expression
    line: int
    column: int


@dataclass
class BinaryOp:
    left: Expression
    op: str
    right: Expression
    line: int
    column: int


@dataclass
class CompareOp:
    left: Expression
    ops: List[str]
    comparators: List[Expression]
    line: int
    column: int


@dataclass
class YieldExpr:
    value: Optional[Expression]
    line: int
    column: int


@dataclass
class YieldFromExpr:
    value: Expression
    line: int
    column: int


@dataclass
class Call:
    func: Expression
    args: List[Expression]
    kwargs: List[Tuple[Optional[str], Expression]]
    line: int
    column: int


@dataclass
class Attribute:
    value: Expression
    name: str
    line: int
    column: int


@dataclass
class Subscript:
    value: Expression
    index: Expression
    line: int
    column: int


@dataclass
class SliceExpr:
    start: Optional[Expression]
    stop: Optional[Expression]
    step: Optional[Expression]
    line: int
    column: int


@dataclass
class PatternWildcard:
    line: int
    column: int


@dataclass
class PatternName:
    identifier: str
    line: int
    column: int


@dataclass
class PatternLiteral:
    value: Any
    line: int
    column: int


@dataclass
class PatternSequence:
    elements: List["Pattern"]
    line: int
    column: int


@dataclass
class PatternStar:
    target: "Pattern"
    line: int
    column: int


@dataclass
class PatternAs:
    pattern: "Pattern"
    name: str
    line: int
    column: int


@dataclass
class PatternValue:
    parts: List[str]
    line: int
    column: int


@dataclass
class PatternMapping:
    items: List[Tuple["PatternKey", "Pattern"]]
    rest: Optional["Pattern"]
    line: int
    column: int


@dataclass
class PatternClass:
    class_path: "PatternValue"
    positional: List["Pattern"]
    keywords: List[Tuple[str, "Pattern"]]
    line: int
    column: int


@dataclass
class PatternOr:
    patterns: List["Pattern"]
    line: int
    column: int


@dataclass
class MatchCase:
    patterns: List["Pattern"]
    guard: Optional[Expression]
    body: List["Statement"]
    line: int
    column: int


@dataclass
class MatchStmt:
    subject: Expression
    cases: List[MatchCase]
    line: int
    column: int


Statement = Union[
    ExpressionStmt,
    AssignStmt,
    AugAssignStmt,
    AnnAssignStmt,
    IfStmt,
    WhileStmt,
    ForStmt,
    WithStmt,
    ClassDef,
    FunctionDef,
    ImportStmt,
    ImportFromStmt,
    GlobalStmt,
    NonlocalStmt,
    DelStmt,
    TypeAliasStmt,
    MatchStmt,
    TryStmt,
    RaiseStmt,
    AssertStmt,
    ReturnStmt,
    PassStmt,
    BreakStmt,
    ContinueStmt,
]

Expression = Union[
    Name,
    AssignmentExpr,
    Starred,
    Literal,
    ListLiteral,
    DictLiteral,
    TupleLiteral,
    SetLiteral,
    FormattedString,
    ListComp,
    SetComp,
    DictComp,
    GeneratorExp,
    UnaryOp,
    AwaitExpr,
    LambdaExpr,
    ConditionalExpr,
    BinaryOp,
    CompareOp,
    YieldExpr,
    YieldFromExpr,
    Call,
    Attribute,
    Subscript,
    SliceExpr,
]

FStringPart = Union[StringText, FormatField]

Pattern = Union[
    PatternWildcard,
    PatternName,
    PatternLiteral,
    PatternSequence,
    PatternOr,
    PatternStar,
    PatternAs,
    PatternValue,
    PatternMapping,
    PatternClass,
]

PatternKey = Union[PatternLiteral, PatternValue]
