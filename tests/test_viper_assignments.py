from ripperdoc.core.viper.ast_nodes import (
    AnnAssignStmt,
    AssignStmt,
    AssignmentExpr,
    AugAssignStmt,
    BinaryOp,
    ClassDef,
    ConditionalExpr,
    LambdaExpr,
    Literal,
    Name,
    Starred,
    TupleLiteral,
    FunctionDef,
    TypeAliasStmt,
)
from ripperdoc.core.viper.parser import parse, parse_expression_source
from ripperdoc.core.viper.tokenizer import tokenize


def _parse_single_stmt(source: str):
    if not source.endswith("\n"):
        source = f"{source}\n"
    module = parse(tokenize(source))
    assert len(module.statements) == 1
    return module.statements[0]


def test_assignment_expression_parses():
    expr = parse_expression_source("x := 1")
    assert isinstance(expr, AssignmentExpr)
    assert isinstance(expr.target, Name)
    assert expr.target.identifier == "x"
    assert isinstance(expr.value, Literal)
    assert expr.value.value == 1


def test_augassign_parses():
    for op in ["+=", "-=", "*=", "/=", "//=", "%=", "**="]:
        stmt = _parse_single_stmt(f"x {op} 2")
        assert isinstance(stmt, AugAssignStmt)
        assert stmt.op == op[:-1]
        assert isinstance(stmt.target, Name)
        assert stmt.target.identifier == "x"
        assert isinstance(stmt.value, Literal)
        assert stmt.value.value == 2


def test_augassign_extended_ops_parse():
    for op in ["@=", "|=", "&=", "^=", "<<=", ">>="]:
        stmt = _parse_single_stmt(f"x {op} 2")
        assert isinstance(stmt, AugAssignStmt)
        assert stmt.op == op[:-1]


def test_annassign_parses_with_value():
    stmt = _parse_single_stmt("x: int = 1")
    assert isinstance(stmt, AnnAssignStmt)
    assert isinstance(stmt.target, Name)
    assert stmt.target.identifier == "x"
    assert isinstance(stmt.annotation, Name)
    assert stmt.annotation.identifier == "int"
    assert isinstance(stmt.value, Literal)
    assert stmt.value.value == 1


def test_annassign_parses_without_value():
    stmt = _parse_single_stmt("x: int")
    assert isinstance(stmt, AnnAssignStmt)
    assert isinstance(stmt.target, Name)
    assert stmt.target.identifier == "x"
    assert isinstance(stmt.annotation, Name)
    assert stmt.annotation.identifier == "int"
    assert stmt.value is None


def test_type_comment_on_assignment():
    stmt = _parse_single_stmt("x = 1  # type: int")
    assert isinstance(stmt, AssignStmt)
    assert stmt.type_comment == "int"


def test_chain_assignment_parses():
    stmt = _parse_single_stmt("a = b = 3")
    assert isinstance(stmt, AssignStmt)
    assert [target.identifier for target in stmt.targets] == ["a", "b"]
    assert isinstance(stmt.value, Literal)
    assert stmt.value.value == 3


def test_starred_assignment_target_parses():
    stmt = _parse_single_stmt("a, *b, c = 1, 2, 3, 4")
    assert isinstance(stmt, AssignStmt)
    assert len(stmt.targets) == 1
    target = stmt.targets[0]
    assert isinstance(target, TupleLiteral)
    assert isinstance(target.elements[0], Name)
    assert isinstance(target.elements[1], Starred)
    assert isinstance(target.elements[2], Name)
    assert target.elements[0].identifier == "a"
    assert target.elements[1].target.identifier == "b"
    assert target.elements[2].identifier == "c"


def test_class_base_keywords_parse():
    stmt = _parse_single_stmt(
        "class A(metaclass=Foo):\n"
        "    pass\n"
    )
    assert isinstance(stmt, ClassDef)
    assert stmt.bases == []
    assert [(name, value.identifier) for name, value in stmt.keywords] == [
        ("metaclass", "Foo")
    ]


def test_class_type_params_parse():
    stmt = _parse_single_stmt(
        "class Box[T, *Ts, **P]:\n"
        "    pass\n"
    )
    assert isinstance(stmt, ClassDef)
    assert [param.name for param in stmt.type_params] == ["T", "Ts", "P"]
    assert [param.kind for param in stmt.type_params] == ["", "*", "**"]


def test_function_type_params_parse():
    stmt = _parse_single_stmt(
        "def make[T](x):\n"
        "    return x\n"
    )
    assert isinstance(stmt, FunctionDef)
    assert [param.name for param in stmt.type_params] == ["T"]
    assert [param.kind for param in stmt.type_params] == [""]


def test_function_params_parse():
    stmt = _parse_single_stmt(
        "def f(a, /, b=2, *args, c, d=4, **kw):\n"
        "    pass\n"
    )
    assert isinstance(stmt, FunctionDef)
    assert [param.name for param in stmt.params] == ["a", "b", "args", "c", "d", "kw"]
    assert [param.kind for param in stmt.params] == [
        "posonly",
        "poskw",
        "varpos",
        "kwonly",
        "kwonly",
        "varkw",
    ]
    assert isinstance(stmt.params[1].default, Literal)
    assert stmt.params[1].default.value == 2
    assert isinstance(stmt.params[4].default, Literal)
    assert stmt.params[4].default.value == 4


def test_lambda_params_parse():
    expr = parse_expression_source("lambda a, /, b=2, *, c: a + b + c")
    assert isinstance(expr, LambdaExpr)
    assert [param.name for param in expr.params] == ["a", "b", "c"]
    assert [param.kind for param in expr.params] == ["posonly", "poskw", "kwonly"]
    assert isinstance(expr.params[1].default, Literal)
    assert expr.params[1].default.value == 2


def test_conditional_expression_parses():
    expr = parse_expression_source("value if cond else other")
    assert isinstance(expr, ConditionalExpr)
    assert isinstance(expr.test, Name)
    assert expr.test.identifier == "cond"


def test_type_alias_type_params_parse():
    stmt = _parse_single_stmt("type Pair[T] = tuple")
    assert isinstance(stmt, TypeAliasStmt)
    assert [param.name for param in stmt.type_params] == ["T"]


def test_bitwise_ops_parse():
    expr = parse_expression_source("a | b ^ c & d")
    assert isinstance(expr, BinaryOp)
    assert expr.op == "|"
    assert isinstance(expr.left, Name)
    assert expr.left.identifier == "a"
    right = expr.right
    assert isinstance(right, BinaryOp)
    assert right.op == "^"
    assert isinstance(right.left, Name)
    assert right.left.identifier == "b"
    inner = right.right
    assert isinstance(inner, BinaryOp)
    assert inner.op == "&"
    assert isinstance(inner.left, Name)
    assert inner.left.identifier == "c"
    assert isinstance(inner.right, Name)
    assert inner.right.identifier == "d"


def test_shift_ops_parse():
    expr = parse_expression_source("a << b + c")
    assert isinstance(expr, BinaryOp)
    assert expr.op == "<<"
    assert isinstance(expr.left, Name)
    assert expr.left.identifier == "a"
    assert isinstance(expr.right, BinaryOp)
    assert expr.right.op == "+"


def test_matmul_parse():
    expr = parse_expression_source("a @ b * c")
    assert isinstance(expr, BinaryOp)
    assert expr.op == "*"
    assert isinstance(expr.left, BinaryOp)
    assert expr.left.op == "@"
