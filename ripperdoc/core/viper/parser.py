"""Parser for the Viper language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ripperdoc.core.viper.ast_nodes import (
    AssignStmt,
    AugAssignStmt,
    AnnAssignStmt,
    Attribute,
    AssignmentExpr,
    AwaitExpr,
    BinaryOp,
    BreakStmt,
    Call,
    ClassDef,
    CompareOp,
    ContinueStmt,
    DictLiteral,
    DelStmt,
    Expression,
    ExpressionStmt,
    ExceptHandler,
    ForStmt,
    GeneratorExp,
    FunctionDef,
    IfStmt,
    ImportAlias,
    ImportFromStmt,
    ImportStmt,
    ListLiteral,
    ListComp,
    Literal,
    LambdaExpr,
    Module,
    Name,
    NonlocalStmt,
    GlobalStmt,
    AssertStmt,
    MatchCase,
    MatchStmt,
    Pattern,
    PatternLiteral,
    PatternName,
    PatternOr,
    PatternSequence,
    PatternStar,
    PatternWildcard,
    PatternAs,
    PatternValue,
    PatternMapping,
    PatternClass,
    PatternKey,
    ReturnStmt,
    RaiseStmt,
    Statement,
    Subscript,
    SetLiteral,
    SetComp,
    TupleLiteral,
    TypeAliasStmt,
    TypeParam,
    Parameter,
    UnaryOp,
    WhileStmt,
    WithItem,
    WithStmt,
    PassStmt,
    FormattedString,
    StringText,
    TryStmt,
    SliceExpr,
    Starred,
    YieldExpr,
    YieldFromExpr,
    ConditionalExpr,
    DictComp,
    Comprehension,
)
from ripperdoc.core.viper.errors import ViperSyntaxError
from ripperdoc.core.viper.tokenizer import Token


@dataclass
class Parser:
    tokens: List[Token]
    position: int = 0

    def parse(self) -> Module:
        statements: List[Statement] = []
        while not self._at("EOF"):
            if self._match("NEWLINE"):
                continue
            statements.append(self._parse_statement())
        return Module(statements)

    def parse_expression_only(self) -> Expression:
        expr = self._parse_expression()
        if self._match("NEWLINE"):
            pass
        self._expect("EOF")
        return expr

    def _parse_statement(self) -> Statement:
        if self._peek_is_keyword("async"):
            return self._parse_async_statement()
        if self._peek_is_keyword("if"):
            return self._parse_if()
        if self._peek_is_keyword("while"):
            return self._parse_while()
        if self._peek_is_keyword("for"):
            return self._parse_for()
        if self._peek_is_keyword("match"):
            return self._parse_match()
        if self._at("OP", "@"):
            decorators = self._parse_decorators()
            if self._peek_is_keyword("class"):
                return self._parse_class(decorators)
            if self._peek_is_keyword("def"):
                return self._parse_function_def(decorators=decorators)
            if self._peek_is_keyword("async"):
                async_token = self._advance()
                if not self._peek_is_keyword("def"):
                    raise ViperSyntaxError(
                        "Expected 'def' after 'async'", async_token.line, async_token.column
                    )
                return self._parse_function_def(is_async=True, decorators=decorators)
            token = self._current()
            raise ViperSyntaxError(
                "Decorators must precede class or function definitions",
                token.line,
                token.column,
            )
        if self._peek_is_keyword("class"):
            return self._parse_class([])
        if self._peek_is_keyword("def"):
            return self._parse_function_def(decorators=[])
        if self._peek_is_keyword("try"):
            return self._parse_try()
        if self._peek_is_keyword("with"):
            return self._parse_with()
        return self._parse_simple_statement()

    def _parse_simple_statement(self) -> Statement:
        token = self._current()
        if token.kind == "KEYWORD":
            if token.value == "import":
                return self._parse_import_stmt()
            if token.value == "from":
                return self._parse_import_from_stmt()
            if token.value == "global":
                return self._parse_global_stmt()
            if token.value == "nonlocal":
                return self._parse_nonlocal_stmt()
            if token.value == "del":
                return self._parse_del_stmt()
            if token.value == "type":
                return self._parse_type_alias_stmt()
            if token.value == "return":
                self._advance()
                value = None
                if not self._at("NEWLINE"):
                    value = self._parse_expression()
                stmt = ReturnStmt(value=value, line=token.line, column=token.column)
                self._expect("NEWLINE")
                return stmt
            if token.value == "raise":
                self._advance()
                exception = None
                cause = None
                if not self._at("NEWLINE"):
                    exception = self._parse_expression()
                    if self._peek_is_keyword("from"):
                        self._advance()
                        cause = self._parse_expression(allow_tuple=False)
                stmt = RaiseStmt(
                    exception=exception,
                    cause=cause,
                    line=token.line,
                    column=token.column,
                )
                self._expect("NEWLINE")
                return stmt
            if token.value == "assert":
                self._advance()
                test = self._parse_expression(allow_tuple=False)
                message = None
                if self._match("OP", ","):
                    message = self._parse_expression(allow_tuple=False)
                stmt = AssertStmt(
                    test=test,
                    message=message,
                    line=token.line,
                    column=token.column,
                )
                self._expect("NEWLINE")
                return stmt
            if token.value == "pass":
                self._advance()
                stmt = PassStmt(line=token.line, column=token.column)
                self._expect("NEWLINE")
                return stmt
            if token.value == "break":
                self._advance()
                stmt = BreakStmt(line=token.line, column=token.column)
                self._expect("NEWLINE")
                return stmt
            if token.value == "continue":
                self._advance()
                stmt = ContinueStmt(line=token.line, column=token.column)
                self._expect("NEWLINE")
                return stmt
        start_pos = self.position
        target = self._parse_assignment_target_list()
        if target is not None:
            if self._match("OP", ":"):
                if not self._is_simple_target(target):
                    raise ViperSyntaxError("Invalid assignment target", target.line, target.column)
                annotation = self._parse_expression(allow_tuple=False)
                value = None
                if self._match("OP", "="):
                    value = self._parse_assignment_value()
                type_comment = self._parse_type_comment()
                stmt = AnnAssignStmt(
                    target=target,
                    annotation=annotation,
                    value=value,
                    type_comment=type_comment,
                    line=target.line,
                    column=target.column,
                )
                self._expect("NEWLINE")
                return stmt
            augassign = self._match_augassign()
            if augassign is not None:
                if not self._is_simple_target(target):
                    raise ViperSyntaxError("Invalid assignment target", target.line, target.column)
                value = self._parse_expression()
                self._parse_type_comment()
                stmt = AugAssignStmt(
                    target=target,
                    op=augassign,
                    value=value,
                    line=target.line,
                    column=target.column,
                )
                self._expect("NEWLINE")
                return stmt
            if self._match("OP", "="):
                if isinstance(target, Starred) or not self._is_assignable(target):
                    raise ViperSyntaxError("Invalid assignment target", target.line, target.column)
                targets = [target]
                expr = self._parse_expression()
                while self._match("OP", "="):
                    if not self._is_assignable(expr):
                        raise ViperSyntaxError("Invalid assignment target", expr.line, expr.column)
                    targets.append(expr)
                    expr = self._parse_expression()
                value = self._parse_assignment_value_tail(expr)
                type_comment = self._parse_type_comment()
                stmt = AssignStmt(
                    targets=targets,
                    value=value,
                    type_comment=type_comment,
                    line=targets[0].line,
                    column=targets[0].column,
                )
                self._expect("NEWLINE")
                return stmt

        self.position = start_pos
        expr = self._parse_expression()
        stmt = ExpressionStmt(expression=expr, line=expr.line, column=expr.column)
        self._expect("NEWLINE")
        return stmt

    def _parse_async_statement(self) -> Statement:
        token = self._expect("KEYWORD", "async")
        if self._peek_is_keyword("def"):
            return self._parse_function_def(is_async=True)
        if self._peek_is_keyword("for"):
            return self._parse_for(is_async=True)
        if self._peek_is_keyword("with"):
            return self._parse_with(is_async=True)
        raise ViperSyntaxError(
            "Expected 'def', 'for', or 'with' after 'async'", token.line, token.column
        )

    def _parse_import_stmt(self) -> ImportStmt:
        token = self._expect("KEYWORD", "import")
        names: List[ImportAlias] = []
        while True:
            name, line, column = self._parse_dotted_name()
            asname: Optional[str] = None
            if self._peek_is_keyword("as"):
                self._advance()
                as_token = self._expect("NAME")
                asname = as_token.value
            names.append(ImportAlias(name=name, asname=asname, line=line, column=column))
            if not self._match("OP", ","):
                break
        self._expect("NEWLINE")
        return ImportStmt(names=names, line=token.line, column=token.column)

    def _parse_import_from_stmt(self) -> ImportFromStmt:
        token = self._expect("KEYWORD", "from")
        level = 0
        while self._match("OP", "."):
            level += 1
        module: Optional[str] = None
        if self._at("NAME"):
            module, _, _ = self._parse_dotted_name()
        elif level == 0:
            raise ViperSyntaxError("Expected module name in from-import", token.line, token.column)
        if not self._peek_is_keyword("import"):
            raise ViperSyntaxError("Expected 'import' in from-import", token.line, token.column)
        self._advance()
        names: List[ImportAlias] = []
        is_star = False
        if self._match("OP", "*"):
            is_star = True
        else:
            if self._match("OP", "("):
                names = self._parse_import_aliases(end_token=")")
                self._expect("OP", ")")
            else:
                names = self._parse_import_aliases(end_token=None)
        self._expect("NEWLINE")
        return ImportFromStmt(
            module=module,
            names=names,
            level=level,
            is_star=is_star,
            line=token.line,
            column=token.column,
        )

    def _parse_import_aliases(self, end_token: Optional[str]) -> List[ImportAlias]:
        aliases: List[ImportAlias] = []
        while True:
            name_token = self._expect("NAME")
            asname: Optional[str] = None
            if self._peek_is_keyword("as"):
                self._advance()
                as_token = self._expect("NAME")
                asname = as_token.value
            aliases.append(
                ImportAlias(
                    name=name_token.value,
                    asname=asname,
                    line=name_token.line,
                    column=name_token.column,
                )
            )
            if not self._match("OP", ","):
                break
            if end_token is not None and self._at("OP", end_token):
                break
        return aliases

    def _parse_global_stmt(self) -> GlobalStmt:
        token = self._expect("KEYWORD", "global")
        names = [self._expect("NAME").value]
        while self._match("OP", ","):
            names.append(self._expect("NAME").value)
        self._expect("NEWLINE")
        return GlobalStmt(names=names, line=token.line, column=token.column)

    def _parse_nonlocal_stmt(self) -> NonlocalStmt:
        token = self._expect("KEYWORD", "nonlocal")
        names = [self._expect("NAME").value]
        while self._match("OP", ","):
            names.append(self._expect("NAME").value)
        self._expect("NEWLINE")
        return NonlocalStmt(names=names, line=token.line, column=token.column)

    def _parse_del_stmt(self) -> DelStmt:
        token = self._expect("KEYWORD", "del")
        targets = [self._parse_expression(allow_tuple=False)]
        while self._match("OP", ","):
            targets.append(self._parse_expression(allow_tuple=False))
        for target in targets:
            if not self._is_assignable(target):
                raise ViperSyntaxError("Invalid delete target", target.line, target.column)
        self._expect("NEWLINE")
        return DelStmt(targets=targets, line=token.line, column=token.column)

    def _parse_type_alias_stmt(self) -> TypeAliasStmt:
        token = self._expect("KEYWORD", "type")
        name_token = self._expect("NAME")
        type_params: List[TypeParam] = []
        if self._at("OP", "["):
            type_params = self._parse_type_params()
        self._expect("OP", "=")
        value = self._parse_expression()
        self._expect("NEWLINE")
        return TypeAliasStmt(
            name=name_token.value,
            type_params=type_params,
            value=value,
            line=token.line,
            column=token.column,
        )

    def _parse_type_params(self) -> List[TypeParam]:
        params: List[TypeParam] = []
        self._expect("OP", "[")
        if self._match("OP", "]"):
            return params
        while True:
            kind = ""
            if self._match("OP", "**"):
                kind = "**"
            elif self._match("OP", "*"):
                kind = "*"
            name_token = self._expect("NAME")
            bound: Optional[Expression] = None
            default: Optional[Expression] = None
            if self._match("OP", ":"):
                bound = self._parse_expression()
            if self._match("OP", "="):
                default = self._parse_expression()
            params.append(
                TypeParam(
                    name=name_token.value,
                    kind=kind,
                    bound=bound,
                    default=default,
                    line=name_token.line,
                    column=name_token.column,
                )
            )
            if self._match("OP", ","):
                if self._match("OP", "]"):
                    break
                continue
            self._expect("OP", "]")
            break
        return params

    def _parse_parameters(self, end_token: str, *, allow_annotations: bool) -> List[Parameter]:
        params: List[Parameter] = []
        seen_names: set[str] = set()
        seen_kwonly = False
        seen_vararg = False
        seen_varkw = False
        seen_pos_default = False
        seen_posonly = False

        def _mark_posonly() -> None:
            for param in params:
                if param.kind == "poskw":
                    param.kind = "posonly"

        while not self._at("OP", end_token):
            if self._match("OP", "/"):
                if seen_posonly:
                    token = self._current()
                    raise ViperSyntaxError(
                        "Multiple '/' in parameter list", token.line, token.column
                    )
                if not params or seen_kwonly or seen_vararg or seen_varkw:
                    token = self._current()
                    raise ViperSyntaxError(
                        "Invalid positional-only marker", token.line, token.column
                    )
                _mark_posonly()
                seen_posonly = True
                if self._match("OP", ","):
                    if self._at("OP", end_token):
                        break
                    continue
                continue
            if self._match("OP", "**"):
                if seen_varkw:
                    token = self._current()
                    raise ViperSyntaxError(
                        "Multiple '**' in parameter list", token.line, token.column
                    )
                name_token = self._expect("NAME")
                annotation: Optional[Expression] = None
                if allow_annotations and self._match("OP", ":"):
                    annotation = self._parse_expression(allow_tuple=False)
                if name_token.value in seen_names:
                    raise ViperSyntaxError(
                        "Duplicate parameter name", name_token.line, name_token.column
                    )
                seen_names.add(name_token.value)
                params.append(
                    Parameter(
                        name=name_token.value,
                        kind="varkw",
                        annotation=annotation,
                        default=None,
                        line=name_token.line,
                        column=name_token.column,
                    )
                )
                seen_varkw = True
                if self._match("OP", ",") and not self._at("OP", end_token):
                    raise ViperSyntaxError(
                        "Parameters after '**' are not allowed",
                        name_token.line,
                        name_token.column,
                    )
                break
            if self._match("OP", "*"):
                if seen_vararg or seen_varkw:
                    token = self._current()
                    raise ViperSyntaxError(
                        "Multiple '*' in parameter list", token.line, token.column
                    )
                if self._at("OP", end_token):
                    token = self._current()
                    raise ViperSyntaxError(
                        "Expected parameter name after '*'", token.line, token.column
                    )
                if self._at("OP", ","):
                    self._advance()
                    seen_kwonly = True
                    if self._at("OP", end_token):
                        token = self._current()
                        raise ViperSyntaxError(
                            "Expected parameter name after '*'", token.line, token.column
                        )
                    continue
                name_token = self._expect("NAME")
                annotation: Optional[Expression] = None
                if allow_annotations and self._match("OP", ":"):
                    annotation = self._parse_expression(allow_tuple=False)
                if name_token.value in seen_names:
                    raise ViperSyntaxError(
                        "Duplicate parameter name", name_token.line, name_token.column
                    )
                seen_names.add(name_token.value)
                params.append(
                    Parameter(
                        name=name_token.value,
                        kind="varpos",
                        annotation=annotation,
                        default=None,
                        line=name_token.line,
                        column=name_token.column,
                    )
                )
                seen_vararg = True
                seen_kwonly = True
                if self._match("OP", ","):
                    if self._at("OP", end_token):
                        break
                    continue
                break
            name_token = self._expect("NAME")
            annotation: Optional[Expression] = None
            if allow_annotations and self._match("OP", ":"):
                annotation = self._parse_expression(allow_tuple=False)
            if name_token.value in seen_names:
                raise ViperSyntaxError(
                    "Duplicate parameter name", name_token.line, name_token.column
                )
            seen_names.add(name_token.value)
            kind = "kwonly" if seen_kwonly else "poskw"
            default: Optional[Expression] = None
            if self._match("OP", "="):
                default = self._parse_expression(allow_tuple=False)
                if kind in {"poskw", "posonly"}:
                    seen_pos_default = True
            else:
                if kind in {"poskw", "posonly"} and seen_pos_default:
                    raise ViperSyntaxError(
                        "Non-default argument follows default argument",
                        name_token.line,
                        name_token.column,
                    )
            params.append(
                Parameter(
                    name=name_token.value,
                    kind=kind,
                    annotation=annotation,
                    default=default,
                    line=name_token.line,
                    column=name_token.column,
                )
            )
            if self._match("OP", ","):
                if self._at("OP", end_token):
                    break
                continue
            if not self._at("OP", end_token):
                token = self._current()
                raise ViperSyntaxError(
                    f"Expected ',' or '{end_token}'", token.line, token.column
                )
        return params

    def _parse_dotted_name(self) -> tuple[str, int, int]:
        first = self._expect("NAME")
        parts = [first.value]
        while self._match("OP", "."):
            part = self._expect("NAME")
            parts.append(part.value)
        return ".".join(parts), first.line, first.column

    def _parse_match(self) -> MatchStmt:
        token = self._expect("KEYWORD", "match")
        subject = self._parse_expression()
        self._expect("OP", ":")
        self._expect("NEWLINE")
        self._expect("INDENT")
        cases: List[MatchCase] = []
        while not self._at("DEDENT"):
            if self._match("NEWLINE"):
                continue
            case_token = self._expect("KEYWORD", "case")
            patterns = self._parse_case_patterns()
            guard: Optional[Expression] = None
            if self._peek_is_keyword("if"):
                self._advance()
                guard = self._parse_expression()
            self._expect("OP", ":")
            body = self._parse_block()
            cases.append(
                MatchCase(
                    patterns=patterns,
                    guard=guard,
                    body=body,
                    line=case_token.line,
                    column=case_token.column,
                )
            )
        self._expect("DEDENT")
        return MatchStmt(subject=subject, cases=cases, line=token.line, column=token.column)

    def _parse_case_patterns(self) -> List[Pattern]:
        start_pos = self.position
        first = self._parse_sequence_pattern_element()
        if self._match("OP", ","):
            elements = [first]
            while True:
                if self._at("OP", ":") or self._peek_is_keyword("if"):
                    break
                elements.append(self._parse_sequence_pattern_element())
                if not self._match("OP", ","):
                    break
            if self._at("OP", "|"):
                token = self._current()
                raise ViperSyntaxError(
                    "Or-pattern cannot follow a sequence pattern",
                    token.line,
                    token.column,
                )
            return [self._build_sequence_pattern(elements, first.line, first.column)]
        self.position = start_pos
        pattern = self._parse_pattern()
        if isinstance(pattern, PatternOr):
            return pattern.patterns
        return [pattern]

    def _parse_open_sequence_pattern(self) -> Pattern:
        first = self._parse_sequence_pattern_element()
        if not self._match("OP", ","):
            if isinstance(first, PatternStar):
                raise ViperSyntaxError(
                    "Starred pattern must be part of a sequence", first.line, first.column
                )
            return first
        elements = [first]
        while True:
            if self._at("OP", ":") or self._peek_is_keyword("if"):
                break
            elements.append(self._parse_sequence_pattern_element())
            if not self._match("OP", ","):
                break
        return self._build_sequence_pattern(elements, first.line, first.column)

    def _parse_sequence_pattern_element(self) -> Pattern:
        if self._at("OP", "*"):
            token = self._advance()
            target = self._parse_pattern_capture_target()
            return PatternStar(target=target, line=token.line, column=token.column)
        return self._parse_pattern()

    def _parse_pattern_capture_target(self) -> Pattern:
        token = self._current()
        if token.kind != "NAME":
            raise ViperSyntaxError("Expected capture target", token.line, token.column)
        self._advance()
        if token.value == "_":
            return PatternWildcard(line=token.line, column=token.column)
        return PatternName(identifier=token.value, line=token.line, column=token.column)

    def _build_sequence_pattern(self, elements: List[Pattern], line: int, column: int) -> PatternSequence:
        star_count = sum(1 for element in elements if isinstance(element, PatternStar))
        if star_count > 1:
            raise ViperSyntaxError("Multiple starred patterns are not allowed", line, column)
        self._validate_pattern_unique_names(elements, line, column)
        return PatternSequence(elements=elements, line=line, column=column)

    def _parse_mapping_key(self) -> PatternKey:
        token = self._current()
        if token.kind == "OP" and token.value in {"+", "-"} and self._peek_next_is("NUMBER"):
            sign = token.value
            self._advance()
            number = self._advance()
            value = self._parse_number_value(f"{sign}{number.value}", number)
            return PatternLiteral(value=value, line=token.line, column=token.column)
        if token.kind == "NUMBER":
            self._advance()
            value = self._parse_number_value(str(token.value), token)
            return PatternLiteral(value=value, line=token.line, column=token.column)
        if token.kind in {"STRING", "BYTES", "FSTRING", "BFSTRING", "TSTRING", "BTSTRING"}:
            expr = self._parse_string_sequence()
            if not isinstance(expr, Literal):
                raise ViperSyntaxError(
                    "Formatted strings are not allowed in mapping keys", token.line, token.column
                )
            return PatternLiteral(value=expr.value, line=expr.line, column=expr.column)
        if token.kind == "KEYWORD" and token.value in {"True", "False", "None"}:
            self._advance()
            value = {"True": True, "False": False, "None": None}[token.value]
            return PatternLiteral(value=value, line=token.line, column=token.column)
        if token.kind == "NAME":
            value_pattern = self._parse_pattern_value()
            if len(value_pattern.parts) < 2:
                raise ViperSyntaxError(
                    "Mapping keys must be literal or dotted names",
                    value_pattern.line,
                    value_pattern.column,
                )
            return value_pattern
        raise ViperSyntaxError("Invalid mapping key pattern", token.line, token.column)

    def _parse_pattern_value(self) -> PatternValue:
        first = self._expect("NAME")
        if first.value == "_" and self._at("OP", "."):
            raise ViperSyntaxError("Invalid wildcard value pattern", first.line, first.column)
        parts = [first.value]
        while self._match("OP", "."):
            part = self._expect("NAME")
            parts.append(part.value)
        return PatternValue(parts=parts, line=first.line, column=first.column)

    def _parse_class_pattern(self, class_path: PatternValue) -> PatternClass:
        self._expect("OP", "(")
        if self._match("OP", ")"):
            return PatternClass(
                class_path=class_path,
                positional=[],
                keywords=[],
                line=class_path.line,
                column=class_path.column,
            )
        positional: List[Pattern] = []
        keywords: List[Tuple[str, Pattern]] = []
        seen_keyword = False
        seen_names: set[str] = set()
        while True:
            if self._at("NAME") and self._peek_next_is("OP", "="):
                name_token = self._advance()
                self._advance()
                if name_token.value in seen_names:
                    raise ViperSyntaxError(
                        "Duplicate keyword in class pattern", name_token.line, name_token.column
                    )
                seen_names.add(name_token.value)
                seen_keyword = True
                keyword_pattern = self._parse_pattern()
                keywords.append((name_token.value, keyword_pattern))
            else:
                if seen_keyword:
                    token = self._current()
                    raise ViperSyntaxError(
                        "Positional pattern follows keyword pattern",
                        token.line,
                        token.column,
                    )
                positional.append(self._parse_pattern())
            if self._match("OP", ","):
                if self._match("OP", ")"):
                    break
                continue
            self._expect("OP", ")")
            break
        all_patterns = positional + [pattern for _, pattern in keywords]
        self._validate_pattern_unique_names(all_patterns, class_path.line, class_path.column)
        return PatternClass(
            class_path=class_path,
            positional=positional,
            keywords=keywords,
            line=class_path.line,
            column=class_path.column,
        )

    def _parse_mapping_pattern(self, line: int, column: int) -> PatternMapping:
        items: List[Tuple[PatternKey, Pattern]] = []
        rest: Optional[Pattern] = None
        seen_keys: set[tuple[str, object]] = set()
        if self._match("OP", "}"):
            return PatternMapping(items=items, rest=rest, line=line, column=column)
        while True:
            if self._at("OP", "**"):
                if rest is not None:
                    raise ViperSyntaxError("Multiple mapping rest patterns", line, column)
                self._advance()
                rest = self._parse_pattern_capture_target()
                if self._match("OP", ","):
                    self._expect("OP", "}")
                else:
                    self._expect("OP", "}")
                break
            key = self._parse_mapping_key()
            key_id: Optional[tuple[str, object]] = None
            if isinstance(key, PatternLiteral):
                key_id = ("literal", key.value)
            elif isinstance(key, PatternValue):
                key_id = ("value", tuple(key.parts))
            if key_id is not None:
                if key_id in seen_keys:
                    raise ViperSyntaxError("Duplicate mapping pattern key", line, column)
                seen_keys.add(key_id)
            self._expect("OP", ":")
            value = self._parse_pattern()
            items.append((key, value))
            if self._match("OP", ","):
                if self._match("OP", "}"):
                    break
                continue
            self._expect("OP", "}")
            break
        patterns_to_check = [value for _, value in items]
        if rest is not None:
            patterns_to_check.append(rest)
        self._validate_pattern_unique_names(patterns_to_check, line, column)
        return PatternMapping(items=items, rest=rest, line=line, column=column)

    def _parse_pattern(self) -> Pattern:
        pattern = self._parse_or_pattern()
        if self._peek_is_keyword("as"):
            self._advance()
            target = self._parse_pattern_capture_target()
            if isinstance(target, PatternWildcard):
                raise ViperSyntaxError("Cannot bind wildcard in as-pattern", target.line, target.column)
            if target.identifier in self._collect_pattern_names(pattern):
                raise ViperSyntaxError(
                    "Multiple assignments to the same name in pattern",
                    target.line,
                    target.column,
                )
            return PatternAs(pattern=pattern, name=target.identifier, line=pattern.line, column=pattern.column)
        return pattern

    def _parse_or_pattern(self) -> Pattern:
        patterns = [self._parse_closed_pattern()]
        while self._match("OP", "|"):
            patterns.append(self._parse_closed_pattern())
        if len(patterns) == 1:
            return patterns[0]
        self._validate_or_pattern_bindings(patterns)
        return PatternOr(patterns=patterns, line=patterns[0].line, column=patterns[0].column)

    def _parse_closed_pattern(self) -> Pattern:
        token = self._current()
        if token.kind == "OP" and token.value in {"+", "-"} and self._peek_next_is("NUMBER"):
            sign = token.value
            self._advance()
            number = self._advance()
            value = self._parse_number_value(f"{sign}{number.value}", number)
            return PatternLiteral(value=value, line=token.line, column=token.column)
        if token.kind == "NAME":
            name_token = token
            class_path = self._parse_pattern_value()
            if class_path.parts == ["_"] and self._at("OP", "("):
                raise ViperSyntaxError(
                    "Invalid class pattern name", name_token.line, name_token.column
                )
            if self._at("OP", "("):
                return self._parse_class_pattern(class_path)
            if len(class_path.parts) > 1:
                return class_path
            if name_token.value == "_":
                return PatternWildcard(line=name_token.line, column=name_token.column)
            return PatternName(identifier=name_token.value, line=name_token.line, column=name_token.column)
        if token.kind == "NUMBER":
            self._advance()
            value = self._parse_number_value(str(token.value), token)
            return PatternLiteral(value=value, line=token.line, column=token.column)
        if token.kind in {"STRING", "BYTES", "FSTRING", "BFSTRING", "TSTRING", "BTSTRING"}:
            expr = self._parse_string_sequence()
            if not isinstance(expr, Literal):
                raise ViperSyntaxError("Formatted strings are not allowed in patterns", token.line, token.column)
            return PatternLiteral(value=expr.value, line=expr.line, column=expr.column)
        if token.kind == "KEYWORD" and token.value in {"True", "False", "None"}:
            self._advance()
            value = {"True": True, "False": False, "None": None}[token.value]
            return PatternLiteral(value=value, line=token.line, column=token.column)
        if self._match("OP", "{"):
            return self._parse_mapping_pattern(token.line, token.column)
        if self._match("OP", "["):
            if self._match("OP", "]"):
                return PatternSequence(elements=[], line=token.line, column=token.column)
            elements = [self._parse_sequence_pattern_element()]
            while True:
                if self._match("OP", ","):
                    if self._match("OP", "]"):
                        break
                    elements.append(self._parse_sequence_pattern_element())
                    continue
                self._expect("OP", "]")
                break
            return self._build_sequence_pattern(elements, token.line, token.column)
        if self._match("OP", "("):
            if self._match("OP", ")"):
                return PatternSequence(elements=[], line=token.line, column=token.column)
            first = self._parse_sequence_pattern_element()
            if self._match("OP", ","):
                elements = [first]
                if self._match("OP", ")"):
                    return self._build_sequence_pattern(elements, token.line, token.column)
                while True:
                    elements.append(self._parse_sequence_pattern_element())
                    if self._match("OP", ","):
                        if self._match("OP", ")"):
                            break
                        continue
                    self._expect("OP", ")")
                    break
                return self._build_sequence_pattern(elements, token.line, token.column)
            if isinstance(first, PatternStar):
                raise ViperSyntaxError(
                    "Starred pattern must be part of a sequence", first.line, first.column
                )
            self._expect("OP", ")")
            return first
        raise ViperSyntaxError("Invalid match pattern", token.line, token.column)

    def _parse_try(self) -> TryStmt:
        token = self._expect("KEYWORD", "try")
        self._expect("OP", ":")
        body = self._parse_block()
        handlers: List[ExceptHandler] = []
        else_body: Optional[List[Statement]] = None
        finally_body: Optional[List[Statement]] = None
        if self._peek_is_keyword("finally"):
            self._advance()
            self._expect("OP", ":")
            finally_body = self._parse_block()
            return TryStmt(
                body=body,
                handlers=handlers,
                else_body=None,
                finally_body=finally_body,
                line=token.line,
                column=token.column,
            )
        if not self._peek_is_keyword("except"):
            raise ViperSyntaxError("Expected 'except' or 'finally' in try statement", token.line, token.column)
        while self._peek_is_keyword("except"):
            handlers.append(self._parse_except_handler())
        if self._peek_is_keyword("else"):
            self._advance()
            self._expect("OP", ":")
            else_body = self._parse_block()
        if self._peek_is_keyword("finally"):
            self._advance()
            self._expect("OP", ":")
            finally_body = self._parse_block()
        return TryStmt(
            body=body,
            handlers=handlers,
            else_body=else_body,
            finally_body=finally_body,
            line=token.line,
            column=token.column,
        )

    def _parse_except_handler(self) -> ExceptHandler:
        token = self._expect("KEYWORD", "except")
        if self._at("OP", ":"):
            self._advance()
            body = self._parse_block()
            return ExceptHandler(
                type=None,
                name=None,
                body=body,
                line=token.line,
                column=token.column,
            )
        exc_type = self._parse_expression(allow_tuple=False)
        exprs = [exc_type]
        while self._match("OP", ","):
            if self._peek_is_keyword("as") or self._at("OP", ":"):
                raise ViperSyntaxError("Expected exception type", token.line, token.column)
            exprs.append(self._parse_expression(allow_tuple=False))
        if len(exprs) > 1:
            exc_type = TupleLiteral(
                elements=exprs,
                line=exprs[0].line,
                column=exprs[0].column,
            )
        name: Optional[str] = None
        if self._peek_is_keyword("as"):
            self._advance()
            name_token = self._expect("NAME")
            name = name_token.value
        self._expect("OP", ":")
        body = self._parse_block()
        return ExceptHandler(
            type=exc_type,
            name=name,
            body=body,
            line=token.line,
            column=token.column,
        )

    def _parse_if(self) -> IfStmt:
        token = self._expect("KEYWORD", "if")
        test = self._parse_expression()
        self._expect("OP", ":")
        body = self._parse_block()
        elif_blocks: List[Tuple[Expression, List[Statement]]] = []
        while self._peek_is_keyword("elif"):
            self._advance()
            elif_test = self._parse_expression()
            self._expect("OP", ":")
            elif_body = self._parse_block()
            elif_blocks.append((elif_test, elif_body))
        else_body: Optional[List[Statement]] = None
        if self._peek_is_keyword("else"):
            self._advance()
            self._expect("OP", ":")
            else_body = self._parse_block()
        return IfStmt(
            test=test,
            body=body,
            elif_blocks=elif_blocks,
            else_body=else_body,
            line=token.line,
            column=token.column,
        )

    def _parse_while(self) -> WhileStmt:
        token = self._expect("KEYWORD", "while")
        test = self._parse_expression()
        self._expect("OP", ":")
        body = self._parse_block()
        else_body: Optional[List[Statement]] = None
        if self._peek_is_keyword("else"):
            self._advance()
            self._expect("OP", ":")
            else_body = self._parse_block()
        return WhileStmt(
            test=test,
            body=body,
            else_body=else_body,
            line=token.line,
            column=token.column,
        )

    def _parse_for(self, *, is_async: bool = False) -> ForStmt:
        token = self._expect("KEYWORD", "for")
        target = self._parse_for_target()
        if not self._peek_is_keyword("in"):
            raise ViperSyntaxError("Expected 'in' in for statement", token.line, token.column)
        self._advance()
        iterable = self._parse_expression()
        self._expect("OP", ":")
        body = self._parse_block()
        else_body: Optional[List[Statement]] = None
        if self._peek_is_keyword("else"):
            self._advance()
            self._expect("OP", ":")
            else_body = self._parse_block()
        return ForStmt(
            target=target,
            iterable=iterable,
            body=body,
            else_body=else_body,
            is_async=is_async,
            line=token.line,
            column=token.column,
        )

    def _parse_for_target(self) -> Expression:
        first = self._parse_target_item()
        if not self._match("OP", ","):
            return first
        elements = [first]
        while True:
            if self._peek_is_keyword("in"):
                break
            elements.append(self._parse_target_item())
            if not self._match("OP", ","):
                break
        return TupleLiteral(elements=elements, line=first.line, column=first.column)

    def _parse_target_item(self) -> Expression:
        return self._parse_postfix()

    def _parse_decorators(self) -> List[Expression]:
        decorators: List[Expression] = []
        while self._match("OP", "@"):
            decorators.append(self._parse_expression(allow_tuple=False))
            self._expect("NEWLINE")
        return decorators

    def _parse_class(self, decorators: List[Expression]) -> ClassDef:
        token = self._expect("KEYWORD", "class")
        name_token = self._expect("NAME")
        type_params: List[TypeParam] = []
        if self._at("OP", "["):
            type_params = self._parse_type_params()
        bases: List[Expression] = []
        keywords: List[Tuple[str, Expression]] = []
        if self._match("OP", "("):
            args, kwargs = self._parse_call_args()
            for arg in args:
                if isinstance(arg, Starred):
                    raise ViperSyntaxError("Starred bases are not supported", arg.line, arg.column)
            for name, value in kwargs:
                if name is None:
                    raise ViperSyntaxError("Unpacking class keywords is not supported", value.line, value.column)
            bases = args
            keywords = kwargs
        self._expect("OP", ":")
        body = self._parse_block()
        return ClassDef(
            name=name_token.value,
            bases=bases,
            keywords=keywords,
            type_params=type_params,
            body=body,
            decorators=decorators,
            line=token.line,
            column=token.column,
        )

    def _parse_function_def(
        self, *, is_async: bool = False, decorators: Optional[List[Expression]] = None
    ) -> FunctionDef:
        token = self._expect("KEYWORD", "def")
        name_token = self._expect("NAME")
        type_params: List[TypeParam] = []
        if self._at("OP", "["):
            type_params = self._parse_type_params()
        self._expect("OP", "(")
        params = (
            self._parse_parameters(")", allow_annotations=True) if not self._at("OP", ")") else []
        )
        self._expect("OP", ")")
        returns: Optional[Expression] = None
        if self._match("OP", "->"):
            returns = self._parse_expression(allow_tuple=False)
        self._expect("OP", ":")
        self._parse_type_comment()
        body = self._parse_block()
        return FunctionDef(
            name=name_token.value,
            params=params,
            type_params=type_params,
            body=body,
            is_async=is_async,
            line=token.line,
            column=token.column,
            decorators=decorators or [],
            returns=returns,
        )

    def _parse_with(self, *, is_async: bool = False) -> WithStmt:
        token = self._expect("KEYWORD", "with")
        if self._match("OP", "("):
            items = self._parse_with_items(end_token=")")
            self._expect("OP", ")")
        else:
            items = self._parse_with_items(end_token=":")
        self._expect("OP", ":")
        body = self._parse_block()
        return WithStmt(
            items=items,
            body=body,
            is_async=is_async,
            line=token.line,
            column=token.column,
        )

    def _parse_with_items(self, end_token: str) -> List[WithItem]:
        items: List[WithItem] = []
        while True:
            items.append(self._parse_with_item())
            if self._match("OP", ","):
                if end_token == ")" and self._at("OP", ")"):
                    break
                if end_token == ":" and self._at("OP", ":"):
                    break
                continue
            break
        return items

    def _parse_with_item(self) -> WithItem:
        expr = self._parse_expression(allow_tuple=False)
        target: Optional[Expression] = None
        if self._peek_is_keyword("as"):
            self._advance()
            name_token = self._expect("NAME")
            target = Name(
                identifier=name_token.value,
                line=name_token.line,
                column=name_token.column,
            )
        return WithItem(context_expr=expr, target=target, line=expr.line, column=expr.column)

    def _parse_block(self) -> List[Statement]:
        self._expect("NEWLINE")
        self._expect("INDENT")
        statements: List[Statement] = []
        while not self._at("DEDENT"):
            if self._match("NEWLINE"):
                continue
            statements.append(self._parse_statement())
        self._expect("DEDENT")
        return statements

    def _parse_expression(self, *, allow_tuple: bool = True) -> Expression:
        if self._peek_is_keyword("lambda"):
            return self._parse_lambda()
        if self._peek_is_keyword("yield"):
            return self._parse_yield()
        expr = self._parse_named_expression()
        if not allow_tuple:
            return expr
        if not self._match("OP", ","):
            return expr
        elements = [expr]
        while True:
            if self._at("NEWLINE") or self._at("DEDENT") or self._at("OP", ")") or self._at("OP", "]") or self._at("OP", "}"):
                break
            elements.append(self._parse_expression(allow_tuple=False))
            if not self._match("OP", ","):
                break
        return TupleLiteral(elements=elements, line=elements[0].line, column=elements[0].column)

    def _parse_lambda(self) -> Expression:
        token = self._expect("KEYWORD", "lambda")
        params = (
            self._parse_parameters(":", allow_annotations=False) if not self._at("OP", ":") else []
        )
        self._expect("OP", ":")
        body = self._parse_expression(allow_tuple=False)
        return LambdaExpr(params=params, body=body, line=token.line, column=token.column)

    def _parse_named_expression(self) -> Expression:
        if self._at("NAME") and self._peek_next_is("OP", ":="):
            name_token = self._advance()
            self._advance()
            value = self._parse_expression(allow_tuple=False)
            return AssignmentExpr(
                target=Name(identifier=name_token.value, line=name_token.line, column=name_token.column),
                value=value,
                line=name_token.line,
                column=name_token.column,
            )
        return self._parse_conditional()

    def _parse_conditional(self) -> Expression:
        expr = self._parse_or()
        if not self._peek_is_keyword("if"):
            return expr
        self._advance()
        test = self._parse_or()
        if not self._peek_is_keyword("else"):
            token = self._current()
            raise ViperSyntaxError("Expected 'else' in conditional expression", token.line, token.column)
        self._advance()
        orelse = self._parse_expression(allow_tuple=False)
        return ConditionalExpr(
            test=test,
            body=expr,
            orelse=orelse,
            line=expr.line,
            column=expr.column,
        )

    def _parse_yield(self) -> Expression:
        token = self._expect("KEYWORD", "yield")
        if self._peek_is_keyword("from"):
            self._advance()
            value = self._parse_expression(allow_tuple=False)
            return YieldFromExpr(value=value, line=token.line, column=token.column)
        if self._at("NEWLINE") or self._at("OP", ")") or self._at("OP", "]") or self._at("OP", "}") or self._at("DEDENT"):
            return YieldExpr(value=None, line=token.line, column=token.column)
        expr = self._parse_expression(allow_tuple=False)
        if not self._match("OP", ","):
            return YieldExpr(value=expr, line=token.line, column=token.column)
        elements = [expr]
        while True:
            if self._at("NEWLINE") or self._at("OP", ")") or self._at("OP", "]") or self._at("OP", "}") or self._at("DEDENT"):
                break
            elements.append(self._parse_expression(allow_tuple=False))
            if not self._match("OP", ","):
                break
        return YieldExpr(
            value=TupleLiteral(elements=elements, line=elements[0].line, column=elements[0].column),
            line=token.line,
            column=token.column,
        )

    def _parse_or(self) -> Expression:
        expr = self._parse_and()
        while self._peek_is_keyword("or"):
            token = self._advance()
            right = self._parse_and()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_and(self) -> Expression:
        expr = self._parse_not()
        while self._peek_is_keyword("and"):
            token = self._advance()
            right = self._parse_not()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_not(self) -> Expression:
        if self._peek_is_keyword("not"):
            token = self._advance()
            operand = self._parse_not()
            return UnaryOp(op=token.value, operand=operand, line=token.line, column=token.column)
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        expr = self._parse_bitwise_or()
        ops: List[str] = []
        comparators: List[Expression] = []
        while True:
            op = self._match_comparison_op()
            if op is None:
                break
            right = self._parse_bitwise_or()
            ops.append(op)
            comparators.append(right)
        if ops:
            return CompareOp(left=expr, ops=ops, comparators=comparators, line=expr.line, column=expr.column)
        return expr

    def _parse_bitwise_or(self) -> Expression:
        expr = self._parse_bitwise_xor()
        while self._at("OP") and self._current().value == "|":
            token = self._advance()
            right = self._parse_bitwise_xor()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_bitwise_xor(self) -> Expression:
        expr = self._parse_bitwise_and()
        while self._at("OP") and self._current().value == "^":
            token = self._advance()
            right = self._parse_bitwise_and()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_bitwise_and(self) -> Expression:
        expr = self._parse_shift_expr()
        while self._at("OP") and self._current().value == "&":
            token = self._advance()
            right = self._parse_shift_expr()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_shift_expr(self) -> Expression:
        expr = self._parse_term()
        while self._at("OP") and self._current().value in {"<<", ">>"}:
            token = self._advance()
            right = self._parse_term()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_term(self) -> Expression:
        expr = self._parse_factor()
        while self._at("OP") and self._current().value in {"+", "-"}:
            token = self._advance()
            right = self._parse_factor()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_factor(self) -> Expression:
        expr = self._parse_unary()
        while self._at("OP") and self._current().value in {"*", "/", "//", "%", "@"}:
            token = self._advance()
            right = self._parse_unary()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_unary(self) -> Expression:
        if self._peek_is_keyword("await"):
            token = self._advance()
            operand = self._parse_unary()
            return AwaitExpr(value=operand, line=token.line, column=token.column)
        if self._at("OP") and self._current().value in {"+", "-", "~"}:
            token = self._advance()
            operand = self._parse_unary()
            return UnaryOp(op=token.value, operand=operand, line=token.line, column=token.column)
        return self._parse_power()

    def _parse_power(self) -> Expression:
        expr = self._parse_postfix()
        if self._at("OP", "**"):
            token = self._advance()
            right = self._parse_unary()
            return BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_postfix(self) -> Expression:
        expr = self._parse_atom()
        while True:
            if self._match("OP", "("):
                args, kwargs = self._parse_call_args()
                expr = Call(func=expr, args=args, kwargs=kwargs, line=expr.line, column=expr.column)
                continue
            if self._match("OP", "."):
                name_token = self._expect("NAME")
                expr = Attribute(value=expr, name=name_token.value, line=name_token.line, column=name_token.column)
                continue
            if self._match("OP", "["):
                index = self._parse_subscript()
                self._expect("OP", "]")
                expr = Subscript(value=expr, index=index, line=expr.line, column=expr.column)
                continue
            break
        return expr

    def _parse_subscript(self) -> Expression:
        items = [self._parse_slice_or_expr()]
        if self._match("OP", ","):
            while True:
                if self._at("OP", "]"):
                    break
                items.append(self._parse_slice_or_expr())
                if not self._match("OP", ","):
                    break
            return TupleLiteral(elements=items, line=items[0].line, column=items[0].column)
        return items[0]

    def _parse_slice_or_expr(self) -> Expression:
        if self._at("OP", ":"):
            colon = self._expect("OP", ":")
            return self._parse_slice_tail(None, colon.line, colon.column)
        expr = self._parse_expression(allow_tuple=False)
        if self._match("OP", ":"):
            return self._parse_slice_tail(expr, expr.line, expr.column)
        return expr

    def _parse_slice_tail(
        self, start: Optional[Expression], line: int, column: int
    ) -> SliceExpr:
        stop: Optional[Expression] = None
        step: Optional[Expression] = None
        if not self._at("OP", "]") and not self._at("OP", ",") and not self._at("OP", ":"):
            stop = self._parse_expression(allow_tuple=False)
        if self._match("OP", ":"):
            if not self._at("OP", "]") and not self._at("OP", ","):
                step = self._parse_expression(allow_tuple=False)
        return SliceExpr(start=start, stop=stop, step=step, line=line, column=column)

    def _parse_comprehension_clauses(self) -> List[Comprehension]:
        clauses: List[Comprehension] = []
        while True:
            is_async = False
            async_token = None
            if self._peek_is_keyword("async"):
                async_token = self._advance()
                is_async = True
            if not self._peek_is_keyword("for"):
                if is_async and async_token is not None:
                    raise ViperSyntaxError(
                        "Expected 'for' after 'async' in comprehension",
                        async_token.line,
                        async_token.column,
                    )
                break
            for_token = self._advance()
            target = self._parse_for_target()
            if not self._peek_is_keyword("in"):
                raise ViperSyntaxError("Expected 'in' in comprehension", for_token.line, for_token.column)
            self._advance()
            iterable = self._parse_or()
            ifs: List[Expression] = []
            while self._peek_is_keyword("if"):
                self._advance()
                ifs.append(self._parse_or())
            clauses.append(
                Comprehension(
                    target=target,
                    iterable=iterable,
                    ifs=ifs,
                    is_async=is_async,
                    line=for_token.line,
                    column=for_token.column,
                )
            )
            if not self._peek_is_keyword("for") and not self._peek_is_keyword("async"):
                break
        if not clauses:
            token = self._current()
            raise ViperSyntaxError("Expected comprehension clause", token.line, token.column)
        return clauses

    def _parse_call_args(self) -> Tuple[List[Expression], List[Tuple[Optional[str], Expression]]]:
        args: List[Expression] = []
        kwargs: List[Tuple[Optional[str], Expression]] = []
        saw_kwarg = False
        if self._match("OP", ")"):
            return args, kwargs
        while True:
            if self._at("OP", "**"):
                token = self._advance()
                value = self._parse_expression(allow_tuple=False)
                kwargs.append((None, value))
                saw_kwarg = True
            elif self._at("OP", "*"):
                token = self._advance()
                if saw_kwarg:
                    raise ViperSyntaxError(
                        "Positional argument follows keyword argument",
                        token.line,
                        token.column,
                    )
                value = self._parse_expression(allow_tuple=False)
                args.append(Starred(target=value, line=token.line, column=token.column))
            elif self._at("NAME") and self._peek_next_is("OP", "="):
                name_token = self._advance()
                self._expect("OP", "=")
                value = self._parse_expression(allow_tuple=False)
                kwargs.append((name_token.value, value))
                saw_kwarg = True
            else:
                if saw_kwarg:
                    token = self._current()
                    raise ViperSyntaxError(
                        "Positional argument follows keyword argument",
                        token.line,
                        token.column,
                    )
                expr = self._parse_expression(allow_tuple=False)
                if self._peek_is_keyword("for") or self._peek_is_keyword("async"):
                    generators = self._parse_comprehension_clauses()
                    expr = GeneratorExp(
                        element=expr,
                        generators=generators,
                        line=expr.line,
                        column=expr.column,
                    )
                args.append(expr)
            if self._match("OP", ","):
                if self._match("OP", ")"):
                    break
                continue
            self._expect("OP", ")")
            break
        return args, kwargs

    def _parse_star_expression(self) -> Expression:
        if self._at("OP", "*"):
            token = self._advance()
            value = self._parse_expression(allow_tuple=False)
            return Starred(target=value, line=token.line, column=token.column)
        return self._parse_expression(allow_tuple=False)

    def _parse_atom(self) -> Expression:
        token = self._current()
        if token.kind in {"STRING", "BYTES", "FSTRING", "BFSTRING", "TSTRING", "BTSTRING"}:
            return self._parse_string_sequence()
        if token.kind == "ELLIPSIS":
            self._advance()
            return Literal(value=Ellipsis, line=token.line, column=token.column)
        if token.kind == "NAME":
            self._advance()
            return Name(identifier=token.value, line=token.line, column=token.column)
        if token.kind == "NUMBER":
            self._advance()
            raw = str(token.value)
            normalized = raw.replace("_", "")
            value: float | int | complex
            try:
                if normalized.endswith(("j", "J")):
                    value = complex(normalized)
                elif any(ch in normalized for ch in ".eE"):
                    value = float(normalized)
                elif normalized.startswith(("0x", "0X", "0b", "0B", "0o", "0O")):
                    value = int(normalized, 0)
                else:
                    value = int(normalized, 10)
            except ValueError as exc:
                raise ViperSyntaxError("Invalid numeric literal", token.line, token.column) from exc
            return Literal(value=value, line=token.line, column=token.column)
        if token.kind == "STRING":
            self._advance()
            return Literal(value=token.value, line=token.line, column=token.column)
        if token.kind == "KEYWORD" and token.value in {"True", "False", "None"}:
            self._advance()
            value = {"True": True, "False": False, "None": None}[token.value]
            return Literal(value=value, line=token.line, column=token.column)
        if self._match("OP", "("):
            if self._match("OP", ")"):
                return TupleLiteral(elements=[], line=token.line, column=token.column)
            expr = self._parse_star_expression()
            if not isinstance(expr, Starred) and (
                self._peek_is_keyword("for") or self._peek_is_keyword("async")
            ):
                generators = self._parse_comprehension_clauses()
                self._expect("OP", ")")
                return GeneratorExp(
                    element=expr,
                    generators=generators,
                    line=expr.line,
                    column=expr.column,
                )
            if self._match("OP", ","):
                elements = [expr]
                if self._match("OP", ")"):
                    return TupleLiteral(elements=elements, line=token.line, column=token.column)
                while True:
                    elements.append(self._parse_star_expression())
                    if self._match("OP", ","):
                        if self._match("OP", ")"):
                            break
                        continue
                    self._expect("OP", ")")
                    break
                return TupleLiteral(elements=elements, line=token.line, column=token.column)
            if isinstance(expr, Starred):
                raise ViperSyntaxError("Starred expression is not allowed here", expr.line, expr.column)
            self._expect("OP", ")")
            return expr
        if self._match("OP", "["):
            elements: List[Expression] = []
            if self._match("OP", "]"):
                return ListLiteral(elements=elements, line=token.line, column=token.column)
            first = self._parse_star_expression()
            if not isinstance(first, Starred) and (
                self._peek_is_keyword("for") or self._peek_is_keyword("async")
            ):
                generators = self._parse_comprehension_clauses()
                self._expect("OP", "]")
                return ListComp(
                    element=first,
                    generators=generators,
                    line=first.line,
                    column=first.column,
                )
            elements.append(first)
            while True:
                if self._match("OP", ","):
                    if self._match("OP", "]"):
                        break
                    elements.append(self._parse_star_expression())
                    continue
                self._expect("OP", "]")
                break
            return ListLiteral(elements=elements, line=token.line, column=token.column)
        if self._match("OP", "{"):
            if self._match("OP", "}"):
                return DictLiteral(items=[], line=token.line, column=token.column)
            if self._at("OP", "**"):
                items: List[Tuple[Optional[Expression], Expression]] = []
                while True:
                    if self._at("OP", "**"):
                        self._advance()
                        value = self._parse_expression(allow_tuple=False)
                        items.append((None, value))
                    else:
                        key = self._parse_expression(allow_tuple=False)
                        self._expect("OP", ":")
                        value = self._parse_expression(allow_tuple=False)
                        items.append((key, value))
                    if self._match("OP", ","):
                        if self._match("OP", "}"):
                            break
                        continue
                    self._expect("OP", "}")
                    break
                return DictLiteral(items=items, line=token.line, column=token.column)
            first = self._parse_star_expression()
            if isinstance(first, Starred):
                elements = [first]
                while True:
                    if self._match("OP", ","):
                        if self._match("OP", "}"):
                            break
                        elements.append(self._parse_star_expression())
                        continue
                    self._expect("OP", "}")
                    break
                return SetLiteral(elements=elements, line=token.line, column=token.column)
            if self._match("OP", ":"):
                value = self._parse_expression(allow_tuple=False)
                if self._peek_is_keyword("for") or self._peek_is_keyword("async"):
                    generators = self._parse_comprehension_clauses()
                    self._expect("OP", "}")
                    return DictComp(
                        key=first,
                        value=value,
                        generators=generators,
                        line=first.line,
                        column=first.column,
                    )
                items = [(first, value)]
                while True:
                    if self._match("OP", ","):
                        if self._match("OP", "}"):
                            break
                        if self._at("OP", "**"):
                            self._advance()
                            value = self._parse_expression(allow_tuple=False)
                            items.append((None, value))
                            continue
                        key = self._parse_expression(allow_tuple=False)
                        self._expect("OP", ":")
                        value = self._parse_expression(allow_tuple=False)
                        items.append((key, value))
                        continue
                    self._expect("OP", "}")
                    break
                return DictLiteral(items=items, line=token.line, column=token.column)
            if self._peek_is_keyword("for") or self._peek_is_keyword("async"):
                generators = self._parse_comprehension_clauses()
                self._expect("OP", "}")
                return SetComp(
                    element=first,
                    generators=generators,
                    line=first.line,
                    column=first.column,
                )
            elements = [first]
            while True:
                if self._match("OP", ","):
                    if self._match("OP", "}"):
                        break
                    elements.append(self._parse_star_expression())
                    continue
                self._expect("OP", "}")
                break
            return SetLiteral(elements=elements, line=token.line, column=token.column)
        raise ViperSyntaxError("Unexpected token", token.line, token.column)

    def _parse_assignment_target_list(self) -> Optional[Expression]:
        start_pos = self.position
        try:
            expr = self._parse_assignment_target()
        except ViperSyntaxError:
            self.position = start_pos
            return None
        elements = [expr]
        saw_comma = False
        while self._match("OP", ","):
            saw_comma = True
            if self._at("OP", "=") or self._at("OP", ":") or self._peek_is_augassign():
                break
            if self._at("NEWLINE"):
                break
            elements.append(self._parse_assignment_target())
        if not saw_comma:
            return expr
        if self._at("OP", "=") or self._at("OP", ":") or self._peek_is_augassign():
            return TupleLiteral(elements=elements, line=elements[0].line, column=elements[0].column)
        self.position = start_pos
        return None

    def _parse_assignment_target(self) -> Expression:
        token = self._current()
        if self._match("OP", "*"):
            target = self._parse_assignment_target()
            return Starred(target=target, line=token.line, column=token.column)
        if self._match("OP", "("):
            if self._match("OP", ")"):
                return TupleLiteral(elements=[], line=token.line, column=token.column)
            first = self._parse_assignment_target()
            if self._match("OP", ","):
                elements = [first]
                if self._match("OP", ")"):
                    return TupleLiteral(elements=elements, line=token.line, column=token.column)
                while True:
                    elements.append(self._parse_assignment_target())
                    if self._match("OP", ","):
                        if self._match("OP", ")"):
                            break
                        continue
                    self._expect("OP", ")")
                    break
                return TupleLiteral(elements=elements, line=token.line, column=token.column)
            self._expect("OP", ")")
            return first
        if self._match("OP", "["):
            elements: List[Expression] = []
            if self._match("OP", "]"):
                return ListLiteral(elements=elements, line=token.line, column=token.column)
            while True:
                elements.append(self._parse_assignment_target())
                if self._match("OP", ","):
                    if self._match("OP", "]"):
                        break
                    continue
                self._expect("OP", "]")
                break
            return ListLiteral(elements=elements, line=token.line, column=token.column)
        return self._parse_postfix()

    def _parse_assignment_value(self) -> Expression:
        expr = self._parse_expression()
        return self._parse_assignment_value_tail(expr)

    def _parse_assignment_value_tail(self, expr: Expression) -> Expression:
        elements = [expr]
        if not self._match("OP", ","):
            return expr
        while True:
            if self._at("NEWLINE"):
                break
            elements.append(self._parse_expression(allow_tuple=False))
            if not self._match("OP", ","):
                break
        return TupleLiteral(elements=elements, line=elements[0].line, column=elements[0].column)

    def _parse_type_comment(self) -> Optional[str]:
        if self._at("TYPE_COMMENT"):
            token = self._advance()
            return str(token.value)
        return None

    def _match_augassign(self) -> Optional[str]:
        if self._at("OP") and self._current().value in {
            "+=",
            "-=",
            "*=",
            "/=",
            "//=",
            "%=",
            "**=",
            "@=",
            "|=",
            "&=",
            "^=",
            "<<=",
            ">>=",
        }:
            token = self._advance()
            return token.value[:-1]
        return None

    def _peek_is_augassign(self) -> bool:
        return self._at("OP") and self._current().value in {
            "+=",
            "-=",
            "*=",
            "/=",
            "//=",
            "%=",
            "**=",
            "@=",
            "|=",
            "&=",
            "^=",
            "<<=",
            ">>=",
        }

    def _parse_string_sequence(self) -> Expression:
        parts: List[Expression] = []
        kinds = set()
        while self._at("STRING") or self._at("BYTES") or self._at("FSTRING") or self._at("BFSTRING") or self._at("TSTRING") or self._at("BTSTRING"):
            token = self._advance()
            kinds.add(token.kind)
            if token.kind in {"STRING", "BYTES"}:
                parts.append(Literal(value=token.value, line=token.line, column=token.column))
                continue
            from ripperdoc.core.viper.fstring_parser import parse_formatted_string

            is_template = token.kind in {"TSTRING", "BTSTRING"}
            is_bytes = token.kind in {"BFSTRING", "BTSTRING"}
            formatted = parse_formatted_string(
                token.value,
                is_template=is_template,
                is_bytes=is_bytes,
                line=token.line,
                column=token.column,
            )
            parts.append(formatted)

        if ({"FSTRING", "BFSTRING"} & kinds) and ({"TSTRING", "BTSTRING"} & kinds):
            token = self._current()
            raise ViperSyntaxError(
                "Cannot mix f-string and t-string literals", token.line, token.column
            )

        if all(isinstance(part, Literal) for part in parts):
            values = [part.value for part in parts]  # type: ignore[attr-defined]
            has_bytes = any(isinstance(value, bytes) for value in values)
            has_str = any(isinstance(value, str) for value in values)
            if has_bytes and has_str:
                raise ViperSyntaxError(
                    "Cannot mix bytes and string literals", parts[0].line, parts[0].column
                )
            if has_bytes:
                combined = b"".join(values)  # type: ignore[arg-type]
            else:
                combined = "".join(values)  # type: ignore[arg-type]
            return Literal(value=combined, line=parts[0].line, column=parts[0].column)

        is_template = "TSTRING" in kinds or "BTSTRING" in kinds
        is_bytes = "BFSTRING" in kinds or "BTSTRING" in kinds
        formatted_parts: List[StringText | FormattedString] = []
        for part in parts:
            if isinstance(part, Literal):
                if isinstance(part.value, bytes):
                    if not is_bytes:
                        raise ViperSyntaxError(
                            "Cannot mix bytes with formatted strings", part.line, part.column
                        )
                formatted_parts.append(StringText(value=part.value, line=part.line, column=part.column))
            elif isinstance(part, FormattedString):
                formatted_parts.append(part)
            else:
                raise ViperSyntaxError("Invalid string literal", part.line, part.column)

        flattened = _flatten_formatted_parts(
            formatted_parts, is_template=is_template, is_bytes=is_bytes
        )
        return FormattedString(
            parts=flattened,
            is_template=is_template,
            is_bytes=is_bytes,
            line=parts[0].line,
            column=parts[0].column,
        )
    def _match_comparison_op(self) -> Optional[str]:
        if self._at("OP") and self._current().value in {"==", "!=", "<", "<=", ">", ">="}:
            return self._advance().value
        if self._peek_is_keyword("is"):
            self._advance()
            if self._peek_is_keyword("not"):
                self._advance()
                return "is not"
            return "is"
        if self._peek_is_keyword("in"):
            self._advance()
            return "in"
        if self._peek_is_keyword("not") and self._peek_next_is("KEYWORD", "in"):
            self._advance()
            self._advance()
            return "not in"
        return None

    def _parse_number_value(self, raw: str, token: Token) -> float | int | complex:
        normalized = raw.replace("_", "")
        try:
            if normalized.endswith(("j", "J")):
                return complex(normalized)
            if any(ch in normalized for ch in ".eE"):
                return float(normalized)
            if normalized.startswith(("0x", "0X", "0b", "0B", "0o", "0O")):
                return int(normalized, 0)
            return int(normalized, 10)
        except ValueError as exc:
            raise ViperSyntaxError("Invalid numeric literal", token.line, token.column) from exc

    def _collect_pattern_names(self, pattern: Pattern) -> set[str]:
        names: set[str] = set()

        def _walk(node: Pattern) -> None:
            if isinstance(node, PatternName):
                names.add(node.identifier)
                return
            if isinstance(node, PatternAs):
                names.add(node.name)
                _walk(node.pattern)
                return
            if isinstance(node, PatternStar):
                if isinstance(node.target, PatternName):
                    names.add(node.target.identifier)
                return
            if isinstance(node, PatternSequence):
                for element in node.elements:
                    _walk(element)
                return
            if isinstance(node, PatternOr):
                for element in node.patterns:
                    _walk(element)
                return
            if isinstance(node, PatternMapping):
                for _, value_pattern in node.items:
                    _walk(value_pattern)
                if node.rest is not None:
                    _walk(node.rest)
                return
            if isinstance(node, PatternClass):
                for item in node.positional:
                    _walk(item)
                for _, item in node.keywords:
                    _walk(item)
                return
            if isinstance(node, (PatternLiteral, PatternWildcard, PatternValue)):
                return

        _walk(pattern)
        return names

    def _validate_or_pattern_bindings(self, patterns: List[Pattern]) -> None:
        expected: Optional[set[str]] = None
        for pattern in patterns:
            names = self._collect_pattern_names(pattern)
            if expected is None:
                expected = names
                continue
            if names != expected:
                raise ViperSyntaxError(
                    "All alternatives in an or-pattern must bind the same names",
                    getattr(pattern, "line", 0),
                    getattr(pattern, "column", 0),
                )

    def _validate_pattern_unique_names(
        self, patterns: List[Pattern], line: int, column: int
    ) -> None:
        seen: set[str] = set()
        for pattern in patterns:
            for name in self._collect_pattern_names(pattern):
                if name in seen:
                    raise ViperSyntaxError(
                        "Multiple assignments to the same name in pattern", line, column
                    )
                seen.add(name)

    def _is_assignable(self, expr: Expression) -> bool:
        if isinstance(expr, (Name, Attribute, Subscript)):
            return True
        if isinstance(expr, Starred):
            return self._is_assignable(expr.target)
        if isinstance(expr, (TupleLiteral, ListLiteral)):
            return all(self._is_assignable(element) for element in expr.elements)
        return False

    def _is_simple_target(self, expr: Expression) -> bool:
        return isinstance(expr, (Name, Attribute, Subscript))

    def _peek_is_keyword(self, value: str) -> bool:
        return self._at("KEYWORD", value)

    def _peek_next_is(self, kind: str, value: Optional[str] = None) -> bool:
        if self.position + 1 >= len(self.tokens):
            return False
        token = self.tokens[self.position + 1]
        if token.kind != kind:
            return False
        return value is None or token.value == value

    def _at(self, kind: str, value: Optional[str] = None) -> bool:
        token = self._current()
        if token.kind != kind:
            return False
        return value is None or token.value == value

    def _current(self) -> Token:
        return self.tokens[self.position]

    def _advance(self) -> Token:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def _match(self, kind: str, value: Optional[str] = None) -> bool:
        if self._at(kind, value):
            self._advance()
            return True
        return False

    def _expect(self, kind: str, value: Optional[str] = None) -> Token:
        if not self._at(kind, value):
            token = self._current()
            expected = f"{kind} {value}" if value else kind
            raise ViperSyntaxError(f"Expected {expected}", token.line, token.column)
        return self._advance()


def _flatten_formatted_parts(
    parts: List[StringText | FormattedString], *, is_template: bool, is_bytes: bool
) -> List[StringText | "FormatField"]:
    flattened: List[StringText | "FormatField"] = []
    for part in parts:
        if isinstance(part, StringText):
            flattened.append(part)
            continue
        if isinstance(part, FormattedString):
            if part.is_template != is_template or part.is_bytes != is_bytes:
                raise ViperSyntaxError(
                    "Cannot mix f-string and t-string literals", part.line, part.column
                )
            flattened.extend(part.parts)
            continue
        raise ViperSyntaxError("Invalid string literal", 0, 0)
    return flattened


def parse(tokens: List[Token]) -> Module:
    """Parse a token stream into a Viper AST."""
    return Parser(tokens).parse()


def parse_expression(tokens: List[Token]) -> Expression:
    """Parse a token stream into a single expression."""
    return Parser(tokens).parse_expression_only()


def parse_expression_source(source: str) -> Expression:
    """Parse a string into a single expression."""
    from ripperdoc.core.viper.tokenizer import tokenize

    return parse_expression(tokenize(source))
