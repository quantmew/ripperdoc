"""Parser for the Viper language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ripperdoc.core.viper.ast_nodes import (
    AssignStmt,
    Attribute,
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
    FunctionDef,
    IfStmt,
    ImportAlias,
    ImportFromStmt,
    ImportStmt,
    ListLiteral,
    Literal,
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
    PatternWildcard,
    ReturnStmt,
    RaiseStmt,
    Statement,
    Subscript,
    TupleLiteral,
    TypeAliasStmt,
    UnaryOp,
    WhileStmt,
    WithItem,
    WithStmt,
    PassStmt,
    FormattedString,
    StringText,
    TryStmt,
    SliceExpr,
    YieldExpr,
    YieldFromExpr,
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
            token = self._current()
            raise ViperSyntaxError("Decorators are only supported for classes", token.line, token.column)
        if self._peek_is_keyword("class"):
            return self._parse_class([])
        if self._peek_is_keyword("def"):
            return self._parse_function_def()
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
                        cause = self._parse_expression()
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
                test = self._parse_expression()
                message = None
                if self._match("OP", ","):
                    message = self._parse_expression()
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
        if target is not None and self._match("OP", "="):
            if not self._is_assignable(target):
                raise ViperSyntaxError("Invalid assignment target", target.line, target.column)
            value = self._parse_assignment_value()
            stmt = AssignStmt(target=target, value=value, line=target.line, column=target.column)
            self._expect("NEWLINE")
            return stmt

        self.position = start_pos
        expr = self._parse_expression()
        if self._match("OP", "="):
            if not self._is_assignable(expr):
                raise ViperSyntaxError("Invalid assignment target", expr.line, expr.column)
            value = self._parse_assignment_value()
            stmt = AssignStmt(target=expr, value=value, line=expr.line, column=expr.column)
        else:
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
        targets = [self._parse_expression()]
        while self._match("OP", ","):
            targets.append(self._parse_expression())
        for target in targets:
            if not self._is_assignable(target):
                raise ViperSyntaxError("Invalid delete target", target.line, target.column)
        self._expect("NEWLINE")
        return DelStmt(targets=targets, line=token.line, column=token.column)

    def _parse_type_alias_stmt(self) -> TypeAliasStmt:
        token = self._expect("KEYWORD", "type")
        name_token = self._expect("NAME")
        self._expect("OP", "=")
        value = self._parse_expression()
        self._expect("NEWLINE")
        return TypeAliasStmt(
            name=name_token.value,
            value=value,
            line=token.line,
            column=token.column,
        )

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
        patterns = [self._parse_pattern()]
        while self._match("OP", "|"):
            patterns.append(self._parse_pattern())
        return patterns

    def _parse_pattern(self) -> Pattern:
        token = self._current()
        if token.kind == "NAME":
            self._advance()
            if token.value == "_":
                return PatternWildcard(line=token.line, column=token.column)
            return PatternName(identifier=token.value, line=token.line, column=token.column)
        if token.kind == "NUMBER":
            self._advance()
            value: float | int
            if "." in token.value or "e" in token.value or "E" in token.value:
                value = float(token.value)
            else:
                value = int(token.value)
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
        if self._match("OP", "["):
            elements: List[Pattern] = []
            if self._match("OP", "]"):
                return PatternSequence(elements=elements, line=token.line, column=token.column)
            while True:
                elements.append(self._parse_pattern())
                if self._match("OP", ","):
                    if self._match("OP", "]"):
                        break
                    continue
                self._expect("OP", "]")
                break
            return PatternSequence(elements=elements, line=token.line, column=token.column)
        if self._match("OP", "("):
            if self._match("OP", ")"):
                return PatternSequence(elements=[], line=token.line, column=token.column)
            first = self._parse_pattern()
            if self._match("OP", ","):
                elements = [first]
                if self._match("OP", ")"):
                    return PatternSequence(elements=elements, line=token.line, column=token.column)
                while True:
                    elements.append(self._parse_pattern())
                    if self._match("OP", ","):
                        if self._match("OP", ")"):
                            break
                        continue
                    self._expect("OP", ")")
                    break
                return PatternSequence(elements=elements, line=token.line, column=token.column)
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
        exc_type = self._parse_expression()
        exprs = [exc_type]
        while self._match("OP", ","):
            if self._peek_is_keyword("as") or self._at("OP", ":"):
                raise ViperSyntaxError("Expected exception type", token.line, token.column)
            exprs.append(self._parse_expression())
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
            decorators.append(self._parse_expression())
            self._expect("NEWLINE")
        return decorators

    def _parse_class(self, decorators: List[Expression]) -> ClassDef:
        token = self._expect("KEYWORD", "class")
        name_token = self._expect("NAME")
        bases: List[Expression] = []
        if self._match("OP", "("):
            args, kwargs = self._parse_call_args()
            if kwargs:
                raise ViperSyntaxError(
                    "Keyword arguments in class bases are not supported",
                    name_token.line,
                    name_token.column,
                )
            bases = args
        self._expect("OP", ":")
        body = self._parse_block()
        return ClassDef(
            name=name_token.value,
            bases=bases,
            body=body,
            decorators=decorators,
            line=token.line,
            column=token.column,
        )

    def _parse_function_def(self, *, is_async: bool = False) -> FunctionDef:
        token = self._expect("KEYWORD", "def")
        name_token = self._expect("NAME")
        self._expect("OP", "(")
        params: List[str] = []
        if not self._at("OP", ")"):
            while True:
                param_token = self._expect("NAME")
                params.append(param_token.value)
                if not self._match("OP", ","):
                    break
                if self._at("OP", ")"):
                    break
        self._expect("OP", ")")
        self._expect("OP", ":")
        body = self._parse_block()
        return FunctionDef(
            name=name_token.value,
            params=params,
            body=body,
            is_async=is_async,
            line=token.line,
            column=token.column,
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
        expr = self._parse_expression()
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

    def _parse_expression(self) -> Expression:
        if self._peek_is_keyword("yield"):
            return self._parse_yield()
        return self._parse_or()

    def _parse_yield(self) -> Expression:
        token = self._expect("KEYWORD", "yield")
        if self._peek_is_keyword("from"):
            self._advance()
            value = self._parse_expression()
            return YieldFromExpr(value=value, line=token.line, column=token.column)
        if self._at("NEWLINE") or self._at("OP", ")") or self._at("OP", "]") or self._at("OP", "}") or self._at("DEDENT"):
            return YieldExpr(value=None, line=token.line, column=token.column)
        expr = self._parse_expression()
        if not self._match("OP", ","):
            return YieldExpr(value=expr, line=token.line, column=token.column)
        elements = [expr]
        while True:
            if self._at("NEWLINE") or self._at("OP", ")") or self._at("OP", "]") or self._at("OP", "}") or self._at("DEDENT"):
                break
            elements.append(self._parse_expression())
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
        expr = self._parse_term()
        ops: List[str] = []
        comparators: List[Expression] = []
        while True:
            op = self._match_comparison_op()
            if op is None:
                break
            right = self._parse_term()
            ops.append(op)
            comparators.append(right)
        if ops:
            return CompareOp(left=expr, ops=ops, comparators=comparators, line=expr.line, column=expr.column)
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
        while self._at("OP") and self._current().value in {"*", "/", "//", "%"}:
            token = self._advance()
            right = self._parse_unary()
            expr = BinaryOp(left=expr, op=token.value, right=right, line=token.line, column=token.column)
        return expr

    def _parse_unary(self) -> Expression:
        if self._at("OP") and self._current().value in {"+", "-"}:
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
        expr = self._parse_expression()
        if self._match("OP", ":"):
            return self._parse_slice_tail(expr, expr.line, expr.column)
        return expr

    def _parse_slice_tail(
        self, start: Optional[Expression], line: int, column: int
    ) -> SliceExpr:
        stop: Optional[Expression] = None
        step: Optional[Expression] = None
        if not self._at("OP", "]") and not self._at("OP", ",") and not self._at("OP", ":"):
            stop = self._parse_expression()
        if self._match("OP", ":"):
            if not self._at("OP", "]") and not self._at("OP", ","):
                step = self._parse_expression()
        return SliceExpr(start=start, stop=stop, step=step, line=line, column=column)

    def _parse_call_args(self) -> Tuple[List[Expression], List[Tuple[str, Expression]]]:
        args: List[Expression] = []
        kwargs: List[Tuple[str, Expression]] = []
        saw_kwarg = False
        if self._match("OP", ")"):
            return args, kwargs
        while True:
            if self._at("NAME") and self._peek_next_is("OP", "="):
                name_token = self._advance()
                self._expect("OP", "=")
                value = self._parse_expression()
                kwargs.append((name_token.value, value))
                saw_kwarg = True
            else:
                if saw_kwarg:
                    token = self._current()
                    raise ViperSyntaxError(
                        "Positional argument follows keyword argument", token.line, token.column
                    )
                args.append(self._parse_expression())
            if self._match("OP", ","):
                if self._match("OP", ")"):
                    break
                continue
            self._expect("OP", ")")
            break
        return args, kwargs

    def _parse_atom(self) -> Expression:
        token = self._current()
        if token.kind in {"STRING", "BYTES", "FSTRING", "BFSTRING", "TSTRING", "BTSTRING"}:
            return self._parse_string_sequence()
        if token.kind == "NAME":
            self._advance()
            return Name(identifier=token.value, line=token.line, column=token.column)
        if token.kind == "NUMBER":
            self._advance()
            value: float | int
            if "." in token.value or "e" in token.value or "E" in token.value:
                value = float(token.value)
            else:
                value = int(token.value)
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
            expr = self._parse_expression()
            if self._match("OP", ","):
                elements = [expr]
                if self._match("OP", ")"):
                    return TupleLiteral(elements=elements, line=token.line, column=token.column)
                while True:
                    elements.append(self._parse_expression())
                    if self._match("OP", ","):
                        if self._match("OP", ")"):
                            break
                        continue
                    self._expect("OP", ")")
                    break
                return TupleLiteral(elements=elements, line=token.line, column=token.column)
            self._expect("OP", ")")
            return expr
        if self._match("OP", "["):
            elements: List[Expression] = []
            if self._match("OP", "]"):
                return ListLiteral(elements=elements, line=token.line, column=token.column)
            while True:
                elements.append(self._parse_expression())
                if self._match("OP", ","):
                    if self._match("OP", "]"):
                        break
                    continue
                self._expect("OP", "]")
                break
            return ListLiteral(elements=elements, line=token.line, column=token.column)
        if self._match("OP", "{"):
            items: List[Tuple[Expression, Expression]] = []
            if self._match("OP", "}"):
                return DictLiteral(items=items, line=token.line, column=token.column)
            while True:
                key = self._parse_expression()
                self._expect("OP", ":")
                value = self._parse_expression()
                items.append((key, value))
                if self._match("OP", ","):
                    if self._match("OP", "}"):
                        break
                    continue
                self._expect("OP", "}")
                break
            return DictLiteral(items=items, line=token.line, column=token.column)
        raise ViperSyntaxError("Unexpected token", token.line, token.column)

    def _parse_assignment_target_list(self) -> Optional[Expression]:
        start_pos = self.position
        expr = self._parse_expression()
        elements = [expr]
        saw_comma = False
        while self._match("OP", ","):
            saw_comma = True
            if self._at("OP", "="):
                break
            elements.append(self._parse_expression())
        if not saw_comma:
            return expr
        if self._at("OP", "="):
            return TupleLiteral(elements=elements, line=elements[0].line, column=elements[0].column)
        self.position = start_pos
        return None

    def _parse_assignment_value(self) -> Expression:
        expr = self._parse_expression()
        elements = [expr]
        if not self._match("OP", ","):
            return expr
        while True:
            if self._at("NEWLINE"):
                break
            elements.append(self._parse_expression())
            if not self._match("OP", ","):
                break
        return TupleLiteral(elements=elements, line=elements[0].line, column=elements[0].column)

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
        if self._peek_is_keyword("in"):
            self._advance()
            return "in"
        if self._peek_is_keyword("not") and self._peek_next_is("KEYWORD", "in"):
            self._advance()
            self._advance()
            return "not in"
        return None

    def _is_assignable(self, expr: Expression) -> bool:
        if isinstance(expr, (Name, Attribute, Subscript)):
            return True
        if isinstance(expr, TupleLiteral):
            return all(self._is_assignable(element) for element in expr.elements)
        return False

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
