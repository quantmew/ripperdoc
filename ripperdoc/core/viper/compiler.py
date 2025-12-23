"""AST to bytecode compiler for Viper."""

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
    ConditionalExpr,
    ContinueStmt,
    DictLiteral,
    DelStmt,
    Expression,
    ExpressionStmt,
    ExceptHandler,
    ForStmt,
    FunctionDef,
    IfStmt,
    ImportFromStmt,
    ImportStmt,
    GlobalStmt,
    NonlocalStmt,
    ListLiteral,
    FormattedString,
    FormatField,
    LambdaExpr,
    Literal,
    Module,
    MatchCase,
    MatchStmt,
    Name,
    PassStmt,
    AssertStmt,
    ReturnStmt,
    RaiseStmt,
    SliceExpr,
    Statement,
    Subscript,
    StringText,
    TupleLiteral,
    TypeAliasStmt,
    Parameter,
    UnaryOp,
    WhileStmt,
    WithStmt,
    TryStmt,
    YieldExpr,
    YieldFromExpr,
    Pattern,
    PatternLiteral,
    PatternName,
    PatternOr,
    PatternSequence,
    PatternWildcard,
    Starred,
)
from ripperdoc.core.viper.bytecode import CodeObject, Instruction, Label, ParamSpec
from ripperdoc.core.viper.errors import ViperSyntaxError


@dataclass
class LoopContext:
    start: Label
    end: Label


class BytecodeCompiler:
    def __init__(self) -> None:
        self._instructions: List[Instruction] = []
        self._loop_stack: List[LoopContext] = []
        self._temp_counter = 0
        self._in_generator = False
        self._in_coroutine = False

    def compile_module(self, module: Module) -> CodeObject:
        self._compile_statements(module.statements)
        self._emit("LOAD_RESULT")
        self._emit("RETURN_VALUE", False)
        code = CodeObject(
            name="<module>",
            instructions=self._instructions,
            params=[],
            is_module=True,
            is_generator=False,
            is_coroutine=False,
        )
        self._resolve_labels(code)
        return code

    def compile_function(self, func: FunctionDef) -> CodeObject:
        contains_yield = _contains_yield(func.body)
        if func.is_async and contains_yield:
            raise ViperSyntaxError("Async generators are not supported", func.line, func.column)
        nested = BytecodeCompiler()
        nested._in_generator = contains_yield
        nested._in_coroutine = func.is_async
        nested._compile_statements(func.body)
        nested._emit("LOAD_CONST", None, func)
        nested._emit("RETURN_VALUE", False, func)
        code = CodeObject(
            name=func.name,
            instructions=nested._instructions,
            params=[param.name for param in func.params],
            param_spec=self._build_param_spec(func.params),
            is_module=False,
            is_generator=contains_yield,
            is_coroutine=func.is_async,
        )
        nested._resolve_labels(code)
        return code

    def _compile_lambda(self, expr: LambdaExpr) -> CodeObject:
        nested = BytecodeCompiler()
        nested._in_generator = False
        nested._in_coroutine = False
        nested._compile_expression(expr.body)
        nested._emit("RETURN_VALUE", True, expr)
        code = CodeObject(
            name="<lambda>",
            instructions=nested._instructions,
            params=[param.name for param in expr.params],
            param_spec=self._build_param_spec(expr.params),
            is_module=False,
            is_generator=False,
            is_coroutine=False,
        )
        nested._resolve_labels(code)
        return code

    def compile_class(self, cls: ClassDef) -> CodeObject:
        nested = BytecodeCompiler()
        nested._compile_statements(cls.body)
        nested._emit("LOAD_RESULT")
        nested._emit("RETURN_VALUE", False, cls)
        code = CodeObject(
            name=f"<class {cls.name}>",
            instructions=nested._instructions,
            params=[],
            is_module=True,
            is_generator=False,
            is_coroutine=False,
        )
        nested._resolve_labels(code)
        return code

    def _compile_statements(self, statements: List[Statement]) -> None:
        for stmt in statements:
            self._compile_statement(stmt)

    def _compile_statement(self, stmt: Statement) -> None:
        if isinstance(stmt, ExpressionStmt):
            self._compile_expression(stmt.expression)
            self._emit("SET_RESULT", None, stmt)
            return
        if isinstance(stmt, AssignStmt):
            self._compile_assignment(stmt)
            return
        if isinstance(stmt, AugAssignStmt):
            self._compile_augassign(stmt)
            return
        if isinstance(stmt, AnnAssignStmt):
            self._compile_annassign(stmt)
            return
        if isinstance(stmt, IfStmt):
            self._compile_if(stmt)
            return
        if isinstance(stmt, WhileStmt):
            self._compile_while(stmt)
            return
        if isinstance(stmt, ForStmt):
            self._compile_for(stmt)
            return
        if isinstance(stmt, ClassDef):
            self._compile_class(stmt)
            return
        if isinstance(stmt, WithStmt):
            self._compile_with(stmt)
            return
        if isinstance(stmt, FunctionDef):
            func_code = self.compile_function(stmt)
            pos_defaults, kw_defaults = self._collect_param_defaults(stmt.params)
            for expr in pos_defaults:
                self._compile_expression(expr)
            kw_default_names = [name for name, _ in kw_defaults]
            for _, expr in kw_defaults:
                self._compile_expression(expr)
            self._emit("LOAD_CONST", func_code, stmt)
            self._emit("MAKE_FUNCTION", (len(pos_defaults), kw_default_names), stmt)
            self._emit("STORE_NAME", stmt.name, stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        if isinstance(stmt, ImportStmt):
            self._compile_import(stmt)
            return
        if isinstance(stmt, ImportFromStmt):
            self._compile_import_from(stmt)
            return
        if isinstance(stmt, GlobalStmt):
            self._emit("DECLARE_GLOBAL", stmt.names, stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        if isinstance(stmt, NonlocalStmt):
            self._emit("DECLARE_NONLOCAL", stmt.names, stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        if isinstance(stmt, DelStmt):
            self._compile_del(stmt)
            return
        if isinstance(stmt, TypeAliasStmt):
            self._compile_type_alias(stmt)
            return
        if isinstance(stmt, MatchStmt):
            self._compile_match(stmt)
            return
        if isinstance(stmt, TryStmt):
            self._compile_try(stmt)
            return
        if isinstance(stmt, RaiseStmt):
            self._compile_raise(stmt)
            return
        if isinstance(stmt, AssertStmt):
            self._compile_assert(stmt)
            return
        if isinstance(stmt, ReturnStmt):
            if stmt.value is None:
                self._emit("LOAD_CONST", None, stmt)
            else:
                self._compile_expression(stmt.value)
            self._emit("RETURN_VALUE", True, stmt)
            return
        if isinstance(stmt, PassStmt):
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        if isinstance(stmt, BreakStmt):
            if not self._loop_stack:
                raise ViperSyntaxError("'break' outside loop", stmt.line, stmt.column)
            self._emit("JUMP", self._loop_stack[-1].end, stmt)
            return
        if isinstance(stmt, ContinueStmt):
            if not self._loop_stack:
                raise ViperSyntaxError("'continue' outside loop", stmt.line, stmt.column)
            self._emit("JUMP", self._loop_stack[-1].start, stmt)
            return
        raise ViperSyntaxError("Unsupported statement", stmt.line, stmt.column)

    def _compile_try(self, stmt: TryStmt) -> None:
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)
        handler_label = Label() if stmt.handlers else None
        finally_label = Label() if stmt.finally_body else None
        end_label = Label()

        if finally_label is not None:
            self._emit("SETUP_FINALLY", finally_label, stmt)
        if handler_label is not None:
            self._emit("SETUP_EXCEPT", handler_label, stmt)

        self._compile_statements(stmt.body)

        if handler_label is not None:
            self._emit("POP_EXCEPT", None, stmt)

        if stmt.else_body is not None:
            self._compile_statements(stmt.else_body)

        if finally_label is not None:
            self._emit("JUMP", finally_label, stmt)
        else:
            self._emit("JUMP", end_label, stmt)

        if handler_label is not None:
            self._mark(handler_label)
            self._compile_except_handlers(
                stmt.handlers,
                finally_label if finally_label is not None else end_label,
                stmt,
            )

        if finally_label is not None:
            self._mark(finally_label)
            self._emit("POP_FINALLY", None, stmt)
            self._compile_statements(stmt.finally_body or [])
            self._emit("END_FINALLY", None, stmt)

        self._mark(end_label)

    def _compile_except_handlers(
        self, handlers: List[ExceptHandler], done_label: Label, stmt: TryStmt
    ) -> None:
        for handler in handlers:
            next_label = Label()
            if handler.type is not None:
                self._emit("LOAD_EXCEPTION", None, handler)
                self._compile_expression(handler.type)
                self._emit("EXC_MATCH", None, handler)
                self._emit("JUMP_IF_FALSE", next_label, handler)
            if handler.name is not None:
                self._emit("LOAD_EXCEPTION", None, handler)
                self._emit("STORE_NAME", handler.name, handler)
            self._emit("CLEAR_EXCEPTION", None, handler)
            self._compile_statements(handler.body)
            self._emit("JUMP", done_label, handler)
            self._mark(next_label)

        self._emit("RAISE", (False, False), stmt)

    def _compile_raise(self, stmt: RaiseStmt) -> None:
        if stmt.exception is not None:
            self._compile_expression(stmt.exception)
        if stmt.cause is not None:
            self._compile_expression(stmt.cause)
        self._emit(
            "RAISE",
            (stmt.exception is not None, stmt.cause is not None),
            stmt,
        )

    def _compile_assert(self, stmt: AssertStmt) -> None:
        fail_label = Label()
        end_label = Label()
        self._compile_expression(stmt.test)
        self._emit("JUMP_IF_FALSE", fail_label, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)
        self._emit("JUMP", end_label, stmt)
        self._mark(fail_label)
        if stmt.message is not None:
            self._compile_expression(stmt.message)
        self._emit("ASSERT", stmt.message is not None, stmt)
        self._mark(end_label)

    def _compile_assignment(self, stmt: AssignStmt) -> None:
        if len(stmt.targets) == 1:
            self._compile_expression(stmt.value)
            self._compile_assign_target(stmt.targets[0], stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return

        temp_name = self._new_temp_name("assign")
        self._compile_expression(stmt.value)
        self._emit("STORE_NAME", temp_name, stmt)
        for target in stmt.targets:
            self._emit("LOAD_NAME", temp_name, stmt)
            self._compile_assign_target(target, stmt)
        self._emit("DELETE_NAME", temp_name, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)

    def _compile_augassign(self, stmt: AugAssignStmt) -> None:
        target = stmt.target
        if isinstance(target, Name):
            self._emit("LOAD_NAME", target.identifier, stmt)
            self._compile_expression(stmt.value)
            self._emit("BINARY_OP", stmt.op, stmt)
            self._emit("STORE_NAME", target.identifier, stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        if isinstance(target, Attribute):
            obj_temp = self._new_temp_name("aug_obj")
            val_temp = self._new_temp_name("aug_val")
            self._compile_expression(target.value)
            self._emit("STORE_NAME", obj_temp, stmt)
            self._emit("LOAD_NAME", obj_temp, stmt)
            self._emit("GET_ATTR", target.name, stmt)
            self._compile_expression(stmt.value)
            self._emit("BINARY_OP", stmt.op, stmt)
            self._emit("STORE_NAME", val_temp, stmt)
            self._emit("LOAD_NAME", obj_temp, stmt)
            self._emit("LOAD_NAME", val_temp, stmt)
            self._emit("SET_ATTR", target.name, stmt)
            self._emit("DELETE_NAME", obj_temp, stmt)
            self._emit("DELETE_NAME", val_temp, stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        if isinstance(target, Subscript):
            obj_temp = self._new_temp_name("aug_obj")
            idx_temp = self._new_temp_name("aug_idx")
            val_temp = self._new_temp_name("aug_val")
            self._compile_expression(target.value)
            self._emit("STORE_NAME", obj_temp, stmt)
            self._compile_expression(target.index)
            self._emit("STORE_NAME", idx_temp, stmt)
            self._emit("LOAD_NAME", obj_temp, stmt)
            self._emit("LOAD_NAME", idx_temp, stmt)
            self._emit("GET_SUBSCRIPT", None, stmt)
            self._compile_expression(stmt.value)
            self._emit("BINARY_OP", stmt.op, stmt)
            self._emit("STORE_NAME", val_temp, stmt)
            self._emit("LOAD_NAME", obj_temp, stmt)
            self._emit("LOAD_NAME", idx_temp, stmt)
            self._emit("LOAD_NAME", val_temp, stmt)
            self._emit("SET_SUBSCRIPT", None, stmt)
            self._emit("DELETE_NAME", obj_temp, stmt)
            self._emit("DELETE_NAME", idx_temp, stmt)
            self._emit("DELETE_NAME", val_temp, stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        raise ViperSyntaxError("Invalid assignment target", stmt.line, stmt.column)

    def _compile_annassign(self, stmt: AnnAssignStmt) -> None:
        if stmt.value is not None:
            self._compile_expression(stmt.value)
            self._compile_assign_target(stmt.target, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)

    def _compile_assign_target(self, target: Expression, stmt: AssignStmt | Statement) -> None:
        if isinstance(target, (TupleLiteral, ListLiteral)):
            self._compile_unpack_targets(target, stmt)
            return
        if isinstance(target, Name):
            self._emit("STORE_NAME", target.identifier, stmt)
            return
        if isinstance(target, Attribute):
            self._compile_expression(target.value)
            self._emit("SWAP", None, stmt)
            self._emit("SET_ATTR", target.name, stmt)
            return
        if isinstance(target, Subscript):
            self._compile_expression(target.value)
            self._compile_expression(target.index)
            self._emit("ROT_THREE", None, stmt)
            self._emit("SET_SUBSCRIPT", None, stmt)
            return
        if isinstance(target, Starred):
            raise ViperSyntaxError("Invalid assignment target", stmt.line, stmt.column)
        raise ViperSyntaxError("Invalid assignment target", stmt.line, stmt.column)

    def _compile_unpack_targets(self, target: Expression, node: AssignStmt) -> None:
        if isinstance(target, (TupleLiteral, ListLiteral)):
            starred_index = None
            for index, element in enumerate(target.elements):
                if isinstance(element, Starred):
                    if starred_index is not None:
                        raise ViperSyntaxError("Multiple starred targets in assignment", node.line, node.column)
                    starred_index = index
            if starred_index is None:
                self._emit("UNPACK_SEQUENCE", len(target.elements), node)
            else:
                before = starred_index
                after = len(target.elements) - starred_index - 1
                self._emit("UNPACK_EX", (before, after), node)
            for element in target.elements:
                self._compile_unpack_targets(element, node)
            return
        if isinstance(target, Starred):
            self._compile_store_target(target.target, node)
            return
        if isinstance(target, Name):
            self._emit("STORE_NAME", target.identifier, node)
            return
        if isinstance(target, Attribute):
            self._compile_expression(target.value)
            self._emit("SWAP", None, node)
            self._emit("SET_ATTR", target.name, node)
            return
        if isinstance(target, Subscript):
            self._compile_expression(target.value)
            self._compile_expression(target.index)
            self._emit("ROT_THREE", None, node)
            self._emit("SET_SUBSCRIPT", None, node)
            return
        raise ViperSyntaxError("Invalid assignment target", node.line, node.column)

    def _compile_if(self, stmt: IfStmt) -> None:
        end_label = Label()
        next_label = Label()
        self._compile_expression(stmt.test)
        self._emit("JUMP_IF_FALSE", next_label, stmt)
        self._compile_statements(stmt.body)
        self._emit("JUMP", end_label, stmt)
        self._mark(next_label)
        for elif_test, elif_body in stmt.elif_blocks:
            next_label = Label()
            self._compile_expression(elif_test)
            self._emit("JUMP_IF_FALSE", next_label, stmt)
            self._compile_statements(elif_body)
            self._emit("JUMP", end_label, stmt)
            self._mark(next_label)
        if stmt.else_body is not None:
            self._compile_statements(stmt.else_body)
        else:
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
        self._mark(end_label)

    def _compile_while(self, stmt: WhileStmt) -> None:
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)
        start_label = Label()
        end_label = Label()
        else_label = end_label if stmt.else_body is None else Label()
        self._loop_stack.append(LoopContext(start=start_label, end=end_label))
        self._mark(start_label)
        self._compile_expression(stmt.test)
        self._emit("JUMP_IF_FALSE", else_label, stmt)
        self._compile_statements(stmt.body)
        self._emit("JUMP", start_label, stmt)
        self._mark(else_label)
        if stmt.else_body is not None:
            self._compile_statements(stmt.else_body)
            self._mark(end_label)
        self._loop_stack.pop()

    def _compile_for(self, stmt: ForStmt) -> None:
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)
        self._compile_expression(stmt.iterable)
        self._emit("BUILD_ITER", None, stmt)
        start_label = Label()
        end_label = Label()
        else_label = end_label if stmt.else_body is None else Label()
        self._loop_stack.append(LoopContext(start=start_label, end=end_label))
        self._mark(start_label)
        self._emit("FOR_ITER", else_label, stmt)
        self._compile_store_target(stmt.target, stmt)
        self._compile_statements(stmt.body)
        self._emit("JUMP", start_label, stmt)
        self._mark(else_label)
        if stmt.else_body is not None:
            self._compile_statements(stmt.else_body)
            self._mark(end_label)
        self._loop_stack.pop()

    def _compile_with(self, stmt: WithStmt) -> None:
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)
        for item in stmt.items:
            self._compile_expression(item.context_expr)
            self._emit("WITH_ENTER", None, item)
            if item.target is None:
                self._emit("POP_TOP", None, item)
            elif isinstance(item.target, Name):
                self._emit("STORE_NAME", item.target.identifier, item)
            else:
                raise ViperSyntaxError("Invalid with-item target", item.line, item.column)
        self._compile_statements(stmt.body)
        for item in reversed(stmt.items):
            self._emit("WITH_EXIT", None, item)

    def _compile_import(self, stmt: ImportStmt) -> None:
        for alias in stmt.names:
            fromlist = ["*"] if alias.asname else None
            self._emit("IMPORT_NAME", (alias.name, fromlist, 0), stmt)
            target_name = alias.asname or alias.name.split(".")[0]
            self._emit("STORE_NAME", target_name, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)

    def _compile_import_from(self, stmt: ImportFromStmt) -> None:
        if stmt.module is None:
            if stmt.is_star:
                raise ViperSyntaxError("from . import * is not supported", stmt.line, stmt.column)
            for alias in stmt.names:
                self._emit("IMPORT_NAME", (alias.name, None, stmt.level), stmt)
                target_name = alias.asname or alias.name
                self._emit("STORE_NAME", target_name, stmt)
            self._emit("LOAD_CONST", None, stmt)
            self._emit("SET_RESULT", None, stmt)
            return
        module_name = stmt.module
        if stmt.is_star:
            fromlist = ["*"]
        else:
            fromlist = [alias.name for alias in stmt.names]
        self._emit("IMPORT_NAME", (module_name, fromlist, stmt.level), stmt)
        if stmt.is_star:
            self._emit("IMPORT_STAR", None, stmt)
        else:
            for alias in stmt.names:
                self._emit("IMPORT_FROM", alias.name, stmt)
                target_name = alias.asname or alias.name
                self._emit("STORE_NAME", target_name, stmt)
            self._emit("POP_TOP", None, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)

    def _compile_del(self, stmt: DelStmt) -> None:
        for target in stmt.targets:
            self._compile_del_target(target, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)

    def _compile_del_target(self, target: Expression, stmt: Statement) -> None:
        if isinstance(target, Name):
            self._emit("DELETE_NAME", target.identifier, stmt)
            return
        if isinstance(target, Attribute):
            self._compile_expression(target.value)
            self._emit("DELETE_ATTR", target.name, stmt)
            return
        if isinstance(target, Subscript):
            self._compile_expression(target.value)
            self._compile_expression(target.index)
            self._emit("DELETE_SUBSCRIPT", None, stmt)
            return
        if isinstance(target, TupleLiteral):
            for element in target.elements:
                self._compile_del_target(element, stmt)
            return
        raise ViperSyntaxError("Invalid delete target", stmt.line, stmt.column)

    def _compile_type_alias(self, stmt: TypeAliasStmt) -> None:
        self._compile_expression(stmt.value)
        self._emit("STORE_NAME", stmt.name, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)

    def _compile_match(self, stmt: MatchStmt) -> None:
        for case in stmt.cases:
            if case.guard is not None:
                raise ViperSyntaxError("Match guards are not supported", case.line, case.column)
        subject_name = self._new_temp_name("match_subject")
        cleanup_label = Label()
        self._compile_expression(stmt.subject)
        self._emit("STORE_NAME", subject_name, stmt)
        for case in stmt.cases:
            case_next = Label()
            for index, pattern in enumerate(case.patterns):
                fail_label = case_next if index == len(case.patterns) - 1 else Label()
                self._compile_match_pattern(pattern, subject_name, case, fail_label, cleanup_label)
                if index < len(case.patterns) - 1:
                    self._mark(fail_label)
            self._mark(case_next)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)
        self._emit("JUMP", cleanup_label, stmt)
        self._mark(cleanup_label)
        self._emit("DELETE_NAME", subject_name, stmt)

    def _compile_match_pattern(
        self,
        pattern: Pattern,
        subject_name: str,
        case: MatchCase,
        fail_label: Label,
        cleanup_label: Label,
    ) -> None:
        bindings_name = self._new_temp_name("match_bindings")
        self._emit("LOAD_NAME", "__viper_match_pattern", case)
        self._emit("LOAD_NAME", subject_name, case)
        self._emit("LOAD_CONST", pattern, case)
        self._emit("CALL_FUNCTION", (2, []), case)
        self._emit("STORE_NAME", bindings_name, case)
        self._emit("LOAD_NAME", bindings_name, case)
        self._emit("LOAD_CONST", None, case)
        self._emit("COMPARE_CHAIN", ["=="], case)
        match_label = Label()
        self._emit("JUMP_IF_FALSE", match_label, case)
        self._emit("DELETE_NAME", bindings_name, case)
        self._emit("JUMP", fail_label, case)
        self._mark(match_label)
        for name in self._collect_pattern_names(pattern):
            self._emit("LOAD_NAME", bindings_name, case)
            self._emit("LOAD_CONST", name, case)
            self._emit("GET_SUBSCRIPT", None, case)
            self._emit("STORE_NAME", name, case)
        self._emit("DELETE_NAME", bindings_name, case)
        self._compile_statements(case.body)
        self._emit("JUMP", cleanup_label, case)

    def _compile_class(self, stmt: ClassDef) -> None:
        class_code = self.compile_class(stmt)
        for base in stmt.bases:
            self._compile_expression(base)
        for _, value in stmt.keywords:
            self._compile_expression(value)
        kwarg_names = [name for name, _ in stmt.keywords]
        self._emit("LOAD_CONST", class_code, stmt)
        self._emit("MAKE_CLASS", (stmt.name, len(stmt.bases), kwarg_names), stmt)
        for decorator in reversed(stmt.decorators):
            self._compile_expression(decorator)
            self._emit("SWAP", None, stmt)
            self._emit("CALL_FUNCTION", (1, []), stmt)
        self._emit("STORE_NAME", stmt.name, stmt)
        self._emit("LOAD_CONST", None, stmt)
        self._emit("SET_RESULT", None, stmt)

    def _compile_store_target(self, target: Expression, stmt: Statement) -> None:
        if isinstance(target, Name):
            self._emit("STORE_NAME", target.identifier, stmt)
            return
        if isinstance(target, Attribute):
            self._compile_expression(target.value)
            self._emit("SWAP", None, stmt)
            self._emit("SET_ATTR", target.name, stmt)
            return
        if isinstance(target, Subscript):
            self._compile_expression(target.value)
            self._compile_expression(target.index)
            self._emit("ROT_THREE", None, stmt)
            self._emit("SET_SUBSCRIPT", None, stmt)
            return
        if isinstance(target, TupleLiteral) or isinstance(target, ListLiteral):
            self._emit("UNPACK_SEQUENCE", len(target.elements), stmt)
            for element in target.elements:
                self._compile_store_target(element, stmt)
            return
        raise ViperSyntaxError("Invalid for-loop target", stmt.line, stmt.column)

    def _compile_yield_from(self, expr: YieldFromExpr) -> None:
        self._compile_expression(expr.value)
        self._emit("BUILD_ITER", None, expr)
        start_label = Label()
        end_label = Label()
        self._mark(start_label)
        self._emit("FOR_ITER", end_label, expr)
        self._emit("YIELD_VALUE", None, expr)
        self._emit("POP_TOP", None, expr)
        self._emit("JUMP", start_label, expr)
        self._mark(end_label)
        self._emit("LOAD_CONST", None, expr)

    def _compile_expression(self, expr: Expression) -> None:
        if isinstance(expr, Name):
            self._emit("LOAD_NAME", expr.identifier, expr)
            return
        if isinstance(expr, AssignmentExpr):
            self._compile_expression(expr.value)
            self._emit("STORE_NAME", expr.target.identifier, expr)
            self._emit("LOAD_NAME", expr.target.identifier, expr)
            return
        if isinstance(expr, AwaitExpr):
            if not self._in_coroutine:
                raise ViperSyntaxError("'await' outside async function", expr.line, expr.column)
            self._compile_expression(expr.value)
            self._emit("AWAIT", None, expr)
            return
        if isinstance(expr, LambdaExpr):
            lambda_code = self._compile_lambda(expr)
            pos_defaults, kw_defaults = self._collect_param_defaults(expr.params)
            for default_expr in pos_defaults:
                self._compile_expression(default_expr)
            kw_default_names = [name for name, _ in kw_defaults]
            for _, default_expr in kw_defaults:
                self._compile_expression(default_expr)
            self._emit("LOAD_CONST", lambda_code, expr)
            self._emit("MAKE_FUNCTION", (len(pos_defaults), kw_default_names), expr)
            return
        if isinstance(expr, Starred):
            raise ViperSyntaxError("Invalid starred expression", expr.line, expr.column)
        if isinstance(expr, Literal):
            self._emit("LOAD_CONST", expr.value, expr)
            return
        if isinstance(expr, FormattedString):
            self._compile_formatted_string(expr)
            return
        if isinstance(expr, ListLiteral):
            if any(isinstance(element, Starred) for element in expr.elements):
                flags: List[bool] = []
                for element in expr.elements:
                    if isinstance(element, Starred):
                        self._compile_expression(element.target)
                        flags.append(True)
                    else:
                        self._compile_expression(element)
                        flags.append(False)
                self._emit("BUILD_LIST_UNPACK", flags, expr)
            else:
                for element in expr.elements:
                    self._compile_expression(element)
                self._emit("BUILD_LIST", len(expr.elements), expr)
            return
        if isinstance(expr, TupleLiteral):
            if any(isinstance(element, Starred) for element in expr.elements):
                flags = []
                for element in expr.elements:
                    if isinstance(element, Starred):
                        self._compile_expression(element.target)
                        flags.append(True)
                    else:
                        self._compile_expression(element)
                        flags.append(False)
                self._emit("BUILD_TUPLE_UNPACK", flags, expr)
            else:
                for element in expr.elements:
                    self._compile_expression(element)
                self._emit("BUILD_TUPLE", len(expr.elements), expr)
            return
        if isinstance(expr, DictLiteral):
            if any(key is None for key, _ in expr.items):
                flags: List[bool] = []
                for key, value in expr.items:
                    if key is None:
                        self._compile_expression(value)
                        flags.append(True)
                    else:
                        self._compile_expression(key)
                        self._compile_expression(value)
                        flags.append(False)
                self._emit("BUILD_DICT_UNPACK", flags, expr)
            else:
                for key, value in expr.items:
                    self._compile_expression(key)
                    self._compile_expression(value)
                self._emit("BUILD_DICT", len(expr.items), expr)
            return
        if isinstance(expr, ConditionalExpr):
            else_label = Label()
            end_label = Label()
            self._compile_expression(expr.test)
            self._emit("JUMP_IF_FALSE", else_label, expr)
            self._compile_expression(expr.body)
            self._emit("JUMP", end_label, expr)
            self._mark(else_label)
            self._compile_expression(expr.orelse)
            self._mark(end_label)
            return
        if isinstance(expr, UnaryOp):
            self._compile_expression(expr.operand)
            self._emit("UNARY_OP", expr.op, expr)
            return
        if isinstance(expr, BinaryOp):
            if expr.op == "and":
                end_label = Label()
                self._compile_expression(expr.left)
                self._emit("JUMP_IF_FALSE_KEEP", end_label, expr)
                self._emit("POP_TOP", None, expr)
                self._compile_expression(expr.right)
                self._mark(end_label)
                return
            if expr.op == "or":
                end_label = Label()
                self._compile_expression(expr.left)
                self._emit("JUMP_IF_TRUE_KEEP", end_label, expr)
                self._emit("POP_TOP", None, expr)
                self._compile_expression(expr.right)
                self._mark(end_label)
                return
            self._compile_expression(expr.left)
            self._compile_expression(expr.right)
            self._emit("BINARY_OP", expr.op, expr)
            return
        if isinstance(expr, CompareOp):
            self._compile_expression(expr.left)
            for comparator in expr.comparators:
                self._compile_expression(comparator)
            self._emit("COMPARE_CHAIN", expr.ops, expr)
            return
        if isinstance(expr, YieldExpr):
            if not self._in_generator:
                raise ViperSyntaxError("'yield' outside function", expr.line, expr.column)
            if expr.value is None:
                self._emit("LOAD_CONST", None, expr)
            else:
                self._compile_expression(expr.value)
            self._emit("YIELD_VALUE", None, expr)
            return
        if isinstance(expr, YieldFromExpr):
            if not self._in_generator:
                raise ViperSyntaxError("'yield' outside function", expr.line, expr.column)
            self._compile_yield_from(expr)
            return
        if isinstance(expr, Call):
            has_star_args = any(isinstance(arg, Starred) for arg in expr.args)
            has_kw_unpack = any(name is None for name, _ in expr.kwargs)
            if not has_star_args and not has_kw_unpack:
                self._compile_expression(expr.func)
                for arg in expr.args:
                    self._compile_expression(arg)
                for name, value in expr.kwargs:
                    if name is None:
                        raise ViperSyntaxError("Invalid keyword unpack", expr.line, expr.column)
                    self._compile_expression(value)
                kwarg_names = [name for name, _ in expr.kwargs if name is not None]
                self._emit("CALL_FUNCTION", (len(expr.args), kwarg_names), expr)
                return
            self._compile_expression(expr.func)
            if expr.args:
                if has_star_args:
                    flags: List[bool] = []
                    for arg in expr.args:
                        if isinstance(arg, Starred):
                            self._compile_expression(arg.target)
                            flags.append(True)
                        else:
                            self._compile_expression(arg)
                            flags.append(False)
                    self._emit("BUILD_LIST_UNPACK", flags, expr)
                else:
                    for arg in expr.args:
                        self._compile_expression(arg)
                    self._emit("BUILD_LIST", len(expr.args), expr)
            else:
                self._emit("BUILD_LIST", 0, expr)
            has_kwargs = False
            if expr.kwargs:
                flags = []
                for name, value in expr.kwargs:
                    if name is None:
                        self._compile_expression(value)
                        flags.append(True)
                    else:
                        self._emit("LOAD_CONST", name, expr)
                        self._compile_expression(value)
                        flags.append(False)
                if any(flags):
                    self._emit("BUILD_DICT_UNPACK", flags, expr)
                else:
                    self._emit("BUILD_DICT", len(expr.kwargs), expr)
                has_kwargs = True
            self._emit("CALL_FUNCTION_EX", has_kwargs, expr)
            return
        if isinstance(expr, Attribute):
            self._compile_expression(expr.value)
            self._emit("GET_ATTR", expr.name, expr)
            return
        if isinstance(expr, Subscript):
            self._compile_expression(expr.value)
            self._compile_expression(expr.index)
            self._emit("GET_SUBSCRIPT", None, expr)
            return
        if isinstance(expr, SliceExpr):
            if expr.start is None:
                self._emit("LOAD_CONST", None, expr)
            else:
                self._compile_expression(expr.start)
            if expr.stop is None:
                self._emit("LOAD_CONST", None, expr)
            else:
                self._compile_expression(expr.stop)
            if expr.step is None:
                self._emit("LOAD_CONST", None, expr)
            else:
                self._compile_expression(expr.step)
            self._emit("BUILD_SLICE", 3, expr)
            return
        raise ViperSyntaxError("Unsupported expression", expr.line, expr.column)

    def _compile_formatted_string(self, expr: FormattedString) -> None:
        part_count = 0
        for part in expr.parts:
            if isinstance(part, StringText):
                literal = part.value
                if expr.is_bytes:
                    if isinstance(literal, bytes):
                        self._emit("LOAD_CONST", literal, part)
                    else:
                        self._emit("LOAD_CONST", str(literal).encode("utf-8"), part)
                else:
                    if isinstance(literal, bytes):
                        raise ViperSyntaxError(
                            "Cannot mix bytes with formatted strings", part.line, part.column
                        )
                    self._emit("LOAD_CONST", literal, part)
                part_count += 1
                continue
            if isinstance(part, FormatField):
                if part.debug and part.expr_text:
                    debug_text = f"{part.expr_text}="
                    if expr.is_bytes:
                        self._emit("LOAD_CONST", debug_text.encode("utf-8"), part)
                    else:
                        self._emit("LOAD_CONST", debug_text, part)
                    part_count += 1
                self._compile_expression(part.expression)
                if expr.is_template:
                    if part.format_spec is None:
                        self._emit("LOAD_CONST", None, part)
                    else:
                        self._compile_format_spec(part.format_spec)
                    self._emit("BUILD_TEMPLATE_FIELD", part.conversion, part)
                    part_count += 1
                else:
                    if part.conversion:
                        self._emit("CONVERT_VALUE", part.conversion, part)
                    if part.format_spec is None:
                        self._emit("LOAD_CONST", None, part)
                    else:
                        self._compile_format_spec(part.format_spec)
                    self._emit("FORMAT_VALUE", None, part)
                    if expr.is_bytes:
                        self._emit("ENCODE_UTF8", None, part)
                    part_count += 1
                continue
            raise ViperSyntaxError("Unsupported formatted string part", expr.line, expr.column)

        if expr.is_template:
            if expr.is_bytes:
                self._emit("BUILD_TEMPLATE_BYTES", part_count, expr)
            else:
                self._emit("BUILD_TEMPLATE", part_count, expr)
        else:
            if expr.is_bytes:
                self._emit("BUILD_BYTES", part_count, expr)
            else:
                self._emit("BUILD_STRING", part_count, expr)

    def _compile_format_spec(self, parts: List[StringText | FormatField]) -> None:
        if not parts:
            self._emit("LOAD_CONST", "", None)
            return
        part_count = 0
        for part in parts:
            if isinstance(part, StringText):
                self._emit("LOAD_CONST", part.value, part)
                part_count += 1
                continue
            if isinstance(part, FormatField):
                if part.debug and part.expr_text:
                    self._emit("LOAD_CONST", f"{part.expr_text}=", part)
                    part_count += 1
                self._compile_expression(part.expression)
                if part.conversion:
                    self._emit("CONVERT_VALUE", part.conversion, part)
                if part.format_spec is None:
                    self._emit("LOAD_CONST", None, part)
                else:
                    self._compile_format_spec(part.format_spec)
                self._emit("FORMAT_VALUE", None, part)
                part_count += 1
                continue
            raise ViperSyntaxError("Unsupported format spec", 0, 0)
        self._emit("BUILD_STRING", part_count, None)

    def _build_param_spec(self, params: List[Parameter]) -> ParamSpec:
        posonly: List[str] = []
        pos_or_kw: List[str] = []
        kwonly: List[str] = []
        vararg: Optional[str] = None
        varkw: Optional[str] = None
        for param in params:
            if param.kind == "posonly":
                posonly.append(param.name)
            elif param.kind == "poskw":
                pos_or_kw.append(param.name)
            elif param.kind == "kwonly":
                kwonly.append(param.name)
            elif param.kind == "varpos":
                vararg = param.name
            elif param.kind == "varkw":
                varkw = param.name
        return ParamSpec(
            posonly=posonly,
            pos_or_kw=pos_or_kw,
            kwonly=kwonly,
            vararg=vararg,
            varkw=varkw,
        )

    def _collect_param_defaults(
        self, params: List[Parameter]
    ) -> Tuple[List[Expression], List[Tuple[str, Expression]]]:
        positional = [param for param in params if param.kind in {"posonly", "poskw"}]
        pos_defaults: List[Expression] = []
        for param in reversed(positional):
            if param.default is None:
                break
            pos_defaults.insert(0, param.default)
        cutoff = len(positional) - len(pos_defaults)
        for param in positional[:cutoff]:
            if param.default is not None:
                raise ViperSyntaxError(
                    "Non-default argument follows default argument", param.line, param.column
                )
        kw_defaults = [
            (param.name, param.default)
            for param in params
            if param.kind == "kwonly" and param.default is not None
        ]
        return pos_defaults, kw_defaults

    def _new_temp_name(self, prefix: str) -> str:
        name = f"__viper_{prefix}_{self._temp_counter}"
        self._temp_counter += 1
        return name

    def _collect_pattern_names(self, pattern: Pattern) -> List[str]:
        names: List[str] = []
        seen: set[str] = set()

        def _walk(node: Pattern) -> None:
            if isinstance(node, PatternName):
                if node.identifier not in seen:
                    seen.add(node.identifier)
                    names.append(node.identifier)
                return
            if isinstance(node, PatternSequence):
                for element in node.elements:
                    _walk(element)
                return
            if isinstance(node, PatternOr):
                for element in node.patterns:
                    _walk(element)
                return
            if isinstance(node, (PatternLiteral, PatternWildcard)):
                return

        _walk(pattern)
        return names

    def _emit(self, op: str, arg: Optional[object] = None, node: Optional[object] = None) -> None:
        line = 0
        column = 0
        if node is not None:
            line = getattr(node, "line", 0)
            column = getattr(node, "column", 0)
        self._instructions.append(Instruction(op=op, arg=arg, line=line, column=column))

    def _mark(self, label: Label) -> None:
        if label.position is not None:
            raise ValueError("Label already marked")
        label.position = len(self._instructions)

    def _resolve_labels(self, code: CodeObject) -> None:
        for instr in code.instructions:
            if isinstance(instr.arg, Label):
                if instr.arg.position is None:
                    raise ValueError("Unresolved label")
                instr.arg = instr.arg.position


def compile_module(module: Module) -> CodeObject:
    """Compile a Viper AST module into bytecode."""
    return BytecodeCompiler().compile_module(module)


def _contains_yield(statements: List[Statement]) -> bool:
    def expr_has_yield(expr: Expression) -> bool:
        if isinstance(expr, (YieldExpr, YieldFromExpr)):
            return True
        if isinstance(expr, AssignmentExpr):
            return expr_has_yield(expr.value)
        if isinstance(expr, Starred):
            return expr_has_yield(expr.target)
        if isinstance(expr, ConditionalExpr):
            return (
                expr_has_yield(expr.test)
                or expr_has_yield(expr.body)
                or expr_has_yield(expr.orelse)
            )
        if isinstance(expr, UnaryOp):
            return expr_has_yield(expr.operand)
        if isinstance(expr, BinaryOp):
            return expr_has_yield(expr.left) or expr_has_yield(expr.right)
        if isinstance(expr, CompareOp):
            return expr_has_yield(expr.left) or any(expr_has_yield(c) for c in expr.comparators)
        if isinstance(expr, Call):
            return expr_has_yield(expr.func) or any(expr_has_yield(a) for a in expr.args) or any(
                expr_has_yield(v) for _, v in expr.kwargs
            )
        if isinstance(expr, Attribute):
            return expr_has_yield(expr.value)
        if isinstance(expr, Subscript):
            return expr_has_yield(expr.value) or expr_has_yield(expr.index)
        if isinstance(expr, SliceExpr):
            return any(
                part is not None and expr_has_yield(part)
                for part in (expr.start, expr.stop, expr.step)
            )
        if isinstance(expr, ListLiteral):
            return any(expr_has_yield(e) for e in expr.elements)
        if isinstance(expr, TupleLiteral):
            return any(expr_has_yield(e) for e in expr.elements)
        if isinstance(expr, DictLiteral):
            return any(
                (k is not None and expr_has_yield(k)) or expr_has_yield(v)
                for k, v in expr.items
            )
        if isinstance(expr, FormattedString):
            for part in expr.parts:
                if isinstance(part, FormatField):
                    if expr_has_yield(part.expression):
                        return True
                    if part.format_spec and any(
                        isinstance(item, FormatField) and expr_has_yield(item.expression)
                        for item in part.format_spec
                    ):
                        return True
            return False
        return False

    def stmt_has_yield(stmt: Statement) -> bool:
        if isinstance(stmt, ExpressionStmt):
            return expr_has_yield(stmt.expression)
        if isinstance(stmt, AssignStmt):
            return any(expr_has_yield(t) for t in stmt.targets) or expr_has_yield(stmt.value)
        if isinstance(stmt, AugAssignStmt):
            return expr_has_yield(stmt.target) or expr_has_yield(stmt.value)
        if isinstance(stmt, AnnAssignStmt):
            return (
                expr_has_yield(stmt.target)
                or expr_has_yield(stmt.annotation)
                or (stmt.value is not None and expr_has_yield(stmt.value))
            )
        if isinstance(stmt, IfStmt):
            return (
                expr_has_yield(stmt.test)
                or any(stmt_has_yield(s) for s in stmt.body)
                or any(expr_has_yield(test) or any(stmt_has_yield(s) for s in body) for test, body in stmt.elif_blocks)
                or (stmt.else_body is not None and any(stmt_has_yield(s) for s in stmt.else_body))
            )
        if isinstance(stmt, WhileStmt):
            return (
                expr_has_yield(stmt.test)
                or any(stmt_has_yield(s) for s in stmt.body)
                or (stmt.else_body is not None and any(stmt_has_yield(s) for s in stmt.else_body))
            )
        if isinstance(stmt, ForStmt):
            return (
                expr_has_yield(stmt.target)
                or expr_has_yield(stmt.iterable)
                or any(stmt_has_yield(s) for s in stmt.body)
                or (stmt.else_body is not None and any(stmt_has_yield(s) for s in stmt.else_body))
            )
        if isinstance(stmt, WithStmt):
            return any(expr_has_yield(item.context_expr) for item in stmt.items) or any(
                stmt_has_yield(s) for s in stmt.body
            )
        if isinstance(stmt, TryStmt):
            return (
                any(stmt_has_yield(s) for s in stmt.body)
                or any(stmt_has_yield(s) for handler in stmt.handlers for s in handler.body)
                or (stmt.else_body is not None and any(stmt_has_yield(s) for s in stmt.else_body))
                or (stmt.finally_body is not None and any(stmt_has_yield(s) for s in stmt.finally_body))
            )
        if isinstance(stmt, MatchStmt):
            return expr_has_yield(stmt.subject) or any(
                any(stmt_has_yield(s) for s in case.body) for case in stmt.cases
            )
        if isinstance(stmt, RaiseStmt):
            return (stmt.exception is not None and expr_has_yield(stmt.exception)) or (
                stmt.cause is not None and expr_has_yield(stmt.cause)
            )
        if isinstance(stmt, AssertStmt):
            return expr_has_yield(stmt.test) or (stmt.message is not None and expr_has_yield(stmt.message))
        if isinstance(stmt, ReturnStmt):
            return stmt.value is not None and expr_has_yield(stmt.value)
        if isinstance(stmt, DelStmt):
            return any(expr_has_yield(target) for target in stmt.targets)
        if isinstance(stmt, TypeAliasStmt):
            return expr_has_yield(stmt.value)
        if isinstance(stmt, (FunctionDef, ClassDef, ImportStmt, ImportFromStmt, GlobalStmt, NonlocalStmt, PassStmt, BreakStmt, ContinueStmt)):
            return False
        return False

    return any(stmt_has_yield(stmt) for stmt in statements)
