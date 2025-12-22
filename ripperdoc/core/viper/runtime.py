"""Bytecode execution engine for Viper."""

from __future__ import annotations

from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ripperdoc.core.viper.bytecode import CodeObject, Instruction
from ripperdoc.core.viper.ast_nodes import (
    Pattern,
    PatternLiteral,
    PatternName,
    PatternOr,
    PatternSequence,
    PatternWildcard,
)
from ripperdoc.core.viper.errors import ViperRuntimeError

_NO_VALUE = object()


@dataclass
class ExecutionResult:
    value: Any
    globals: Dict[str, Any]


class Environment:
    def __init__(self, parent: Optional[Environment] = None) -> None:
        self._values: Dict[str, Any] = {}
        self._parent = parent

    def get(self, name: str, line: int, column: int) -> Any:
        if name in self._values:
            return self._values[name]
        if self._parent is not None:
            return self._parent.get(name, line, column)
        raise ViperRuntimeError(f"Undefined name '{name}'", line, column)

    def set(self, name: str, value: Any) -> None:
        self._values[name] = value

    def delete(self, name: str) -> bool:
        if name in self._values:
            del self._values[name]
            return True
        return False

    def set_nonlocal(self, name: str, value: Any) -> bool:
        if self._parent is None:
            return False
        if name in self._parent._values:
            self._parent._values[name] = value
            return True
        return self._parent.set_nonlocal(name, value)

    def delete_nonlocal(self, name: str) -> bool:
        if self._parent is None:
            return False
        if name in self._parent._values:
            del self._parent._values[name]
            return True
        return self._parent.delete_nonlocal(name)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._values)


@dataclass
class ViperFunction:
    code: CodeObject
    closure: Environment
    vm: "VirtualMachine"

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        return BoundViperMethod(func=self, instance=instance, vm=self.vm)


@dataclass
class BoundViperMethod:
    func: ViperFunction
    instance: Any
    vm: "VirtualMachine"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call_with_instr(list(args), kwargs, None)

    def call_with_instr(
        self, args: List[Any], kwargs: Dict[str, Any], instr: Optional[Instruction]
    ) -> Any:
        if instr is None:
            instr = Instruction(
                op="CALL_FUNCTION",
                arg=(len(args) + 1, list(kwargs.keys())),
                line=0,
                column=0,
            )
        return self.vm._call_viper_function(
            self.func,
            [self.instance] + list(args),
            kwargs,
            instr,
        )


@dataclass
class YieldSignal:
    value: Any


@dataclass
class GeneratorReturn:
    value: Any


@dataclass
class ViperGenerator:
    frame: "Frame"
    vm: "VirtualMachine"
    started: bool = False
    finished: bool = False

    def __iter__(self) -> "ViperGenerator":
        return self

    def __next__(self) -> Any:
        return self.send(None)

    def send(self, value: Any) -> Any:
        if self.finished:
            raise StopIteration
        if not self.started:
            self.started = True
            if value is not None:
                raise TypeError("Cannot send non-None value to a just-started generator")
        result = self.vm._run_frame(self.frame, stop_on_yield=True, send_value=value)
        if isinstance(result, YieldSignal):
            return result.value
        if isinstance(result, GeneratorReturn):
            self.finished = True
            raise StopIteration(result.value)
        self.finished = True
        raise StopIteration(result)


@dataclass
class ViperCoroutine(Coroutine):
    frame: "Frame"
    vm: "VirtualMachine"
    finished: bool = False
    result: Any = None

    def _run(self) -> Any:
        if not self.finished:
            self.result = self.vm._run_frame(self.frame)
            self.finished = True
        return self.result

    def __await__(self):
        result = self._run()
        if False:  # pragma: no cover
            yield None
        return result

    def __iter__(self):
        return self.__await__()

    def send(self, value: Any) -> Any:
        if value is not None:
            raise TypeError("Cannot send non-None value to a just-started coroutine")
        if self.finished:
            raise StopIteration(self.result)
        result = self._run()
        raise StopIteration(result)

    def throw(self, typ, val=None, tb=None):
        self.finished = True
        if val is None:
            val = typ()
        if tb is not None:
            raise val.with_traceback(tb)
        raise val

    def close(self) -> None:
        self.finished = True


@dataclass
class TemplateField:
    value: Any
    conversion: Optional[object]
    format_spec: Optional[str]

    def render(self) -> str:
        value = apply_conversion(self.value, self.conversion)
        return format_value(value, self.format_spec)


@dataclass
class TemplateString:
    parts: List[str | TemplateField]

    def render(self) -> str:
        rendered: List[str] = []
        for part in self.parts:
            if isinstance(part, TemplateField):
                rendered.append(part.render())
            else:
                rendered.append(str(part))
        return "".join(rendered)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"TemplateString({self.parts!r})"


@dataclass
class TemplateBytes:
    parts: List[bytes | TemplateField]

    def render_bytes(self) -> bytes:
        rendered: List[bytes] = []
        for part in self.parts:
            if isinstance(part, TemplateField):
                rendered.append(part.render().encode("utf-8"))
            else:
                rendered.append(part)
        return b"".join(rendered)

    def __bytes__(self) -> bytes:
        return self.render_bytes()

    def __str__(self) -> str:
        return self.render_bytes().decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"TemplateBytes({self.parts!r})"


@dataclass
class Frame:
    code: CodeObject
    env: Environment
    stack: List[Any]
    ip: int
    last_value: Any
    global_names: set[str] = field(default_factory=set)
    nonlocal_names: set[str] = field(default_factory=set)
    awaiting_send: bool = False
    with_stack: List[Any] = field(default_factory=list)
    try_stack: List["TryHandler"] = field(default_factory=list)
    pending_exception: Optional[BaseException] = None
    last_exception: Optional[BaseException] = None


@dataclass
class TryHandler:
    kind: str
    handler_ip: int
    stack_depth: int
    with_depth: int


class VirtualMachine:
    def __init__(
        self,
        globals: Optional[Mapping[str, Any]] = None,
        builtins: Optional[Mapping[str, Any]] = None,
    ) -> None:
        builtins_env = Environment()
        for name, value in (builtins or default_builtins()).items():
            builtins_env.set(name, value)
        self.globals = Environment(builtins_env)
        if globals:
            for name, value in globals.items():
                self.globals.set(name, value)

    def execute(self, code: CodeObject) -> ExecutionResult:
        frame = Frame(code=code, env=self.globals, stack=[], ip=0, last_value=None)
        value = self._run_frame(frame)
        return ExecutionResult(value=value, globals=self.globals.snapshot())

    def _run_frame(self, frame: Frame, *, stop_on_yield: bool = False, send_value: Any = _NO_VALUE) -> Any:
        instructions = frame.code.instructions
        instr = Instruction(op="LOAD_CONST", arg=None, line=0, column=0)
        if stop_on_yield and frame.awaiting_send:
            if send_value is _NO_VALUE:
                send_value = None
            frame.stack.append(send_value)
            frame.awaiting_send = False
        while frame.ip < len(instructions):
            instr = instructions[frame.ip]
            frame.ip += 1
            op = instr.op
            try:
                if op == "LOAD_CONST":
                    frame.stack.append(instr.arg)
                elif op == "LOAD_NAME":
                    frame.stack.append(self._load_name(frame, instr.arg, instr))
                elif op == "STORE_NAME":
                    value = frame.stack.pop()
                    self._store_name(frame, instr.arg, value, instr)
                elif op == "DELETE_NAME":
                    self._delete_name(frame, instr.arg, instr)
                elif op == "SET_RESULT":
                    frame.last_value = frame.stack.pop()
                elif op == "LOAD_RESULT":
                    frame.stack.append(frame.last_value)
                elif op == "POP_TOP":
                    frame.stack.pop()
                elif op == "BUILD_STRING":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    frame.stack.append("".join(str(item) for item in items))
                elif op == "BUILD_BYTES":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    rendered: List[bytes] = []
                    for item in items:
                        if isinstance(item, bytes):
                            rendered.append(item)
                        elif isinstance(item, str):
                            rendered.append(item.encode("utf-8"))
                        else:
                            rendered.append(str(item).encode("utf-8"))
                    frame.stack.append(b"".join(rendered))
                elif op == "ENCODE_UTF8":
                    value = frame.stack.pop()
                    if isinstance(value, bytes):
                        frame.stack.append(value)
                    elif isinstance(value, str):
                        frame.stack.append(value.encode("utf-8"))
                    else:
                        frame.stack.append(str(value).encode("utf-8"))
                elif op == "CONVERT_VALUE":
                    value = frame.stack.pop()
                    frame.stack.append(self._apply_conversion(instr.arg, value, instr))
                elif op == "FORMAT_VALUE":
                    format_spec = frame.stack.pop()
                    value = frame.stack.pop()
                    frame.stack.append(self._format_value(value, format_spec, instr))
                elif op == "BUILD_TEMPLATE_FIELD":
                    format_spec = frame.stack.pop()
                    value = frame.stack.pop()
                    frame.stack.append(
                        TemplateField(
                            value=value,
                            conversion=self._resolve_conversion(instr.arg, instr),
                            format_spec=format_spec,
                        )
                    )
                elif op == "BUILD_TEMPLATE":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    frame.stack.append(TemplateString(parts=list(items)))
                elif op == "BUILD_TEMPLATE_BYTES":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    parts: List[bytes | TemplateField] = []
                    for item in items:
                        if isinstance(item, TemplateField):
                            parts.append(item)
                        elif isinstance(item, bytes):
                            parts.append(item)
                        elif isinstance(item, str):
                            parts.append(item.encode("utf-8"))
                        else:
                            parts.append(str(item).encode("utf-8"))
                    frame.stack.append(TemplateBytes(parts=parts))
                elif op == "BUILD_LIST":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    frame.stack.append(list(items))
                elif op == "BUILD_TUPLE":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    frame.stack.append(tuple(items))
                elif op == "BUILD_DICT":
                    count = int(instr.arg)
                    items: List[tuple[Any, Any]] = []
                    for _ in range(count):
                        value = frame.stack.pop()
                        key = frame.stack.pop()
                        items.append((key, value))
                    items.reverse()
                    frame.stack.append(dict(items))
                elif op == "BUILD_SLICE":
                    step = frame.stack.pop()
                    stop = frame.stack.pop()
                    start = frame.stack.pop()
                    frame.stack.append(slice(start, stop, step))
                elif op == "UNPACK_SEQUENCE":
                    count = int(instr.arg)
                    iterable = frame.stack.pop()
                    try:
                        items = list(iterable)
                    except Exception as exc:  # noqa: BLE001
                        raise ViperRuntimeError("Unpack failed", instr.line, instr.column) from exc
                    if len(items) != count:
                        raise ViperRuntimeError("Unpack mismatch", instr.line, instr.column)
                    for item in reversed(items):
                        frame.stack.append(item)
                elif op == "UNARY_OP":
                    operand = frame.stack.pop()
                    frame.stack.append(self._apply_unary(instr.arg, operand, instr))
                elif op == "BINARY_OP":
                    right = frame.stack.pop()
                    left = frame.stack.pop()
                    frame.stack.append(self._apply_binary(instr.arg, left, right, instr))
                elif op == "COMPARE_CHAIN":
                    ops = list(instr.arg or [])
                    values = [frame.stack.pop() for _ in range(len(ops))]
                    values.reverse()
                    left = frame.stack.pop()
                    frame.stack.append(self._apply_compare_chain(ops, left, values, instr))
                elif op == "JUMP":
                    frame.ip = int(instr.arg)
                elif op == "JUMP_IF_FALSE":
                    value = frame.stack.pop()
                    if not self._truthy(value):
                        frame.ip = int(instr.arg)
                elif op == "JUMP_IF_TRUE":
                    value = frame.stack.pop()
                    if self._truthy(value):
                        frame.ip = int(instr.arg)
                elif op == "JUMP_IF_FALSE_KEEP":
                    value = frame.stack[-1]
                    if not self._truthy(value):
                        frame.ip = int(instr.arg)
                elif op == "JUMP_IF_TRUE_KEEP":
                    value = frame.stack[-1]
                    if self._truthy(value):
                        frame.ip = int(instr.arg)
                elif op == "GET_ATTR":
                    obj = frame.stack.pop()
                    frame.stack.append(self._get_attr(obj, instr.arg, instr))
                elif op == "SET_ATTR":
                    value = frame.stack.pop()
                    obj = frame.stack.pop()
                    self._set_attr(obj, instr.arg, value, instr)
                elif op == "DELETE_ATTR":
                    obj = frame.stack.pop()
                    try:
                        delattr(obj, instr.arg)
                    except Exception as exc:  # noqa: BLE001
                        raise ViperRuntimeError("Delete attribute failed", instr.line, instr.column) from exc
                elif op == "GET_SUBSCRIPT":
                    index = frame.stack.pop()
                    obj = frame.stack.pop()
                    frame.stack.append(self._get_subscript(obj, index, instr))
                elif op == "SET_SUBSCRIPT":
                    value = frame.stack.pop()
                    index = frame.stack.pop()
                    obj = frame.stack.pop()
                    self._set_subscript(obj, index, value, instr)
                elif op == "DELETE_SUBSCRIPT":
                    index = frame.stack.pop()
                    obj = frame.stack.pop()
                    try:
                        del obj[index]
                    except Exception as exc:  # noqa: BLE001
                        raise ViperRuntimeError("Delete subscript failed", instr.line, instr.column) from exc
                elif op == "IMPORT_NAME":
                    module_name, fromlist, level = instr.arg
                    try:
                        module = __import__(
                            module_name,
                            {},
                            {},
                            [] if fromlist is None else list(fromlist),
                            int(level),
                        )
                    except Exception as exc:  # noqa: BLE001
                        raise ViperRuntimeError("Import failed", instr.line, instr.column) from exc
                    frame.stack.append(module)
                elif op == "IMPORT_FROM":
                    module = frame.stack[-1]
                    try:
                        value = getattr(module, instr.arg)
                    except Exception as exc:  # noqa: BLE001
                        raise ViperRuntimeError("Import from failed", instr.line, instr.column) from exc
                    frame.stack.append(value)
                elif op == "IMPORT_STAR":
                    module = frame.stack.pop()
                    namespace = getattr(module, "__dict__", {})
                    for name, value in namespace.items():
                        if name.startswith("_"):
                            continue
                        frame.env.set(name, value)
                elif op == "DECLARE_GLOBAL":
                    for name in instr.arg:
                        if name in frame.nonlocal_names:
                            raise ViperRuntimeError(
                                "Name declared nonlocal and global", instr.line, instr.column
                            )
                        frame.global_names.add(name)
                elif op == "DECLARE_NONLOCAL":
                    if frame.code.is_module:
                        raise ViperRuntimeError("Nonlocal declaration in module", instr.line, instr.column)
                    for name in instr.arg:
                        if name in frame.global_names:
                            raise ViperRuntimeError(
                                "Name declared global and nonlocal", instr.line, instr.column
                            )
                        frame.nonlocal_names.add(name)
                elif op == "BUILD_ITER":
                    iterable = frame.stack.pop()
                    frame.stack.append(self._build_iter(iterable, instr))
                elif op == "FOR_ITER":
                    iterator = frame.stack[-1]
                    try:
                        item = next(iterator)
                    except StopIteration:
                        frame.stack.pop()
                        frame.ip = int(instr.arg)
                    else:
                        frame.stack.append(item)
                elif op == "WITH_ENTER":
                    manager = frame.stack.pop()
                    frame.stack.append(self._with_enter(manager, frame, instr))
                elif op == "WITH_EXIT":
                    self._with_exit(frame, instr)
                elif op == "SETUP_EXCEPT":
                    frame.try_stack.append(
                        TryHandler(
                            kind="except",
                            handler_ip=int(instr.arg),
                            stack_depth=len(frame.stack),
                            with_depth=len(frame.with_stack),
                        )
                    )
                elif op == "POP_EXCEPT":
                    if not frame.try_stack or frame.try_stack[-1].kind != "except":
                        raise ViperRuntimeError("Exception stack underflow", instr.line, instr.column)
                    frame.try_stack.pop()
                elif op == "SETUP_FINALLY":
                    frame.try_stack.append(
                        TryHandler(
                            kind="finally",
                            handler_ip=int(instr.arg),
                            stack_depth=len(frame.stack),
                            with_depth=len(frame.with_stack),
                        )
                    )
                elif op == "POP_FINALLY":
                    if not frame.try_stack or frame.try_stack[-1].kind != "finally":
                        raise ViperRuntimeError("Finally stack underflow", instr.line, instr.column)
                    frame.try_stack.pop()
                elif op == "END_FINALLY":
                    if frame.pending_exception is not None:
                        exc = frame.pending_exception
                        frame.pending_exception = None
                        raise exc
                elif op == "LOAD_EXCEPTION":
                    if frame.pending_exception is None:
                        raise ViperRuntimeError("No active exception", instr.line, instr.column)
                    frame.stack.append(frame.pending_exception)
                elif op == "CLEAR_EXCEPTION":
                    frame.pending_exception = None
                elif op == "EXC_MATCH":
                    exc_type = frame.stack.pop()
                    exc = frame.stack.pop()
                    try:
                        frame.stack.append(isinstance(exc, exc_type))
                    except Exception as exc_inner:  # noqa: BLE001
                        raise ViperRuntimeError(
                            "Invalid exception type", instr.line, instr.column
                        ) from exc_inner
                elif op == "RAISE":
                    has_exc, has_cause = instr.arg
                    cause = None
                    if has_cause:
                        cause = frame.stack.pop()
                    exc_obj = None
                    if has_exc:
                        exc_obj = frame.stack.pop()
                        exc = self._coerce_exception(exc_obj, instr)
                    else:
                        if frame.last_exception is None:
                            raise ViperRuntimeError("No active exception", instr.line, instr.column)
                        exc = frame.last_exception
                    if has_cause:
                        if cause is None:
                            try:
                                exc.__cause__ = None
                                exc.__suppress_context__ = True
                            except Exception:
                                pass
                        else:
                            cause_exc = self._coerce_exception(cause, instr)
                            try:
                                exc.__cause__ = cause_exc
                                exc.__suppress_context__ = True
                            except Exception:
                                pass
                    frame.last_exception = exc
                    raise exc
                elif op == "ASSERT":
                    has_message = bool(instr.arg)
                    message = frame.stack.pop() if has_message else None
                    raise AssertionError(message)
                elif op == "CALL_FUNCTION":
                    argc, kwarg_names = instr.arg
                    kwargs = {}
                    for name in reversed(list(kwarg_names)):
                        kwargs[name] = frame.stack.pop()
                    args = [frame.stack.pop() for _ in range(argc)]
                    args.reverse()
                    func = frame.stack.pop()
                    frame.stack.append(self._call(func, args, kwargs, instr))
                elif op == "MAKE_FUNCTION":
                    code_obj = frame.stack.pop()
                    frame.stack.append(ViperFunction(code=code_obj, closure=frame.env, vm=self))
                elif op == "MAKE_CLASS":
                    name, base_count = instr.arg
                    code_obj = frame.stack.pop()
                    bases = [frame.stack.pop() for _ in range(base_count)]
                    bases.reverse()
                    frame.stack.append(self._make_class(name, bases, code_obj, instr))
                elif op == "SWAP":
                    if len(frame.stack) < 2:
                        raise ViperRuntimeError("Stack underflow", instr.line, instr.column)
                    frame.stack[-1], frame.stack[-2] = frame.stack[-2], frame.stack[-1]
                elif op == "ROT_THREE":
                    if len(frame.stack) < 3:
                        raise ViperRuntimeError("Stack underflow", instr.line, instr.column)
                    first, second, third = frame.stack[-3:]
                    frame.stack[-3:] = [second, third, first]
                elif op == "YIELD_VALUE":
                    value = frame.stack.pop()
                    frame.awaiting_send = True
                    if stop_on_yield:
                        return YieldSignal(value=value)
                    raise ViperRuntimeError("Yield outside generator", instr.line, instr.column)
                elif op == "RETURN_VALUE":
                    explicit = bool(instr.arg)
                    if frame.code.is_module and explicit:
                        raise ViperRuntimeError("Return outside function", instr.line, instr.column)
                    value = frame.stack.pop() if frame.stack else None
                    self._close_with_stack(frame, instr)
                    if stop_on_yield and frame.code.is_generator:
                        return GeneratorReturn(value=value)
                    return value
                else:
                    raise ViperRuntimeError("Unknown instruction", instr.line, instr.column)
            except Exception as exc:  # noqa: BLE001
                if self._handle_exception(frame, exc, instr):
                    continue
                self._drain_with_stack(frame, exc, instr)
                if isinstance(exc, ViperRuntimeError):
                    raise
                raise ViperRuntimeError("Execution failed", instr.line, instr.column) from exc
        self._close_with_stack(frame, instr)
        if stop_on_yield and frame.code.is_generator:
            return GeneratorReturn(value=frame.last_value)
        return frame.last_value

    def _call(self, func: Any, args: List[Any], kwargs: Dict[str, Any], instr: Instruction) -> Any:
        if isinstance(func, BoundViperMethod):
            return func.call_with_instr(args, kwargs, instr)
        if isinstance(func, ViperFunction):
            return self._call_viper_function(func, args, kwargs, instr)
        if callable(func):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise ViperRuntimeError("Function call failed", instr.line, instr.column) from exc
        raise ViperRuntimeError("Object is not callable", instr.line, instr.column)

    def _load_name(self, frame: Frame, name: str, instr: Instruction) -> Any:
        if name in frame.global_names:
            return self.globals.get(name, instr.line, instr.column)
        if name in frame.nonlocal_names:
            if frame.env._parent is None:
                raise ViperRuntimeError("Nonlocal name not found", instr.line, instr.column)
            return frame.env._parent.get(name, instr.line, instr.column)
        return frame.env.get(name, instr.line, instr.column)

    def _store_name(self, frame: Frame, name: str, value: Any, instr: Instruction) -> None:
        if name in frame.global_names:
            self.globals.set(name, value)
            return
        if name in frame.nonlocal_names:
            if not frame.env.set_nonlocal(name, value):
                raise ViperRuntimeError("Nonlocal name not found", instr.line, instr.column)
            return
        frame.env.set(name, value)

    def _delete_name(self, frame: Frame, name: str, instr: Instruction) -> None:
        deleted = False
        if name in frame.global_names:
            deleted = self.globals.delete(name)
        elif name in frame.nonlocal_names:
            deleted = frame.env.delete_nonlocal(name)
        else:
            deleted = frame.env.delete(name)
        if not deleted:
            raise ViperRuntimeError(f"Undefined name '{name}'", instr.line, instr.column)

    def _handle_exception(self, frame: Frame, exc: Exception, instr: Instruction) -> bool:
        if not frame.try_stack:
            return False
        handler = frame.try_stack[-1]
        if handler.kind == "except":
            frame.try_stack.pop()
        frame.last_exception = exc if isinstance(exc, BaseException) else None
        frame.pending_exception = exc if isinstance(exc, BaseException) else None
        if len(frame.stack) > handler.stack_depth:
            del frame.stack[handler.stack_depth :]
        self._unwind_with_stack(frame, handler.with_depth, exc, instr)
        frame.ip = int(handler.handler_ip)
        return True

    def _call_viper_function(
        self,
        func: ViperFunction,
        args: List[Any],
        kwargs: Dict[str, Any],
        instr: Instruction,
    ) -> Any:
        params = func.code.params
        if len(args) + len(kwargs) != len(params):
            raise ViperRuntimeError("Argument mismatch", instr.line, instr.column)
        local_env = Environment(func.closure)
        for name, value in zip(params, args):
            local_env.set(name, value)
        for name, value in kwargs.items():
            if name not in params:
                raise ViperRuntimeError("Unknown parameter", instr.line, instr.column)
            if name in local_env._values:
                raise ViperRuntimeError("Duplicate parameter", instr.line, instr.column)
            local_env.set(name, value)
        frame = Frame(code=func.code, env=local_env, stack=[], ip=0, last_value=None)
        if func.code.is_generator:
            return ViperGenerator(frame=frame, vm=self)
        if func.code.is_coroutine:
            return ViperCoroutine(frame=frame, vm=self)
        return self._run_frame(frame)

    def _apply_unary(self, op: str, operand: Any, instr: Instruction) -> Any:
        try:
            if op == "+":
                return +operand
            if op == "-":
                return -operand
            if op == "not":
                return not self._truthy(operand)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Unary operation failed", instr.line, instr.column) from exc
        raise ViperRuntimeError("Unknown unary operator", instr.line, instr.column)

    def _apply_binary(self, op: str, left: Any, right: Any, instr: Instruction) -> Any:
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "//":
                return left // right
            if op == "%":
                return left % right
            if op == "**":
                return left**right
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Binary operation failed", instr.line, instr.column) from exc
        raise ViperRuntimeError("Unknown binary operator", instr.line, instr.column)

    def _apply_compare_chain(
        self, ops: List[str], left: Any, comparators: List[Any], instr: Instruction
    ) -> bool:
        current = left
        for op, right in zip(ops, comparators):
            try:
                if op == "==":
                    result = current == right
                elif op == "!=":
                    result = current != right
                elif op == "<":
                    result = current < right
                elif op == "<=":
                    result = current <= right
                elif op == ">":
                    result = current > right
                elif op == ">=":
                    result = current >= right
                elif op == "in":
                    result = current in right
                elif op == "not in":
                    result = current not in right
                else:
                    raise ViperRuntimeError("Unknown comparison operator", instr.line, instr.column)
            except Exception as exc:  # noqa: BLE001
                raise ViperRuntimeError("Comparison failed", instr.line, instr.column) from exc
            if not result:
                return False
            current = right
        return True

    def _make_class(
        self, name: str, bases: List[Any], code_obj: CodeObject, instr: Instruction
    ) -> type:
        if not isinstance(code_obj, CodeObject):
            raise ViperRuntimeError("Invalid class body", instr.line, instr.column)
        class_env = Environment(parent=self.globals)
        frame = Frame(code=code_obj, env=class_env, stack=[], ip=0, last_value=None)
        self._run_frame(frame)
        namespace = class_env.snapshot()
        if "__module__" not in namespace:
            namespace["__module__"] = "__main__"
        try:
            return type(name, tuple(bases) if bases else (object,), namespace)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Class creation failed", instr.line, instr.column) from exc

    def _with_enter(self, manager: Any, frame: Frame, instr: Instruction) -> Any:
        try:
            enter = getattr(manager, "__enter__")
            exit_func = getattr(manager, "__exit__")
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError(
                "Context manager missing __enter__ or __exit__", instr.line, instr.column
            ) from exc
        if not callable(enter) or not callable(exit_func):
            raise ViperRuntimeError(
                "Context manager missing __enter__ or __exit__", instr.line, instr.column
            )
        try:
            entered = enter()
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError(
                "Context manager __enter__ failed", instr.line, instr.column
            ) from exc
        frame.with_stack.append(exit_func)
        return entered

    def _with_exit(self, frame: Frame, instr: Instruction) -> None:
        if not frame.with_stack:
            raise ViperRuntimeError("Context manager stack underflow", instr.line, instr.column)
        exit_func = frame.with_stack.pop()
        try:
            exit_func(None, None, None)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Context manager exit failed", instr.line, instr.column) from exc

    def _close_with_stack(self, frame: Frame, instr: Instruction) -> None:
        while frame.with_stack:
            exit_func = frame.with_stack.pop()
            try:
                exit_func(None, None, None)
            except Exception as exc:  # noqa: BLE001
                raise ViperRuntimeError("Context manager exit failed", instr.line, instr.column) from exc

    def _drain_with_stack(self, frame: Frame, exc: Exception, instr: Instruction) -> None:
        if not frame.with_stack:
            return
        exc_type = type(exc)
        while frame.with_stack:
            exit_func = frame.with_stack.pop()
            try:
                exit_func(exc_type, exc, None)
            except Exception:
                continue

    def _unwind_with_stack(
        self, frame: Frame, target_depth: int, exc: Exception, instr: Instruction
    ) -> None:
        exc_type = type(exc)
        while len(frame.with_stack) > target_depth:
            exit_func = frame.with_stack.pop()
            try:
                exit_func(exc_type, exc, None)
            except Exception:
                continue

    def _get_attr(self, obj: Any, name: str, instr: Instruction) -> Any:
        try:
            return getattr(obj, name)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Attribute not found", instr.line, instr.column) from exc

    def _set_attr(self, obj: Any, name: str, value: Any, instr: Instruction) -> None:
        try:
            setattr(obj, name, value)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Attribute assignment failed", instr.line, instr.column) from exc

    def _get_subscript(self, obj: Any, index: Any, instr: Instruction) -> Any:
        try:
            return obj[index]
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Invalid subscript access", instr.line, instr.column) from exc

    def _set_subscript(self, obj: Any, index: Any, value: Any, instr: Instruction) -> None:
        try:
            obj[index] = value
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Subscript assignment failed", instr.line, instr.column) from exc

    def _build_iter(self, iterable: Any, instr: Instruction) -> Any:
        try:
            return iter(iterable)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Object is not iterable", instr.line, instr.column) from exc

    def _apply_conversion(self, conversion: Optional[str], value: Any, instr: Instruction) -> Any:
        try:
            resolved = self._resolve_conversion(conversion, instr)
            return apply_conversion(value, resolved)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Invalid conversion", instr.line, instr.column) from exc

    def _resolve_conversion(self, conversion: Optional[str], instr: Instruction) -> Optional[object]:
        if conversion in {None, "r", "s", "a"}:
            return conversion
        try:
            candidate = self.globals.get(conversion, instr.line, instr.column)
        except ViperRuntimeError:
            return "s"
        if callable(candidate):
            return candidate
        return "s"

    def _format_value(self, value: Any, format_spec: Optional[str], instr: Instruction) -> str:
        try:
            return format_value(value, format_spec)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Invalid format spec", instr.line, instr.column) from exc

    def _truthy(self, value: Any) -> bool:
        return bool(value)

    def _coerce_exception(self, value: Any, instr: Instruction) -> BaseException:
        if isinstance(value, BaseException):
            return value
        if isinstance(value, type) and issubclass(value, BaseException):
            try:
                return value()
            except Exception as exc:  # noqa: BLE001
                raise ViperRuntimeError("Exception construction failed", instr.line, instr.column) from exc
        raise ViperRuntimeError("Exceptions must derive from BaseException", instr.line, instr.column)


class Interpreter(VirtualMachine):
    """Backward-compatible alias for the VM-based interpreter."""


def apply_conversion(value: Any, conversion: Optional[object]) -> Any:
    if conversion is None:
        return value
    if callable(conversion):
        return conversion(value)
    if conversion == "r":
        return repr(value)
    if conversion == "s":
        return str(value)
    if conversion == "a":
        return ascii(value)
    return str(value)


def format_value(value: Any, format_spec: Optional[str]) -> str:
    spec = "" if format_spec is None else format_spec
    return format(value, spec)


def match_pattern(value: Any, pattern: Pattern) -> Optional[Dict[str, Any]]:
    bindings: Dict[str, Any] = {}
    if _match_pattern(value, pattern, bindings):
        return bindings
    return None


def _match_pattern(value: Any, pattern: Pattern, bindings: Dict[str, Any]) -> bool:
    if isinstance(pattern, PatternWildcard):
        return True
    if isinstance(pattern, PatternLiteral):
        return value == pattern.value
    if isinstance(pattern, PatternName):
        if pattern.identifier in bindings:
            return bindings[pattern.identifier] == value
        bindings[pattern.identifier] = value
        return True
    if isinstance(pattern, PatternSequence):
        if not isinstance(value, (list, tuple)):
            return False
        if len(value) != len(pattern.elements):
            return False
        for item, subpattern in zip(value, pattern.elements):
            if not _match_pattern(item, subpattern, bindings):
                return False
        return True
    if isinstance(pattern, PatternOr):
        for subpattern in pattern.patterns:
            temp = dict(bindings)
            if _match_pattern(value, subpattern, temp):
                bindings.clear()
                bindings.update(temp)
                return True
        return False
    return False


def default_builtins() -> Dict[str, Any]:
    """Minimal builtins for Viper programs."""
    return {
        "abs": abs,
        "len": len,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "sum": sum,
        "print": print,
        "__viper_match_pattern": match_pattern,
    }


def run(source: str, globals: Optional[Mapping[str, Any]] = None) -> ExecutionResult:
    """Convenience entry point that tokenizes, parses, compiles, and executes Viper code."""
    from ripperdoc.core.viper.compiler import compile_module
    from ripperdoc.core.viper.parser import parse
    from ripperdoc.core.viper.tokenizer import tokenize

    tokens = tokenize(source)
    module = parse(tokens)
    code = compile_module(module)
    interpreter = Interpreter(globals=globals)
    return interpreter.execute(code)
