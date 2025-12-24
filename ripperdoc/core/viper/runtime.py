"""Bytecode execution engine for Viper."""

from __future__ import annotations

from collections.abc import Coroutine
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ripperdoc.core.viper.bytecode import CodeObject, Instruction
from ripperdoc.core.viper.ast_nodes import (
    Pattern,
    PatternLiteral,
    PatternName,
    PatternOr,
    PatternSequence,
    PatternWildcard,
    PatternStar,
    PatternAs,
    PatternValue,
    PatternMapping,
    PatternClass,
    PatternKey,
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
    defaults: List[Any] = field(default_factory=list)
    kw_defaults: Dict[str, Any] = field(default_factory=dict)

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
    with_stack: List["WithExit"] = field(default_factory=list)
    try_stack: List["TryHandler"] = field(default_factory=list)
    pending_exception: Optional[BaseException] = None
    last_exception: Optional[BaseException] = None


@dataclass
class WithExit:
    func: Any
    is_async: bool


@dataclass
class TryHandler:
    kind: str
    handler_ip: int
    stack_depth: int
    with_depth: int


@dataclass
class ViperModule:
    name: str
    namespace: Dict[str, Any]
    file: Optional[str]

    def __getattribute__(self, name: str) -> Any:
        if name in {"name", "namespace", "file", "__dict__", "__class__"}:
            return object.__getattribute__(self, name)
        namespace = object.__getattribute__(self, "namespace")
        if name in namespace:
            return namespace[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"name", "namespace", "file"}:
            object.__setattr__(self, name, value)
            return
        self.namespace[name] = value

    @property
    def __dict__(self) -> Dict[str, Any]:  # type: ignore[override]
        return self.namespace

    def __repr__(self) -> str:
        return f"<viper module {self.name}>"


class SafeModule:
    def __init__(self, module: Any, allowed: Optional[Sequence[str]] = None) -> None:
        object.__setattr__(self, "_module", module)
        if allowed is None:
            allowed = [name for name in dir(module) if not name.startswith("_")]
        object.__setattr__(self, "_allowed", set(allowed))

    def __getattribute__(self, name: str) -> Any:
        if name in {"_module", "_allowed", "allowed_names", "__class__"}:
            return object.__getattribute__(self, name)
        allowed = object.__getattribute__(self, "_allowed")
        if name in allowed:
            module = object.__getattribute__(self, "_module")
            return getattr(module, name)
        raise AttributeError(name)

    @property
    def allowed_names(self) -> set[str]:
        return set(object.__getattribute__(self, "_allowed"))

    def __repr__(self) -> str:
        module = object.__getattribute__(self, "_module")
        name = getattr(module, "__name__", "module")
        return f"<safe module {name}>"


class VirtualMachine:
    def __init__(
        self,
        globals: Optional[Mapping[str, Any]] = None,
        builtins: Optional[Mapping[str, Any]] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        builtins_env = Environment()
        for name, value in (builtins or default_builtins()).items():
            builtins_env.set(name, value)
        self.builtins = builtins_env
        self.globals = Environment(builtins_env)
        if globals:
            for name, value in globals.items():
                self.globals.set(name, value)
        base = base_path or Path.cwd()
        self.module_paths = [base] + list(base.parents)
        self.modules_by_path: Dict[Path, ViperModule] = {}
        self.modules_by_name: Dict[str, ViperModule] = {}
        self.loading_modules: set[Path] = set()
        self.system_modules = self._build_system_modules()

    def execute(self, code: CodeObject, *, source_path: Optional[Path] = None) -> ExecutionResult:
        if source_path is not None:
            self.globals.set("__file__", str(source_path))
            self.globals.set("__name__", "__main__")
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
                elif op == "LOAD_ENV":
                    frame.stack.append(frame.env)
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
                elif op == "BUILD_LIST_UNPACK":
                    flags = list(instr.arg or [])
                    items = []
                    for flag in reversed(flags):
                        value = frame.stack.pop()
                        items.append((bool(flag), value))
                    items.reverse()
                    result: List[Any] = []
                    for is_star, value in items:
                        if is_star:
                            try:
                                result.extend(list(value))
                            except Exception as exc:  # noqa: BLE001
                                raise ViperRuntimeError(
                                    "List unpack failed", instr.line, instr.column
                                ) from exc
                        else:
                            result.append(value)
                    frame.stack.append(result)
                elif op == "BUILD_TUPLE":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    frame.stack.append(tuple(items))
                elif op == "BUILD_TUPLE_UNPACK":
                    flags = list(instr.arg or [])
                    items = []
                    for flag in reversed(flags):
                        value = frame.stack.pop()
                        items.append((bool(flag), value))
                    items.reverse()
                    result: List[Any] = []
                    for is_star, value in items:
                        if is_star:
                            try:
                                result.extend(list(value))
                            except Exception as exc:  # noqa: BLE001
                                raise ViperRuntimeError(
                                    "Tuple unpack failed", instr.line, instr.column
                                ) from exc
                        else:
                            result.append(value)
                    frame.stack.append(tuple(result))
                elif op == "BUILD_SET":
                    count = int(instr.arg)
                    items = frame.stack[-count:] if count else []
                    if count:
                        del frame.stack[-count:]
                    frame.stack.append(set(items))
                elif op == "BUILD_DICT":
                    count = int(instr.arg)
                    items: List[tuple[Any, Any]] = []
                    for _ in range(count):
                        value = frame.stack.pop()
                        key = frame.stack.pop()
                        items.append((key, value))
                    items.reverse()
                    frame.stack.append(dict(items))
                elif op == "BUILD_DICT_UNPACK":
                    flags = list(instr.arg or [])
                    items = []
                    for flag in reversed(flags):
                        if flag:
                            value = frame.stack.pop()
                            items.append((True, value))
                        else:
                            value = frame.stack.pop()
                            key = frame.stack.pop()
                            items.append((False, (key, value)))
                    items.reverse()
                    merged: Dict[Any, Any] = {}
                    for is_unpack, payload in items:
                        if is_unpack:
                            if not isinstance(payload, Mapping):
                                raise ViperRuntimeError(
                                    "Dict unpack failed", instr.line, instr.column
                                )
                            merged.update(payload)
                        else:
                            key, value = payload
                            merged[key] = value
                    frame.stack.append(merged)
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
                elif op == "UNPACK_EX":
                    before, after = instr.arg
                    iterable = frame.stack.pop()
                    try:
                        items = list(iterable)
                    except Exception as exc:  # noqa: BLE001
                        raise ViperRuntimeError("Unpack failed", instr.line, instr.column) from exc
                    if len(items) < before + after:
                        raise ViperRuntimeError("Unpack mismatch", instr.line, instr.column)
                    before_items = items[:before]
                    after_items = items[len(items) - after :] if after else []
                    star_items = items[before : len(items) - after]
                    for item in reversed(after_items):
                        frame.stack.append(item)
                    frame.stack.append(star_items)
                    for item in reversed(before_items):
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
                    module = self._import_module(module_name, fromlist, int(level), frame, instr)
                    frame.stack.append(module)
                elif op == "IMPORT_FROM":
                    module = frame.stack[-1]
                    frame.stack.append(self._import_from(module, instr.arg, instr))
                elif op == "IMPORT_STAR":
                    module = frame.stack.pop()
                    self._import_star(module, frame, instr)
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
                elif op == "BUILD_AITER":
                    iterable = frame.stack.pop()
                    frame.stack.append(self._build_aiter(iterable, instr))
                elif op == "FOR_ITER":
                    iterator = frame.stack[-1]
                    try:
                        item = next(iterator)
                    except StopIteration:
                        frame.stack.pop()
                        frame.ip = int(instr.arg)
                    else:
                        frame.stack.append(item)
                elif op == "FOR_AITER":
                    iterator = frame.stack[-1]
                    try:
                        awaitable = self._get_anext(iterator, instr)
                    except StopAsyncIteration:
                        frame.stack.pop()
                        frame.ip = int(instr.arg)
                    else:
                        try:
                            item = self._await_value(awaitable, instr)
                        except StopAsyncIteration:
                            frame.stack.pop()
                            frame.ip = int(instr.arg)
                        else:
                            frame.stack.append(item)
                elif op == "WITH_ENTER":
                    manager = frame.stack.pop()
                    frame.stack.append(self._with_enter(manager, frame, instr))
                elif op == "WITH_EXIT":
                    self._with_exit(frame, instr)
                elif op == "ASYNC_WITH_ENTER":
                    manager = frame.stack.pop()
                    frame.stack.append(self._async_with_enter(manager, frame, instr))
                elif op == "ASYNC_WITH_EXIT":
                    self._async_with_exit(frame, instr)
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
                elif op == "CALL_FUNCTION_EX":
                    has_kwargs = bool(instr.arg)
                    kwargs = frame.stack.pop() if has_kwargs else {}
                    args = frame.stack.pop()
                    func = frame.stack.pop()
                    try:
                        args_list = list(args)
                    except Exception as exc:  # noqa: BLE001
                        raise ViperRuntimeError(
                            "Function call args unpack failed", instr.line, instr.column
                        ) from exc
                    if not isinstance(kwargs, dict):
                        raise ViperRuntimeError(
                            "Function call kwargs unpack failed", instr.line, instr.column
                        )
                    frame.stack.append(self._call(func, args_list, kwargs, instr))
                elif op == "AWAIT":
                    value = frame.stack.pop()
                    frame.stack.append(self._await_value(value, instr))
                elif op == "MAKE_FUNCTION":
                    pos_defaults = 0
                    kw_default_names: List[str] = []
                    has_annotations = False
                    if instr.arg is not None:
                        if isinstance(instr.arg, tuple) and len(instr.arg) == 3:
                            pos_defaults, kw_default_names, has_annotations = instr.arg
                        else:
                            pos_defaults, kw_default_names = instr.arg
                    code_obj = frame.stack.pop()
                    annotations = frame.stack.pop() if has_annotations else None
                    kw_defaults: Dict[str, Any] = {}
                    if kw_default_names:
                        values = [frame.stack.pop() for _ in range(len(kw_default_names))]
                        values.reverse()
                        for name, value in zip(kw_default_names, values):
                            kw_defaults[name] = value
                    defaults: List[Any] = []
                    if pos_defaults:
                        values = [frame.stack.pop() for _ in range(int(pos_defaults))]
                        values.reverse()
                        defaults = values
                    func = (
                        ViperFunction(
                            code=code_obj,
                            closure=frame.env,
                            vm=self,
                            defaults=defaults,
                            kw_defaults=kw_defaults,
                        )
                    )
                    if annotations is not None:
                        setattr(func, "__annotations__", annotations)
                    frame.stack.append(func)
                elif op == "MAKE_CLASS":
                    arg = instr.arg
                    if isinstance(arg, tuple) and len(arg) == 3:
                        name, base_count, kwarg_names = arg
                    else:
                        name, base_count = arg
                        kwarg_names = []
                    code_obj = frame.stack.pop()
                    kwargs = {}
                    for kw_name in reversed(kwarg_names):
                        kwargs[kw_name] = frame.stack.pop()
                    bases = [frame.stack.pop() for _ in range(base_count)]
                    bases.reverse()
                    frame.stack.append(self._make_class(name, bases, code_obj, instr, kwargs=kwargs))
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

    def _import_module(
        self,
        name: str,
        fromlist: Optional[Sequence[str]],
        level: int,
        frame: Frame,
        instr: Instruction,
    ) -> Any:
        if not name:
            raise ViperRuntimeError("Missing module name", instr.line, instr.column)
        if level < 0:
            raise ViperRuntimeError("Invalid import level", instr.line, instr.column)
        if level == 0:
            module_path = self._find_module_path(name, None)
            if module_path is None:
                return self._import_system_module(name, fromlist, instr)
            module = self._load_viper_module(name, module_path, instr)
        else:
            base = self._relative_base(frame, level, instr)
            module_path = self._find_module_path(name, base)
            if module_path is None:
                raise ViperRuntimeError("Module not found", instr.line, instr.column)
            module = self._load_viper_module(name, module_path, instr)
        return self._apply_import_fromlist(name, module, fromlist)

    def _apply_import_fromlist(
        self, name: str, module: Any, fromlist: Optional[Sequence[str]]
    ) -> Any:
        if fromlist:
            return module
        if "." not in name:
            return module
        top_name = name.split(".", 1)[0]
        return self.modules_by_name.get(top_name, module)

    def _import_system_module(
        self, name: str, fromlist: Optional[Sequence[str]], instr: Instruction
    ) -> Any:
        top_name = name.split(".", 1)[0]
        if top_name not in self.system_modules:
            raise ViperRuntimeError("Module not found", instr.line, instr.column)
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Import failed", instr.line, instr.column) from exc
        if fromlist:
            return SafeModule(module)
        return self.system_modules[top_name]

    def _find_module_path(self, name: str, base: Optional[Path]) -> Optional[Path]:
        parts = name.split(".")
        bases = [base] if base is not None else self.module_paths
        for root in bases:
            if root is None:
                continue
            module_path = root.joinpath(*parts).with_suffix(".vp")
            if module_path.is_file():
                return module_path.resolve()
            package_path = root.joinpath(*parts, "__init__.vp")
            if package_path.is_file():
                return package_path.resolve()
        return None

    def _relative_base(self, frame: Frame, level: int, instr: Instruction) -> Path:
        current_file = self._current_file(frame, instr)
        if current_file is None:
            raise ViperRuntimeError("Relative import requires __file__", instr.line, instr.column)
        base = current_file.parent
        for _ in range(level - 1):
            if base.parent == base:
                raise ViperRuntimeError("Relative import beyond top-level", instr.line, instr.column)
            base = base.parent
        return base

    def _current_file(self, frame: Frame, instr: Instruction) -> Optional[Path]:
        try:
            value = frame.env.get("__file__", instr.line, instr.column)
        except ViperRuntimeError:
            return None
        if not value:
            return None
        try:
            return Path(str(value)).resolve()
        except Exception:
            return None

    def _load_viper_module(self, name: str, path: Path, instr: Instruction) -> ViperModule:
        if path in self.modules_by_path:
            module = self.modules_by_path[path]
            self._register_module_hierarchy(name, module, path)
            return module
        if path in self.loading_modules:
            raise ViperRuntimeError("Circular import detected", instr.line, instr.column)
        self.loading_modules.add(path)
        try:
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ViperRuntimeError("Failed to read module", instr.line, instr.column) from exc
            from ripperdoc.core.viper.compiler import compile_module
            from ripperdoc.core.viper.parser import parse
            from ripperdoc.core.viper.tokenizer import tokenize

            tokens = tokenize(source)
            module_ast = parse(tokens)
            code = compile_module(module_ast)
            module_env = Environment(self.builtins)
            module_env.set("__file__", str(path))
            module_env.set("__name__", name)
            module_frame = Frame(code=code, env=module_env, stack=[], ip=0, last_value=None)
            self._run_frame(module_frame)
            module = ViperModule(name=name, namespace=module_env.snapshot(), file=str(path))
            self.modules_by_path[path] = module
            self.modules_by_name[name] = module
            self._register_module_hierarchy(name, module, path)
            return module
        finally:
            self.loading_modules.discard(path)

    def _register_module_hierarchy(self, name: str, module: ViperModule, path: Path) -> None:
        parts = name.split(".")
        if len(parts) == 1:
            return
        parent_module: Optional[ViperModule] = None
        for index, part in enumerate(parts[:-1]):
            prefix = ".".join(parts[: index + 1])
            pkg = self.modules_by_name.get(prefix)
            if pkg is None:
                depth = len(parts) - index - 2
                package_dir = path.parent
                for _ in range(depth):
                    package_dir = package_dir.parent
                pkg = self._load_package_module(prefix, package_dir)
                self.modules_by_name[prefix] = pkg
            if parent_module is not None:
                parent_module.namespace[part] = pkg
            parent_module = pkg
        if parent_module is not None:
            parent_module.namespace[parts[-1]] = module

    def _load_package_module(self, name: str, package_dir: Path) -> ViperModule:
        package_init = package_dir / "__init__.vp"
        if package_init.is_file():
            return self._load_viper_module(
                name,
                package_init.resolve(),
                Instruction(op="IMPORT_NAME", arg=None, line=0, column=0),
            )
        return ViperModule(name=name, namespace={"__name__": name}, file=None)

    def _import_from(self, module: Any, name: str, instr: Instruction) -> Any:
        try:
            return getattr(module, name)
        except Exception:
            if isinstance(module, ViperModule):
                sub_name = f"{module.name}.{name}"
                module_path = self._find_module_path(sub_name, None)
                if module_path is not None:
                    sub_module = self._load_viper_module(sub_name, module_path, instr)
                    module.namespace[name] = sub_module
                    return sub_module
            if isinstance(module, SafeModule):
                module_name = getattr(module._module, "__name__", "")
                if module_name:
                    sub_name = f"{module_name}.{name}"
                    return self._import_system_module(sub_name, ["*"], instr)
            raise ViperRuntimeError("Import failed", instr.line, instr.column)

    def _import_star(self, module: Any, frame: Frame, instr: Instruction) -> None:
        names = self._module_names(module)
        for name in names:
            frame.env.set(name, getattr(module, name))

    def _module_names(self, module: Any) -> List[str]:
        if isinstance(module, SafeModule):
            return sorted(module.allowed_names)
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict):
            if "__all__" in namespace and isinstance(namespace["__all__"], list):
                return list(namespace["__all__"])
            return [name for name in namespace.keys() if not name.startswith("_")]
        return [name for name in dir(module) if not name.startswith("_")]

    def _build_system_modules(self) -> Dict[str, SafeModule]:
        modules: Dict[str, SafeModule] = {}
        for name in ("math", "collections", "random"):
            try:
                modules[name] = SafeModule(importlib.import_module(name))
            except Exception:
                continue
        return modules

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
        if func.code.param_spec is None:
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
        spec = func.code.param_spec
        local_env = Environment(func.closure)
        pos_params = list(spec.posonly) + list(spec.pos_or_kw)
        if spec.vararg:
            vararg_name = spec.vararg
        else:
            vararg_name = None
        if spec.varkw:
            varkw_name = spec.varkw
        else:
            varkw_name = None
        bound: Dict[str, Any] = {}
        if len(args) > len(pos_params) and vararg_name is None:
            raise ViperRuntimeError("Too many positional arguments", instr.line, instr.column)
        for name, value in zip(pos_params, args):
            bound[name] = value
        extra_pos = args[len(pos_params) :]
        if vararg_name is not None:
            bound[vararg_name] = tuple(extra_pos)
        elif extra_pos:
            raise ViperRuntimeError("Too many positional arguments", instr.line, instr.column)
        extra_kwargs: Dict[str, Any] = {}
        for name, value in kwargs.items():
            if name in spec.posonly:
                raise ViperRuntimeError(
                    "Positional-only argument passed as keyword", instr.line, instr.column
                )
            if name in bound:
                raise ViperRuntimeError("Duplicate parameter", instr.line, instr.column)
            if name in spec.pos_or_kw or name in spec.kwonly:
                bound[name] = value
                continue
            if varkw_name is not None:
                extra_kwargs[name] = value
                continue
            raise ViperRuntimeError("Unknown parameter", instr.line, instr.column)
        if varkw_name is not None:
            bound[varkw_name] = extra_kwargs
        elif extra_kwargs:
            raise ViperRuntimeError("Unknown parameter", instr.line, instr.column)
        default_offset = len(pos_params) - len(func.defaults)
        for index, name in enumerate(pos_params):
            if name in bound:
                continue
            if index >= default_offset:
                bound[name] = func.defaults[index - default_offset]
            else:
                raise ViperRuntimeError("Missing required argument", instr.line, instr.column)
        for name in spec.kwonly:
            if name in bound:
                continue
            if name in func.kw_defaults:
                bound[name] = func.kw_defaults[name]
            else:
                raise ViperRuntimeError(
                    "Missing required keyword-only argument", instr.line, instr.column
                )
        if vararg_name is not None and vararg_name not in bound:
            bound[vararg_name] = ()
        if varkw_name is not None and varkw_name not in bound:
            bound[varkw_name] = {}
        for name, value in bound.items():
            local_env.set(name, value)
        frame = Frame(code=func.code, env=local_env, stack=[], ip=0, last_value=None)
        if func.code.is_generator:
            return ViperGenerator(frame=frame, vm=self)
        if func.code.is_coroutine:
            return ViperCoroutine(frame=frame, vm=self)
        return self._run_frame(frame)

    def _await_value(self, value: Any, instr: Instruction) -> Any:
        if isinstance(value, ViperCoroutine):
            return value._run()
        if hasattr(value, "__await__"):
            iterator = value.__await__()
            while True:
                try:
                    yielded = next(iterator)
                except StopIteration as exc:
                    return exc.value
                if yielded is not None:
                    raise ViperRuntimeError(
                        "Awaitable yielded; event loop not supported", instr.line, instr.column
                    )
        raise ViperRuntimeError("Object is not awaitable", instr.line, instr.column)

    def _apply_unary(self, op: str, operand: Any, instr: Instruction) -> Any:
        try:
            if op == "+":
                return +operand
            if op == "-":
                return -operand
            if op == "~":
                return ~operand
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
            if op == "@":
                return left @ right
            if op == "|":
                return left | right
            if op == "&":
                return left & right
            if op == "^":
                return left ^ right
            if op == "<<":
                return left << right
            if op == ">>":
                return left >> right
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
                elif op == "is":
                    result = current is right
                elif op == "is not":
                    result = current is not right
                else:
                    raise ViperRuntimeError("Unknown comparison operator", instr.line, instr.column)
            except Exception as exc:  # noqa: BLE001
                raise ViperRuntimeError("Comparison failed", instr.line, instr.column) from exc
            if not result:
                return False
            current = right
        return True

    def _make_class(
        self,
        name: str,
        bases: List[Any],
        code_obj: CodeObject,
        instr: Instruction,
        *,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> type:
        kwargs = kwargs or {}
        import builtins

        if kwargs:
            def _viper_type(
                name: str,
                bases: tuple[Any, ...],
                namespace: dict[str, Any],
            ) -> type:
                return builtins.type(name, bases, namespace, **kwargs)

            type = _viper_type
        else:
            type = builtins.type
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
        frame.with_stack.append(WithExit(func=exit_func, is_async=False))
        return entered

    def _with_exit(self, frame: Frame, instr: Instruction) -> None:
        if not frame.with_stack:
            raise ViperRuntimeError("Context manager stack underflow", instr.line, instr.column)
        try:
            exit_entry = frame.with_stack.pop()
            self._call_with_exit(exit_entry, None, None, instr)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Context manager exit failed", instr.line, instr.column) from exc

    def _close_with_stack(self, frame: Frame, instr: Instruction) -> None:
        while frame.with_stack:
            try:
                exit_entry = frame.with_stack.pop()
                self._call_with_exit(exit_entry, None, None, instr)
            except Exception as exc:  # noqa: BLE001
                raise ViperRuntimeError("Context manager exit failed", instr.line, instr.column) from exc

    def _drain_with_stack(self, frame: Frame, exc: Exception, instr: Instruction) -> None:
        if not frame.with_stack:
            return
        exc_type = type(exc)
        while frame.with_stack:
            exit_entry = frame.with_stack.pop()
            try:
                self._call_with_exit(exit_entry, exc_type, exc, instr)
            except Exception:
                continue

    def _unwind_with_stack(
        self, frame: Frame, target_depth: int, exc: Exception, instr: Instruction
    ) -> None:
        exc_type = type(exc)
        while len(frame.with_stack) > target_depth:
            exit_entry = frame.with_stack.pop()
            try:
                self._call_with_exit(exit_entry, exc_type, exc, instr)
            except Exception:
                continue

    def _async_with_enter(self, manager: Any, frame: Frame, instr: Instruction) -> Any:
        try:
            enter = getattr(manager, "__aenter__")
            exit_func = getattr(manager, "__aexit__")
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError(
                "Context manager missing __aenter__ or __aexit__", instr.line, instr.column
            ) from exc
        if not callable(enter) or not callable(exit_func):
            raise ViperRuntimeError(
                "Context manager missing __aenter__ or __aexit__", instr.line, instr.column
            )
        try:
            entered = self._await_value(enter(), instr)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError(
                "Context manager __aenter__ failed", instr.line, instr.column
            ) from exc
        frame.with_stack.append(WithExit(func=exit_func, is_async=True))
        return entered

    def _async_with_exit(self, frame: Frame, instr: Instruction) -> None:
        if not frame.with_stack:
            raise ViperRuntimeError("Context manager stack underflow", instr.line, instr.column)
        exit_entry = frame.with_stack.pop()
        try:
            self._call_with_exit(exit_entry, None, None, instr)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Context manager exit failed", instr.line, instr.column) from exc

    def _call_with_exit(
        self,
        exit_entry: WithExit,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        instr: Instruction,
    ) -> None:
        if exit_entry.is_async:
            self._await_value(exit_entry.func(exc_type, exc, None), instr)
        else:
            exit_entry.func(exc_type, exc, None)

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

    def _build_aiter(self, iterable: Any, instr: Instruction) -> Any:
        try:
            if hasattr(iterable, "__aiter__"):
                iterator = iterable.__aiter__()
                if hasattr(iterator, "__await__"):
                    iterator = self._await_value(iterator, instr)
                return iterator
            if hasattr(iterable, "__anext__"):
                return iterable
        except StopAsyncIteration:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Object is not async iterable", instr.line, instr.column) from exc
        raise ViperRuntimeError("Object is not async iterable", instr.line, instr.column)

    def _get_anext(self, iterator: Any, instr: Instruction) -> Any:
        try:
            anext = getattr(iterator, "__anext__")
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Object is not async iterable", instr.line, instr.column) from exc
        if not callable(anext):
            raise ViperRuntimeError("Object is not async iterable", instr.line, instr.column)
        return anext()

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


def match_pattern(
    value: Any, pattern: Pattern, env: Optional[Environment] = None
) -> Optional[Dict[str, Any]]:
    bindings: Dict[str, Any] = {}
    if _match_pattern(value, pattern, bindings, env):
        return bindings
    return None


def _resolve_pattern_value(pattern: PatternValue, env: Optional[Environment]) -> Any:
    if env is None:
        raise ViperRuntimeError("Pattern value resolution requires an environment", pattern.line, pattern.column)
    current = env.get(pattern.parts[0], pattern.line, pattern.column)
    for part in pattern.parts[1:]:
        try:
            current = getattr(current, part)
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Attribute not found", pattern.line, pattern.column) from exc
    return current


def _resolve_pattern_key(key: PatternKey, env: Optional[Environment]) -> Any:
    if isinstance(key, PatternLiteral):
        return key.value
    if isinstance(key, PatternValue):
        return _resolve_pattern_value(key, env)
    return key


def _match_pattern(
    value: Any, pattern: Pattern, bindings: Dict[str, Any], env: Optional[Environment]
) -> bool:
    if isinstance(pattern, PatternAs):
        if not _match_pattern(value, pattern.pattern, bindings, env):
            return False
        if pattern.name in bindings:
            return bindings[pattern.name] == value
        bindings[pattern.name] = value
        return True
    if isinstance(pattern, PatternWildcard):
        return True
    if isinstance(pattern, PatternLiteral):
        return value == pattern.value
    if isinstance(pattern, PatternValue):
        return value == _resolve_pattern_value(pattern, env)
    if isinstance(pattern, PatternName):
        if pattern.identifier in bindings:
            return bindings[pattern.identifier] == value
        bindings[pattern.identifier] = value
        return True
    if isinstance(pattern, PatternSequence):
        if not isinstance(value, (list, tuple)):
            return False
        star_index = None
        for index, element in enumerate(pattern.elements):
            if isinstance(element, PatternStar):
                star_index = index
                break
        if star_index is None:
            if len(value) != len(pattern.elements):
                return False
            for item, subpattern in zip(value, pattern.elements):
                if not _match_pattern(item, subpattern, bindings, env):
                    return False
            return True
        prefix = pattern.elements[:star_index]
        suffix = pattern.elements[star_index + 1 :]
        if len(value) < len(prefix) + len(suffix):
            return False
        for item, subpattern in zip(value[: len(prefix)], prefix):
            if not _match_pattern(item, subpattern, bindings, env):
                return False
        if suffix:
            for item, subpattern in zip(value[-len(suffix) :], suffix):
                if not _match_pattern(item, subpattern, bindings, env):
                    return False
        star_pattern = pattern.elements[star_index]
        captured = list(value[len(prefix) : len(value) - len(suffix)])
        if isinstance(star_pattern, PatternStar):
            if isinstance(star_pattern.target, PatternWildcard):
                return True
            if isinstance(star_pattern.target, PatternName):
                name = star_pattern.target.identifier
                if name in bindings:
                    return bindings[name] == captured
                bindings[name] = captured
                return True
        return False
    if isinstance(pattern, PatternMapping):
        if not isinstance(value, Mapping):
            return False
        used_keys: set[Any] = set()
        for key_pattern, value_pattern in pattern.items:
            key = _resolve_pattern_key(key_pattern, env)
            try:
                present = key in value
            except Exception as exc:  # noqa: BLE001
                raise ViperRuntimeError(
                    "Invalid mapping key pattern", pattern.line, pattern.column
                ) from exc
            if not present:
                return False
            used_keys.add(key)
            if not _match_pattern(value[key], value_pattern, bindings, env):
                return False
        if pattern.rest is None:
            return True
        remaining = {k: v for k, v in value.items() if k not in used_keys}
        if isinstance(pattern.rest, PatternWildcard):
            return True
        if isinstance(pattern.rest, PatternName):
            name = pattern.rest.identifier
            if name in bindings:
                return bindings[name] == remaining
            bindings[name] = remaining
            return True
        return False
    if isinstance(pattern, PatternClass):
        cls = _resolve_pattern_value(pattern.class_path, env)
        try:
            if not isinstance(value, cls):
                return False
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Invalid class pattern", pattern.line, pattern.column) from exc
        has_match_args = hasattr(cls, "__match_args__")
        try:
            match_args = getattr(cls, "__match_args__", ())
        except Exception as exc:  # noqa: BLE001
            raise ViperRuntimeError("Invalid __match_args__", pattern.line, pattern.column) from exc
        if has_match_args:
            if not isinstance(match_args, tuple):
                raise ViperRuntimeError("Invalid __match_args__", pattern.line, pattern.column)
            for item in match_args:
                if not isinstance(item, str):
                    raise ViperRuntimeError("Invalid __match_args__", pattern.line, pattern.column)
            if len(set(match_args)) != len(match_args):
                raise ViperRuntimeError("Duplicate __match_args__", pattern.line, pattern.column)
        else:
            match_args = ()
        if len(pattern.positional) > len(match_args):
            raise ViperRuntimeError("Too many positional patterns", pattern.line, pattern.column)
        used_names: set[str] = set()
        for name, subpattern in zip(match_args, pattern.positional):
            used_names.add(name)
            if not hasattr(value, name):
                return False
            if not _match_pattern(getattr(value, name), subpattern, bindings, env):
                return False
        for name, subpattern in pattern.keywords:
            if name in used_names:
                raise ViperRuntimeError("Duplicate class pattern attribute", pattern.line, pattern.column)
            if not hasattr(value, name):
                return False
            if not _match_pattern(getattr(value, name), subpattern, bindings, env):
                return False
        return True
    if isinstance(pattern, PatternStar):
        return False
    if isinstance(pattern, PatternOr):
        for subpattern in pattern.patterns:
            temp = dict(bindings)
            if _match_pattern(value, subpattern, temp, env):
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


def run(
    source: str,
    globals: Optional[Mapping[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> ExecutionResult:
    """Convenience entry point that tokenizes, parses, compiles, and executes Viper code."""
    from ripperdoc.core.viper.compiler import compile_module
    from ripperdoc.core.viper.parser import parse
    from ripperdoc.core.viper.tokenizer import tokenize

    tokens = tokenize(source)
    module = parse(tokens)
    code = compile_module(module)
    base_path = source_path.parent if source_path is not None else None
    interpreter = Interpreter(globals=globals, base_path=base_path)
    return interpreter.execute(code, source_path=source_path)
