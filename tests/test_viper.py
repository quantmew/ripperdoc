import asyncio
import pytest

from ripperdoc.core.viper import (
    Interpreter,
    ViperRuntimeError,
    ViperSyntaxError,
    compile_module,
    disassemble,
    parse,
    run,
    tokenize,
)
from ripperdoc.core.viper.runtime import TemplateBytes, TemplateString
from ripperdoc.cli.viper_cli import run_file


def _run_with_compile(code: str, globals_dict=None):
    tokens = tokenize(code)
    module = parse(tokens)
    compiled = compile_module(module)
    vm = Interpreter(globals=globals_dict)
    return vm.execute(compiled)


def _run_file(path, globals_dict=None):
    tokens = tokenize(path.read_text(encoding="utf-8"))
    module = parse(tokens)
    compiled = compile_module(module)
    vm = Interpreter(globals=globals_dict, base_path=path.parent)
    return vm.execute(compiled, source_path=path)


def test_tokenizer_indentation_tokens():
    code = """
value = 1
if value:
    inner = 2
    inner = inner + 1
value = 3
"""
    tokens = tokenize(code)
    kinds = [token.kind for token in tokens]
    assert kinds.count("INDENT") == 1
    assert kinds.count("DEDENT") == 1
    assert kinds[-1] == "EOF"


def test_tuple_literal_trailing_comma():
    code = """
value = (1,)
value
"""
    result = run(code)
    assert result.value == (1,)
    assert result.globals["value"] == (1,)


def test_missing_trailing_newline():
    code = "value = 1\nvalue"
    result = run(code)
    assert result.value == 1


def test_operator_precedence():
    code = """
value = 1 + 2 * 3
value
"""
    result = run(code)
    assert result.value == 7


def test_lambda_expression():
    code = """
inc = lambda x: x + 1
inc(2)
"""
    result = run(code)
    assert result.value == 3


def test_lambda_closure():
    code = """
base = 3
adder = lambda x: x + base
adder(4)
"""
    result = run(code)
    assert result.value == 7


def test_short_circuit_and_or_not():
    code = """
value = 0 and unknown
value
"""
    result = run(code)
    assert result.value == 0

    code = """
value = 1 or unknown
value
"""
    result = run(code)
    assert result.value == 1

    code = """
value = not False
value
"""
    result = run(code)
    assert result.value is True


def test_await_in_async_function():
    code = """
async def inner():
    return 5

async def outer():
    return await inner()

result = outer()
"""
    result = run(code)

    async def _run(coro):
        return await coro

    assert asyncio.run(_run(result.globals["result"])) == 5


def test_await_outside_async_raises():
    code = """
async def inner():
    return 1

await inner()
"""
    with pytest.raises(ViperSyntaxError):
        run(code)


def test_compare_chain():
    code = """
value = 1 < 2 < 3
value
"""
    result = run(code)
    assert result.value is True


def test_for_loop_break_continue():
    code = """
values = [1, 2, 3, 4]
result = []
for item in values:
    if item == 2:
        continue
    if item == 4:
        break
    result = result + [item]
result
"""
    result = run(code)
    assert result.value == [1, 3]


def test_for_loop_unpack_targets():
    code = """
steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
result = []
for dx, dy in steps:
    result = result + [dx + dy]
result
"""
    result = run(code)
    assert result.value == [1, -1, 1, -1]


def test_slice_subscript():
    code = """
values = [0, 1, 2, 3, 4]
part = values[1:4]
part
"""
    result = run(code)
    assert result.value == [1, 2, 3]

    code = """
values = [0, 1, 2, 3, 4]
rev = values[::-1]
rev
"""
    result = run(code)
    assert result.value == [4, 3, 2, 1, 0]


def test_while_loop_break_continue():
    code = """
count = 0
result = 0
while count < 5:
    count = count + 1
    if count == 2:
        continue
    if count == 4:
        break
    result = result + count
result
"""
    result = run(code)
    assert result.value == 4


def test_with_statement_assigns_and_exits():
    events = []

    class DummyContext:
        def __enter__(self):
            events.append("enter")
            return 7

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    ctx = DummyContext()
    code = """
with ctx as value:
    result = value
result
"""
    result = run(code, globals={"ctx": ctx})
    assert result.value == 7
    assert events == ["enter", "exit"]


def test_with_statement_multiple_items():
    events = []

    class DummyContext:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            events.append(f"enter-{self.label}")
            return self.label

        def __exit__(self, exc_type, exc, tb):
            events.append(f"exit-{self.label}")
            return False

    left = DummyContext(2)
    right = DummyContext(3)
    code = """
with left as a, right as b:
    result = a + b
result
"""
    result = run(code, globals={"left": left, "right": right})
    assert result.value == 5
    assert events == ["enter-2", "enter-3", "exit-3", "exit-2"]


def test_class_definition_and_method_binding():
    code = """
class Greeter:
    kind = "greeter"
    def __init__(self, name):
        self.name = name
    def describe(self):
        return self.kind + ":" + self.name

g = Greeter("Ada")
g.describe()
"""
    result = run(code)
    assert result.value == "greeter:Ada"


def test_class_decorator_applies():
    def add_flag(cls):
        cls.decorated = True
        return cls

    code = """
@add_flag
class Box:
    pass

box = Box()
box.decorated
"""
    result = run(code, globals={"add_flag": add_flag})
    assert result.value is True


def test_function_decorator_applies():
    code = """
def wrap(fn):
    def inner(value):
        return fn(value) + 1
    return inner

@wrap
def add_one(x):
    return x

add_one(2)
"""
    result = run(code)
    assert result.value == 3


def test_function_annotations_set():
    code = """
def add(x: int, y: int) -> int:
    return x + y

add.__annotations__
"""
    result = run(code, globals={"int": int})
    annotations = result.value
    assert annotations["x"] is int
    assert annotations["y"] is int
    assert annotations["return"] is int


def test_class_inheritance_overrides_method():
    code = """
class Base:
    def who(self):
        return "base"

class Child(Base):
    def who(self):
        return "child"

item = Child()
item.who()
"""
    result = run(code)
    assert result.value == "child"


def test_try_except_else_finally_flow():
    events = []

    class Boom(Exception):
        pass

    code = """
try:
    value = 1
except:
    value = 2
else:
    value = 3
finally:
    events.append("finally")
value
"""
    result = run(code, globals={"events": events})
    assert result.value == 3
    assert events == ["finally"]

    events = []
    code = """
try:
    raise Boom("fail")
except Boom as exc:
    events.append(exc.args[0])
finally:
    events.append("cleanup")
events
"""
    result = run(code, globals={"events": events, "Boom": Boom})
    assert result.value == ["fail", "cleanup"]


def test_raise_and_reraise():
    class Boom(Exception):
        pass

    code = """
try:
    raise Boom("again")
except Boom as exc:
    try:
        raise
    except Boom as again:
        result = again.args[0]
result
"""
    result = run(code, globals={"Boom": Boom})
    assert result.value == "again"


def test_assert_statement():
    code = """
value = 1
assert value == 1
value
"""
    result = run(code)
    assert result.value == 1

    code = """
try:
    assert 0, "bad"
except AssertionError as err:
    result = err.args[0]
result
"""
    result = run(code, globals={"AssertionError": AssertionError})
    assert result.value == "bad"


def test_function_call_and_kwargs():
    code = """

def add(a, b):
    return a + b

value = add(2, b=3)
value
"""
    result = run(code)
    assert result.value == 5


def test_function_params_posonly_kwonly_kwargs():
    code = """
def combine(a, /, b, *, c, **kw):
    return a, b, c, kw
value = combine(1, 2, c=3, d=4)
value
"""
    result = run(code)
    assert result.value == (1, 2, 3, {"d": 4})


def test_varargs_and_kwargs_binding():
    code = """
def pack(*args, **kwargs):
    return args, kwargs
value = pack(1, 2, a=3)
value
"""
    result = run(code)
    assert result.value == ((1, 2), {"a": 3})


def test_posonly_keyword_error():
    code = """
def add(a, /, b):
    return a + b
add(a=1, b=2)
"""
    with pytest.raises(ViperRuntimeError):
        run(code)


def test_kwonly_missing_error():
    code = """
def add(*, value):
    return value
add()
"""
    with pytest.raises(ViperRuntimeError):
        run(code)


def test_call_and_literal_unpacking():
    code = """
def add(a, b, c):
    return a + b + c
values = [1, 2]
data = {"c": 3}
result = add(*values, **data)
combo = [0, *values, 3]
mapping = {"a": 1}
merged = {**mapping, "b": 2}
result, combo, merged
"""
    result = run(code)
    assert result.value == (6, [0, 1, 2, 3], {"a": 1, "b": 2})


def test_tuple_unpack_and_conditional_is_unary():
    code = """
values = [1, 2]
result = (*values, 3)
a = []
b = a
c = []
value = (1 if True else 2, a is b, a is not c, ~1)
result, value
"""
    result = run(code)
    assert result.value == ((1, 2, 3), (1, True, True, -2))


def test_lambda_params_defaults_and_kwonly():
    code = """
adder = lambda x, /, y=2, *, z=3: x + y + z
result = adder(1, z=4)
result
"""
    result = run(code)
    assert result.value == 7


def test_positional_after_keyword_is_error():
    code = """

def add(a, b):
    return a + b

value = add(a=1, 2)
"""
    with pytest.raises(ViperSyntaxError):
        run(code)


def test_attribute_and_subscript_assignment():
    class Box:
        def __init__(self):
            self.value = 0

    box = Box()
    code = """
box.value = 3
box.value
"""
    result = run(code, globals={"box": box})
    assert result.value == 3
    assert box.value == 3

    code = """
items = {"a": 1}
items["b"] = 2
items["b"]
"""
    result = run(code)
    assert result.value == 2


def test_tuple_unpack_assignment():
    code = """
a, b = 1, 2
a + b
"""
    result = run(code)
    assert result.value == 3
    assert result.globals["a"] == 1
    assert result.globals["b"] == 2


def test_nested_tuple_unpack_assignment():
    code = """
values = (1, (2, 3))
a, (b, c) = values
b * c
"""
    result = run(code)
    assert result.value == 6


def test_tuple_unpack_with_attribute_and_subscript():
    class Box:
        def __init__(self):
            self.value = 0

    box = Box()
    code = """
items = {"a": 1, "b": 2}
box.value, items["b"] = (3, 4)
box.value + items["b"]
"""
    result = run(code, globals={"box": box})
    assert result.value == 7
    assert box.value == 3


def test_tuple_unpack_length_mismatch_error():
    code = """
a, b = (1,)
"""
    with pytest.raises(ViperRuntimeError):
        run(code)


def test_manual_compile_executes():
    code = """
value = 5
value
"""
    result = _run_with_compile(code)
    assert result.value == 5


def test_return_outside_function_errors():
    code = """
return 3
"""
    with pytest.raises(ViperRuntimeError):
        run(code)


def test_undefined_name_errors():
    code = """
value = missing
"""
    with pytest.raises(ViperRuntimeError):
        run(code)


def test_non_iterable_for_loop_errors():
    code = """
for item in 3:
    item
"""
    with pytest.raises(ViperRuntimeError):
        run(code)


def test_tool_call_injection():
    def tool_call(name, args):
        return {"tool": name, "args": args}

    code = """
result = tool_call("search", {"query": "hi"})
result
"""
    result = run(code, globals={"tool_call": tool_call})
    assert result.value == {"tool": "search", "args": {"query": "hi"}}


def test_fstring_basic_and_escape():
    code = """
name = "Ripper"
value = 3
text = f"Hi {name}! {{value}}={value}"
text
"""
    result = run(code)
    assert result.value == "Hi Ripper! {value}=3"


def test_fstring_debug_and_conversion():
    code = """
value = 7
text = f"{value=}"
text
"""
    result = run(code)
    assert result.value == "value=7"

    code = """
value = "hi"
text = f"{value!r}"
text
"""
    result = run(code)
    assert result.value == repr("hi")


def test_fstring_format_spec_with_nested_expr():
    code = """
value = 12
width = 4
text = f"{value:{width}}"
text
"""
    result = run(code)
    assert result.value == "  12"


def test_tstring_template_object():
    code = """
name = "Ada"
text = t"Hello {name}"
text
"""
    result = run(code)
    assert isinstance(result.value, TemplateString)
    assert str(result.value) == "Hello Ada"


def test_triple_quoted_strings():
    code = '''
text = """line1
line2"""
text
'''
    result = run(code)
    assert result.value == "line1\nline2"


def test_triple_quoted_fstring():
    code = '''
name = "Ada"
text = f"""Hi
{name}"""
text
'''
    result = run(code)
    assert result.value == "Hi\nAda"


def test_raw_string_prefix():
    code = r'''
text = r"\n"
text
'''
    result = run(code)
    assert result.value == "\\n"


def test_bytes_string_prefix():
    code = r'''
value = b"hi"
value
'''
    result = run(code)
    assert result.value == b"hi"

    code = r'''
value = b"\n"
value
'''
    result = run(code)
    assert result.value == b"\n"

    code = r'''
value = b"ab" b"cd"
value
'''
    result = run(code)
    assert result.value == b"abcd"

def test_raw_bytes_prefix():
    code = r'''
value = rb"\n"
value
'''
    result = run(code)
    assert result.value == b"\\n"


def test_unicode_prefix():
    code = r'''
value = u"hi"
value
'''
    result = run(code)
    assert result.value == "hi"


def test_raw_fstring_prefix():
    code = r'''
name = "Neo"
value = fr"\n{name}"
value
'''
    result = run(code)
    assert result.value == "\\nNeo"


def test_bytes_fstring_prefix():
    code = r'''
value = bf"{1}A"
value
'''
    result = run(code)
    assert result.value == b"1A"

    code = r'''
value = brf"\n{2}"
value
'''
    result = run(code)
    assert result.value == b"\\n2"


def test_template_bytes():
    code = r'''
name = "Ada"
text = bt"Hello {name}"
text
'''
    result = run(code)
    assert isinstance(result.value, TemplateBytes)
    assert bytes(result.value) == b"Hello Ada"


def test_mixed_bytes_and_string_literals_error():
    code = r'''
value = "hi" b"there"
'''
    with pytest.raises(ViperSyntaxError):
        run(code)


def test_custom_conversion_function():
    def upper(value):
        return str(value).upper()

    code = """
name = "Ada"
text = f"{name!upper}"
text
"""
    result = run(code, globals={"upper": upper})
    assert result.value == "ADA"


def test_unknown_conversion_falls_back_to_str():
    code = """
value = 3
text = f"{value!unknown}"
text
"""
    result = run(code)
    assert result.value == "3"


def test_escape_hex_and_unicode_sequences():
    code = r'''
text = "\x41\u0042\U00000043"
text
'''
    result = run(code)
    assert result.value == "ABC"

    code = r'''
text = "\N{GREEK SMALL LETTER ALPHA}"
text
'''
    result = run(code)
    assert result.value == "α"

    code = r'''
value = b"\x41"
value
'''
    result = run(code)
    assert result.value == b"A"

    code = r'''
value = b"\u03b1"
'''
    with pytest.raises(ViperSyntaxError):
        run(code)


def test_bytes_non_ascii_literal_error():
    code = 'value = b"é"'
    with pytest.raises(ViperSyntaxError):
        run(code)


def test_disassemble_includes_opcodes():
    code = """
value = 1
if value:
    value = value + 1
value
"""
    tokens = tokenize(code)
    module = parse(tokens)
    compiled = compile_module(module)
    output = disassemble(compiled)
    assert "LOAD_CONST" in output
    assert "JUMP_IF_FALSE" in output


def test_viper_cli_run_file(tmp_path, capsys):
    script = tmp_path / "sample.vp"
    script.write_text(
        """
value = 2 + 3
value
""",
        encoding="utf-8",
    )
    exit_code = run_file(script, show_disassembly=True)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "LOAD_CONST" in captured.out
    assert "Result: 5" in captured.out


def test_semicolon_separates_statements():
    code = "a = 1; b = 2; a + b"
    result = run(code)
    assert result.value == 3


def test_for_else_and_while_else():
    code = """
values = []
for item in [1, 2]:
    values = values + [item]
else:
    values = values + [3]
values
"""
    result = run(code)
    assert result.value == [1, 2, 3]

    code = """
values = []
for item in [1, 2]:
    break
else:
    values = values + [3]
values
"""
    result = run(code)
    assert result.value == []

    code = """
count = 0
values = []
while count < 2:
    count = count + 1
    values = values + [count]
else:
    values = values + [99]
values
"""
    result = run(code)
    assert result.value == [1, 2, 99]


def test_import_and_from_import():
    code = """
import math
value = math.sqrt(9)
value
"""
    result = run(code)
    assert result.value == 3.0

    code = """
from math import sqrt
value = sqrt(16)
value
"""
    result = run(code)
    assert result.value == 4.0


def test_local_module_imports(tmp_path):
    module_path = tmp_path / "util.vp"
    module_path.write_text(
        """
value = 5
""",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.vp"
    main_path.write_text(
        """
import util
result = util.value
result
""",
        encoding="utf-8",
    )
    result = _run_file(main_path)
    assert result.value == 5

    main_path.write_text(
        """
from util import value
value
""",
        encoding="utf-8",
    )
    result = _run_file(main_path)
    assert result.value == 5

    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    rel_mod = package_dir / "mod.vp"
    rel_mod.write_text(
        """
answer = 42
""",
        encoding="utf-8",
    )
    rel_main = package_dir / "main.vp"
    rel_main.write_text(
        """
from .mod import answer
answer
""",
        encoding="utf-8",
    )
    result = _run_file(rel_main)
    assert result.value == 42


def test_circular_import_detection(tmp_path):
    mod_a = tmp_path / "a.vp"
    mod_b = tmp_path / "b.vp"
    mod_a.write_text(
        """
import b
value = 1
""",
        encoding="utf-8",
    )
    mod_b.write_text(
        """
import a
value = 2
""",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.vp"
    main_path.write_text(
        """
import a
""",
        encoding="utf-8",
    )
    with pytest.raises(ViperRuntimeError):
        _run_file(main_path)


def test_global_and_nonlocal():
    code = """
value = 1
def bump():
    global value
    value = value + 1
bump()
value
"""
    result = run(code)
    assert result.value == 2

    code = """
def outer():
    value = 2
    def inner():
        nonlocal value
        value = 3
    inner()
    return value
result = outer()
result
"""
    result = run(code)
    assert result.value == 3


def test_del_statement_and_type_alias():
    code = """
value = 1
del value
"""
    result = run(code)
    assert "value" not in result.globals

    code = """
type Alias = 5
Alias
"""
    result = run(code)
    assert result.value == 5


def test_match_case_basic_and_sequence_capture():
    code = """
value = 2
match value:
    case 1:
        result = "one"
    case 2:
        result = "two"
result
"""
    result = run(code)
    assert result.value == "two"

    code = """
value = [1, 2]
match value:
    case [a, b]:
        result = a + b
result
"""
    result = run(code)
    assert result.value == 3


def test_match_case_star_and_as_patterns():
    code = """
value = [1, 2, 3, 4]
match value:
    case [head, *tail]:
        result = (head, tail)
result
"""
    result = run(code)
    assert result.value == (1, [2, 3, 4])

    code = """
value = [1, 2]
match value:
    case [a, b] as whole:
        result = (a + b, whole)
result
"""
    result = run(code)
    assert result.value == (3, [1, 2])


def test_match_case_mapping_and_class_patterns():
    code = """
class Keys:
    a = "a"

value = {"a": 1, "b": 2}
match value:
    case {Keys.a: x, "b": y, **rest}:
        result = (x + y, rest)
result
"""
    result = run(code)
    assert result.value == (3, {})

    code = """
class Point:
    __match_args__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
match p:
    case Point(1, y=val):
        result = val
result
"""
    result = run(code)
    assert result.value == 2

    code = """
class Colors:
    Red = 1

value = 1
match value:
    case Colors.Red:
        result = "red"
    case _:
        result = "other"
result
"""
    result = run(code)
    assert result.value == "red"


def test_match_pattern_semantics_errors():
    code = """
value = {"a": 1, "b": 2}
match value:
    case {"a": x, "a": y}:
        result = x + y
"""
    with pytest.raises(ViperSyntaxError):
        run(code)

    code = """
class Keys:
    a = "a"

value = {"a": 1}
match value:
    case {Keys.a: x, Keys.a: y}:
        result = x + y
"""
    with pytest.raises(ViperSyntaxError):
        run(code)

    code = """
class Point:
    __match_args__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
match p:
    case Point(1, x=2):
        result = 1
"""
    with pytest.raises(ViperRuntimeError):
        run(code)

    code = """
class Bad:
    __match_args__ = ("x", 1)
    def __init__(self, x):
        self.x = x

value = Bad(1)
match value:
    case Bad(1):
        result = 1
"""
    with pytest.raises(ViperRuntimeError):
        run(code)

    code = """
value = {"a": 1, "b": 2}
match value:
    case {"a": 1, **_}:
        result = 1
result
"""
    result = run(code)
    assert result.value == 1

    code = """
value = [1, 2]
match value:
    case [x, x]:
        result = x
"""
    with pytest.raises(ViperSyntaxError):
        run(code)

    code = """
value = 1
match value:
    case 1 as x:
        result = x
    case 2:
        result = 2
result
"""
    result = run(code)
    assert result.value == 1

    code = """
value = 1
match value:
    case 1 as x | 2:
        result = x
"""
    with pytest.raises(ViperSyntaxError):
        run(code)

    code = """
value = [1]
match value:
    case [x] | [y, z]:
        result = x
"""
    with pytest.raises(ViperSyntaxError):
        run(code)


def test_yield_and_yield_from():
    code = """
def gen():
    yield 1
    yield 2
g = gen()
"""
    result = run(code)
    assert list(result.globals["g"]) == [1, 2]

    code = """
def gen():
    yield from [1, 2]
g = gen()
"""
    result = run(code)
    assert list(result.globals["g"]) == [1, 2]


def test_async_def_and_async_for():
    code = """
async def add():
    return 3
value = add()
"""
    result = run(code)
    assert asyncio.run(result.globals["value"]) == 3

    code = """
async def total():
    result = 0
    async for item in [1, 2]:
        result = result + item
    return result
value = total()
"""
    result = run(code)
    assert asyncio.run(result.globals["value"]) == 3
