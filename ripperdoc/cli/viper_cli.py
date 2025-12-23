"""CLI for running Viper scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from ripperdoc.core.viper import (
    Interpreter,
    ViperError,
    compile_module,
    disassemble,
    parse,
    tokenize,
)
from ripperdoc.core.viper.diagnostics import format_viper_diagnostic

_SUPPORTED_EXTENSIONS = {".vp"}
_LEGACY_EXTENSIONS = {".viper"}


def _validate_script_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in _SUPPORTED_EXTENSIONS:
        return
    if suffix in _LEGACY_EXTENSIONS:
        click.echo(
            "Warning: .viper extension is deprecated; prefer .vp.",
            err=True,
        )
        return
    raise SystemExit("Viper scripts must use the .vp extension.")


def run_file(path: Path, *, show_disassembly: bool = False) -> int:
    _validate_script_path(path)
    source = path.read_text(encoding="utf-8")
    tokens = tokenize(source)
    module = parse(tokens)
    code = compile_module(module)
    if show_disassembly:
        click.echo(disassemble(code))
        click.echo("")
    vm = Interpreter(base_path=path.parent)
    result = vm.execute(code, source_path=path)
    if result.value is not None:
        click.echo(f"Result: {result.value!r}")
    return 0


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("script", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--disassemble", "show_disassembly", is_flag=True, help="Print bytecode.")
def main(script: Path, show_disassembly: bool) -> None:
    """Run a Viper script through tokenize -> parse -> compile -> execute."""
    try:
        raise SystemExit(run_file(script, show_disassembly=show_disassembly))
    except ViperError as exc:
        source = ""
        try:
            source = script.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            source = ""
        message = format_viper_diagnostic(exc, source, script)
        raise SystemExit(message) from exc
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Unhandled error: {exc}") from exc
