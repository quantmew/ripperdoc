#!/usr/bin/env python3
"""Generate Python model catalog module from JSON source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat


def generate_module(source_json: Path, output_py: Path) -> None:
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    payload_literal = pformat(payload, width=120, sort_dicts=True)

    module_body = (
        '"""Generated from assets/model_prices_and_context_window.json.\n\n'
        'Do not edit manually. Run scripts/generate_model_prices_module.py to regenerate.\n'
        '"""\n\n'
        'from __future__ import annotations\n\n'
        f"MODEL_PRICES_AND_CONTEXT_WINDOW = {payload_literal}\n\n"
        '__all__ = ["MODEL_PRICES_AND_CONTEXT_WINDOW"]\n'
    )

    output_py.parent.mkdir(parents=True, exist_ok=True)
    output_py.write_text(module_body, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Generate ripperdoc/data/model_prices_and_context_window.py from JSON"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "assets" / "model_prices_and_context_window.json",
        help="Source JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "ripperdoc" / "data" / "model_prices_and_context_window.py",
        help="Output Python module path",
    )
    args = parser.parse_args()

    source_json = args.source.resolve()
    output_py = args.output.resolve()

    if not source_json.is_file():
        raise SystemExit(f"Source JSON not found: {source_json}")

    generate_module(source_json, output_py)
    print(f"Generated: {output_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
