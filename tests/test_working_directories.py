"""Tests for additional working directory helpers."""

from pathlib import Path

from ripperdoc.utils.working_directories import (
    coerce_directory_list,
    extract_additional_directories,
    normalize_directory_inputs,
)


def test_coerce_directory_list_handles_scalar_and_list_inputs():
    assert coerce_directory_list(None) == []
    assert coerce_directory_list("  ./foo  ") == ["./foo"]
    assert coerce_directory_list(["a", " ", None, Path("b")]) == ["a", "b"]


def test_extract_additional_directories_uses_working_directories_key():
    payload = {"working_directories": ["./foo"]}
    assert extract_additional_directories(payload) == ["./foo"]


def test_extract_additional_directories_ignores_legacy_aliases():
    payload = {"additionalDirectories": ["./foo"], "additional_directories": ["./bar"]}
    assert extract_additional_directories(payload) == []


def test_normalize_directory_inputs_resolves_and_deduplicates(tmp_path: Path):
    base_dir = tmp_path
    first = base_dir / "first"
    second = base_dir / "second"
    first.mkdir()
    second.mkdir()

    normalized, errors = normalize_directory_inputs(
        ["first", "./first", str(second)],
        base_dir=base_dir,
        require_exists=True,
    )

    assert errors == []
    assert normalized == [str(first.resolve()), str(second.resolve())]


def test_normalize_directory_inputs_reports_missing_directories(tmp_path: Path):
    normalized, errors = normalize_directory_inputs(
        ["missing-dir"],
        base_dir=tmp_path,
        require_exists=True,
    )

    assert normalized == []
    assert errors and "missing-dir" in errors[0]
