"""Tests for grep output parsing with Windows paths."""

from ripperdoc.tools.grep_tool import (
    _normalize_glob_for_grep,
    _parse_content_line,
    _parse_count_line,
    _split_globs,
)


def test_parse_count_line_windows_path():
    line = r"C:\\repo\\file.txt:12"
    parsed = _parse_count_line(line)
    assert parsed == (r"C:\\repo\\file.txt", 12)


def test_parse_content_line_windows_path_with_colons():
    line = r"C:\\repo\\file.txt:42:hello:world"
    parsed = _parse_content_line(line)
    assert parsed == (r"C:\\repo\\file.txt", 42, "hello:world")


def test_parse_content_line_non_windows_path():
    line = "/tmp/file.txt:7:match"
    parsed = _parse_content_line(line)
    assert parsed == ("/tmp/file.txt", 7, "match")


def test_parse_count_line_single_value():
    parsed = _parse_count_line("5")
    assert parsed == ("", 5)


def test_parse_content_line_single_file_format():
    parsed = _parse_content_line("12:hello")
    assert parsed == ("", 12, "hello")


def test_parse_content_line_invalid():
    parsed = _parse_content_line("nope")
    assert parsed is None


def test_split_globs_whitespace_and_commas():
    globs = _split_globs("**/*.py,**/*.md  *.txt")
    assert globs == ["**/*.py", "**/*.md", "*.txt"]


def test_normalize_glob_for_grep_strips_paths():
    assert _normalize_glob_for_grep("**/*.py") == "*.py"
    assert _normalize_glob_for_grep("src/**/*.ts") == "*.ts"
    assert _normalize_glob_for_grep("*.md") == "*.md"
