"""Tests for output truncation helpers."""

from ripperdoc.utils.output_utils import truncate_output


def test_truncate_output_respects_max_length():
    """Truncated output should never exceed the requested limit."""
    text = "x" * 80
    result = truncate_output(text, max_chars=50)

    assert result["is_truncated"] is True
    assert result["original_length"] == 80
    assert len(result["truncated_content"]) <= 50
    assert "truncated" in result["truncated_content"]


def test_truncate_output_handles_tiny_budgets():
    """Even very small budgets should return a bounded string."""
    text = "abcdef" * 10
    result = truncate_output(text, max_chars=8)

    assert result["is_truncated"] is True
    assert len(result["truncated_content"]) <= 8


def test_truncate_output_reports_omitted_location_metadata():
    """Truncation metadata should include omitted character and line ranges."""
    text = "x" * 200
    result = truncate_output(text, max_chars=64)

    assert result["is_truncated"] is True
    assert result["omitted_chars"] > 0
    assert result["omitted_start_line"] == 1
    assert result["omitted_end_line"] == 1
    assert result["omitted_start_char"] == result["kept_start_chars"] + 1
    assert result["omitted_end_char"] == result["original_length"] - result["kept_end_chars"]
