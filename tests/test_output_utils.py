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
