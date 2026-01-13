"""Comprehensive tests for utility modules.

Tests cover:
- coerce.py: Type coercion functions
- json_utils.py: JSON parsing utilities
- safe_get_cwd.py: Safe directory operations
- bash_constants.py: Bash configuration constants
- token_estimation.py: Token counting utilities
- shell_token_utils.py: Shell command parsing
- exit_code_handlers.py: Exit code interpretation
- output_utils.py: Output processing utilities
- message_formatting.py: Message content formatting
"""

import json
import math
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# coerce.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.coerce import parse_boolish, parse_optional_int


class TestParseBoolish:
    """Tests for parse_boolish function."""

    def test_none_returns_default(self):
        """None should return the default value."""
        assert parse_boolish(None) is False
        assert parse_boolish(None, default=True) is True

    def test_bool_passthrough(self):
        """Boolean values should pass through unchanged."""
        assert parse_boolish(True) is True
        assert parse_boolish(False) is False

    def test_numeric_truthy(self):
        """Non-zero numbers should be truthy."""
        assert parse_boolish(1) is True
        assert parse_boolish(42) is True
        assert parse_boolish(-1) is True
        assert parse_boolish(1.5) is True

    def test_numeric_falsey(self):
        """Zero should be falsey."""
        assert parse_boolish(0) is False
        assert parse_boolish(0.0) is False

    def test_string_truthy_values(self):
        """String truthy values should return True."""
        assert parse_boolish("1") is True
        assert parse_boolish("true") is True
        assert parse_boolish("TRUE") is True
        assert parse_boolish("True") is True
        assert parse_boolish("yes") is True
        assert parse_boolish("YES") is True
        assert parse_boolish("on") is True
        assert parse_boolish("ON") is True

    def test_string_falsey_values(self):
        """String falsey values should return False."""
        assert parse_boolish("0") is False
        assert parse_boolish("false") is False
        assert parse_boolish("FALSE") is False
        assert parse_boolish("False") is False
        assert parse_boolish("no") is False
        assert parse_boolish("NO") is False
        assert parse_boolish("off") is False
        assert parse_boolish("OFF") is False

    def test_string_with_whitespace(self):
        """Strings with whitespace should be trimmed."""
        assert parse_boolish("  true  ") is True
        assert parse_boolish("  false  ") is False
        assert parse_boolish("\tyes\n") is True

    def test_unknown_string_returns_default(self):
        """Unknown string values should return default."""
        assert parse_boolish("maybe") is False
        assert parse_boolish("maybe", default=True) is True
        assert parse_boolish("unknown") is False
        assert parse_boolish("") is False

    def test_other_types_return_default(self):
        """Other types should return default."""
        assert parse_boolish([]) is False
        assert parse_boolish({}) is False
        assert parse_boolish(object()) is False


class TestParseOptionalInt:
    """Tests for parse_optional_int function."""

    def test_none_returns_none(self):
        """None should return None."""
        assert parse_optional_int(None) is None

    def test_int_passthrough(self):
        """Integer values should pass through."""
        assert parse_optional_int(42) == 42
        assert parse_optional_int(0) == 0
        assert parse_optional_int(-5) == -5

    def test_string_int(self):
        """String integers should be parsed."""
        assert parse_optional_int("42") == 42
        assert parse_optional_int("0") == 0
        assert parse_optional_int("-10") == -10

    def test_string_with_whitespace(self):
        """Strings with whitespace should be trimmed."""
        assert parse_optional_int("  42  ") == 42
        assert parse_optional_int("\t100\n") == 100

    def test_float_returns_none(self):
        """Float values (via string conversion) should return None."""
        # parse_optional_int converts to string first, so "3.7" can't parse as int
        assert parse_optional_int(3.7) is None
        assert parse_optional_int(3.2) is None

    def test_bool_converts(self):
        """Boolean should convert to 0 or 1."""
        assert parse_optional_int(True) == 1
        assert parse_optional_int(False) == 0

    def test_invalid_string_returns_none(self):
        """Invalid strings should return None."""
        assert parse_optional_int("abc") is None
        assert parse_optional_int("12.5") is None
        assert parse_optional_int("") is None

    def test_other_types_return_none(self):
        """Other types should return None."""
        assert parse_optional_int([]) is None
        assert parse_optional_int({}) is None


# ─────────────────────────────────────────────────────────────────────────────
# json_utils.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.json_utils import safe_parse_json


class TestSafeParseJson:
    """Tests for safe_parse_json function."""

    def test_valid_json_object(self):
        """Valid JSON object should be parsed."""
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        """Valid JSON array should be parsed."""
        result = safe_parse_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_valid_json_primitives(self):
        """Valid JSON primitives should be parsed."""
        assert safe_parse_json('"hello"') == "hello"
        assert safe_parse_json('42') == 42
        assert safe_parse_json('true') is True
        assert safe_parse_json('false') is False
        assert safe_parse_json('null') is None

    def test_none_returns_none(self):
        """None input should return None."""
        assert safe_parse_json(None) is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert safe_parse_json("") is None

    def test_invalid_json_returns_none(self):
        """Invalid JSON should return None."""
        assert safe_parse_json("{invalid}") is None
        assert safe_parse_json("not json") is None
        assert safe_parse_json("{") is None

    def test_nested_json(self):
        """Nested JSON should be parsed."""
        result = safe_parse_json('{"nested": {"key": [1, 2, 3]}}')
        assert result == {"nested": {"key": [1, 2, 3]}}

    def test_log_error_false_suppresses_logging(self):
        """log_error=False should not log errors."""
        result = safe_parse_json("invalid", log_error=False)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# safe_get_cwd.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.safe_get_cwd import safe_get_cwd, get_original_cwd


class TestSafeGetCwd:
    """Tests for safe_get_cwd functions."""

    def test_get_original_cwd_returns_string(self):
        """get_original_cwd should return a string path."""
        result = get_original_cwd()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_safe_get_cwd_returns_current_directory(self):
        """safe_get_cwd should return current directory."""
        result = safe_get_cwd()
        assert isinstance(result, str)
        assert Path(result).exists()

    def test_safe_get_cwd_fallback_on_oserror(self, monkeypatch):
        """safe_get_cwd should fall back on OSError."""
        def raise_oserror():
            raise OSError("Directory deleted")

        monkeypatch.setattr(os, "getcwd", raise_oserror)
        result = safe_get_cwd()
        # Should return original cwd as fallback
        assert isinstance(result, str)

    def test_safe_get_cwd_fallback_on_runtime_error(self, monkeypatch):
        """safe_get_cwd should fall back on RuntimeError."""
        def raise_runtime():
            raise RuntimeError("Some error")

        monkeypatch.setattr(os, "getcwd", raise_runtime)
        result = safe_get_cwd()
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# bash_constants.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.bash_constants import (
    get_bash_max_output_length,
    get_bash_default_timeout_ms,
    get_bash_max_timeout_ms,
)


class TestBashConstants:
    """Tests for bash constants functions."""

    def test_default_max_output_length(self, monkeypatch):
        """Default max output length should be 30000."""
        monkeypatch.delenv("BASH_MAX_OUTPUT_LENGTH", raising=False)
        assert get_bash_max_output_length() == 30000

    def test_override_max_output_length(self, monkeypatch):
        """Environment variable should override max output length."""
        monkeypatch.setenv("BASH_MAX_OUTPUT_LENGTH", "50000")
        assert get_bash_max_output_length() == 50000

    def test_invalid_max_output_length_uses_default(self, monkeypatch):
        """Invalid environment value should use default."""
        monkeypatch.setenv("BASH_MAX_OUTPUT_LENGTH", "invalid")
        assert get_bash_max_output_length() == 30000

    def test_negative_max_output_length_uses_default(self, monkeypatch):
        """Negative value should use default."""
        monkeypatch.setenv("BASH_MAX_OUTPUT_LENGTH", "-100")
        assert get_bash_max_output_length() == 30000

    def test_zero_max_output_length_uses_default(self, monkeypatch):
        """Zero value should use default."""
        monkeypatch.setenv("BASH_MAX_OUTPUT_LENGTH", "0")
        assert get_bash_max_output_length() == 30000

    def test_default_timeout(self, monkeypatch):
        """Default timeout should be 120000ms."""
        monkeypatch.delenv("BASH_DEFAULT_TIMEOUT_MS", raising=False)
        assert get_bash_default_timeout_ms() == 120000

    def test_override_default_timeout(self, monkeypatch):
        """Environment variable should override default timeout."""
        monkeypatch.setenv("BASH_DEFAULT_TIMEOUT_MS", "60000")
        assert get_bash_default_timeout_ms() == 60000

    def test_default_max_timeout(self, monkeypatch):
        """Default max timeout should be 600000ms."""
        monkeypatch.delenv("BASH_MAX_TIMEOUT_MS", raising=False)
        monkeypatch.delenv("BASH_DEFAULT_TIMEOUT_MS", raising=False)
        assert get_bash_max_timeout_ms() == 600000

    def test_override_max_timeout(self, monkeypatch):
        """Environment variable should override max timeout."""
        monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", "900000")
        monkeypatch.delenv("BASH_DEFAULT_TIMEOUT_MS", raising=False)
        assert get_bash_max_timeout_ms() == 900000

    def test_max_timeout_never_less_than_default(self, monkeypatch):
        """Max timeout should never be less than default timeout."""
        monkeypatch.setenv("BASH_DEFAULT_TIMEOUT_MS", "300000")
        monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", "100000")
        result = get_bash_max_timeout_ms()
        assert result >= 300000


# ─────────────────────────────────────────────────────────────────────────────
# token_estimation.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.token_estimation import estimate_tokens


class TestTokenEstimation:
    """Tests for estimate_tokens function."""

    def test_empty_string_returns_zero(self):
        """Empty string should return 0 tokens."""
        assert estimate_tokens("") == 0

    def test_short_text(self):
        """Short text should return positive token count."""
        result = estimate_tokens("Hello world")
        assert result > 0

    def test_long_text(self):
        """Long text should return proportional token count."""
        short_result = estimate_tokens("Hello")
        long_result = estimate_tokens("Hello " * 100)
        assert long_result > short_result

    def test_heuristic_fallback(self):
        """Heuristic should estimate ~4 chars per token."""
        # Without tiktoken, should use heuristic
        text = "a" * 100  # 100 chars
        result = estimate_tokens(text)
        # Should be around 25 tokens (100/4)
        assert result >= 1
        assert result <= 100  # Upper bound sanity check

    def test_minimum_one_token(self):
        """Non-empty text should have at least 1 token."""
        assert estimate_tokens("a") >= 1
        assert estimate_tokens("ab") >= 1


# ─────────────────────────────────────────────────────────────────────────────
# shell_token_utils.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.shell_token_utils import (
    parse_shell_tokens,
    filter_valid_tokens,
    parse_and_clean_shell_tokens,
    SHELL_OPERATORS_WITH_REDIRECTION,
)


class TestParseShellTokens:
    """Tests for parse_shell_tokens function."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert parse_shell_tokens("") == []

    def test_simple_command(self):
        """Simple command should be tokenized."""
        assert parse_shell_tokens("ls -la") == ["ls", "-la"]

    def test_command_with_pipe(self):
        """Command with pipe should include pipe token."""
        tokens = parse_shell_tokens("cat file | grep pattern")
        assert "|" in tokens

    def test_command_with_quotes(self):
        """Quoted strings should be preserved."""
        tokens = parse_shell_tokens('echo "hello world"')
        assert "hello world" in tokens

    def test_command_with_redirect(self):
        """Redirect operators should be tokenized."""
        tokens = parse_shell_tokens("echo hello > file.txt")
        assert ">" in tokens

    def test_complex_command(self):
        """Complex commands should be tokenized correctly."""
        tokens = parse_shell_tokens("git log --oneline | head -10 | grep feat")
        assert "git" in tokens
        assert "--oneline" in tokens
        assert "|" in tokens

    def test_invalid_quotes_fallback(self):
        """Invalid quotes should fall back to simple split."""
        tokens = parse_shell_tokens('echo "unclosed')
        assert len(tokens) > 0


class TestFilterValidTokens:
    """Tests for filter_valid_tokens function."""

    def test_removes_operators(self):
        """Should remove shell operators."""
        tokens = ["ls", "|", "grep", "pattern"]
        result = filter_valid_tokens(tokens)
        assert "|" not in result
        assert "ls" in result
        assert "grep" in result

    def test_removes_all_operators(self):
        """Should remove all operator types."""
        operators = list(SHELL_OPERATORS_WITH_REDIRECTION)
        result = filter_valid_tokens(operators)
        assert len(result) == 0

    def test_preserves_regular_tokens(self):
        """Should preserve regular command tokens."""
        tokens = ["ls", "-la", "/tmp"]
        result = filter_valid_tokens(tokens)
        assert result == tokens


class TestParseAndCleanShellTokens:
    """Tests for parse_and_clean_shell_tokens function."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert parse_and_clean_shell_tokens("") == []

    def test_simple_command(self):
        """Simple command should be parsed and cleaned."""
        result = parse_and_clean_shell_tokens("ls -la")
        assert "ls" in result
        assert "-la" in result

    def test_removes_dev_null_redirect(self):
        """Should remove /dev/null redirections."""
        result = parse_and_clean_shell_tokens("command 2>/dev/null")
        assert "/dev/null" not in result
        assert "2>/dev/null" not in result

    def test_removes_file_descriptor_redirect(self):
        """Should remove file descriptor redirections."""
        result = parse_and_clean_shell_tokens("command 2>&1")
        assert "2>&1" not in result

    def test_removes_operators(self):
        """Should remove operators after cleaning."""
        result = parse_and_clean_shell_tokens("ls | grep pattern")
        assert "|" not in result

    def test_complex_command_cleaning(self):
        """Complex command should be cleaned properly."""
        result = parse_and_clean_shell_tokens("cmd1 && cmd2 || cmd3 2>/dev/null")
        assert "&&" not in result
        assert "||" not in result


# ─────────────────────────────────────────────────────────────────────────────
# exit_code_handlers.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.exit_code_handlers import (
    ExitCodeResult,
    default_handler,
    grep_handler,
    diff_handler,
    test_handler as shell_test_handler,  # Renamed to avoid pytest collection
    find_handler,
    normalize_command,
    classify_command,
    get_exit_code_handler,
    interpret_exit_code,
    create_exit_result,
    COMMON_COMMANDS,
)


class TestExitCodeResult:
    """Tests for ExitCodeResult dataclass."""

    def test_default_values(self):
        """Default values should be correct."""
        result = ExitCodeResult(is_error=False)
        assert result.is_error is False
        assert result.message is None
        assert result.semantic_meaning is None

    def test_with_all_values(self):
        """All values should be settable."""
        result = ExitCodeResult(
            is_error=True,
            message="Error occurred",
            semantic_meaning="Command failed",
        )
        assert result.is_error is True
        assert result.message == "Error occurred"
        assert result.semantic_meaning == "Command failed"


class TestDefaultHandler:
    """Tests for default_handler function."""

    def test_exit_0_is_not_error(self):
        """Exit code 0 should not be error."""
        result = default_handler(0, "", "")
        assert result.is_error is False

    def test_non_zero_is_error(self):
        """Non-zero exit code should be error."""
        result = default_handler(1, "", "")
        assert result.is_error is True
        assert "exit code 1" in result.message


class TestGrepHandler:
    """Tests for grep_handler function."""

    def test_exit_0_found_matches(self):
        """Exit code 0 means matches found."""
        result = grep_handler(0, "match", "")
        assert result.is_error is False
        assert result.semantic_meaning is None

    def test_exit_1_no_matches(self):
        """Exit code 1 means no matches found."""
        result = grep_handler(1, "", "")
        assert result.is_error is False
        assert result.semantic_meaning == "No matches found"

    def test_exit_2_is_error(self):
        """Exit code 2+ is error."""
        result = grep_handler(2, "", "Error")
        assert result.is_error is True


class TestDiffHandler:
    """Tests for diff_handler function."""

    def test_exit_0_files_identical(self):
        """Exit code 0 means files identical."""
        result = diff_handler(0, "", "")
        assert result.is_error is False
        assert result.semantic_meaning == "Files are identical"

    def test_exit_1_files_differ(self):
        """Exit code 1 means files differ."""
        result = diff_handler(1, "differences", "")
        assert result.is_error is False
        assert result.semantic_meaning == "Files differ"

    def test_exit_2_is_error(self):
        """Exit code 2+ is error."""
        result = diff_handler(2, "", "Error")
        assert result.is_error is True


class TestShellTestHandler:
    """Tests for shell test_handler function."""

    def test_exit_0_condition_true(self):
        """Exit code 0 means condition is true."""
        result = shell_test_handler(0, "", "")
        assert result.is_error is False
        assert result.semantic_meaning == "Condition is true"

    def test_exit_1_condition_false(self):
        """Exit code 1 means condition is false."""
        result = shell_test_handler(1, "", "")
        assert result.is_error is False
        assert result.semantic_meaning == "Condition is false"

    def test_exit_2_is_error(self):
        """Exit code 2+ is error."""
        result = shell_test_handler(2, "", "Error")
        assert result.is_error is True


class TestFindHandler:
    """Tests for find_handler function."""

    def test_exit_0_success(self):
        """Exit code 0 means success."""
        result = find_handler(0, "/path/file", "")
        assert result.is_error is False

    def test_exit_1_partial(self):
        """Exit code 1 means some dirs inaccessible."""
        result = find_handler(1, "", "Permission denied")
        assert result.is_error is False
        assert "inaccessible" in result.semantic_meaning

    def test_exit_2_is_error(self):
        """Exit code 2+ is error."""
        result = find_handler(2, "", "Error")
        assert result.is_error is True


class TestNormalizeCommand:
    """Tests for normalize_command function."""

    def test_simple_command(self):
        """Simple command should return first word."""
        assert normalize_command("git status") == "git"
        assert normalize_command("ls -la") == "ls"

    def test_command_with_pipe(self):
        """Piped command should return last command."""
        assert normalize_command("cat file | grep pattern") == "grep"
        assert normalize_command("ls | head | tail") == "tail"

    def test_empty_command(self):
        """Empty command should return empty string."""
        assert normalize_command("") == ""
        assert normalize_command("   ") == ""


class TestClassifyCommand:
    """Tests for classify_command function."""

    def test_known_commands(self):
        """Known commands should be classified."""
        assert classify_command("npm install") == "npm"
        assert classify_command("python script.py") == "python"
        assert classify_command("pytest tests/") == "pytest"

    def test_unknown_command(self):
        """Unknown commands should return 'other'."""
        assert classify_command("someunknowncommand") == "other"

    def test_empty_command(self):
        """Empty command should return 'other'."""
        assert classify_command("") == "other"

    def test_command_with_operators(self):
        """Commands with operators should be classified."""
        result = classify_command("npm install && npm test")
        assert result == "npm"


class TestGetExitCodeHandler:
    """Tests for get_exit_code_handler function."""

    def test_grep_commands(self):
        """grep-like commands should use grep handler."""
        assert get_exit_code_handler("grep pattern") == grep_handler
        assert get_exit_code_handler("rg pattern") == grep_handler

    def test_diff_command(self):
        """diff command should use diff handler."""
        assert get_exit_code_handler("diff file1 file2") == diff_handler

    def test_test_command(self):
        """test command should use test handler."""
        assert get_exit_code_handler("test -f file") == shell_test_handler
        assert get_exit_code_handler("[ -f file ]") == shell_test_handler

    def test_find_command(self):
        """find command should use find handler."""
        assert get_exit_code_handler("find . -name '*'") == find_handler

    def test_unknown_uses_default(self):
        """Unknown commands should use default handler."""
        assert get_exit_code_handler("somecommand") == default_handler


class TestInterpretExitCode:
    """Tests for interpret_exit_code function."""

    def test_grep_interpretation(self):
        """grep exit codes should be interpreted correctly."""
        result = interpret_exit_code("grep pattern", 1, "", "")
        assert result.is_error is False
        assert result.semantic_meaning == "No matches found"

    def test_default_interpretation(self):
        """Default commands should use standard interpretation."""
        result = interpret_exit_code("unknowncommand", 1, "", "")
        assert result.is_error is True


class TestCreateExitResult:
    """Tests for create_exit_result function."""

    def test_wrapper_works(self):
        """create_exit_result should be alias for interpret_exit_code."""
        result = create_exit_result("grep", 0, "match", "")
        assert result.is_error is False


# ─────────────────────────────────────────────────────────────────────────────
# output_utils.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.output_utils import (
    trim_blank_lines,
    is_image_data,
    truncate_output,
    format_duration,
    is_output_large,
    count_lines,
    get_last_n_lines,
    sanitize_output,
    MAX_OUTPUT_CHARS,
    LARGE_OUTPUT_THRESHOLD,
)


class TestTrimBlankLines:
    """Tests for trim_blank_lines function."""

    def test_no_blank_lines(self):
        """Text without blank lines should be unchanged."""
        text = "line1\nline2\nline3"
        assert trim_blank_lines(text) == text

    def test_leading_blank_lines(self):
        """Leading blank lines should be removed."""
        text = "\n\n\nline1\nline2"
        assert trim_blank_lines(text) == "line1\nline2"

    def test_trailing_blank_lines(self):
        """Trailing blank lines should be removed."""
        text = "line1\nline2\n\n\n"
        assert trim_blank_lines(text) == "line1\nline2"

    def test_both_leading_and_trailing(self):
        """Both leading and trailing should be removed."""
        text = "\n\nline1\nline2\n\n"
        assert trim_blank_lines(text) == "line1\nline2"

    def test_preserves_internal_blanks(self):
        """Internal blank lines should be preserved."""
        text = "line1\n\nline2\n\nline3"
        assert trim_blank_lines(text) == text

    def test_empty_string(self):
        """Empty string should return empty."""
        assert trim_blank_lines("") == ""

    def test_only_blank_lines(self):
        """Only blank lines should return empty."""
        assert trim_blank_lines("\n\n\n") == ""


class TestIsImageData:
    """Tests for is_image_data function."""

    def test_empty_string(self):
        """Empty string should not be image data."""
        assert is_image_data("") is False
        assert is_image_data(None) is False

    def test_data_uri_scheme(self):
        """Data URI with image type should be detected."""
        assert is_image_data("data:image/png;base64,iVBORw0KGgo=") is True
        assert is_image_data("data:image/jpeg;base64,/9j/4AAQSkZ=") is True

    def test_short_text_not_image(self):
        """Short text should not be detected as image."""
        assert is_image_data("Hello world") is False
        assert is_image_data("abc123") is False

    def test_regular_text_not_image(self):
        """Regular long text should not be detected as image."""
        # Text with special characters that aren't base64
        text = "This is regular text! With special chars @#$%"
        assert is_image_data(text * 100) is False

    def test_limited_charset_not_image(self):
        """Text with limited charset should not be image."""
        # Only uses a few base64 chars repeatedly
        assert is_image_data("aaaa" * 5000) is False


class TestTruncateOutput:
    """Tests for truncate_output function."""

    def test_empty_text(self):
        """Empty text should return empty result."""
        result = truncate_output("")
        assert result["truncated_content"] == ""
        assert result["is_truncated"] is False
        assert result["original_length"] == 0

    def test_short_text_not_truncated(self):
        """Short text should not be truncated."""
        text = "Hello world"
        result = truncate_output(text)
        assert result["truncated_content"] == text
        assert result["is_truncated"] is False

    def test_long_text_truncated(self):
        """Long text should be truncated."""
        text = "a" * (MAX_OUTPUT_CHARS + 1000)
        result = truncate_output(text)
        assert result["is_truncated"] is True
        assert len(result["truncated_content"]) <= MAX_OUTPUT_CHARS
        assert result["original_length"] == len(text)

    def test_truncation_marker_present(self):
        """Truncation marker should be present."""
        text = "a" * (MAX_OUTPUT_CHARS + 1000)
        result = truncate_output(text)
        assert "truncated" in result["truncated_content"].lower()

    def test_custom_max_chars(self):
        """Custom max_chars should be respected."""
        text = "a" * 100
        result = truncate_output(text, max_chars=50)
        assert result["is_truncated"] is True
        assert len(result["truncated_content"]) <= 50


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_milliseconds(self):
        """Sub-second durations should show ms."""
        assert format_duration(100) == "100ms"
        assert format_duration(999) == "999ms"

    def test_seconds(self):
        """Second+ durations should show seconds."""
        assert format_duration(1000) == "1.00s"
        assert format_duration(1500) == "1.50s"
        assert format_duration(12345) == "12.35s"


class TestIsOutputLarge:
    """Tests for is_output_large function."""

    def test_small_output(self):
        """Small output should not be large."""
        assert is_output_large("hello") is False
        assert is_output_large("a" * 100) is False

    def test_large_output(self):
        """Large output should be detected."""
        assert is_output_large("a" * (LARGE_OUTPUT_THRESHOLD + 1)) is True


class TestCountLines:
    """Tests for count_lines function."""

    def test_empty_string(self):
        """Empty string should have 0 lines."""
        assert count_lines("") == 0

    def test_single_line(self):
        """Single line should return 1."""
        assert count_lines("hello") == 1

    def test_multiple_lines(self):
        """Multiple lines should be counted."""
        assert count_lines("a\nb\nc") == 3
        assert count_lines("a\nb\nc\n") == 4


class TestGetLastNLines:
    """Tests for get_last_n_lines function."""

    def test_empty_string(self):
        """Empty string should return empty."""
        assert get_last_n_lines("", 5) == ""

    def test_fewer_lines_than_n(self):
        """Fewer lines than n should return all."""
        text = "a\nb\nc"
        assert get_last_n_lines(text, 10) == text

    def test_exact_n_lines(self):
        """Exact n lines should return all."""
        text = "a\nb\nc"
        assert get_last_n_lines(text, 3) == text

    def test_more_lines_than_n(self):
        """More lines than n should return last n."""
        text = "a\nb\nc\nd\ne"
        assert get_last_n_lines(text, 2) == "d\ne"


class TestSanitizeOutput:
    """Tests for sanitize_output function."""

    def test_plain_text_unchanged(self):
        """Plain text should be unchanged."""
        text = "Hello world"
        assert sanitize_output(text) == text

    def test_removes_ansi_colors(self):
        """ANSI color codes should be removed."""
        text = "\x1b[31mRed text\x1b[0m"
        result = sanitize_output(text)
        assert "\x1b" not in result
        assert "Red text" in result

    def test_removes_control_characters(self):
        """Control characters should be removed."""
        text = "Hello\x00\x01\x02World"
        result = sanitize_output(text)
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_preserves_newlines_and_tabs(self):
        """Newlines and tabs should be preserved."""
        text = "Line1\nLine2\tTabbed"
        result = sanitize_output(text)
        assert "\n" in result
        assert "\t" in result


# ─────────────────────────────────────────────────────────────────────────────
# message_formatting.py Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.utils.message_formatting import (
    stringify_message_content,
    format_tool_use_detail,
    format_tool_result_detail,
    format_reasoning_preview,
)


class MockBlock:
    """Mock content block for testing."""
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestStringifyMessageContent:
    """Tests for stringify_message_content function."""

    def test_none_returns_empty(self):
        """None should return empty string."""
        assert stringify_message_content(None) == ""

    def test_string_passthrough(self):
        """String should pass through."""
        assert stringify_message_content("Hello") == "Hello"

    def test_text_block(self):
        """Text block should be extracted."""
        block = MockBlock("text", text="Hello world")
        result = stringify_message_content([block])
        assert result == "Hello world"

    def test_tool_use_block_without_details(self):
        """Tool use block should show placeholder."""
        block = MockBlock("tool_use", name="Bash", input={"command": "ls"})
        result = stringify_message_content([block], include_tool_details=False)
        assert "[Called Bash]" in result

    def test_tool_use_block_with_details(self):
        """Tool use block with details should show input."""
        block = MockBlock("tool_use", name="Bash", input={"command": "ls -la"})
        result = stringify_message_content([block], include_tool_details=True)
        assert "Bash" in result
        assert "ls -la" in result

    def test_tool_result_block_without_details(self):
        """Tool result block should show placeholder."""
        block = MockBlock("tool_result", text="output", is_error=False)
        result = stringify_message_content([block], include_tool_details=False)
        assert "[Tool result]" in result

    def test_tool_result_block_with_details(self):
        """Tool result block with details should show output."""
        block = MockBlock("tool_result", text="file1\nfile2", is_error=False)
        result = stringify_message_content([block], include_tool_details=True)
        assert "file1" in result

    def test_multiple_blocks(self):
        """Multiple blocks should be combined."""
        blocks = [
            MockBlock("text", text="Hello"),
            MockBlock("text", text="World"),
        ]
        result = stringify_message_content(blocks)
        assert "Hello" in result
        assert "World" in result


class TestFormatToolUseDetail:
    """Tests for format_tool_use_detail function."""

    def test_no_input(self):
        """No input should return simple format."""
        result = format_tool_use_detail("Bash", None)
        assert result == "[Called Bash]"

    def test_bash_command(self):
        """Bash command should show command."""
        result = format_tool_use_detail("Bash", {"command": "ls -la"})
        assert "Bash" in result
        assert "ls -la" in result

    def test_read_file(self):
        """Read tool should show file path."""
        result = format_tool_use_detail("Read", {"file_path": "/tmp/test.txt"})
        assert "file=/tmp/test.txt" in result

    def test_glob_pattern(self):
        """Glob tool should show pattern."""
        result = format_tool_use_detail("Glob", {"pattern": "*.py"})
        assert "pattern=*.py" in result

    def test_task_tool(self):
        """Task tool should show subagent and description."""
        result = format_tool_use_detail("Task", {
            "subagent_type": "Explore",
            "description": "Find files",
        })
        assert "subagent=Explore" in result
        assert "desc=Find files" in result

    def test_long_command_truncated(self):
        """Long command should be truncated."""
        long_cmd = "x" * 300
        result = format_tool_use_detail("Bash", {"command": long_cmd})
        assert "..." in result
        assert len(result) < 350


class TestFormatToolResultDetail:
    """Tests for format_tool_result_detail function."""

    def test_empty_result(self):
        """Empty result should return prefix only."""
        assert format_tool_result_detail("") == "[Tool result]"

    def test_error_result(self):
        """Error result should show error prefix."""
        result = format_tool_result_detail("Error occurred", is_error=True)
        assert "[Tool error]" in result
        assert "Error occurred" in result

    def test_normal_result(self):
        """Normal result should show content."""
        result = format_tool_result_detail("Success output")
        assert "[Tool result]" in result
        assert "Success output" in result

    def test_long_result_truncated(self):
        """Long result should be truncated."""
        long_result = "x" * 1000
        result = format_tool_result_detail(long_result)
        assert "truncated" in result
        assert len(result) < 600


class TestFormatReasoningPreview:
    """Tests for format_reasoning_preview function."""

    def test_none_returns_empty(self):
        """None should return empty string."""
        assert format_reasoning_preview(None) == ""

    def test_string_reasoning(self):
        """String reasoning should be returned."""
        result = format_reasoning_preview("Thinking about this...")
        assert "Thinking" in result

    def test_long_reasoning_truncated(self):
        """Long reasoning should be truncated by default."""
        long_text = "x" * 500
        result = format_reasoning_preview(long_text)
        assert "..." in result
        assert len(result) <= 254

    def test_show_full_thinking(self):
        """show_full_thinking should return full content."""
        long_text = "x" * 500
        result = format_reasoning_preview(long_text, show_full_thinking=True)
        assert result == long_text

    def test_list_reasoning(self):
        """List reasoning should be combined."""
        blocks = [
            {"thinking": "First thought"},
            {"thinking": "Second thought"},
        ]
        result = format_reasoning_preview(blocks, show_full_thinking=True)
        assert "First thought" in result
        assert "Second thought" in result
