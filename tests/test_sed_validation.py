"""Tests for sed command validation."""

from ripperdoc.tools.bash.sed_validation import (
    sed_command_is_allowed_by_allowlist,
    is_line_printing_command,
)


class TestSedValidation:
    def test_line_printing(self):
        assert sed_command_is_allowed_by_allowlist("sed -n '1p' file.txt")

    def test_line_printing_range(self):
        assert sed_command_is_allowed_by_allowlist("sed -n '1,10p' file.txt")

    def test_substitution(self):
        assert sed_command_is_allowed_by_allowlist("sed 's/foo/bar/'")

    def test_substitution_with_flags(self):
        assert sed_command_is_allowed_by_allowlist("sed 's/foo/bar/g'")

    def test_block_write_command(self):
        assert not sed_command_is_allowed_by_allowlist("sed 'w output.txt'")

    def test_block_execute_command(self):
        assert not sed_command_is_allowed_by_allowlist("sed 'e echo hi'")


class TestLinePrintingCommand:
    def test_valid(self):
        assert is_line_printing_command("sed -n '1p'", ["1p"])

    def test_missing_n(self):
        assert not is_line_printing_command("sed '1p'", ["1p"])
