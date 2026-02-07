"""Tests for permission rule syntax parsing and matching."""

from pathlib import Path

from ripperdoc.utils.permissions.rule_syntax import (
    match_parsed_permission_rule,
    normalize_permission_rule,
    parse_permission_rule,
)


def test_parse_tool_and_tool_specifier_forms():
    parsed_tool = parse_permission_rule("Bash")
    assert parsed_tool is not None
    assert parsed_tool.tool_name == "Bash"
    assert parsed_tool.specifier is None
    assert parsed_tool.canonical_rule == "Bash"

    parsed_with_spec = parse_permission_rule("Read(./.env)")
    assert parsed_with_spec is not None
    assert parsed_with_spec.tool_name == "Read"
    assert parsed_with_spec.specifier == "./.env"
    assert parsed_with_spec.canonical_rule == "Read(./.env)"


def test_bash_star_is_equivalent_to_tool_only():
    parsed = parse_permission_rule("Bash(*)")
    assert parsed is not None
    assert parsed.tool_name == "Bash"
    assert parsed.specifier is None
    assert parsed.canonical_rule == "Bash"


def test_legacy_bash_suffix_is_normalized():
    parsed = parse_permission_rule("Bash(ls:*)")
    assert parsed is not None
    assert parsed.tool_name == "Bash"
    assert parsed.specifier == "ls *"
    assert parsed.canonical_rule == "Bash(ls *)"
    assert parsed.used_legacy_bash_suffix is True

    assert normalize_permission_rule("ls:*") == "Bash(ls *)"


def test_path_specifier_matches_relative_rule(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n")

    parsed = parse_permission_rule("Read(./.env)")
    assert parsed is not None
    assert match_parsed_permission_rule(
        parsed,
        tool_name="Read",
        parsed_input={"file_path": str(env_file)},
        cwd=tmp_path,
    )


def test_domain_specifier_matches_url():
    parsed = parse_permission_rule("WebFetch(domain:example.com)")
    assert parsed is not None
    assert match_parsed_permission_rule(
        parsed,
        tool_name="WebFetch",
        parsed_input={"url": "https://example.com/api/v1"},
    )
    assert not match_parsed_permission_rule(
        parsed,
        tool_name="WebFetch",
        parsed_input={"url": "https://other.com/api/v1"},
    )

