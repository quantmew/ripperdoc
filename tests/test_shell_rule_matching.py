"""Tests for shell rule matching."""

from ripperdoc.utils.permissions.shell_rule_matching import (
    parse_permission_rule,
    match_wildcard_pattern,
)


class TestParsePermissionRule:
    def test_exact(self):
        rule = parse_permission_rule("git status")
        assert rule.type == "exact"
        assert rule.command == "git status"

    def test_prefix(self):
        rule = parse_permission_rule("npm:*")
        assert rule.type == "prefix"
        assert rule.prefix == "npm"

    def test_wildcard(self):
        rule = parse_permission_rule("git *")
        assert rule.type == "wildcard"
        assert rule.pattern == "git *"


class TestMatchWildcardPattern:
    def test_glob_match(self):
        assert match_wildcard_pattern("git *", "git status")
        assert match_wildcard_pattern("npm *", "npm install")

    def test_no_match(self):
        assert not match_wildcard_pattern("git *", "ls -la")
