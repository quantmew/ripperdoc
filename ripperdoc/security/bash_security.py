"""Comprehensive bash command security validation.


Provides 24 security check validators that detect potentially dangerous
shell constructs before execution. Uses a FAIL-CLOSED approach: if any
validator returns 'ask', the command requires user approval.

Each validator returns a PermissionResult with one of:
- 'passthrough': Command passed this check (safe or not applicable)
- 'ask': Command requires user approval
- 'allow': Command is explicitly allowed (rare, for safe patterns)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# PermissionResult type (shared across security/permissions modules)
# ---------------------------------------------------------------------------

@dataclass
class PermissionResult:
    """Result of a permission or security check.

    Attributes:
        behavior: 'allow' | 'deny' | 'ask' | 'passthrough'
        message: Human-readable message explaining the result.
        updated_input: Optional updated input (for allow results with transformations).
        decision_reason: Structured reason for the decision.
        suggestions: List of rule suggestions for the user.
    """
    behavior: str  # 'allow' | 'deny' | 'ask' | 'passthrough'
    message: str = ""
    updated_input: Optional[Any] = None
    decision_reason: Optional[Dict] = None
    suggestions: Optional[List[Any]] = None

    @staticmethod
    def passthrough(msg: str = "") -> "PermissionResult":
        return PermissionResult(behavior="passthrough", message=msg)

    @staticmethod
    def ask(msg: str, reason: Optional[dict] = None) -> "PermissionResult":
        return PermissionResult(behavior="ask", message=msg, decision_reason=reason)

    @staticmethod
    def deny(msg: str, reason: Optional[dict] = None) -> "PermissionResult":
        return PermissionResult(behavior="deny", message=msg, decision_reason=reason)

    @staticmethod
    def allow(updated_input: Optional[Any] = None, reason: Optional[dict] = None) -> "PermissionResult":
        return PermissionResult(
            behavior="allow",
            message="",
            updated_input=updated_input,
            decision_reason=reason,
        )


# ---------------------------------------------------------------------------
# Security check IDs
# ---------------------------------------------------------------------------

BASH_SECURITY_CHECK_IDS = {
    "INCOMPLETE_COMMANDS": 1,
    "JQ_SYSTEM_FUNCTION": 2,
    "JQ_FILE_ARGUMENTS": 3,
    "OBFUSCATED_FLAGS": 4,
    "SHELL_METACHARACTERS": 5,
    "DANGEROUS_VARIABLES": 6,
    "NEWLINES": 7,
    "DANGEROUS_PATTERNS_COMMAND_SUBSTITUTION": 8,
    "DANGEROUS_PATTERNS_INPUT_REDIRECTION": 9,
    "DANGEROUS_PATTERNS_OUTPUT_REDIRECTION": 10,
    "IFS_INJECTION": 11,
    "GIT_COMMIT_SUBSTITUTION": 12,
    "PROC_ENVIRON_ACCESS": 13,
    "MALFORMED_TOKEN_INJECTION": 14,
    "BACKSLASH_ESCAPED_WHITESPACE": 15,
    "BRACE_EXPANSION": 16,
    "CONTROL_CHARACTERS": 17,
    "UNICODE_WHITESPACE": 18,
    "MID_WORD_HASH": 19,
    "ZSH_DANGEROUS_COMMANDS": 20,
    "BACKSLASH_ESCAPED_OPERATORS": 21,
    "COMMENT_QUOTE_DESYNC": 22,
    "QUOTED_NEWLINE": 23,
    "CARRIAGE_RETURN": 24,
}

# ---------------------------------------------------------------------------
# Command substitution patterns
# ---------------------------------------------------------------------------

COMMAND_SUBSTITUTION_PATTERNS = [
    (re.compile(r"<\(/"), "Process substitution <()"),
    (re.compile(r">\(/"), "Process substitution >()"),
    (re.compile(r"=\(/"), "Zsh process substitution =()"),
    (re.compile(r"(?:^|[\s;&|])=[a-zA-Z_]"), "Zsh equals expansion (=cmd)"),
    (re.compile(r"\$\("), "$() command substitution"),
    (re.compile(r"\$\{"), "${} parameter substitution"),
    (re.compile(r"\$\["), "$[] legacy arithmetic expansion"),
    (re.compile(r"~\["), "Zsh-style parameter expansion"),
    (re.compile(r"\(e:/"), "Zsh-style glob qualifiers"),
    (re.compile(r"\(\+/"), "Zsh glob qualifier with command execution"),
    (re.compile(r"\}\s*always\s*\{"), "Zsh always block (try/always construct)"),
    (re.compile(r"<#"), "PowerShell comment syntax (defense in depth)"),
]

# HEREDOC_IN_SUBSTITUTION pattern
HEREDOC_IN_SUBSTITUTION = re.compile(r"\$\(.*<<")

# ---------------------------------------------------------------------------
# Zsh dangerous commands
# ---------------------------------------------------------------------------

ZSH_DANGEROUS_COMMANDS = frozenset({
    "zmodload",
    "emulate",
    "sysopen",
    "sysread",
    "syswrite",
    "sysseek",
    "zpty",
    "ztcp",
    "zsocket",
    "mapfile",
    "zf_rm",
    "zf_mv",
    "zf_ln",
    "zf_chmod",
    "zf_chown",
    "zf_mkdir",
    "zf_rmdir",
    "zf_chgrp",
})

# ---------------------------------------------------------------------------
# Validation context
# ---------------------------------------------------------------------------


@dataclass
class ValidationContext:
    """Context data for security validators."""
    original_command: str
    base_command: str
    unquoted_content: str  # With double quotes preserved
    fully_unquoted_content: str  # No quotes of any kind
    fully_unquoted_pre_strip: str  # Before stripSafeRedirections
    unquoted_keep_quote_chars: str  # Quote chars preserved but content stripped
    tree_sitter: Optional[Any] = None


# ---------------------------------------------------------------------------
# Quote extraction
# ---------------------------------------------------------------------------


def extract_quoted_content(command: str, is_jq: bool = False) -> Dict[str, str]:
    """Extract content from the command, tracking quote states.

    Returns three versions:
    - with_double_quotes: Single-quote content stripped, double-quote preserved
    - fully_unquoted: All quoted content stripped
    - unquoted_keep_quote_chars: Like fully_unquoted but quote delimiters remain

    Args:
        command: The command string.
        is_jq: If True, include quotes in extraction for jq analysis.

    Returns:
        Dict with 'with_double_quotes', 'fully_unquoted', 'unquoted_keep_quote_chars'.
    """
    with_double_quotes = []
    fully_unquoted = []
    unquoted_keep_quote_chars = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for char in command:
        if escaped:
            escaped = False
            if not in_single_quote:
                with_double_quotes.append(char)
            if not in_single_quote and not in_double_quote:
                fully_unquoted.append(char)
                unquoted_keep_quote_chars.append(char)
            continue

        if char == "\\" and not in_single_quote:
            escaped = True
            if not in_single_quote:
                with_double_quotes.append(char)
            if not in_single_quote and not in_double_quote:
                fully_unquoted.append(char)
                unquoted_keep_quote_chars.append(char)
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            unquoted_keep_quote_chars.append(char)
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            unquoted_keep_quote_chars.append(char)
            if not is_jq:
                continue

        if not in_single_quote:
            with_double_quotes.append(char)
        if not in_single_quote and not in_double_quote:
            fully_unquoted.append(char)
            unquoted_keep_quote_chars.append(char)

    return {
        "with_double_quotes": "".join(with_double_quotes),
        "fully_unquoted": "".join(fully_unquoted),
        "unquoted_keep_quote_chars": "".join(unquoted_keep_quote_chars),
    }


# ---------------------------------------------------------------------------
# Safe redirection stripping
# ---------------------------------------------------------------------------


def strip_safe_redirections(content: str) -> str:
    """Strip safe redirections (2>&1, >/dev/null, </dev/null) from content.

    All three patterns MUST have a trailing boundary (?=\\s|$) to prevent
    partial matches like `>/dev/nullo` matching as `>/dev/null`.

    Args:
        content: The string to process.

    Returns:
        Content with safe redirections removed.
    """
    result = content
    result = re.sub(r"\s+2\s*>&\s*1(?=\s|$)", "", result)
    result = re.sub(r"[012]?\s*>\s*/dev/null(?=\s|$)", "", result)
    result = re.sub(r"\s*<\s*/dev/null(?=\s|$)", "", result)
    return result


# ---------------------------------------------------------------------------
# Has unescaped char
# ---------------------------------------------------------------------------


def has_unescaped_char(content: str, char: str) -> bool:
    """Check if content contains an unescaped occurrence of a single character.

    Handles bash escape sequences where a backslash escapes the following character.

    Args:
        content: The string to search.
        char: Single character to search for.

    Returns:
        True if an unescaped occurrence is found.
    """
    if len(char) != 1:
        raise ValueError("has_unescaped_char only works with single characters")

    i = 0
    while i < len(content):
        if content[i] == "\\" and i + 1 < len(content):
            i += 2
            continue
        if content[i] == char:
            return True
        i += 1
    return False


# ============================================================================
# VALIDATORS
# ============================================================================


def validate_empty(context: ValidationContext) -> PermissionResult:
    """Check if command is empty — safe."""
    if not context.original_command.strip():
        return PermissionResult.allow(
            updated_input={"command": context.original_command},
            reason={"type": "other", "reason": "Empty command is safe"},
        )
    return PermissionResult.passthrough("Command is not empty")


def validate_incomplete_commands(context: ValidationContext) -> PermissionResult:
    """Check for incomplete command fragments."""
    original = context.original_command
    trimmed = original.strip()

    if re.match(r"^\s*\t", original):
        return PermissionResult.ask(
            "Command appears to be an incomplete fragment (starts with tab)"
        )

    if trimmed.startswith("-"):
        return PermissionResult.ask(
            "Command appears to be an incomplete fragment (starts with flags)"
        )

    if re.match(r"^\s*(&&|\|\||;|>>?|<)", original):
        return PermissionResult.ask(
            "Command appears to be a continuation line (starts with operator)"
        )

    return PermissionResult.passthrough("Command appears complete")


def validate_safe_heredoc(command: str) -> PermissionResult:
    """Early-allow path for safe $(cat <<'DELIM'...DELIM) patterns.

    This is an EARLY-ALLOW validator: if this returns allow, it bypasses
    ALL subsequent validators. The check must be PROVABLY safe.

    Args:
        command: The original command string.

    Returns:
        PermissionResult.allow if the command is a safe heredoc pattern,
        PermissionResult.passthrough otherwise.
    """
    if not HEREDOC_IN_SUBSTITUTION.search(command):
        return PermissionResult.passthrough("No heredoc in substitution")

    if _is_safe_heredoc(command):
        return PermissionResult.allow(
            updated_input={"command": command},
            reason={"type": "other", "reason": "Safe heredoc substitution"},
        )

    return PermissionResult.passthrough("Not a safe heredoc pattern")


def _is_safe_heredoc(command: str) -> bool:
    """Check if command is a safe heredoc-in-substitution pattern.

    The only pattern we allow is:
      [prefix] $(cat <<'DELIM'\\n[body]\\nDELIM\\n) [suffix]

    Where:
    - Delimiter is single-quoted or backslash-escaped (body is literal)
    - Closing delimiter is on a line by itself
    - There is non-whitespace text BEFORE the $(
    - Remaining text (with heredoc stripped) passes all validators
    """
    # Find all $(cat <<'DELIM' or $(cat <<\\DELIM patterns
    pattern = re.compile(
        r'\$\(cat[ \t]*<<(-?)[ \t]*(?:\'+([A-Za-z_]\w*)\'+|\\([A-Za-z_]\w*))'
    )

    matches: list[dict[str, int | str | bool]] = []
    for match in pattern.finditer(command):
        delim = match.group(2) or match.group(3)
        if not delim:
            continue
        matches.append({
            "start": match.start(),
            "end": match.end(),
            "delimiter": delim,
            "is_dash": match.group(1) == "-",
        })

    if not matches:
        return False

    # Verify each heredoc match using line-based matching
    verified = []
    for m in matches:
        start_byte = m["start"]
        start_byte_i: int = start_byte if isinstance(start_byte, int) else 0
        operator_end = m["end"]
        operator_end_i: int = operator_end if isinstance(operator_end, int) else 0
        delimiter: str = str(m["delimiter"])
        is_dash: bool = bool(m["is_dash"])

        # Check the opening line ends immediately after the delimiter
        after_operator = command[operator_end_i:]
        open_line_end = after_operator.find("\n")
        if open_line_end == -1:
            return False

        open_line_tail = after_operator[:open_line_end]
        if not re.match(r"^[ \t]*$", open_line_tail):
            return False

        # Body starts after the newline
        body_start = operator_end_i + open_line_end + 1
        body = command[body_start:]
        body_lines = body.split("\n")

        # Find the FIRST line that closes the heredoc
        closing_line_idx = -1
        close_paren_line_idx = -1

        for i, line in enumerate(body_lines):
            check_line = line
            if is_dash:
                check_line = line.lstrip("\t")

            if check_line == delimiter:
                closing_line_idx = i
                # Check next line for )
                if i + 1 < len(body_lines):
                    next_line = body_lines[i + 1]
                    paren_match = re.match(r"^([ \t]*)\)", next_line)
                    if paren_match:
                        close_paren_line_idx = i + 1
                        break
                break

            # Form 2: delimiter followed by ) on same line
            if check_line.startswith(delimiter):
                after_delim = check_line[len(delimiter):]
                paren_match = re.match(r"^([ \t]*)\)", after_delim)
                if paren_match:
                    closing_line_idx = i
                    close_paren_line_idx = i
                    break

        if closing_line_idx == -1:
            return False

        # Calculate end position
        end_pos = body_start
        for i in range(close_paren_line_idx):
            end_pos += len(body_lines[i]) + 1
        # Find the )
        rest_line = body_lines[close_paren_line_idx] if close_paren_line_idx < len(body_lines) else ""
        paren_match_end = re.search(r"\)", rest_line)
        if paren_match_end:
            end_pos += paren_match_end.start() + 1
        else:
            return False

        verified.append({"start": start_byte_i, "end": end_pos})

    # Check for nested matches (reject)
    for outer in verified:
        for inner in verified:
            if inner is outer:
                continue
            if inner["start"] > outer["start"] and inner["start"] < outer["end"]:
                return False

    # Strip all verified heredocs
    sorted_verified = sorted(verified, key=lambda x: x["start"], reverse=True)
    remaining = command
    for v in sorted_verified:
        remaining = remaining[:v["start"]] + remaining[v["end"]:]

    # The remaining text must not start with only whitespace before $(
    trimmed_remaining = remaining.strip()
    if trimmed_remaining:
        first_heredoc_start = min(v["start"] for v in verified)
        prefix = command[:first_heredoc_start]
        if not prefix.strip():
            return False

    # Remaining text must contain only safe characters
    if not re.match(r"^[a-zA-Z0-9 \t\"'.\-/_@=,:+~]*$", remaining):
        return False

    # Check remaining text passes security validators (prevent recursion)
    if bash_command_is_safe(remaining).behavior != "passthrough":
        return False

    return True


def validate_obfuscated_flags(context: ValidationContext) -> PermissionResult:
    """Check for obfuscated flags that bypass permission checks.

    Detects flags that look different to our parser than to the actual command.
    """
    from ripperdoc.utils.bash.shell_quote import try_parse_shell_command
    from ripperdoc.utils.bash.commands import split_command

    command = context.original_command

    # Check for isolated equals signs in positions that suggest obfuscation
    # Example: --exec=... vs --exec ...
    if re.search(r"--\w+='?\$?\(", command):
        return PermissionResult.ask(
            "Command contains obfuscated flag with command substitution"
        )

    # Check for flags with embedded newlines (using $'\\n' ANSI-C quoting)
    if "$'" in command and re.search(r"\\[nrt]", command):
        return PermissionResult.ask(
            "Command contains ANSI-C quoted escape sequences that may obfuscate arguments"
        )

    return PermissionResult.passthrough("No obfuscated flags detected")


def validate_shell_metachars(context: ValidationContext) -> PermissionResult:
    """Check for shell metacharacters (;, |, &) outside of quoted context."""
    unquoted = context.fully_unquoted_content

    # These are the "operators" we already know about and handle elsewhere.
    # But here we also catch non-command-chain uses like inline ; or bare &.
    # We only flag patterns that are clearly dangerous, not basic command chaining.

    # Flag standalone semicolons (not part of ;; which is case terminator)
    if re.search(r"(?<![;]);(?![;])", unquoted):
        # But only if not inside a case statement structure
        if "case" not in context.original_command:
            return PermissionResult.ask(
                "Command uses shell metacharacters (;) outside of quoted context"
            )

    return PermissionResult.passthrough("No dangerous shell metacharacters")


def validate_dangerous_variables(context: ValidationContext) -> PermissionResult:
    """Check for dangerous shell variables that affect execution behavior."""
    unquoted = context.fully_unquoted_content

    # Dangerous variables that modify shell behavior
    dangerous_vars = {
        "IFS": "IFS (Internal Field Separator) manipulation detected",
        "SHELLOPTS": "SHELLOPTS manipulation detected (changes shell behavior)",
        "BASH_ENV": "BASH_ENV detected (can execute arbitrary code at shell startup)",
        "BASH_FUNC_": "BASH_FUNC_ prefix detected (can define shell functions via env vars)",
        "ENV": "ENV detected (sh alternative to BASH_ENV)",
        "LD_PRELOAD": "LD_PRELOAD detected (library injection)",
        "LD_LIBRARY_PATH": "LD_LIBRARY_PATH detected (library path injection)",
        "PYTHONPATH": "PYTHONPATH detected (Python module injection)",
        "NODE_PATH": "NODE_PATH detected (Node module injection)",
        "PERL5LIB": "PERL5LIB detected (Perl module injection)",
        "BASH_SOURCE": "BASH_SOURCE manipulation attempt",
        "PIP_REQUIRE_VIRTUALENV": "PIP_REQUIRE_VIRTUALENV manipulation",
    }

    for var_name, message in dangerous_vars.items():
        pattern = re.compile(
            rf"(?:^|[\s;&|])(?:export\s+)?{re.escape(var_name)}\s*=",
            re.IGNORECASE,
        )
        if pattern.search(unquoted):
            return PermissionResult.ask(
                f"Command contains dangerous variable assignment: {message}"
            )

    return PermissionResult.passthrough("No dangerous variables detected")


def validate_newlines(context: ValidationContext) -> PermissionResult:
    """Check for dangerous newline usage."""
    command = context.original_command

    # Check for newlines outside of quotes
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for char in command:
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if char in ("\n", "\r") and not in_single_quote and not in_double_quote:
            return PermissionResult.ask(
                "Command contains unquoted newlines (potential injection)"
            )

    return PermissionResult.passthrough("No dangerous newlines detected")


def validate_carriage_return(context: ValidationContext) -> PermissionResult:
    """Detect carriage return (\r) injection attacks.

    CR outside double quotes causes shell-quote/bash tokenization differential.
    """
    if "\r" not in context.original_command:
        return PermissionResult.passthrough("No carriage return")

    in_single_quote = False
    in_double_quote = False
    escaped = False
    for c in context.original_command:
        if escaped:
            escaped = False
            continue
        if c == "\\" and not in_single_quote:
            escaped = True
            continue
        if c == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if c == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if c == "\r" and not in_double_quote:
            return PermissionResult.ask(
                "Command contains carriage return (\\r) which shell-quote and bash tokenize differently",
                reason={"type": "safetyCheck", "check_id": "CARRIAGE_RETURN"},
            )

    return PermissionResult.passthrough("CR only inside double quotes")


_jq_dangerous_flag_pattern = re.compile(
    r"(?:^|\s)(?:-f\b|--from-file|--rawfile|--slurpfile|-L\b|--library-path)"
)


def validate_jq_command(context: ValidationContext) -> PermissionResult:
    """Detect dangerous jq command patterns.

    Blocks: system() function, dangerous file-reading flags.
    """
    if context.base_command != "jq":
        return PermissionResult.passthrough("Not jq")

    if "system" in context.original_command and "system(" in context.original_command:
        return PermissionResult.ask(
            "jq command contains system() function which executes arbitrary commands",
            reason={"type": "safetyCheck", "check_id": "JQ_SYSTEM_FUNCTION"},
        )

    after_jq = context.original_command[len("jq"):].strip()
    if _jq_dangerous_flag_pattern.search(after_jq):
        return PermissionResult.ask(
            "jq command contains dangerous flags that could execute code or read arbitrary files",
            reason={"type": "safetyCheck", "check_id": "JQ_FILE_ARGUMENTS"},
        )

    return PermissionResult.passthrough("jq command is safe")


def validate_command_substitution(context: ValidationContext) -> PermissionResult:
    """Check for command substitution patterns."""
    unquoted = context.fully_unquoted_content

    for pattern, message in COMMAND_SUBSTITUTION_PATTERNS:
        if pattern.search(unquoted):
            # Skip if this is a safe heredoc pattern
            if pattern.pattern.startswith("\\$\\(") and "<<" in context.original_command:
                if _is_safe_heredoc(context.original_command):
                    continue
            return PermissionResult.ask(
                f"Command contains potentially dangerous pattern: {message}"
            )

    return PermissionResult.passthrough("No command substitution detected")


def validate_input_redirection(context: ValidationContext) -> PermissionResult:
    """Check for dangerous input redirection patterns."""
    unquoted = context.fully_unquoted_content

    # Check for destructive input redirection (reading from /dev/zero, /dev/random)
    # that could be used for denial of service
    if re.search(r"<\s*/dev/(zero|random|urandom)\b", unquoted):
        return PermissionResult.ask(
            "Command reads from a device file that could cause denial of service"
        )

    # Check for reading from sensitive files via input redirection
    sensitive_files = [
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/passwd",
        "/proc/self/environ",
        "/proc/self/fd",
    ]
    for sensitive in sensitive_files:
        if re.search(rf"<\s*{re.escape(sensitive)}\b", unquoted):
            return PermissionResult.ask(
                f"Command reads from sensitive file: {sensitive}"
            )

    return PermissionResult.passthrough("No dangerous input redirection")


def validate_output_redirection(context: ValidationContext) -> PermissionResult:
    """Check for dangerous output redirection patterns."""
    unquoted = context.fully_unquoted_content

    # Check for output redirection to sensitive system files
    sensitive_paths = [
        "/etc/",
        "/boot/",
        "/usr/",
        "/bin/",
        "/sbin/",
        "/lib/",
        "/dev/",
        "/proc/",
        "/sys/",
    ]
    # Match patterns like >/etc/shadow, >>/etc/hosts, 2>/etc/crontab
    for path in sensitive_paths:
        escaped_path = re.escape(path)
        if re.search(rf"(?:^|[\s])[012]?\s*(?:>|>>)\s*{escaped_path}", unquoted):
            return PermissionResult.ask(
                f"Command writes to a sensitive system path: {path}"
            )

    return PermissionResult.passthrough("No dangerous output redirection")


def validate_ifs_injection(context: ValidationContext) -> PermissionResult:
    """Check for IFS injection attacks."""
    unquoted = context.fully_unquoted_content

    # IFS attack pattern: IFS=... command where the command contains characters
    # that would be split differently under a non-standard IFS
    if re.search(r"(?:^|[\s;&|])IFS\s*=\s*", unquoted):
        return PermissionResult.ask(
            "Command contains IFS assignment (potential word-splitting injection)"
        )

    # Check for IFS manipulation via read-only variable tricks
    if re.search(r"(?:^|[\s;&|])readonly\s+IFS", unquoted):
        return PermissionResult.ask(
            "Command attempts to manipulate IFS as a readonly variable"
        )

    return PermissionResult.passthrough("No IFS injection detected")


def validate_git_commit_substitution(context: ValidationContext) -> PermissionResult:
    """Check for git commit metadata substitution attacks."""
    command = context.original_command
    fully_unquoted = context.fully_unquoted_content

    # Check for GIT_AUTHOR_DATE, GIT_COMMITTER_DATE manipulation
    # that could be used to bypass commit signing or verification
    if re.search(r"GIT_AUTHOR_DATE\s*=", fully_unquoted, re.IGNORECASE):
        return PermissionResult.ask(
            "Command sets GIT_AUTHOR_DATE (potential commit metadata manipulation)"
        )

    if re.search(r"GIT_COMMITTER_DATE\s*=", fully_unquoted, re.IGNORECASE):
        return PermissionResult.ask(
            "Command sets GIT_COMMITTER_DATE (potential commit metadata manipulation)"
        )

    # Check for git -c with dangerous config options
    if re.search(r"git\s+.*?\s+-c\s+", command):
        dangerous_git_configs = [
            "core.fsmonitor",
            "core.hookspath",
            "core.ssh",
            "core.alternateObjectDirectories",
            "diff.external",
            "diff.textconv",
            "filter.*.clean",
            "filter.*.smudge",
            "protocol.ext.helper",
        ]
        for config in dangerous_git_configs:
            pattern = re.compile(
                rf"-c\s+{re.escape(config)}\s*=", re.IGNORECASE
            )
            if pattern.search(command):
                return PermissionResult.ask(
                    f"Command uses git -c with dangerous config option: {config}"
                )

    return PermissionResult.passthrough("No git commit substitution detected")


def validate_proc_environ_access(context: ValidationContext) -> PermissionResult:
    """Check for /proc/self/environ access (environment variable exfiltration)."""
    unquoted = context.fully_unquoted_content

    if re.search(r"/proc/self/environ", unquoted):
        return PermissionResult.ask(
            "Command accesses /proc/self/environ (potential environment variable exfiltration)"
        )

    # Also check /proc/*/environ patterns
    if re.search(r"/proc/\d+/environ", unquoted):
        return PermissionResult.ask(
            "Command accesses process environment (potential environment variable exfiltration)"
        )

    return PermissionResult.passthrough("No /proc/environ access detected")


def validate_malformed_token_injection(context: ValidationContext) -> PermissionResult:
    """Check for malformed token injection via shell-quote parser differentials."""
    from ripperdoc.utils.bash.shell_quote import (
        try_parse_shell_command,
        has_malformed_tokens,
        has_shell_quote_single_quote_bug,
    )

    command = context.original_command

    if has_shell_quote_single_quote_bug(command):
        return PermissionResult.ask(
            "Command contains backslash inside single quotes (potential shell-quote parser differential)"
        )

    if has_malformed_tokens(command):
        return PermissionResult.ask(
            "Command contains malformed tokens (potential parser differential)"
        )

    # Check for unbalanced quotes (bash will fail, but could indicate an attack)
    single_quotes = command.count("'")
    double_quotes = command.count('"')
    if single_quotes % 2 != 0:
        return PermissionResult.ask(
            "Command has unbalanced single quotes (potential injection)"
        )
    if double_quotes % 2 != 0:
        return PermissionResult.ask(
            "Command has unbalanced double quotes (potential injection)"
        )

    return PermissionResult.passthrough("No malformed token injection detected")


def validate_backslash_escaped_whitespace(context: ValidationContext) -> PermissionResult:
    """Check for backslash-escaped whitespace that could obfuscate commands."""
    command = context.original_command

    # Look for patterns like: \<newline> or \<tab> or \<space>
    # In bash, backslash + newline is a line continuation (safe).
    # But backslash + other whitespace isn't meaningful and could be obfuscation.
    if re.search(r"\\(?:\t| )", command):
        return PermissionResult.ask(
            "Command contains backslash-escaped whitespace (potential obfuscation)"
        )

    return PermissionResult.passthrough("No backslash-escaped whitespace")


def validate_brace_expansion(context: ValidationContext) -> PermissionResult:
    """Check for brace expansion that could bypass security checks."""
    fully_unquoted = context.fully_unquoted_pre_strip

    # Brace expansion {a,b} creates multiple arguments from one pattern.
    # This can be used to bypass path-based or flag-based security checks.
    # Example: rm -rf /tmp/{evil,innocent} — the /tmp/evil path is hidden
    # from path validation but expanded by bash.

    # Look for { and , or { and .. in the unquoted content
    # Must have both opening brace and a separator (, or ..)
    i = 0
    while i < len(fully_unquoted):
        if fully_unquoted[i] == "{":
            # Check for comma or .. in the braces
            brace_depth = 1
            j = i + 1
            has_comma = False
            has_dots = False
            while j < len(fully_unquoted) and brace_depth > 0:
                if fully_unquoted[j] == "{":
                    brace_depth += 1
                elif fully_unquoted[j] == "}":
                    brace_depth -= 1
                elif fully_unquoted[j] == ",":
                    has_comma = True
                elif fully_unquoted[j:j+2] == "..":
                    has_dots = True
                    j += 1
                j += 1

            if brace_depth == 0 and (has_comma or has_dots):
                return PermissionResult.ask(
                    "Command contains brace expansion that could bypass security checks"
                )
        i += 1

    return PermissionResult.passthrough("No brace expansion detected")


def validate_control_characters(context: ValidationContext) -> PermissionResult:
    """Check for raw control characters that could be used for injection."""
    command = context.original_command

    # Check for control characters (except \n, \t, \r which are normal)
    for char in command:
        code = ord(char)
        if code < 0x20 and code not in (0x09, 0x0A, 0x0D):  # Not tab, newline, CR
            return PermissionResult.ask(
                f"Command contains control character 0x{code:02x} (potential injection)"
            )
        if code == 0x7F:  # DEL
            return PermissionResult.ask(
                "Command contains DEL character (0x7F) (potential injection)"
            )

    return PermissionResult.passthrough("No control characters detected")


def validate_unicode_whitespace(context: ValidationContext) -> PermissionResult:
    """Check for Unicode whitespace characters used for obfuscation."""
    command = context.original_command

    # Non-ASCII whitespace that bash treats as normal text
    # but could look like ASCII whitespace to a human
    unicode_whitespace = re.compile(
        "[\u00A0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
        "\u2008\u2009\u200A\u202F\u205F\u3000\uFEFF]"
    )

    if unicode_whitespace.search(command):
        return PermissionResult.ask(
            "Command contains Unicode whitespace characters (potential obfuscation)"
        )

    return PermissionResult.passthrough("No Unicode whitespace detected")


def validate_mid_word_hash(context: ValidationContext) -> PermissionResult:
    """Check for mid-word # that could indicate comment-based injection."""
    unquoted = context.unquoted_keep_quote_chars

    # Look for # that appears mid-word (after alphanumeric chars without space)
    # In bash, # starts a comment. If it appears mid-word due to quote removal,
    # it could be used for injection.
    if re.search(r"[a-zA-Z0-9_]#", unquoted):
        return PermissionResult.ask(
            "Command contains mid-word hash character (potential comment injection)"
        )

    return PermissionResult.passthrough("No mid-word hash detected")


def validate_zsh_dangerous_commands(context: ValidationContext) -> PermissionResult:
    """Check for dangerous zsh-specific commands."""
    command = context.original_command
    tokens = command.strip().split()
    if not tokens:
        return PermissionResult.passthrough("No zsh dangerous commands")

    base = tokens[0].lower()
    if base in ZSH_DANGEROUS_COMMANDS:
        return PermissionResult.ask(
            f"Command uses dangerous zsh builtin: {base}"
        )

    # Also check for zsh-specific syntax after pipes and operators
    for token in tokens:
        cleaned = token.lower()
        if cleaned in ZSH_DANGEROUS_COMMANDS:
            return PermissionResult.ask(
                f"Command contains dangerous zsh builtin: {cleaned}"
            )

    return PermissionResult.passthrough("No zsh dangerous commands detected")


def validate_backslash_escaped_operators(context: ValidationContext) -> PermissionResult:
    """Check for backslash-escaped shell operators that could bypass checks."""
    unquoted = context.fully_unquoted_content

    # In bash, \ before an operator makes it literal.
    # But some parsers don't handle this, creating a differential.
    # Example: \; or \| or \&
    if re.search(r"\\(?:;|\||&|&&|\|\||\$|`|#|!)", unquoted):
        return PermissionResult.ask(
            "Command contains backslash-escaped shell operators (potential parser differential)"
        )

    return PermissionResult.passthrough("No backslash-escaped operators detected")


def validate_comment_quote_desync(context: ValidationContext) -> PermissionResult:
    """Check for comment-based quote desynchronization."""
    command = context.original_command

    # Pattern: # inside a command that starts a comment, but the quotes
    # become desynchronized. Example: echo 'x'# (the # is outside quotes)
    # This creates a parser differential between our parser and bash.

    # Look for # outside of quotes
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for i, char in enumerate(command):
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if char == "#" and not in_single_quote and not in_double_quote and i > 0:
            # # at start of a token (after whitespace) is a normal comment
            # But # mid-token after quotes could be desync
            if command[i - 1] not in (" ", "\t", "\n", ";", "|", "&", "("):
                return PermissionResult.ask(
                    "Command contains hash character that may cause quote desynchronization"
                )

    return PermissionResult.passthrough("No comment quote desync detected")


def validate_quoted_newline(context: ValidationContext) -> PermissionResult:
    """Check for newlines inside double quotes that could enable injection."""
    command = context.original_command

    in_double_quote = False
    escaped = False

    for char in command:
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_double_quote = not in_double_quote
            continue

        if char in ("\n", "\r") and in_double_quote:
            return PermissionResult.ask(
                "Command contains newline inside double quotes (potential injection)"
            )

    return PermissionResult.passthrough("No quoted newlines detected")


# ============================================================================
# Main entry points
# ============================================================================


def _build_context(command: str) -> ValidationContext:
    """Build a ValidationContext from a command string."""
    quotes = extract_quoted_content(command)
    fully_unquoted_pre_strip = quotes["fully_unquoted"]

    # Apply safe redirection stripping
    stripped = strip_safe_redirections(quotes["fully_unquoted"])

    tokens = command.strip().split()
    base_cmd = tokens[0] if tokens else ""

    return ValidationContext(
        original_command=command,
        base_command=base_cmd,
        unquoted_content=strip_safe_redirections(quotes["with_double_quotes"]),
        fully_unquoted_content=stripped,
        fully_unquoted_pre_strip=fully_unquoted_pre_strip,
        unquoted_keep_quote_chars=quotes["unquoted_keep_quote_chars"],
    )


def bash_command_is_safe(command: str) -> PermissionResult:
    """Synchronous security check for a bash command.

    Runs all validators in order. Returns the first non-passthrough result,
    or passthrough if all checks pass.

    Args:
        command: The bash command string to validate.

    Returns:
        PermissionResult with the security check outcome.
    """
    context = _build_context(command)

    # Run validators in order
    validators = [
        ("empty", validate_empty),
        ("incomplete", validate_incomplete_commands),
        ("safe_heredoc", lambda ctx: validate_safe_heredoc(ctx.original_command)),
        ("obfuscated_flags", validate_obfuscated_flags),
        ("shell_metachars", validate_shell_metachars),
        ("dangerous_variables", validate_dangerous_variables),
        ("newlines", validate_newlines),
        ("command_substitution", validate_command_substitution),
        ("input_redirection", validate_input_redirection),
        ("output_redirection", validate_output_redirection),
        ("ifs_injection", validate_ifs_injection),
        ("git_commit_substitution", validate_git_commit_substitution),
        ("proc_environ_access", validate_proc_environ_access),
        ("malformed_token_injection", validate_malformed_token_injection),
        ("backslash_escaped_whitespace", validate_backslash_escaped_whitespace),
        ("brace_expansion", validate_brace_expansion),
        ("control_characters", validate_control_characters),
        ("unicode_whitespace", validate_unicode_whitespace),
        ("mid_word_hash", validate_mid_word_hash),
        ("zsh_dangerous_commands", validate_zsh_dangerous_commands),
        ("backslash_escaped_operators", validate_backslash_escaped_operators),
        ("comment_quote_desync", validate_comment_quote_desync),
        ("quoted_newline", validate_quoted_newline),
        ("carriage_return", validate_carriage_return),
        ("jq_command", validate_jq_command),
    ]

    for name, validator in validators:
        result = validator(context)
        if result.behavior != "passthrough":
            return result

    return PermissionResult.passthrough("Command passed all security checks")


def bash_command_is_safe_async(command: str) -> PermissionResult:
    """Async security check (currently synchronous, same as bash_command_is_safe).

    Async wrapper
    for future compatibility but currently runs synchronously.

    Args:
        command: The bash command string to validate.

    Returns:
        PermissionResult with the security check outcome.
    """
    return bash_command_is_safe(command)


def strip_safe_heredoc_substitutions(command: str) -> Optional[str]:
    """Strip safe heredoc substitutions from a command.

    Returns the command with safe heredocs removed, or None if no safe
    heredocs were found.

    Args:
        command: The command string.

    Returns:
        Stripped command string, or None.
    """
    if not HEREDOC_IN_SUBSTITUTION.search(command):
        return None

    pattern = re.compile(
        r'\$\(cat[ \t]*<<(-?)[ \t]*(?:\'+([A-Za-z_]\w*)\'+|\\([A-Za-z_]\w*))'
    )

    result = command
    found = False
    ranges = []

    for m in pattern.finditer(command):
        if m.start() > 0 and command[m.start() - 1] == "\\":
            continue
        delimiter = m.group(2) or m.group(3)
        if not delimiter:
            continue
        is_dash = m.group(1) == "-"
        operator_end = m.start() + len(m.group(0))

        after_operator = command[operator_end:]
        open_line_end = after_operator.find("\n")
        if open_line_end == -1:
            continue
        if not re.match(r"^[ \t]*$", after_operator[:open_line_end]):
            continue

        body_start = operator_end + open_line_end + 1
        body_lines = command[body_start:].split("\n")

        for i, line in enumerate(body_lines):
            check = line
            if is_dash:
                check = line.lstrip("\t")
            if check.startswith(delimiter):
                after = check[len(delimiter):]
                close_pos = -1
                if re.match(r"^[ \t]*\)", after):
                    line_start = body_start + sum(len(body_lines[j]) + 1 for j in range(i))
                    close_pos = command.index(")", line_start)
                elif after == "":
                    if i + 1 < len(body_lines):
                        next_line = body_lines[i + 1]
                        if re.match(r"^[ \t]*\)", next_line):
                            next_line_start = body_start + sum(len(body_lines[j]) + 1 for j in range(i + 1))
                            close_pos = command.index(")", next_line_start)
                if close_pos != -1:
                    ranges.append({"start": m.start(), "end": close_pos + 1})
                    found = True
                break

    if not found:
        return None

    for r in sorted(ranges, key=lambda x: x["start"], reverse=True):
        result = result[:r["start"]] + result[r["end"]:]

    return result


def has_safe_heredoc_substitution(command: str) -> bool:
    """Check if a command contains a safe heredoc substitution.

    Args:
        command: The command string.

    Returns:
        True if a safe heredoc substitution exists.
    """
    return strip_safe_heredoc_substitutions(command) is not None


__all__ = [
    "PermissionResult",
    "ValidationContext",
    "bash_command_is_safe",
    "bash_command_is_safe_async",
    "strip_safe_heredoc_substitutions",
    "has_safe_heredoc_substitution",
    "extract_quoted_content",
    "strip_safe_redirections",
    "has_unescaped_char",
    "COMMAND_SUBSTITUTION_PATTERNS",
    "ZSH_DANGEROUS_COMMANDS",
]
