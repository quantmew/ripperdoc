"""Shell command validation with comprehensive security checks.

This module implements security checks for shell commands to detect
potentially dangerous constructs before execution.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of shell command validation."""

    behavior: str  # 'passthrough' | 'ask' | 'allow' | 'deny'
    message: str
    rule_suggestions: Optional[List[str]] = None


def _strip_single_quotes(shell_command: str, first_token: str) -> str:
    """Strip content inside single quotes, handling escapes properly.

    Single-quoted content in shell is literal and cannot contain command
    substitution, so we can safely ignore it for security analysis.

    Double quotes are kept for analysis since they can contain variable
    expansions and command substitutions.
    """
    in_single_quote = False
    escaped = False
    result = []

    i = 0
    while i < len(shell_command):
        char = shell_command[i]

        if escaped:
            escaped = False
            result.append(char)
            i += 1
            continue

        if char == "\\":
            escaped = True
            result.append(char)
            i += 1
            continue

        if char == "'":
            in_single_quote = not in_single_quote
            i += 1
            continue

        if not in_single_quote:
            result.append(char)

        i += 1

    return "".join(result)


def _strip_quotes_for_analysis(command: str) -> str:
    """Strip content inside both single and double quotes for security analysis.

    This is used for checking shell metacharacters in arguments.
    Double quotes are stripped because they can contain variable expansions
    and command substitutions that need to be analyzed.
    """
    result = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    i = 0
    while i < len(command):
        char = command[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if char == "\\":
            escaped = True
            i += 1
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            i += 1
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            i += 1
            continue

        if not in_single_quote and not in_double_quote:
            result.append(char)

        i += 1

    return "".join(result)


def _is_safe_command_pattern(command: str) -> bool:
    """Check if command matches known safe patterns.

    These are commands that are commonly used and known to be safe
    even if they contain characters that might otherwise trigger warnings.
    """
    import re

    safe_patterns = [
        # Common version checks
        r"^\s*(python|python3|node|npm|git|bash|sh)\s+--version\s*$",
        r"^\s*(python|python3|node|npm|git|bash|sh)\s+-v\s*$",
        r"^\s*(python|python3|node|npm|git|bash|sh)\s+-V\s*$",
        # Common help commands
        r"^\s*\w+\s+--help\s*$",
        r"^\s*\w+\s+-h\s*$",
        r"^\s*\w+\s+help\s*$",
        # Simple echo/print commands
        r'^\s*echo\s+["\'].*["\']\s*$',
        r"^\s*print(env|f)?\s+.*$",
        # Directory listing with common options
        r'^\s*ls\s+(-[a-zA-Z]*[lhtr]*\s*)*["\']?[^;&|<>]*["\']?\s*$',
        r"^\s*dir\s+.*$",
        # Current directory
        r"^\s*pwd\s*$",
        # Environment variable checks
        r"^\s*env\s*$",
        r"^\s*printenv\s*$",
        # Which/whereis commands
        r"^\s*which\s+\w+\s*$",
        r"^\s*whereis\s+\w+\s*$",
        # Type/command commands
        r"^\s*type\s+\w+\s*$",
        r"^\s*command\s+-v\s+\w+\s*$",
    ]

    for pattern in safe_patterns:
        if re.match(pattern, command, re.IGNORECASE):
            return True

    return False


def _sanitize_safe_redirections(command: str) -> str:
    """Remove safe redirection patterns that don't need security checks."""
    # Remove stderr to stdout redirection (2>&1)
    sanitized = re.sub(r"\s+2\s*>&\s*1(?=\s|$)", "", command)
    # Remove redirections to /dev/null
    sanitized = re.sub(r"[012]?\s*>\s*/dev/null", "", sanitized)
    # Remove input from /dev/null
    sanitized = re.sub(r"\s*<\s*/dev/null", "", sanitized)
    return sanitized


# Dangerous patterns to check in sanitized commands
_DANGEROUS_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Newlines can break out of quotes
    (re.compile(r"[\n\r]"), "Command contains newlines which could break out of quotes"),
    # Process substitution
    (re.compile(r"<\("), "Command contains process substitution <()"),
    (re.compile(r">\("), "Command contains process substitution >()"),
    # Command substitution
    (re.compile(r"[`‵]"), "Command contains backticks (`) for command substitution"),
    (re.compile(r"\$\("), "Command contains $() command substitution"),
    # Parameter substitution
    (re.compile(r"\$\{"), "Command contains ${} parameter substitution"),
    # Input/output redirection
    (
        re.compile(r"<(?!\()"),
        "Command contains input redirection (<) which could read sensitive files",
    ),
    (
        re.compile(r">(?!\()"),
        "Command contains output redirection (>) which could write to arbitrary files",
    ),
    # Zsh-specific patterns
    (re.compile(r"~\["), "Command contains Zsh-style parameter expansion"),
    (re.compile(r"\(e:"), "Command contains Zsh-style glob qualifiers"),
]

# Patterns that indicate dangerous metacharacters in find/grep arguments
_DANGEROUS_METACHARACTER_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r'-name\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
    re.compile(r'-path\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
    re.compile(r'-iname\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
    re.compile(r'-regex\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
]

# =============================================================================
# Destructive Command Detection (Cross-platform)
# =============================================================================
# These patterns detect commands that can cause irreversible data loss.
# Inspired by the Gemini incident where `cmd /c "rmdir /s /q ..."` deleted C:\

# Windows destructive commands
_WINDOWS_DESTRUCTIVE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # rmdir /s - Recursive directory deletion (Windows)
    (
        re.compile(r"\brmdir\s+.*(/s|/S)", re.IGNORECASE),
        "Command contains 'rmdir /s' which recursively deletes directories",
    ),
    # del /s or del /q - Recursive or quiet file deletion (Windows)
    (
        re.compile(r"\bdel\s+.*(/s|/S|/q|/Q)", re.IGNORECASE),
        "Command contains 'del' with dangerous flags (/s or /q)",
    ),
    # rd /s - Alias for rmdir /s (Windows)
    (
        re.compile(r"\brd\s+.*(/s|/S)", re.IGNORECASE),
        "Command contains 'rd /s' which recursively deletes directories",
    ),
    # format command (Windows)
    (
        re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE),
        "Command contains 'format' which erases entire drives",
    ),
    # cmd /c with destructive subcommand
    (
        re.compile(r"\bcmd\s+/[cC]\s+.*\b(rmdir|rd|del|format)\b", re.IGNORECASE),
        "Command uses 'cmd /c' to execute a destructive subcommand",
    ),
    # PowerShell Remove-Item -Recurse
    (
        re.compile(r"\b(Remove-Item|rm|ri|del)\s+.*-Recurse", re.IGNORECASE),
        "Command contains 'Remove-Item -Recurse' which recursively deletes items",
    ),
    # PowerShell with -Force flag on destructive commands
    (
        re.compile(r"\b(Remove-Item|rm|ri|del)\s+.*-Force", re.IGNORECASE),
        "Command contains destructive command with -Force flag",
    ),
]

# Unix/Linux destructive commands
_UNIX_DESTRUCTIVE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # rm -rf or rm -r (recursive deletion) - must be at word boundary and followed by space/path
    (
        re.compile(r'(?<!["\'])\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|\s*-[a-zA-Z]*r[a-zA-Z]*$)'),
        "Command contains 'rm -r' which recursively deletes files and directories",
    ),
    # rm with force flag on system paths
    (
        re.compile(
            r'(?<!["\'])\brm\s+-[a-zA-Z]*f[a-zA-Z]*\s+(/|~|/home|/usr|/var|/etc|/root|\$HOME)'
        ),
        "Command contains 'rm -f' targeting a critical system path",
    ),
    # dd command (can overwrite disks)
    (re.compile(r"\bdd\s+.*of=/dev/"), "Command contains 'dd' writing to a device file"),
    # mkfs (creates filesystem, destroys data)
    (re.compile(r"\bmkfs\b"), "Command contains 'mkfs' which formats storage devices"),
    # shred (secure deletion)
    (re.compile(r"\bshred\s+"), "Command contains 'shred' which irreversibly destroys file data"),
    # chmod 777 on sensitive paths
    (
        re.compile(r"\bchmod\s+777\s+(/|/etc|/usr|/var|/home)"),
        "Command contains 'chmod 777' on a sensitive system path",
    ),
    # chown on system paths
    (
        re.compile(r"\bchown\s+.*\s+(/etc|/usr|/var|/bin|/sbin)"),
        "Command contains 'chown' on a critical system path",
    ),
]

# Path patterns that should trigger extra scrutiny
_CRITICAL_PATH_PATTERNS: List[re.Pattern[str]] = [
    # Windows critical paths
    re.compile(r'["\']?[A-Za-z]:\\(?:Windows|Program Files|Users)\\?["\']?', re.IGNORECASE),
    re.compile(r'["\']?[A-Za-z]:\\["\']?(?:\s|$)', re.IGNORECASE),  # Root of any drive
    # Unix critical paths
    re.compile(r'["\']?/(?:etc|usr|var|bin|sbin|lib|boot|root|home)["\']?(?:\s|$)'),
    re.compile(r'["\']?/["\']?(?:\s|$)'),  # Root directory
    re.compile(r'["\']?~["\']?(?:\s|$)'),  # Home directory
]

# Commands with nested/escaped quotes that might cause parsing issues
_NESTED_QUOTE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Windows cmd with escaped quotes inside
    (
        re.compile(r'\bcmd\s+/[cC]\s+"[^"]*\\"[^"]*"'),
        "Command contains 'cmd /c' with nested escaped quotes which may cause unexpected parsing",
    ),
    # PowerShell with complex quoting
    (
        re.compile(r'\bpowershell\s+.*-[Cc]ommand\s+["\'][^"\']*["\'][^"\']*["\']'),
        "Command contains PowerShell with complex nested quotes",
    ),
]


def _check_destructive_commands(command: str) -> Optional[ValidationResult]:
    """Check for destructive commands that could cause irreversible data loss.

    This function specifically addresses scenarios like the Gemini incident
    where a command with improper quoting deleted an entire drive.
    """
    # First, strip content inside quotes for Unix destructive pattern matching
    # to avoid false positives like 'find . -name "*.py;rm -rf /"'
    # But for Windows commands, we check the full command since quoting is different

    # Check for nested quote issues first (like the Gemini incident)
    for pattern, message in _NESTED_QUOTE_PATTERNS:
        if pattern.search(command):
            # If nested quotes AND targets critical path, deny
            has_critical_path = any(p.search(command) for p in _CRITICAL_PATH_PATTERNS)
            if has_critical_path:
                return ValidationResult(
                    behavior="deny",
                    message=f"BLOCKED: {message} targeting a critical system path",
                    rule_suggestions=None,
                )
            return ValidationResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    # Check if command targets critical paths with any destructive operation
    has_critical_path = any(p.search(command) for p in _CRITICAL_PATH_PATTERNS)

    # For interpreter commands, we need to check the code string for destructive commands
    # First, extract the first token to check if it's an interpreter
    trimmed = command.strip()
    first_token = trimmed.split()[0] if trimmed.split() else ""

    # Check if it's an interpreter command
    if _is_interpreter_command(command, first_token):
        # Extract and check the code string
        code_string = _extract_code_string(command, first_token)
        if code_string:
            # Check the code string for destructive patterns
            # We need to check both the code string itself and any commands it might execute
            code_check_result = _check_destructive_commands_in_code_string(code_string, first_token)
            if code_check_result:
                if has_critical_path:
                    return ValidationResult(
                        behavior="deny",
                        message=f"BLOCKED: {code_check_result.message} targeting a critical system path",
                        rule_suggestions=None,
                    )
                return code_check_result

    # Strip quoted content to avoid false positives like 'echo "rmdir /s /q folder"'
    command_without_quotes = _strip_quoted_content_for_destructive_check(command)

    # Check Windows destructive patterns on stripped command
    for pattern, message in _WINDOWS_DESTRUCTIVE_PATTERNS:
        if pattern.search(command_without_quotes):
            if has_critical_path:
                return ValidationResult(
                    behavior="deny",
                    message=f"BLOCKED: {message} targeting a critical system path",
                    rule_suggestions=None,
                )
            return ValidationResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    # Check Unix destructive patterns on the stripped command
    for pattern, message in _UNIX_DESTRUCTIVE_PATTERNS:
        if pattern.search(command_without_quotes):
            # Re-check critical path on original command
            if has_critical_path:
                return ValidationResult(
                    behavior="deny",
                    message=f"BLOCKED: {message} targeting a critical system path",
                    rule_suggestions=None,
                )
            return ValidationResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    # Special check for the exact Gemini incident pattern
    # cmd /c "rmdir /s /q \"path with spaces\""
    if re.search(r'\bcmd\s+/[cC]\s+"[^"]*\\[""][^"]*"', command):
        if has_critical_path:
            return ValidationResult(
                behavior="deny",
                message="BLOCKED: Command contains 'cmd /c' with escaped quotes - "
                "this pattern has caused data loss incidents",
                rule_suggestions=None,
            )
        return ValidationResult(
            behavior="ask",
            message="Command contains 'cmd /c' with escaped quotes inside double quotes - "
            "this pattern has caused data loss incidents due to quote parsing issues",
            rule_suggestions=None,
        )

    return None


def _check_destructive_commands_in_code_string(
    code_string: str, interpreter: str
) -> Optional[ValidationResult]:
    """Check for destructive commands in interpreter code strings.

    This handles cases like `bash -c "rm -rf /"` where the destructive
    command is inside the code string.
    """
    import re

    # For shell interpreters (bash, sh, zsh), the code string is shell code
    if interpreter in ("bash", "sh", "zsh"):
        # Check for destructive patterns in the shell code
        stripped_code = _strip_quoted_content_for_destructive_check(code_string)

        for pattern, message in _UNIX_DESTRUCTIVE_PATTERNS:
            if pattern.search(stripped_code):
                return ValidationResult(
                    behavior="ask",
                    message=f"Code string contains {message}",
                    rule_suggestions=None,
                )

    # For Python, check for os.system, subprocess, etc.
    elif interpreter in ("python", "python3"):
        # Check for system calls that execute shell commands
        system_patterns = [
            (
                r'\bos\.system\s*\(\s*["\'][^"\']*rm\s+-[a-zA-Z]*r',
                "Python code executes destructive shell command",
            ),
            (
                r"\bsubprocess\.(run|call|Popen)\s*\(\s*[^)]*rm\s+-[a-zA-Z]*r",
                "Python code executes destructive shell command",
            ),
        ]

        for pattern_str, message in system_patterns:
            if re.search(pattern_str, code_string):
                return ValidationResult(
                    behavior="ask",
                    message=message,
                    rule_suggestions=None,
                )

    # For other interpreters, we could add more checks as needed
    return None


def _strip_quoted_content_for_destructive_check(command: str) -> str:
    """Strip content inside quotes for destructive command checking.

    This prevents false positives like 'find . -name "rm -rf /"' from
    triggering the rm -rf detection.
    """
    result = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for char in command:
        if escaped:
            escaped = False
            if not in_single_quote and not in_double_quote:
                result.append(char)
            continue

        if char == "\\":
            escaped = True
            if not in_single_quote and not in_double_quote:
                result.append(char)
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if not in_single_quote and not in_double_quote:
            result.append(char)

    return "".join(result)


def _is_interpreter_command(command: str, first_token: str) -> bool:
    """Check if the command is an interpreter command that executes code strings.

    Interpreter commands like `python -c "code"`, `node -e "code"`, `bash -c "code"`
    should have different validation rules for their code strings.
    """
    interpreter_tokens = {"python", "python3", "node", "bash", "sh", "zsh", "perl", "ruby"}

    if first_token not in interpreter_tokens:
        return False

    # Check for -c or -e flag (execute code string)
    # Pattern: command -c "code" or command -e "code"
    import re

    pattern = rf'\b{re.escape(first_token)}\s+-(c|e)\s+["\']'
    return bool(re.search(pattern, command))


def _extract_code_string(command: str, first_token: str) -> str:
    """Extract the code string from an interpreter command.

    Returns the code string without the surrounding quotes, or empty string
    if not an interpreter command or no code string found.
    """
    if not _is_interpreter_command(command, first_token):
        return ""

    import re

    # Find the code string after -c or -e flag
    # Match: command -c "code" or command -e 'code'
    pattern = rf'{re.escape(first_token)}\s+-(c|e)\s+(["\'])(.*?)(?<!\\)\2'
    match = re.search(pattern, command, re.DOTALL)

    if match:
        code_string = match.group(3)
        # Remove escape characters
        code_string = code_string.replace('\\"', '"').replace("\\'", "'")
        return code_string

    return ""


def _strip_interpreter_code_strings(command: str, first_token: str) -> str:
    """Strip code strings from interpreter commands for validation.

    This allows us to validate the shell command structure while
    ignoring the content of code strings which may contain shell-like
    characters that are actually part of the code language.
    """
    if not _is_interpreter_command(command, first_token):
        return command

    import re

    # Replace code strings with placeholder
    # Match: command -c "code" or command -e 'code'
    # The (?<!\\) negative lookbehind ensures we don't match escaped quotes
    pattern = rf'({re.escape(first_token)}\s+-(c|e)\s+)(["\'])(.*?)(?<!\\)\3'

    def replace_code_string(match: re.Match[str]) -> str:
        prefix = match.group(1)
        quote = match.group(3)
        return f"{prefix}{quote}__CODE_STRING__{quote}"

    result = re.sub(pattern, replace_code_string, command, flags=re.DOTALL)
    return result


def validate_shell_command(shell_command: str) -> ValidationResult:
    """Validate a shell command for security risks.

    This function checks for potentially dangerous shell constructs that could
    be used for command injection or other security issues.

    Args:
        shell_command: The shell command to validate.

    Returns:
        ValidationResult with behavior indicating whether the command is safe.
    """
    if not shell_command or not shell_command.strip():
        return ValidationResult(
            behavior="passthrough",
            message="Empty command is safe",
            rule_suggestions=None,
        )

    trimmed = shell_command.strip()
    first_token = trimmed.split()[0] if trimmed.split() else ""

    # Check for safe command patterns first
    if _is_safe_command_pattern(trimmed):
        return ValidationResult(
            behavior="passthrough",
            message="Command matches safe pattern",
            rule_suggestions=None,
        )

    # FIRST: Check for destructive commands (highest priority)
    # This catches dangerous patterns like the Gemini incident
    destructive_result = _check_destructive_commands(trimmed)
    if destructive_result:
        return destructive_result

    # Special handling for jq commands
    if first_token == "jq":
        # Check for system() function which can execute arbitrary commands
        if re.search(r"\bsystem\s*\(", trimmed):
            return ValidationResult(
                behavior="ask",
                message="jq command contains system() function which executes arbitrary commands",
                rule_suggestions=None,
            )

        # Check if jq is reading from files (should only read from stdin)
        jq_args = trimmed[3:].strip() if len(trimmed) > 3 else ""
        if re.search(r'(?:^|\s)(?:[^\'"\s-][^\s]*\s+)?(?:/|~|\w+\.\w+)', jq_args):
            if not re.match(r"^\.[^\s]+$", jq_args):
                return ValidationResult(
                    behavior="ask",
                    message="jq command contains file arguments - jq should only read from stdin in read-only mode",
                    rule_suggestions=None,
                )

    # Allow git commit with single-quoted heredoc (common pattern for commit messages)
    if re.search(r"git\s+commit\s+.*-m\s+\"?\$\(cat\s*<<'[^']+'[\s\S]*?\)\"?", trimmed):
        return ValidationResult(
            behavior="passthrough",
            message="Git commit with single-quoted heredoc is allowed",
            rule_suggestions=None,
        )

    # Check for heredoc with command substitution (dangerous)
    if re.search(r"['\"]?\$\(cat\s*<<(?!')", trimmed):
        return ValidationResult(
            behavior="ask",
            message="Command contains heredoc with command substitution",
            rule_suggestions=None,
        )

    # Check for heredoc patterns that could run arbitrary content
    if re.search(r"\b(cat|tee)\s+<<\s*['\"]?EOF", trimmed):
        return ValidationResult(
            behavior="ask",
            message="Command contains heredoc which may run arbitrary content",
            rule_suggestions=None,
        )

    # Strip single-quoted content for further analysis
    sanitized = _strip_single_quotes(trimmed, first_token)

    # For interpreter commands, strip code strings before checking shell metacharacters
    # This allows code strings to contain language-specific characters like ;
    sanitized_for_metachar_check = sanitized
    if _is_interpreter_command(trimmed, first_token):
        sanitized_for_metachar_check = _strip_interpreter_code_strings(sanitized, first_token)

    # Remove safe redirections
    sanitized = _sanitize_safe_redirections(sanitized)
    sanitized_for_metachar_check = _sanitize_safe_redirections(sanitized_for_metachar_check)

    # Check for shell metacharacters outside of quotes
    # We'll parse the command and check for ; & characters that are not inside quotes
    # Special handling for find -exec escaped semicolon (\;)
    def has_metachars_outside_quotes(cmd: str) -> bool:
        # Use shlex for proper shell tokenization and quote handling
        lex = shlex.shlex(cmd, posix=True)
        lex.whitespace_split = True  # Split on whitespace, better for argument parsing
        lex.commenters = ""  # Don't treat # as comment for security analysis
        
        tokens = []
        try:
            # Get all tokens
            while True:
                token = lex.get_token()
                if token == lex.eof:
                    break
                tokens.append(token)
        except ValueError:
            # If shlex fails (e.g., unmatched quotes), be cautious and return True
            # This treats malformed commands as potentially dangerous
            return True

        # Check for dangerous operators in tokens
        # shlex will separate operators like ; & | as individual tokens
        # even when they're not surrounded by spaces
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in (";", "&", "|"):
                # Check if it's part of a safe operator (&& or ||)
                if token in ("&", "|") and i + 1 < len(tokens) and tokens[i + 1] == token:
                    # This is && or ||, skip both tokens
                    i += 2
                    continue
                # Single ; & | are dangerous
                return True
            i += 1
        
        # Also check for find -exec escaped semicolon pattern
        # shlex will have already parsed \; as separate token ';' (since escaped)
        # We need to check if this ; is part of find -exec pattern
        # by looking at the token context
        for i, token in enumerate(tokens):
            if token == ";":
                # Check if previous tokens contain "-exec"
                # Look backward through tokens to find "-exec"
                j = i - 1
                found_exec = False
                while j >= 0:
                    if tokens[j] == "-exec":
                        found_exec = True
                        break
                    j -= 1
                if found_exec:
                    # This is likely find -exec ... ;, check if it's escaped in original
                    # We need to check the original string to confirm it's \;
                    # Build a regex to find this specific semicolon
                    # For now, we'll assume it's the find -exec semicolon
                    # and continue checking other tokens
                    continue
                # Not part of find -exec, so it's dangerous
                return True
        
        return False

    if has_metachars_outside_quotes(sanitized_for_metachar_check):
        return ValidationResult(
            behavior="ask",
            message="Command contains shell metacharacters (;, |, or &) outside of quoted arguments",
            rule_suggestions=None,
        )

    # Check for dangerous metacharacters in find/grep arguments
    # Use the version with quotes stripped for this check
    stripped_for_pattern_check = _strip_quotes_for_analysis(sanitized)
    for pattern in _DANGEROUS_METACHARACTER_PATTERNS:
        if pattern.search(stripped_for_pattern_check):
            return ValidationResult(
                behavior="ask",
                message="Command contains shell metacharacters (;, |, or &) in find/grep arguments",
                rule_suggestions=None,
            )

    # Check for variables in dangerous contexts (redirections or pipes)
    if re.search(r"[<>|]\s*\$[A-Za-z_]", sanitized) or re.search(
        r"\$[A-Za-z_][A-Za-z0-9_]*\s*[|<>]", sanitized
    ):
        return ValidationResult(
            behavior="ask",
            message="Command contains variables in dangerous contexts (redirections or pipes)",
            rule_suggestions=None,
        )

    # Check for sourcing scripts
    # Note: `. ` followed by a path-like string (e.g., `. script.sh`, `. ./script`)
    # but not `. -name` (find argument) or similar
    if re.search(r"(^|\s)source\s+", sanitized):
        return ValidationResult(
            behavior="ask",
            message="Command sources another script which may execute arbitrary code",
            rule_suggestions=None,
        )
    # Match `. ` followed by a path (starts with /, ./, ~, or alphanumeric)
    if re.search(r"(^|\s)\.\s+[/~\w]", sanitized):
        # Exclude common find arguments like `. -name`
        if not re.search(r"\.\s+-[a-z]", sanitized):
            return ValidationResult(
                behavior="ask",
                message="Command sources another script which may execute arbitrary code",
                rule_suggestions=None,
            )

    # Check for eval
    if re.search(r"(^|\s)eval\s+", sanitized):
        return ValidationResult(
            behavior="ask",
            message="Command uses eval which executes arbitrary code",
            rule_suggestions=None,
        )

    # Check all dangerous patterns
    for pattern, message in _DANGEROUS_PATTERNS:
        if pattern.search(sanitized):
            # Special handling for newlines
            if "newlines" in message:
                # Check if newlines are in quotes or code strings
                in_quote = False
                quote_char = None
                escaped = False
                newline_outside_quotes = False

                i = 0
                while i < len(trimmed):
                    char = trimmed[i]

                    if escaped:
                        escaped = False
                        i += 1
                        continue

                    if char == "\\":
                        escaped = True
                        i += 1
                        continue

                    if char in ("'", '"') and not escaped:
                        if not in_quote:
                            in_quote = True
                            quote_char = char
                        elif char == quote_char:
                            in_quote = False
                            quote_char = None

                    if char in ("\n", "\r") and not in_quote:
                        newline_outside_quotes = True
                        break

                    i += 1

                if not newline_outside_quotes:
                    # Newlines are inside quotes, which is safer
                    # For interpreter commands, check if newlines are in code strings
                    if _is_interpreter_command(trimmed, first_token):
                        code_string = _extract_code_string(trimmed, first_token)
                        if code_string and any(c in code_string for c in ("\n", "\r")):
                            # Newlines are in code string, which is allowed for interpreter commands
                            continue
                    else:
                        # For non-interpreter commands, newlines in quotes are questionable but not blocked
                        continue

            return ValidationResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    return ValidationResult(
        behavior="passthrough",
        message="Command passed all security checks",
        rule_suggestions=None,
    )


def is_complex_unsafe_shell_command(command: str) -> bool:
    """Check if a command contains complex shell operators that need special handling.

    This detects commands with control operators like &&, ||, ;, etc. that
    combine multiple commands.
    """
    if not command:
        return False

    # Simple check for control operators outside of quotes
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

        # Outside of quotes, check for control operators
        if not in_single_quote and not in_double_quote:
            # Check for && or ||
            if char in ("&", "|") and i + 1 < len(command):
                next_char = command[i + 1]
                if (char == "&" and next_char == "&") or (char == "|" and next_char == "|"):
                    return True

            # Check for ; (but not ;;)
            if char == ";":
                if i + 1 >= len(command) or command[i + 1] != ";":
                    return True

    return False


__all__ = ["validate_shell_command", "ValidationResult", "is_complex_unsafe_shell_command"]
