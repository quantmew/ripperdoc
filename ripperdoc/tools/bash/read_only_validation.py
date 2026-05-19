"""Read-only validation for bash commands.


Provides the unified read-only constraint checking system with:
- Per-command flag allowlist validation (COMMAND_ALLOWLIST)
- Simple command regex-based validation (READONLY_COMMAND_REGEXES)
- Unquoted expansion detection (glob + $ expansion)
- Git sandbox escape detection
"""

from __future__ import annotations

import re
from typing import Dict, List

from ripperdoc.utils.bash.shell_quote import try_parse_shell_command
from ripperdoc.utils.bash.commands import (
    split_command,
    extract_output_redirections,
)
from ripperdoc.security import PermissionResult, bash_command_is_safe
from ripperdoc.tools.bash.sed_validation import sed_command_is_allowed_by_allowlist
from ripperdoc.utils.permissions.read_only_command_validation import (
    CommandConfig,
    FLAG_ARG_NONE, FLAG_ARG_NUMBER, FLAG_ARG_STRING, FLAG_ARG_CHAR,
    FLAG_ARG_BRACES, FLAG_ARG_EOF,
    validate_flags,
    contains_vulnerable_unc_path,
    GIT_READ_ONLY_COMMANDS,
    RIPGREP_READ_ONLY_COMMANDS,
    PYRIGHT_READ_ONLY_COMMANDS,
    DOCKER_READ_ONLY_COMMANDS,
    EXTERNAL_READONLY_COMMANDS,
)


# ============================================================================
# Helper: regex for safe commands
# ============================================================================


def _make_regex_for_safe_command(command: str) -> re.Pattern:
    """Create a regex pattern matching safe invocations of a command.

    Blovks: < > ( ) $ ` | { } & ; \\n \\r

    Args:
        command: The command name.

    Returns:
        Compiled regex pattern.
    """
    return re.compile(rf"^{command}(?:\s|$)[^<>()$\`|{{}}&;\\n\\r]*$")


# ============================================================================
# FD file descriptor flags (shared by fd and fdfind)
# ============================================================================

FD_SAFE_FLAGS: Dict[str, str] = {
    "-h": FLAG_ARG_NONE, "--help": FLAG_ARG_NONE,
    "-V": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    "-H": FLAG_ARG_NONE, "--hidden": FLAG_ARG_NONE,
    "-I": FLAG_ARG_NONE, "--no-ignore": FLAG_ARG_NONE,
    "--no-ignore-vcs": FLAG_ARG_NONE, "--no-ignore-parent": FLAG_ARG_NONE,
    "-s": FLAG_ARG_NONE, "--case-sensitive": FLAG_ARG_NONE,
    "-i": FLAG_ARG_NONE, "--ignore-case": FLAG_ARG_NONE,
    "-g": FLAG_ARG_NONE, "--glob": FLAG_ARG_NONE,
    "--regex": FLAG_ARG_NONE, "-F": FLAG_ARG_NONE, "--fixed-strings": FLAG_ARG_NONE,
    "-a": FLAG_ARG_NONE, "--absolute-path": FLAG_ARG_NONE,
    "-L": FLAG_ARG_NONE, "--follow": FLAG_ARG_NONE,
    "-p": FLAG_ARG_NONE, "--full-path": FLAG_ARG_NONE,
    "-0": FLAG_ARG_NONE, "--print0": FLAG_ARG_NONE,
    "-d": FLAG_ARG_NUMBER, "--max-depth": FLAG_ARG_NUMBER,
    "--min-depth": FLAG_ARG_NUMBER, "--exact-depth": FLAG_ARG_NUMBER,
    "-t": FLAG_ARG_STRING, "--type": FLAG_ARG_STRING,
    "-e": FLAG_ARG_STRING, "--extension": FLAG_ARG_STRING,
    "-S": FLAG_ARG_STRING, "--size": FLAG_ARG_STRING,
    "--changed-within": FLAG_ARG_STRING, "--changed-before": FLAG_ARG_STRING,
    "-o": FLAG_ARG_STRING, "--owner": FLAG_ARG_STRING,
    "-E": FLAG_ARG_STRING, "--exclude": FLAG_ARG_STRING,
    "--ignore-file": FLAG_ARG_STRING,
    "-c": FLAG_ARG_STRING, "--color": FLAG_ARG_STRING,
    "-j": FLAG_ARG_NUMBER, "--threads": FLAG_ARG_NUMBER,
    "--max-results": FLAG_ARG_NUMBER,
    "-1": FLAG_ARG_NONE, "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
    "--show-errors": FLAG_ARG_NONE, "--strip-cwd-prefix": FLAG_ARG_NONE,
    "--one-file-system": FLAG_ARG_NONE,
    "--search-path": FLAG_ARG_STRING, "--base-directory": FLAG_ARG_STRING,
    "--no-require-git": FLAG_ARG_NONE,
    "--format": FLAG_ARG_STRING,
}


# ============================================================================
# COMMAND_ALLOWLIST — per-command flag-level allowlist
# ============================================================================

COMMAND_ALLOWLIST: Dict[str, CommandConfig] = {}

# Populate from shared maps
COMMAND_ALLOWLIST.update(GIT_READ_ONLY_COMMANDS)
COMMAND_ALLOWLIST.update(RIPGREP_READ_ONLY_COMMANDS)
COMMAND_ALLOWLIST.update(PYRIGHT_READ_ONLY_COMMANDS)
COMMAND_ALLOWLIST.update(DOCKER_READ_ONLY_COMMANDS)

# fd / fdfind
COMMAND_ALLOWLIST["fd"] = CommandConfig(safe_flags=dict(FD_SAFE_FLAGS))
COMMAND_ALLOWLIST["fdfind"] = CommandConfig(safe_flags=dict(FD_SAFE_FLAGS))

# file
COMMAND_ALLOWLIST["file"] = CommandConfig(safe_flags={
    "-b": FLAG_ARG_NONE, "--brief": FLAG_ARG_NONE,
    "-i": FLAG_ARG_NONE, "--mime": FLAG_ARG_NONE,
    "--mime-type": FLAG_ARG_NONE, "--mime-encoding": FLAG_ARG_NONE,
    "-c": FLAG_ARG_NONE, "--check-encoding": FLAG_ARG_NONE,
    "-e": FLAG_ARG_STRING, "--exclude": FLAG_ARG_STRING,
    "--exclude-quiet": FLAG_ARG_STRING,
    "-0": FLAG_ARG_NONE, "--print0": FLAG_ARG_NONE,
    "-f": FLAG_ARG_STRING, "-F": FLAG_ARG_STRING,
    "--separator": FLAG_ARG_STRING,
    "-h": FLAG_ARG_NONE, "--no-dereference": FLAG_ARG_NONE,
    "-L": FLAG_ARG_NONE, "--dereference": FLAG_ARG_NONE,
    "-m": FLAG_ARG_STRING, "--magic-file": FLAG_ARG_STRING,
    "-k": FLAG_ARG_NONE, "--keep-going": FLAG_ARG_NONE,
    "-l": FLAG_ARG_NONE, "--list": FLAG_ARG_NONE,
    "-n": FLAG_ARG_NONE, "--no-buffer": FLAG_ARG_NONE,
    "-p": FLAG_ARG_NONE, "--preserve-date": FLAG_ARG_NONE,
    "-r": FLAG_ARG_NONE, "--raw": FLAG_ARG_NONE,
    "-s": FLAG_ARG_NONE, "--special-files": FLAG_ARG_NONE,
    "-z": FLAG_ARG_NONE, "--uncompress": FLAG_ARG_NONE,
    "--help": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
})

# grep
COMMAND_ALLOWLIST["grep"] = CommandConfig(safe_flags={
    "-e": FLAG_ARG_STRING, "--regexp": FLAG_ARG_STRING,
    "-f": FLAG_ARG_STRING, "--file": FLAG_ARG_STRING,
    "-F": FLAG_ARG_NONE, "--fixed-strings": FLAG_ARG_NONE,
    "-G": FLAG_ARG_NONE, "--basic-regexp": FLAG_ARG_NONE,
    "-E": FLAG_ARG_NONE, "--extended-regexp": FLAG_ARG_NONE,
    "-P": FLAG_ARG_NONE, "--perl-regexp": FLAG_ARG_NONE,
    "-i": FLAG_ARG_NONE, "--ignore-case": FLAG_ARG_NONE,
    "-v": FLAG_ARG_NONE, "--invert-match": FLAG_ARG_NONE,
    "-w": FLAG_ARG_NONE, "--word-regexp": FLAG_ARG_NONE,
    "-x": FLAG_ARG_NONE, "--line-regexp": FLAG_ARG_NONE,
    "-c": FLAG_ARG_NONE, "--count": FLAG_ARG_NONE,
    "--color": FLAG_ARG_STRING, "--colour": FLAG_ARG_STRING,
    "-L": FLAG_ARG_NONE, "--files-without-match": FLAG_ARG_NONE,
    "-l": FLAG_ARG_NONE, "--files-with-matches": FLAG_ARG_NONE,
    "-m": FLAG_ARG_NUMBER, "--max-count": FLAG_ARG_NUMBER,
    "-o": FLAG_ARG_NONE, "--only-matching": FLAG_ARG_NONE,
    "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE, "--silent": FLAG_ARG_NONE,
    "-s": FLAG_ARG_NONE, "--no-messages": FLAG_ARG_NONE,
    "-b": FLAG_ARG_NONE, "--byte-offset": FLAG_ARG_NONE,
    "-H": FLAG_ARG_NONE, "--with-filename": FLAG_ARG_NONE,
    "-h": FLAG_ARG_NONE, "--no-filename": FLAG_ARG_NONE,
    "-n": FLAG_ARG_NONE, "--line-number": FLAG_ARG_NONE,
    "-T": FLAG_ARG_NONE, "--initial-tab": FLAG_ARG_NONE,
    "-u": FLAG_ARG_NONE, "--unix-byte-offsets": FLAG_ARG_NONE,
    "-Z": FLAG_ARG_NONE, "--null": FLAG_ARG_NONE,
    "-z": FLAG_ARG_NONE, "--null-data": FLAG_ARG_NONE,
    "-A": FLAG_ARG_NUMBER, "--after-context": FLAG_ARG_NUMBER,
    "-B": FLAG_ARG_NUMBER, "--before-context": FLAG_ARG_NUMBER,
    "-C": FLAG_ARG_NUMBER, "--context": FLAG_ARG_NUMBER,
    "-a": FLAG_ARG_NONE, "--text": FLAG_ARG_NONE,
    "--binary-files": FLAG_ARG_STRING,
    "-D": FLAG_ARG_STRING, "--devices": FLAG_ARG_STRING,
    "-d": FLAG_ARG_STRING, "--directories": FLAG_ARG_STRING,
    "--exclude": FLAG_ARG_STRING, "--exclude-from": FLAG_ARG_STRING,
    "--exclude-dir": FLAG_ARG_STRING, "--include": FLAG_ARG_STRING,
    "-r": FLAG_ARG_NONE, "--recursive": FLAG_ARG_NONE,
    "-R": FLAG_ARG_NONE, "--dereference-recursive": FLAG_ARG_NONE,
    "--line-buffered": FLAG_ARG_NONE,
    "--help": FLAG_ARG_NONE, "-V": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
})

COMMAND_ALLOWLIST["sed"] = CommandConfig(
    safe_flags={
        "-e": FLAG_ARG_STRING, "--expression": FLAG_ARG_STRING,
        "-n": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE, "--silent": FLAG_ARG_NONE,
        "-r": FLAG_ARG_NONE, "--regexp-extended": FLAG_ARG_NONE,
        "-E": FLAG_ARG_NONE, "--posix": FLAG_ARG_NONE,
        "-l": FLAG_ARG_NUMBER, "--line-length": FLAG_ARG_NUMBER,
        "-z": FLAG_ARG_NONE, "--zero-terminated": FLAG_ARG_NONE,
        "-s": FLAG_ARG_NONE, "--separate": FLAG_ARG_NONE,
        "-u": FLAG_ARG_NONE, "--unbuffered": FLAG_ARG_NONE,
        "--debug": FLAG_ARG_NONE,
        "--help": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    },
    additional_command_is_dangerous_callback=lambda raw_cmd, args: (
        not sed_command_is_allowed_by_allowlist(raw_cmd)
    ),
)

# sort
COMMAND_ALLOWLIST["sort"] = CommandConfig(safe_flags={
    "-b": FLAG_ARG_NONE, "--ignore-leading-blanks": FLAG_ARG_NONE,
    "-d": FLAG_ARG_NONE, "--dictionary-order": FLAG_ARG_NONE,
    "-f": FLAG_ARG_NONE, "--ignore-case": FLAG_ARG_NONE,
    "-g": FLAG_ARG_NONE, "--general-numeric-sort": FLAG_ARG_NONE,
    "-h": FLAG_ARG_NONE, "--human-numeric-sort": FLAG_ARG_NONE,
    "-i": FLAG_ARG_NONE, "--ignore-nonprinting": FLAG_ARG_NONE,
    "-M": FLAG_ARG_NONE, "--month-sort": FLAG_ARG_NONE,
    "-n": FLAG_ARG_NONE, "--numeric-sort": FLAG_ARG_NONE,
    "-R": FLAG_ARG_NONE, "--random-sort": FLAG_ARG_NONE,
    "-r": FLAG_ARG_NONE, "--reverse": FLAG_ARG_NONE,
    "--sort": FLAG_ARG_STRING,
    "-s": FLAG_ARG_NONE, "--stable": FLAG_ARG_NONE,
    "-u": FLAG_ARG_NONE, "--unique": FLAG_ARG_NONE,
    "-V": FLAG_ARG_NONE, "--version-sort": FLAG_ARG_NONE,
    "-z": FLAG_ARG_NONE, "--zero-terminated": FLAG_ARG_NONE,
    "-k": FLAG_ARG_STRING, "--key": FLAG_ARG_STRING,
    "-t": FLAG_ARG_STRING, "--field-separator": FLAG_ARG_STRING,
    "-c": FLAG_ARG_NONE, "--check": FLAG_ARG_NONE,
    "-C": FLAG_ARG_NONE, "--check-char-order": FLAG_ARG_NONE,
    "-m": FLAG_ARG_NONE, "--merge": FLAG_ARG_NONE,
    "-S": FLAG_ARG_STRING, "--buffer-size": FLAG_ARG_STRING,
    "--parallel": FLAG_ARG_NUMBER, "--batch-size": FLAG_ARG_NUMBER,
    "--help": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
})

# man
COMMAND_ALLOWLIST["man"] = CommandConfig(safe_flags={
    "-a": FLAG_ARG_NONE, "--all": FLAG_ARG_NONE,
    "-d": FLAG_ARG_NONE, "-f": FLAG_ARG_NONE, "--whatis": FLAG_ARG_NONE,
    "-h": FLAG_ARG_NONE, "-k": FLAG_ARG_NONE, "--apropos": FLAG_ARG_NONE,
    "-l": FLAG_ARG_STRING, "-w": FLAG_ARG_NONE,
    "-S": FLAG_ARG_STRING, "-s": FLAG_ARG_STRING,
})

# help
COMMAND_ALLOWLIST["help"] = CommandConfig(safe_flags={
    "-d": FLAG_ARG_NONE, "-m": FLAG_ARG_NONE, "-s": FLAG_ARG_NONE,
})

# ps (with callback for BSD-style 'e' modifier)
COMMAND_ALLOWLIST["ps"] = CommandConfig(
    safe_flags={
        "-e": FLAG_ARG_NONE, "-A": FLAG_ARG_NONE, "-a": FLAG_ARG_NONE, "-d": FLAG_ARG_NONE,
        "-N": FLAG_ARG_NONE, "--deselect": FLAG_ARG_NONE,
        "-f": FLAG_ARG_NONE, "-F": FLAG_ARG_NONE, "-l": FLAG_ARG_NONE, "-j": FLAG_ARG_NONE, "-y": FLAG_ARG_NONE,
        "-w": FLAG_ARG_NONE, "-ww": FLAG_ARG_NONE, "--width": FLAG_ARG_NUMBER,
        "-c": FLAG_ARG_NONE, "-H": FLAG_ARG_NONE, "--forest": FLAG_ARG_NONE,
        "--headers": FLAG_ARG_NONE, "--no-headers": FLAG_ARG_NONE,
        "-n": FLAG_ARG_STRING, "--sort": FLAG_ARG_STRING,
        "-L": FLAG_ARG_NONE, "-T": FLAG_ARG_NONE, "-m": FLAG_ARG_NONE,
        "-C": FLAG_ARG_STRING, "-G": FLAG_ARG_STRING, "-g": FLAG_ARG_STRING,
        "-p": FLAG_ARG_STRING, "--pid": FLAG_ARG_STRING,
        "-q": FLAG_ARG_STRING, "--quick-pid": FLAG_ARG_STRING,
        "-s": FLAG_ARG_STRING, "--sid": FLAG_ARG_STRING,
        "-t": FLAG_ARG_STRING, "--tty": FLAG_ARG_STRING,
        "-U": FLAG_ARG_STRING, "-u": FLAG_ARG_STRING, "--user": FLAG_ARG_STRING,
        "--help": FLAG_ARG_NONE, "--info": FLAG_ARG_NONE,
        "-V": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    },
    additional_command_is_dangerous_callback=lambda raw, args: (
        any(a for a in args if not a.startswith("-") and re.match(r"^[a-zA-Z]*e[a-zA-Z]*$", a))
    ),
)

# base64
COMMAND_ALLOWLIST["base64"] = CommandConfig(
    respects_double_dash=False,
    safe_flags={
        "-d": FLAG_ARG_NONE, "-D": FLAG_ARG_NONE, "--decode": FLAG_ARG_NONE,
        "-b": FLAG_ARG_NUMBER, "--break": FLAG_ARG_NUMBER,
        "-w": FLAG_ARG_NUMBER, "--wrap": FLAG_ARG_NUMBER,
        "-i": FLAG_ARG_STRING, "--input": FLAG_ARG_STRING,
        "--ignore-garbage": FLAG_ARG_NONE,
        "-h": FLAG_ARG_NONE, "--help": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    },
)

# xargs (with safe target commands)
SAFE_TARGET_COMMANDS_FOR_XARGS = ["echo", "printf", "wc", "grep", "head", "tail"]

COMMAND_ALLOWLIST["xargs"] = CommandConfig(
    safe_flags={
        "-I": FLAG_ARG_BRACES,
        "-n": FLAG_ARG_NUMBER, "-P": FLAG_ARG_NUMBER, "-L": FLAG_ARG_NUMBER,
        "-s": FLAG_ARG_NUMBER,
        "-E": FLAG_ARG_EOF,
        "-0": FLAG_ARG_NONE, "-t": FLAG_ARG_NONE, "-r": FLAG_ARG_NONE, "-x": FLAG_ARG_NONE,
        "-d": FLAG_ARG_CHAR,
    },
)

# sha256sum / sha1sum / md5sum
for _sum_cmd in ["sha256sum", "sha1sum", "md5sum"]:
    COMMAND_ALLOWLIST[_sum_cmd] = CommandConfig(safe_flags={
        "-b": FLAG_ARG_NONE, "--binary": FLAG_ARG_NONE,
        "-t": FLAG_ARG_NONE, "--text": FLAG_ARG_NONE,
        "-c": FLAG_ARG_NONE, "--check": FLAG_ARG_NONE,
        "--ignore-missing": FLAG_ARG_NONE,
        "--quiet": FLAG_ARG_NONE, "--status": FLAG_ARG_NONE,
        "--strict": FLAG_ARG_NONE,
        "-w": FLAG_ARG_NONE, "--warn": FLAG_ARG_NONE,
        "--tag": FLAG_ARG_NONE,
        "-z": FLAG_ARG_NONE, "--zero": FLAG_ARG_NONE,
        "--help": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    })

# tree (excluding -o/--output which writes files)
COMMAND_ALLOWLIST["tree"] = CommandConfig(safe_flags={
    "-a": FLAG_ARG_NONE, "-d": FLAG_ARG_NONE, "-l": FLAG_ARG_NONE,
    "-f": FLAG_ARG_NONE, "-x": FLAG_ARG_NONE, "-L": FLAG_ARG_NUMBER,
    "-P": FLAG_ARG_STRING, "-I": FLAG_ARG_STRING,
    "--gitignore": FLAG_ARG_NONE, "--ignore-case": FLAG_ARG_NONE,
    "--prune": FLAG_ARG_NONE, "--noreport": FLAG_ARG_NONE,
    "--charset": FLAG_ARG_STRING, "--filelimit": FLAG_ARG_NUMBER,
    "-q": FLAG_ARG_NONE, "-N": FLAG_ARG_NONE, "-Q": FLAG_ARG_NONE,
    "-p": FLAG_ARG_NONE, "-u": FLAG_ARG_NONE, "-g": FLAG_ARG_NONE,
    "-s": FLAG_ARG_NONE, "-h": FLAG_ARG_NONE, "--si": FLAG_ARG_NONE, "--du": FLAG_ARG_NONE,
    "-D": FLAG_ARG_NONE, "--timefmt": FLAG_ARG_STRING,
    "-F": FLAG_ARG_NONE, "--inodes": FLAG_ARG_NONE, "--device": FLAG_ARG_NONE,
    "-v": FLAG_ARG_NONE, "-t": FLAG_ARG_NONE, "-c": FLAG_ARG_NONE,
    "-U": FLAG_ARG_NONE, "-r": FLAG_ARG_NONE,
    "--dirsfirst": FLAG_ARG_NONE, "--sort": FLAG_ARG_STRING,
    "-i": FLAG_ARG_NONE, "-A": FLAG_ARG_NONE, "-S": FLAG_ARG_NONE,
    "-n": FLAG_ARG_NONE, "-C": FLAG_ARG_NONE,
    "-X": FLAG_ARG_NONE, "-J": FLAG_ARG_NONE,
    "-H": FLAG_ARG_STRING, "--nolinks": FLAG_ARG_NONE,
    "-T": FLAG_ARG_STRING, "--hyperlink": FLAG_ARG_NONE,
    "--fromfile": FLAG_ARG_NONE,
    "--help": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
})

# date (excluding -s/--set which sets system time)
COMMAND_ALLOWLIST["date"] = CommandConfig(
    safe_flags={
        "-d": FLAG_ARG_STRING, "--date": FLAG_ARG_STRING,
        "-r": FLAG_ARG_STRING, "--reference": FLAG_ARG_STRING,
        "-u": FLAG_ARG_NONE, "--utc": FLAG_ARG_NONE, "--universal": FLAG_ARG_NONE,
        "-I": FLAG_ARG_NONE, "--iso-8601": FLAG_ARG_STRING,
        "-R": FLAG_ARG_NONE, "--rfc-email": FLAG_ARG_NONE,
        "--rfc-3339": FLAG_ARG_STRING,
        "--debug": FLAG_ARG_NONE,
        "--help": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    },
    additional_command_is_dangerous_callback=lambda _raw, args: (
        _date_is_dangerous(args)
    ),
)

def _date_is_dangerous(args: List[str]) -> bool:
    """Check if date positional args could set system time."""
    flags_with_args = {"-d", "--date", "-r", "--reference", "--iso-8601", "--rfc-3339"}
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--") and "=" in token:
            i += 1
        elif token.startswith("-"):
            if token in flags_with_args:
                i += 2
            else:
                i += 1
        else:
            if not token.startswith("+"):
                return True
            i += 1
    return False

# hostname (block positional args)
COMMAND_ALLOWLIST["hostname"] = CommandConfig(
    safe_flags={
        "-f": FLAG_ARG_NONE, "--fqdn": FLAG_ARG_NONE, "--long": FLAG_ARG_NONE,
        "-s": FLAG_ARG_NONE, "--short": FLAG_ARG_NONE,
        "-i": FLAG_ARG_NONE, "--ip-address": FLAG_ARG_NONE,
        "-I": FLAG_ARG_NONE, "--all-ip-addresses": FLAG_ARG_NONE,
        "-a": FLAG_ARG_NONE, "--alias": FLAG_ARG_NONE,
        "-d": FLAG_ARG_NONE, "--domain": FLAG_ARG_NONE,
        "-A": FLAG_ARG_NONE, "--all-fqdns": FLAG_ARG_NONE,
        "-v": FLAG_ARG_NONE, "--verbose": FLAG_ARG_NONE,
        "-h": FLAG_ARG_NONE, "--help": FLAG_ARG_NONE,
        "-V": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    },
    regex=re.compile(r"^hostname(?:\s+(?:-[a-zA-Z]|--[a-zA-Z-]+))*\s*$"),
)

# pgrep
COMMAND_ALLOWLIST["pgrep"] = CommandConfig(safe_flags={
    "-d": FLAG_ARG_STRING, "--delimiter": FLAG_ARG_STRING,
    "-l": FLAG_ARG_NONE, "--list-name": FLAG_ARG_NONE,
    "-a": FLAG_ARG_NONE, "--list-full": FLAG_ARG_NONE,
    "-v": FLAG_ARG_NONE, "--inverse": FLAG_ARG_NONE,
    "-c": FLAG_ARG_NONE, "--count": FLAG_ARG_NONE,
    "-f": FLAG_ARG_NONE, "--full": FLAG_ARG_NONE,
    "-g": FLAG_ARG_STRING, "--pgroup": FLAG_ARG_STRING,
    "-G": FLAG_ARG_STRING, "--group": FLAG_ARG_STRING,
    "-i": FLAG_ARG_NONE, "--ignore-case": FLAG_ARG_NONE,
    "-n": FLAG_ARG_NONE, "--newest": FLAG_ARG_NONE,
    "-o": FLAG_ARG_NONE, "--oldest": FLAG_ARG_NONE,
    "-P": FLAG_ARG_STRING, "--parent": FLAG_ARG_STRING,
    "-s": FLAG_ARG_STRING, "--session": FLAG_ARG_STRING,
    "-t": FLAG_ARG_STRING, "--terminal": FLAG_ARG_STRING,
    "-u": FLAG_ARG_STRING, "--euid": FLAG_ARG_STRING,
    "-U": FLAG_ARG_STRING, "--uid": FLAG_ARG_STRING,
    "-x": FLAG_ARG_NONE, "--exact": FLAG_ARG_NONE,
    "-F": FLAG_ARG_STRING, "--pidfile": FLAG_ARG_STRING,
    "--help": FLAG_ARG_NONE, "-V": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
})


# ============================================================================
# READONLY_COMMANDS — simple commands matched by regex
# ============================================================================

READONLY_COMMANDS: List[str] = [
    *EXTERNAL_READONLY_COMMANDS,
    # Unix-specific
    "cal", "uptime",
    "cat", "head", "tail", "wc", "stat", "strings", "hexdump", "od", "nl",
    "id", "uname", "free", "df", "du", "locale", "groups", "nproc",
    "basename", "dirname", "realpath",
    "cut", "paste", "tr", "column", "tac", "rev", "fold", "expand",
    "unexpand", "fmt", "comm", "cmp", "numfmt",
    "readlink",
    "diff",
    "true", "false",
    "sleep", "which", "type", "expr", "test", "getconf", "seq", "tsort", "pr",
]

# ============================================================================
# READONLY_COMMAND_REGEXES — complex regex patterns
# ============================================================================

READONLY_COMMAND_REGEXES: List[re.Pattern] = [
    # Simple commands as regexes
    *(re.compile(rf"^{c}(?:\s|$)[^<>()$`|{{}}&;\\n\\r]*$") for c in READONLY_COMMANDS),
    # echo with safe quoting
    re.compile(
        r"^echo(?:\s+(?:'[^']*'|\"[^\"$<>\n\r]*\"|[^|;&`$(){}><#\\!\"'\s]+))*(?:\s+2>&1)?\s*$"
    ),
    # pwd, whoami
    re.compile(r"^pwd$"),
    re.compile(r"^whoami$"),
    # history
    re.compile(r"^history(?:\s+\d+)?\s*$"),
    # alias
    re.compile(r"^alias$"),
    # arch
    re.compile(r"^arch(?:\s+(?:--help|-h))?\s*$"),
    # ip addr, ifconfig
    re.compile(r"^ip addr$"),
    re.compile(r"^ifconfig(?:\s+[a-zA-Z][a-zA-Z0-9_-]*)?\s*$"),
    # jq
    re.compile(
        r"^jq(?!\s+.*(?:-f\b|--from-file|--rawfile|--slurpfile|--run-tests|-L\b|--library-path|\benv\b|\$ENV\b))"
        r"(?:\s+(?:-[a-zA-Z]+|--[a-zA-Z-]+(?:=\S+)?))*"
        r"(?:\s+'[^'`]*'|\s+\"[^\"`]*\"|\s+[^-\s'\"][^\s]*)+\s*$"
    ),
    # cd
    re.compile(r"^cd(?:\s+(?:'[^']*'|\"[^\"]*\"|[^\s;|&`$(){}><#\\]+))?$"),
    # ls
    re.compile(r"^ls(?:\s+[^<>()$`|{}&;\n\r]*)?$"),
    # find (block -delete, -exec, -execdir, -ok, -okdir, -fprint, -fls, -fprintf)
    re.compile(
        r"^find(?:\s+(?:\\[()]|(?!-delete\b|-exec\b|-execdir\b|-ok\b|-okdir\b"
        r"|-fprint0?\b|-fls\b|-fprintf\b)[^<>()$`|{}&;\n\r\s]|\s)+)?$"
    ),
    # node/python version checks (anchored)
    re.compile(r"^node -v$"),
    re.compile(r"^node --version$"),
    re.compile(r"^python --version$"),
    re.compile(r"^python3 --version$"),
]


# ============================================================================
# Unquoted expansion detection
# ============================================================================


def contains_unquoted_expansion(command: str) -> bool:
    """Check for glob characters and expandable $ outside quotes.

    These could expand at runtime to bypass regex-based security checks.

    Args:
        command: The command string.

    Returns:
        True if unquoted glob or $ expansion is detected.
    """
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for i, char in enumerate(command):
        if escaped:
            escaped = False
            continue

        # SECURITY: backslash outside single quotes
        if char == "\\" and not in_single_quote:
            escaped = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if in_single_quote:
            continue

        # Check $ followed by variable-name char
        if char == "$":
            if i + 1 < len(command) and re.match(r"[A-Za-z_@*#?!$0-9-]", command[i + 1]):
                return True

        if in_double_quote:
            continue

        # Glob chars outside all quotes
        if char in "?*[]":
            return True

    return False


# ============================================================================
# Git-specific helpers
# ============================================================================


def is_normalized_git_command(command: str) -> bool:
    """Check if a command is a git subcommand.

    Args:
        command: Trimmed command string.

    Returns:
        True if the command starts with 'git'.
    """
    return command.startswith("git ") or command == "git"


def command_has_any_git(command: str) -> bool:
    """Check if a compound command contains any git subcommand.

    Args:
        command: The full command string.

    Returns:
        True if any subcommand is git.
    """
    return any(is_normalized_git_command(subcmd.strip()) for subcmd in split_command(command))


GIT_INTERNAL_PATTERNS = [
    re.compile(r"^HEAD$"),
    re.compile(r"^objects(?:\/|$)"),
    re.compile(r"^refs(?:\/|$)"),
    re.compile(r"^hooks(?:\/|$)"),
]


def is_git_internal_path(path: str) -> bool:
    """Check if a path is a git-internal path.

    Args:
        path: The path to check.

    Returns:
        True if the path matches git internal patterns.
    """
    normalized = re.sub(r"^\.?\/", "", path)
    return any(p.search(normalized) for p in GIT_INTERNAL_PATTERNS)


NON_CREATING_WRITE_COMMANDS = frozenset({"rm", "rmdir", "sed"})
WRITE_COMMANDS = frozenset({
    "cp", "mv", "touch", "mkdir", "ln",
    "cat", "tee", "dd",
})


def _command_writes_to_git_internal_paths(command: str) -> bool:
    """Check if a compound command writes to git-internal paths.

    Args:
        command: The full command string.

    Returns:
        True if any subcommand writes to git-internal paths.
    """
    subcommands = split_command(command)
    for subcmd in subcommands:
        trimmed = subcmd.strip()
        tokens = trimmed.split()
        if not tokens:
            continue
        base = tokens[0]

        if base in WRITE_COMMANDS:
            # Check positional args as paths
            for arg in tokens[1:]:
                if not arg.startswith("-") and is_git_internal_path(arg):
                    return True

        # Check output redirections
        result = extract_output_redirections(trimmed)
        for r in result.redirections:
            if is_git_internal_path(r.target):
                return True

    return False


# ============================================================================
# Main entry point: checkReadOnlyConstraints
# ============================================================================


def is_command_safe_via_flag_parsing(command: str) -> bool:
    """Check if a command is safe by validating its flags against the allowlist.

    Unified command validation that replaces individual validator functions.

    Args:
        command: The command string to validate.

    Returns:
        True if the command is safe via flag parsing.
    """
    parse_result = try_parse_shell_command(command)
    if not parse_result.success:
        return False

    tokens = [str(t) for t in parse_result.tokens]

    if not tokens:
        return False

    # Check multi-word commands first
    config = None
    cmd_token_count = 0
    for cmd_pattern, cfg in COMMAND_ALLOWLIST.items():
        pattern_tokens = cmd_pattern.split()
        if len(tokens) >= len(pattern_tokens):
            matches = all(tokens[i] == pattern_tokens[i] for i in range(len(pattern_tokens)))
            if matches:
                config = cfg
                cmd_token_count = len(pattern_tokens)
                break

    if config is None:
        return False

    # SECURITY: Reject tokens containing $ (variable expansion)
    for i in range(cmd_token_count, len(tokens)):
        token = tokens[i]
        if not token:
            continue
        if "$" in token:
            return False
        if "{" in token and ("," in token or ".." in token):
            return False

    # Validate flags
    if not validate_flags(
        tokens, cmd_token_count, config,
        command_name=tokens[0],
        raw_command=command,
        xargs_target_commands=(
            SAFE_TARGET_COMMANDS_FOR_XARGS if tokens[0] == "xargs" else None
        ),
    ):
        return False

    # Run regex check
    if config.regex and not config.regex.search(command):
        return False

    # Block backticks (unless regex handles it)
    if not config.regex and "`" in command:
        return False

    # Block newlines in grep/rg
    if not config.regex and tokens[0] in ("rg", "grep") and re.search(r"[\n\r]", command):
        return False

    # Run custom callback
    if config.additional_command_is_dangerous_callback:
        if config.additional_command_is_dangerous_callback(command, tokens[cmd_token_count:]):
            return False

    return True


def is_command_read_only(command: str) -> bool:
    """Check if a single command string is read-only.

    Args:
        command: The command string.

    Returns:
        True if the command is read-only.
    """
    test_cmd = command.strip()

    # Handle 2>&1 at the end
    if test_cmd.endswith(" 2>&1"):
        test_cmd = test_cmd[:-5].strip()

    # Check UNC paths
    if contains_vulnerable_unc_path(test_cmd):
        return False

    # Check unquoted expansion
    if contains_unquoted_expansion(test_cmd):
        return False

    # First try allowlist-based flag parsing
    if is_command_safe_via_flag_parsing(test_cmd):
        return True

    # Then try regex matching
    for regex in READONLY_COMMAND_REGEXES:
        if regex.search(test_cmd):
            # Extra git checks
            if "git" in test_cmd:
                if re.search(r"\s-c[\s=]", test_cmd):
                    return False
                if re.search(r"\s--exec-path[\s=]", test_cmd):
                    return False
                if re.search(r"\s--config-env[\s=]", test_cmd):
                    return False
            return True

    return False


def check_read_only_constraints(
    command: str,
    compound_command_has_cd: bool = False,
) -> PermissionResult:
    """Check read-only constraints for a bash command.

    This is the main exported entry point for read-only validation.

    Args:
        command: The bash command string.
        compound_command_has_cd: Whether any cd command exists in the compound command.

    Returns:
        PermissionResult indicating whether the command is read-only.
    """
    # Parse command via shell-quote
    parse_result = try_parse_shell_command(command)
    if not parse_result.success:
        return PermissionResult.passthrough(
            "Command cannot be parsed, requires further permission checks"
        )

    # Check bash security first (before splitting)
    security_result = bash_command_is_safe(command)
    if security_result.behavior != "passthrough":
        return PermissionResult.passthrough(
            "Command is not read-only, requires further permission checks"
        )

    # Check UNC paths (before transformation)
    if contains_vulnerable_unc_path(command):
        return PermissionResult.ask(
            "Command contains Windows UNC path that could be vulnerable to WebDAV attacks"
        )

    # Check for git commands
    has_git = command_has_any_git(command)

    # cd + git compound command check (sandbox escape prevention)
    if compound_command_has_cd and has_git:
        return PermissionResult.passthrough(
            "Compound commands with cd and git require permission checks for enhanced security"
        )

    # SECURITY: Block compound commands that write to git-internal paths AND run git
    if has_git and _command_writes_to_git_internal_paths(command):
        return PermissionResult.passthrough(
            "Compound commands that create git internal files and run git require permission checks"
        )

    # Check all subcommands are read-only
    all_read_only = all(
        bash_command_is_safe(subcmd).behavior == "passthrough"
        and is_command_read_only(subcmd)
        for subcmd in split_command(command)
    )

    if all_read_only:
        return PermissionResult.allow(
            updated_input={"command": command},
        )

    return PermissionResult.passthrough(
        "Command is not read-only, requires further permission checks"
    )


__all__ = [
    "check_read_only_constraints",
    "is_command_read_only",
    "is_command_safe_via_flag_parsing",
    "contains_unquoted_expansion",
    "command_has_any_git",
    "COMMAND_ALLOWLIST",
]
