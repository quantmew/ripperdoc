"""Heredoc extraction and restoration utilities.

Extracts heredocs before parsing to avoid shell-quote bugs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class HeredocInfo:
    """Information about an extracted heredoc."""
    delimiter: str
    body: str
    start: int
    end: int
    is_dash: bool  # True if <<- (tabs stripped)
    is_quoted: bool  # True if <<'EOF' or <<"EOF" (no expansion in body)
    placeholder: str


# Pattern to match heredoc operators: <<EOF, <<'EOF', <<"EOF", <<-EOF, <<-'EOF'
_HEREDOC_OPEN_RE = re.compile(
    r"""<<(-)?               # << or <<-
    \s*                     # optional whitespace
    (?:                     # delimiter variations
        '([A-Za-z_]\w*)'   |  # single-quoted: 'EOF'
        "([A-Za-z_]\w*)"   |  # double-quoted: "EOF"
        ([A-Za-z_]\w*)        # unquoted: EOF
    )""",
    re.VERBOSE,
)


def extract_heredocs(command: str) -> Tuple[str, List[HeredocInfo]]:
    """Extract heredocs from a command, replacing them with placeholders.

    This must be done before shell-quote parsing because shell-quote
    handles << incorrectly.

    Args:
        command: The command string possibly containing heredocs.

    Returns:
        Tuple of (processed_command, list_of_heredoc_info).
    """
    heredocs: List[HeredocInfo] = []
    result = command

    for i, match in enumerate(reversed(list(_HEREDOC_OPEN_RE.finditer(command)))):
        # Adjust index since we're iterating in reverse
        actual_start = len(command) - match.end()
        is_dash = match.group(1) == "-"
        delimiter = match.group(2) or match.group(3) or match.group(4) or ""
        is_quoted = match.group(2) is not None or match.group(3) is not None

        if not delimiter:
            continue

        # Find the closing delimiter
        body_start = match.end()
        rest = command[body_start:]
        lines = rest.split("\n")

        closing_line_idx = -1
        for j, line in enumerate(lines):
            check = line
            if is_dash:
                check = line.lstrip("\t")
            if check.strip() == delimiter:
                closing_line_idx = j
                break

        if closing_line_idx == -1:
            continue

        # Calculate positions in original string (reverse-order safe)
        body_end = body_start + sum(len(lines[k]) + 1 for k in range(closing_line_idx))
        heredoc_end = body_end + len(lines[closing_line_idx]) + 1

        # Extract the heredoc body (without the closing delimiter line)
        body_lines = lines[:closing_line_idx]
        body = "\n".join(body_lines)

        placeholder = f"__HEREDOC_{i}__"
        heredocs.append(HeredocInfo(
            delimiter=delimiter,
            body=body,
            start=actual_start,
            end=heredoc_end,
            is_dash=is_dash,
            is_quoted=is_quoted,
            placeholder=placeholder,
        ))

    # Apply replacements in reverse order to preserve offsets
    for h in reversed(heredocs):
        result = result[:h.start] + h.placeholder + result[h.end:]

    return result, heredocs


def restore_heredocs(command: str, heredocs: List[HeredocInfo]) -> str:
    """Restore heredocs from their placeholders back into the command.

    Args:
        command: The command string with heredoc placeholders.
        heredocs: The list of extracted HeredocInfo.

    Returns:
        The command with heredocs restored.
    """
    result = command
    for h in heredocs:
        heredoc_text = f"<<{'-' if h.is_dash else ''}{h.delimiter}\n{h.body}\n{h.delimiter}"
        result = result.replace(h.placeholder, heredoc_text, 1)
    return result


__all__ = ["HeredocInfo", "extract_heredocs", "restore_heredocs"]
