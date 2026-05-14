"""Secret detection utilities for blocking sensitive content writes."""

from __future__ import annotations

import re
from typing import Optional

# Patterns for common secret formats
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key ID", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("AWS Secret Access Key", re.compile(r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key[^\S\n]*[=:]\s*[A-Za-z0-9/+=]{40}")),
    ("GitHub Token", re.compile(r"gh[ps]_[A-Za-z0-9_]{36,}")),
    ("GitHub OAuth", re.compile(r"gho_[A-Za-z0-9]{36}")),
    ("Private Key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("Slack Token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("Stripe Key", re.compile(r"(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,}")),
    ("Generic API Key", re.compile(r"(?i)(?:api[_\-]?key|apikey|secret[_\-]?key|auth[_\-]?token|access[_\-]?token)[^\S\n]*[=:]\s*['\"]?[A-Za-z0-9\-_.]{20,}['\"]?")),
    ("Bearer Token", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*")),
]


def detect_secrets(content: str) -> Optional[str]:
    """Scan content for common secret patterns.

    Returns a human-readable description of the first secret found,
    or None if no secrets were detected.
    """
    for name, pattern in _SECRET_PATTERNS:
        match = pattern.search(content)
        if match:
            return f"Potential {name} detected in content"
    return None
