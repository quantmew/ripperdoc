"""Sandbox helpers.

The reference uses macOS sandbox-exec profiles; in this environment we
surface the same API surface but report unavailability by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import shlex
from typing import Callable


@dataclass
class SandboxWrapper:
    """Represents a wrapped command plus a cleanup callback."""

    final_command: str
    cleanup: Callable[[], None]


def is_sandbox_available() -> bool:
    """Return whether sandboxed execution is available on this host."""
    return shutil.which("srt") is not None


def create_sandbox_wrapper(command: str) -> SandboxWrapper:
    """Wrap a command for sandboxed execution or raise if unsupported."""
    if not is_sandbox_available():
        raise RuntimeError(
            "Sandbox mode requested but not available (install @anthropic-ai/sandbox-runtime and ensure 'srt' is on PATH)"
        )
    quoted = shlex.quote(command)
    return SandboxWrapper(final_command=f"srt {quoted}", cleanup=lambda: None)


__all__ = ["SandboxWrapper", "is_sandbox_available", "create_sandbox_wrapper"]
