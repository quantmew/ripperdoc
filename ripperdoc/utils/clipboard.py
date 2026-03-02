"""Clipboard helpers with SSH-aware OSC52 fallback.

Behavior mirrors Claude Code's clipboard routing:
- Prefer OSC52 over SSH when terminal likely supports it
- Use native clipboard commands when available
- Fall back between methods on failure and cache the chosen method
"""

from __future__ import annotations

import base64
import os
import platform
import shutil
import subprocess
import sys
from typing import Literal, Optional


ClipboardOS = Literal["macos", "linux", "wsl", "windows", "unknown"]
ClipboardMethod = Literal["osc52", "native", "none"]

_method_cache: Optional[ClipboardMethod] = None
_native_command_cache: Optional[list[str]] = None


def _is_ssh_session() -> bool:
    return bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"))


def _get_os() -> ClipboardOS:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    if "microsoft" in platform.release().lower() or os.environ.get("WSL_DISTRO_NAME"):
        return "wsl"
    if sys.platform.startswith("linux"):
        return "linux"
    return "unknown"


def _native_command_candidates() -> list[list[str]]:
    os_name = _get_os()
    by_os: dict[ClipboardOS, list[list[str]]] = {
        "macos": [["pbcopy"]],
        "linux": [["xclip", "-selection", "clipboard"], ["wl-copy"]],
        "wsl": [["clip.exe"]],
        "windows": [["clip"]],
        "unknown": [["xclip", "-selection", "clipboard"], ["wl-copy"]],
    }
    return by_os[os_name]


def _find_native_command() -> Optional[list[str]]:
    for cmd in _native_command_candidates():
        if shutil.which(cmd[0]) is not None:
            return cmd
    return None


def _terminal_supports_osc52() -> bool:
    if not sys.stdout.isatty():
        return False
    try:
        result = subprocess.run(
            ["tput", "Ms"],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )
        if "]52" in (result.stdout or ""):
            return True
    except (subprocess.SubprocessError, OSError):
        pass

    for key in ("ITERM_SESSION_ID", "WT_SESSION", "KONSOLE_VERSION"):
        if os.environ.get(key):
            return True
    return False


def _select_clipboard_method() -> ClipboardMethod:
    global _method_cache, _native_command_cache

    if _method_cache is not None:
        return _method_cache

    is_ssh = _is_ssh_session()
    has_osc52_terminal = _terminal_supports_osc52()
    _native_command_cache = _find_native_command()
    has_native = _native_command_cache is not None

    if is_ssh and has_osc52_terminal:
        _method_cache = "osc52"
    elif is_ssh and has_native:
        _method_cache = "native"
    elif is_ssh and sys.stdout.isatty():
        _method_cache = "osc52"
    elif (not is_ssh) and has_native:
        _method_cache = "native"
    elif has_osc52_terminal:
        _method_cache = "osc52"
    else:
        _method_cache = "none"

    return _method_cache


def _wrap_for_tmux_or_screen(escape_seq: str) -> str:
    if os.environ.get("TMUX"):
        return f"\x1bPtmux;{escape_seq.replace('\x1b', '\x1b\x1b')}\x1b\\"
    if os.environ.get("STY"):
        return f"\x1bP{escape_seq}\x1b\\"
    return escape_seq


def _copy_via_osc52(text: str) -> bool:
    global _method_cache

    if not sys.stdout.isatty():
        return False

    try:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        osc52 = f"\x1b]52;c;{encoded}\x07"
        wrapped = _wrap_for_tmux_or_screen(osc52)
        sys.stdout.write(wrapped)
        sys.stdout.flush()
        return True
    except Exception:
        _method_cache = "native" if _native_command_cache is not None else "none"
        return False


def _copy_via_native(text: str, command: list[str]) -> bool:
    global _method_cache

    try:
        subprocess.run(command, input=text, text=True, check=True)
        return True
    except (subprocess.SubprocessError, OSError):
        _method_cache = "osc52" if _terminal_supports_osc52() else "none"
        return False


def clipboard_failure_message() -> str:
    if _is_ssh_session():
        return (
            "Failed to copy to clipboard. Over SSH, clipboard access requires a terminal "
            "that supports OSC52 (iTerm2, Kitty, Ghostty, WezTerm, Alacritty, etc.). "
            "If using tmux, ensure `set-clipboard` is enabled and `allow-passthrough` is on."
        )

    os_name = _get_os()
    messages = {
        "macos": (
            "Failed to copy to clipboard. Make sure the `pbcopy` command is available "
            "on your system and try again."
        ),
        "windows": (
            "Failed to copy to clipboard. Make sure the `clip` command is available on your "
            "system and try again."
        ),
        "wsl": (
            "Failed to copy to clipboard. Make sure the `clip.exe` command is available in "
            "your WSL environment and try again."
        ),
        "linux": (
            "Failed to copy to clipboard. Make sure `xclip` or `wl-copy` is installed on your "
            "system and try again."
        ),
        "unknown": (
            "Failed to copy to clipboard. Make sure `xclip` or `wl-copy` is installed on your "
            "system and try again."
        ),
    }
    return messages[os_name]


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    method = _select_clipboard_method()

    if method == "osc52":
        if _copy_via_osc52(text):
            return True, ""
        return False, clipboard_failure_message()

    if method == "native":
        if _native_command_cache is not None and _copy_via_native(text, _native_command_cache):
            return True, ""
        return False, clipboard_failure_message()

    return False, clipboard_failure_message()

