"""Prompt helpers for interactive input."""

from getpass import getpass


def prompt_secret(prompt_text: str, prompt_suffix: str = ": ") -> str:
    """Prompt for sensitive input, masking characters when possible.

    Falls back to getpass (no echo) if prompt_toolkit is unavailable.
    """
    full_prompt = f"{prompt_text}{prompt_suffix}"
    try:
        from prompt_toolkit import prompt as pt_prompt

        return pt_prompt(full_prompt, is_password=True)
    except (ImportError, OSError, RuntimeError, EOFError):
        return getpass(full_prompt)
