"""Prompt suggestion helpers for the Rich interactive UI."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

from ripperdoc.core.query import query_llm
from ripperdoc.utils.message_formatting import extract_assistant_text, stringify_message_content
from ripperdoc.utils.messages import create_user_message

PROMPT_SUGGESTION_ENV = "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION"
_CACHE_COLD_THRESHOLD = 0.5
_MAX_TRANSCRIPT_CHARS = 4000
_MAX_TRANSCRIPT_MESSAGES = 12

SUGGESTION_SYSTEM_PROMPT = """[SUGGESTION MODE: Suggest what the user might naturally type next.]

FIRST: Look at the user's recent messages and original request.

Your job is to predict what THEY would type, not what you think they should do.

THE TEST: Would they think "I was just about to type that"?

EXAMPLES:
User asked "fix the bug and run tests", bug is fixed -> "run the tests"
After code written -> "try it out"
Assistant asks to continue -> "yes" or "go ahead"
Task complete, obvious follow-up -> "commit this" or "push it"
After error or misunderstanding -> silence

Be specific: "run the tests" beats "continue".

NEVER SUGGEST:
- Evaluative ("looks good", "thanks")
- Questions ("what about...?")
- Assistant-voice ("Let me...", "I'll...", "Here's...")
- New ideas they didn't ask about
- Multiple sentences

Stay silent if the next step isn't obvious from what the user said.

Format: 2-12 words, match the user's style. Or nothing.
Reply with ONLY the suggestion, no quotes or explanation."""


def resolve_prompt_suggestion_enabled(
    *,
    env_value: Optional[str],
    config_enabled: Optional[bool],
    interactive: bool,
) -> bool:
    """Resolve whether prompt suggestions should be enabled for this session."""
    normalized_env = (env_value or "").strip().lower()
    if normalized_env in {"0", "false", "off", "no"}:
        return False
    if normalized_env in {"1", "true", "on", "yes"}:
        return True
    if not interactive:
        return False
    return config_enabled is not False


def is_cache_cold(last_assistant_message: Any, *, threshold: float = _CACHE_COLD_THRESHOLD) -> bool:
    """Return True when cache creation dominates input tokens for the previous assistant turn."""
    try:
        input_tokens = int(getattr(last_assistant_message, "input_tokens", 0) or 0)
        cache_read_tokens = int(getattr(last_assistant_message, "cache_read_tokens", 0) or 0)
        cache_creation_tokens = int(
            getattr(last_assistant_message, "cache_creation_tokens", 0) or 0
        )
    except (TypeError, ValueError):
        return False

    total_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
    if total_tokens <= 0:
        return False
    return (cache_creation_tokens / total_tokens) > threshold


def build_suggestion_transcript(
    messages: Sequence[Any],
    *,
    max_messages: int = _MAX_TRANSCRIPT_MESSAGES,
    max_chars: int = _MAX_TRANSCRIPT_CHARS,
) -> str:
    """Build a compact `User:`/`Assistant:` transcript for suggestion generation."""
    rows: list[str] = []
    for message in messages:
        message_type = getattr(message, "type", "")
        if message_type not in {"user", "assistant"}:
            continue
        payload = getattr(message, "message", None)
        content = getattr(payload, "content", None) if payload else None
        text = stringify_message_content(content, include_tool_details=False).strip()
        if not text:
            continue
        # Tool-result-only user messages do not represent real user intent.
        if message_type == "user" and text == "[Tool result]":
            continue
        compact = " ".join(segment.strip() for segment in text.splitlines() if segment.strip())
        if not compact:
            continue
        if len(compact) > 400:
            compact = compact[:397].rstrip() + "..."
        role = "User" if message_type == "user" else "Assistant"
        rows.append(f"{role}: {compact}")

    if len(rows) > max_messages:
        rows = rows[-max_messages:]

    transcript = "\n\n".join(rows).strip()
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:].lstrip()
    return transcript


def normalize_generated_suggestion(raw_text: str) -> Optional[str]:
    """Normalize and validate model output for prompt suggestions."""
    text = (raw_text or "").strip()
    if not text:
        return None
    if "\n" in text:
        text = text.splitlines()[0].strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    if not text:
        return None
    if _is_suggestion_rejected(text):
        return None
    return text


def _is_suggestion_rejected(text: str) -> bool:
    lowered = text.lower()
    word_count = len(text.strip().split())

    if lowered == "done":
        return True
    if (
        lowered in {"nothing found", "nothing found."}
        or lowered.startswith("nothing to suggest")
        or lowered.startswith("no suggestion")
    ):
        return True
    if any(
        lowered.startswith(prefix)
        for prefix in (
            "api error:",
            "prompt is too long",
            "request timed out",
            "invalid api key",
            "image was too large",
        )
    ):
        return True
    if re.match(r"^\w+:\s", text):
        return True
    if word_count < 2 and not text.startswith("/"):
        if lowered not in {
            "yes",
            "yeah",
            "yep",
            "yea",
            "yup",
            "sure",
            "ok",
            "okay",
            "push",
            "commit",
            "deploy",
            "stop",
            "continue",
            "check",
            "exit",
            "quit",
            "no",
        }:
            return True
    if word_count > 12:
        return True
    if len(text) >= 100:
        return True
    if re.search(r"[.!?]\s+[A-Z]", text):
        return True
    if "\n" in text or "*" in text:
        return True
    if re.search(
        r"thanks|thank you|looks good|sounds good|that works|that worked|that's all|nice|great|"
        r"perfect|makes sense|awesome|excellent",
        lowered,
    ):
        return True
    if re.match(
        r"^(let me|i'll|i've|i'm|i can|i would|i think|i notice|here's|here is|here are|that's|"
        r"this is|this will|you can|you should|you could|sure,|of course|certainly)",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return False


async def generate_prompt_suggestion(
    *,
    messages: Sequence[Any],
    model: str,
    request_timeout_sec: float = 20.0,
    max_retries: int = 1,
) -> Optional[str]:
    """Generate a suggestion via a lightweight background LLM request."""
    transcript = build_suggestion_transcript(messages)
    if not transcript:
        return None

    suggestion_request = (
        "Recent conversation:\n\n"
        f"{transcript}\n\n"
        "Predict the next user input text only."
    )
    response = await query_llm(
        [create_user_message(suggestion_request)],
        SUGGESTION_SYSTEM_PROMPT,
        tools=[],
        max_thinking_tokens=0,
        model=model,
        stream=False,
        request_timeout=request_timeout_sec,
        max_retries=max_retries,
    )
    return normalize_generated_suggestion(extract_assistant_text(response))


def suggest_initial_prompt_from_git_history(project_path: Path) -> Optional[str]:
    """Generate a first-turn suggestion from recent git activity."""
    if not _is_git_repo(project_path):
        return None

    candidates = _collect_recent_paths(project_path)
    if not candidates:
        return None

    best = _shorten_path(candidates[0])
    return normalize_generated_suggestion(f"review changes in {best}")


def _is_git_repo(project_path: Path) -> bool:
    result = _run_git(project_path, "rev-parse", "--is-inside-work-tree")
    return result is not None and result.returncode == 0


def _collect_recent_paths(project_path: Path) -> list[str]:
    """Collect likely relevant paths from staged/unstaged files first, then git log."""
    paths: list[str] = []
    seen: set[str] = set()

    status = _run_git(project_path, "status", "--porcelain", "--untracked-files=no")
    if status and status.returncode == 0:
        for line in status.stdout.splitlines():
            if len(line) < 4:
                continue
            raw_path = line[3:].strip()
            if raw_path and raw_path not in seen:
                seen.add(raw_path)
                paths.append(raw_path)

    log = _run_git(project_path, "log", "--name-only", "--pretty=format:", "-n", "40")
    if log and log.returncode == 0:
        for line in log.stdout.splitlines():
            candidate = line.strip()
            if (
                not candidate
                or candidate.startswith(".")
                or candidate.startswith(" ")
                or candidate in seen
            ):
                continue
            if "/" not in candidate and "." not in candidate:
                continue
            seen.add(candidate)
            paths.append(candidate)
            if len(paths) >= 30:
                break
    return paths


def _shorten_path(path: str, max_len: int = 48) -> str:
    if len(path) <= max_len:
        return path
    filename = Path(path).name
    if len(filename) <= max_len:
        return filename
    return filename[: max_len - 3] + "..."


def _run_git(project_path: Path, *args: str) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return subprocess.run(
            ["git", "-C", str(project_path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.2,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
