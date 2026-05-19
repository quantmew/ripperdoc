"""Output style definitions and loading for Ripperdoc."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ripperdoc.utils.filesystem.config_paths import project_config_dir, user_config_dir
from ripperdoc.utils.log import get_logger

logger = get_logger()

from ripperdoc.constants.output_styles import OUTPUT_STYLE_DIR_NAME, OUTPUT_STYLE_FILE_SUFFIX  # noqa: E402


class OutputStyleLocation(str, Enum):
    """Where an output style is sourced from."""

    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True)
class OutputStyleDefinition:
    """Resolved output style metadata and instructions."""

    key: str
    name: str
    description: str
    instructions: str
    include_efficiency_instructions: bool
    keep_coding_instructions: bool
    location: OutputStyleLocation
    path: Optional[Path] = None

    @property
    def is_custom(self) -> bool:
        return self.location in {OutputStyleLocation.USER, OutputStyleLocation.PROJECT}


@dataclass(frozen=True)
class OutputStyleLoadError:
    """Error encountered while loading an output style file."""

    path: Path
    reason: str


@dataclass(frozen=True)
class OutputStyleLoadResult:
    """Loaded output styles and non-fatal loader errors."""

    styles: List[OutputStyleDefinition]
    errors: List[OutputStyleLoadError]

    def by_key(self) -> Dict[str, OutputStyleDefinition]:
        return {style.key: style for style in self.styles}


_BUILTIN_OUTPUT_STYLES: Tuple[OutputStyleDefinition, ...] = (
    OutputStyleDefinition(
        key="default",
        name="Default",
        description="Complete coding tasks efficiently with concise responses.",
        instructions=dedent(
            """\
            # Output Style: Default
            Stay concise and execution-focused while completing software engineering tasks.

            ## Output efficiency
            IMPORTANT: Go straight to the point. Try the simplest approach first without going in circles. Do not overdo it. Be extra concise.

            Keep your text output brief and direct. Lead with the answer or action, not the reasoning. Skip filler words, preamble, and unnecessary transitions. Do not restate what the user said - just do it. When explaining, include only what is necessary for the user to understand.

            Focus text output on:
            - Decisions that need the user's input
            - High-level status updates at natural milestones
            - Errors or blockers that change the plan

            If you can say it in one sentence, don't use three. Prefer short, direct sentences over long explanations. This does not apply to code or tool calls.

            ## Code References
            When referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location."""
        ).strip(),
        include_efficiency_instructions=True,
        keep_coding_instructions=True,
        location=OutputStyleLocation.BUILTIN,
    ),
    OutputStyleDefinition(
        key="explanatory",
        name="Explanatory",
        description="Explain implementation choices and codebase patterns while coding.",
        instructions=dedent(
            """\
            # Output Style: Explanatory
            In addition to completing tasks, provide educational insights about the codebase along the way.

            You should be clear and educational, providing helpful explanations while remaining focused on the task. Balance educational content with task completion. When providing insights, you may exceed typical length constraints, but remain focused and relevant.

            ## Insights
            In order to encourage learning, before and after writing code, always provide brief educational explanations about implementation choices:
            - Why a solution was chosen over alternatives.
            - How the touched code fits existing project patterns.
            - What tradeoffs or constraints influenced decisions.

            These insights should be included in the conversation, not in the codebase. Focus on interesting insights that are specific to the codebase or the code you just wrote, rather than general programming concepts."""
        ).strip(),
        include_efficiency_instructions=False,
        keep_coding_instructions=True,
        location=OutputStyleLocation.BUILTIN,
    ),
    OutputStyleDefinition(
        key="learning",
        name="Learning",
        description="Collaborative, hands-on mode with guided user coding practice.",
        instructions=dedent(
            """\
            # Output Style: Learning
            Use a learn-by-doing workflow. Be collaborative and encouraging. Balance task completion with learning by requesting user input for meaningful design decisions while handling routine implementation yourself.

            ## Insights
            In order to encourage learning, before and after writing code, always provide brief educational explanations about implementation choices:
            - Why a solution was chosen over alternatives.
            - How the touched code fits existing project patterns.
            - What tradeoffs or constraints influenced decisions.

            ## Requesting Human Contributions
            In order to encourage learning, ask the human to contribute code pieces when generating significant portions involving:
            - Design decisions (error handling, data structures)
            - Business logic with multiple valid approaches
            - Key algorithms or interface definitions

            ### Request Format
            - **Context**: what's built and why this decision matters
            - **Your Task**: specific function/section in file
            - **Guidance**: trade-offs and constraints to consider

            ### Key Guidelines
            - Frame contributions as valuable design decisions, not busy work
            - Insert a TODO(human) section into the codebase before making the request
            - Don't take any action or output anything after the request. Wait for human implementation.
            - After contributions, share one insight connecting their code to broader patterns."""
        ).strip(),
        include_efficiency_instructions=False,
        keep_coding_instructions=True,
        location=OutputStyleLocation.BUILTIN,
    ),
)


def builtin_output_styles() -> List[OutputStyleDefinition]:
    """Return built-in output styles in fixed display order."""
    return list(_BUILTIN_OUTPUT_STYLES)


def output_style_directories(
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> List[Tuple[Path, OutputStyleLocation]]:
    """Return user/project output-style directories in precedence order."""
    home_dir = user_config_dir(home=home)
    project_dir = project_config_dir(project_path)
    return [
        (home_dir / OUTPUT_STYLE_DIR_NAME, OutputStyleLocation.USER),
        (project_dir / OUTPUT_STYLE_DIR_NAME, OutputStyleLocation.PROJECT),
    ]


def _normalize_style_key(raw: str) -> str:
    key = raw.strip().lower()
    if not key:
        return ""
    translated = []
    for char in key:
        if char.isalnum() or char in {"-", "_", ":"}:
            translated.append(char)
        elif char in {" ", "/", "\\"}:
            translated.append("-")
    collapsed = "".join(translated).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed


def _split_frontmatter(raw_text: str) -> Tuple[Dict[str, Any], str]:
    lines = raw_text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                frontmatter_text = "\n".join(lines[1:idx])
                body = "\n".join(lines[idx + 1 :])
                try:
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                except (yaml.YAMLError, ValueError, TypeError) as exc:
                    return {"__error__": f"Invalid frontmatter: {exc}"}, body
                return frontmatter, body
    return {}, raw_text


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _derive_style_key(path: Path, base_dir: Path) -> str:
    relative = path.relative_to(base_dir)
    parts = list(relative.parts)
    if parts:
        parts[-1] = parts[-1].removesuffix(OUTPUT_STYLE_FILE_SUFFIX)
    return _normalize_style_key(":".join(parts))


def _load_style_file(
    path: Path,
    location: OutputStyleLocation,
    base_dir: Path,
) -> Tuple[Optional[OutputStyleDefinition], Optional[OutputStyleLoadError]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, IOError, UnicodeDecodeError) as exc:
        logger.warning(
            "[output_styles] Failed to read style file: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return None, OutputStyleLoadError(path=path, reason=f"Failed to read file: {exc}")

    frontmatter, body = _split_frontmatter(text)
    if "__error__" in frontmatter:
        return None, OutputStyleLoadError(path=path, reason=str(frontmatter["__error__"]))

    style_key = _derive_style_key(path, base_dir)
    if not style_key:
        return None, OutputStyleLoadError(path=path, reason="Style key resolved to empty string")

    raw_name = frontmatter.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        display_name = raw_name.strip()
    else:
        display_name = path.stem.replace("-", " ").replace("_", " ").strip().title() or style_key

    raw_description = frontmatter.get("description")
    description = ""
    if isinstance(raw_description, str):
        description = raw_description.strip()
    elif raw_description is not None:
        description = str(raw_description).strip()

    keep_coding_instructions = _coerce_bool(
        frontmatter.get("keep-coding-instructions")
        if "keep-coding-instructions" in frontmatter
        else frontmatter.get("keep_coding_instructions"),
        False,
    )

    instructions = body.strip()
    if not instructions:
        return None, OutputStyleLoadError(path=path, reason="Style instructions cannot be empty")

    style = OutputStyleDefinition(
        key=style_key,
        name=display_name,
        description=description,
        instructions=instructions,
        include_efficiency_instructions=False,
        keep_coding_instructions=keep_coding_instructions,
        location=location,
        path=path,
    )
    return style, None


def _load_styles_from_dir(
    styles_dir: Path,
    location: OutputStyleLocation,
) -> Tuple[List[OutputStyleDefinition], List[OutputStyleLoadError]]:
    styles: List[OutputStyleDefinition] = []
    errors: List[OutputStyleLoadError] = []
    if not styles_dir.exists() or not styles_dir.is_dir():
        return styles, errors

    try:
        for style_file in styles_dir.rglob(f"*{OUTPUT_STYLE_FILE_SUFFIX}"):
            if not style_file.is_file():
                continue
            style, error = _load_style_file(style_file, location, styles_dir)
            if style:
                styles.append(style)
            elif error:
                errors.append(error)
    except OSError as exc:
        errors.append(OutputStyleLoadError(path=styles_dir, reason=f"Failed to scan directory: {exc}"))
    return styles, errors


def load_all_output_styles(
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> OutputStyleLoadResult:
    """Load built-in and custom output styles.

    Precedence:
    1) Built-in styles
    2) User styles (~/.ripperdoc/output-styles)
    3) Project styles (.ripperdoc/output-styles), overriding same-key user styles
    """

    styles_by_key: Dict[str, OutputStyleDefinition] = {s.key: s for s in _BUILTIN_OUTPUT_STYLES}
    errors: List[OutputStyleLoadError] = []

    for directory, location in output_style_directories(project_path=project_path, home=home):
        loaded, load_errors = _load_styles_from_dir(directory, location)
        errors.extend(load_errors)
        for style in loaded:
            styles_by_key[style.key] = style

    ordered: List[OutputStyleDefinition] = list(_BUILTIN_OUTPUT_STYLES)
    builtin_keys = {style.key for style in _BUILTIN_OUTPUT_STYLES}
    custom_styles = [style for key, style in styles_by_key.items() if key not in builtin_keys]
    ordered.extend(sorted(custom_styles, key=lambda style: (style.name.lower(), style.key)))

    return OutputStyleLoadResult(styles=ordered, errors=errors)


def find_output_style(
    style_name: str,
    *,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Tuple[Optional[OutputStyleDefinition], OutputStyleLoadResult]:
    """Find an output style by key or display name."""
    result = load_all_output_styles(project_path=project_path, home=home)
    candidate = _normalize_style_key(style_name)
    if not candidate:
        return None, result

    for style in result.styles:
        if style.key == candidate:
            return style, result

    for style in result.styles:
        normalized_name = _normalize_style_key(style.name)
        if normalized_name == candidate:
            return style, result
    return None, result


def resolve_output_style(
    style_name: Optional[str],
    *,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Tuple[OutputStyleDefinition, OutputStyleLoadResult]:
    """Resolve a style name and fall back to built-in default when needed."""
    result = load_all_output_styles(project_path=project_path, home=home)
    if isinstance(style_name, str):
        candidate = _normalize_style_key(style_name)
        if candidate:
            for style in result.styles:
                if style.key == candidate:
                    return style, result
            for style in result.styles:
                if _normalize_style_key(style.name) == candidate:
                    return style, result

    default_style = next((style for style in result.styles if style.key == "default"), None)
    if default_style is None:
        default_style = _BUILTIN_OUTPUT_STYLES[0]
    return default_style, result


def style_adherence_reminder(style: OutputStyleDefinition) -> str:
    """Reminder block appended to every prompt to keep the selected style active."""
    return dedent(
        f"""\
        # Output style reminder
        Active output style: {style.name} ({style.key}).
        Follow this style for every response in this conversation."""
    ).strip()


__all__ = [
    "OUTPUT_STYLE_DIR_NAME",
    "OUTPUT_STYLE_FILE_SUFFIX",
    "OutputStyleLocation",
    "OutputStyleDefinition",
    "OutputStyleLoadError",
    "OutputStyleLoadResult",
    "builtin_output_styles",
    "output_style_directories",
    "load_all_output_styles",
    "find_output_style",
    "resolve_output_style",
    "style_adherence_reminder",
]
