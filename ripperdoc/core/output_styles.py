"""Output style definitions and loading for Ripperdoc."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ripperdoc.utils.log import get_logger

logger = get_logger()

OUTPUT_STYLE_DIR_NAME = "output-styles"
OUTPUT_STYLE_FILE_SUFFIX = ".md"


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


_BUILTIN_OUTPUT_STYLES: tuple[OutputStyleDefinition, ...] = (
    OutputStyleDefinition(
        key="default",
        name="Default",
        description="Complete coding tasks efficiently with concise responses.",
        instructions=dedent(
            """\
            # Output Style: Default
            Stay concise and execution-focused while completing software engineering tasks."""
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
            While solving tasks, include concise "Insight" notes that explain:
            - Why a solution was chosen over alternatives.
            - How the touched code fits existing project patterns.
            - What tradeoffs or constraints influenced decisions.
            Keep insights practical and tied to concrete code changes."""
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
            Use a learn-by-doing workflow:
            - Share concise "Insight" notes about implementation decisions and patterns.
            - Pause at strategic points and ask the user to implement small code changes.
            - Insert TODO(human) markers where the user should contribute code.
            - Resume and integrate the user's contributions after they respond."""
        ).strip(),
        include_efficiency_instructions=False,
        keep_coding_instructions=True,
        location=OutputStyleLocation.BUILTIN,
    ),
)


def builtin_output_styles() -> list[OutputStyleDefinition]:
    """Return built-in output styles in fixed display order."""
    return list(_BUILTIN_OUTPUT_STYLES)


def output_style_directories(
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> List[Tuple[Path, OutputStyleLocation]]:
    """Return user/project output-style directories in precedence order."""
    home_dir = (home or Path.home()).expanduser()
    project_dir = (project_path or Path.cwd()).resolve()
    return [
        (home_dir / ".ripperdoc" / OUTPUT_STYLE_DIR_NAME, OutputStyleLocation.USER),
        (project_dir / ".ripperdoc" / OUTPUT_STYLE_DIR_NAME, OutputStyleLocation.PROJECT),
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
) -> tuple[OutputStyleDefinition | None, OutputStyleLoadError | None]:
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
) -> tuple[list[OutputStyleDefinition], list[OutputStyleLoadError]]:
    styles: list[OutputStyleDefinition] = []
    errors: list[OutputStyleLoadError] = []
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
    errors: list[OutputStyleLoadError] = []

    for directory, location in output_style_directories(project_path=project_path, home=home):
        loaded, load_errors = _load_styles_from_dir(directory, location)
        errors.extend(load_errors)
        for style in loaded:
            styles_by_key[style.key] = style

    ordered: list[OutputStyleDefinition] = list(_BUILTIN_OUTPUT_STYLES)
    builtin_keys = {style.key for style in _BUILTIN_OUTPUT_STYLES}
    custom_styles = [style for key, style in styles_by_key.items() if key not in builtin_keys]
    ordered.extend(sorted(custom_styles, key=lambda style: (style.name.lower(), style.key)))

    return OutputStyleLoadResult(styles=ordered, errors=errors)


def find_output_style(
    style_name: str,
    *,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> tuple[OutputStyleDefinition | None, OutputStyleLoadResult]:
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
) -> tuple[OutputStyleDefinition, OutputStyleLoadResult]:
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
