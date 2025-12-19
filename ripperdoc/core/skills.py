"""Agent Skill loading helpers for Ripperdoc.

Skills are small capability bundles defined by SKILL.md files that live under
`~/.ripperdoc/skills` or `.ripperdoc/skills` in a project. Only the skill
metadata (name + description) should be added to the system prompt up front;
the full content is loaded on demand via the Skill tool. Optional frontmatter
fields include:
- allowed-tools: Comma-separated list of tools that are allowed/preferred.
- model: Model pointer hint for this skill.
- max-thinking-tokens: Reasoning budget hint for this skill.
- disable-model-invocation: If true, block the Skill tool from loading this
  skill.
- type: Skill kind (defaults to "prompt").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from ripperdoc.utils.coerce import parse_boolish, parse_optional_int
from ripperdoc.utils.log import get_logger

logger = get_logger()

SKILL_DIR_NAME = "skills"
SKILL_FILE_NAME = "SKILL.md"
_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")


class SkillLocation(str, Enum):
    """Where a skill definition is sourced from."""

    USER = "user"
    PROJECT = "project"
    OTHER = "other"


@dataclass
class SkillDefinition:
    """Parsed representation of a skill."""

    name: str
    description: str
    content: str
    path: Path
    base_dir: Path
    location: SkillLocation
    allowed_tools: List[str]
    model: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    skill_type: str = "prompt"
    disable_model_invocation: bool = False


@dataclass
class SkillLoadError:
    """Error encountered while loading a skill file."""

    path: Path
    reason: str


@dataclass
class SkillLoadResult:
    """Aggregated result of loading skills."""

    skills: List[SkillDefinition]
    errors: List[SkillLoadError]


def _split_frontmatter(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and body content from a markdown file."""
    lines = raw_text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                frontmatter_text = "\n".join(lines[1:idx])
                body = "\n".join(lines[idx + 1 :])
                try:
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                except (
                    yaml.YAMLError,
                    ValueError,
                    TypeError,
                ) as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "[skills] Invalid frontmatter in SKILL.md: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    return {"__error__": f"Invalid frontmatter: {exc}"}, body
                return frontmatter, body
    return {}, raw_text


def _normalize_allowed_tools(value: object) -> List[str]:
    """Normalize allowed-tools values to a clean list of tool names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        tools: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                tools.append(item.strip())
        return tools
    return []


def _load_skill_file(
    path: Path, location: SkillLocation
) -> Tuple[Optional[SkillDefinition], Optional[SkillLoadError]]:
    """Parse a single SKILL.md file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, IOError, UnicodeDecodeError) as exc:
        logger.warning(
            "[skills] Failed to read skill file: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return None, SkillLoadError(path=path, reason=f"Failed to read file: {exc}")

    frontmatter, body = _split_frontmatter(text)
    if "__error__" in frontmatter:
        return None, SkillLoadError(path=path, reason=str(frontmatter["__error__"]))

    raw_name = frontmatter.get("name")
    raw_description = frontmatter.get("description")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, SkillLoadError(path=path, reason='Missing required "name" field')
    if not _SKILL_NAME_RE.match(raw_name.strip()):
        return None, SkillLoadError(
            path=path,
            reason='Invalid "name" format. Use lowercase letters, numbers, and hyphens only (max 64 chars).',
        )
    if not isinstance(raw_description, str) or not raw_description.strip():
        return None, SkillLoadError(path=path, reason='Missing required "description" field')

    allowed_tools = _normalize_allowed_tools(
        frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    )
    model_value = frontmatter.get("model")
    model = model_value if isinstance(model_value, str) and model_value.strip() else None
    max_thinking_tokens = parse_optional_int(
        frontmatter.get("max-thinking-tokens") or frontmatter.get("max_thinking_tokens")
    )
    raw_type = (
        frontmatter.get("type")
        or frontmatter.get("skill-type")
        or frontmatter.get("skill_type")
        or "prompt"
    )
    skill_type = str(raw_type).strip().lower() if isinstance(raw_type, str) else "prompt"
    disable_model_invocation = parse_boolish(
        frontmatter.get("disable-model-invocation") or frontmatter.get("disable_model_invocation")
    )

    skill = SkillDefinition(
        name=raw_name.strip(),
        description=raw_description.strip(),
        content=body.strip(),
        path=path,
        base_dir=path.parent,
        location=location,
        allowed_tools=allowed_tools,
        model=model,
        max_thinking_tokens=max_thinking_tokens,
        skill_type=skill_type or "prompt",
        disable_model_invocation=disable_model_invocation,
    )
    return skill, None


def _load_skill_dir(
    path: Path, location: SkillLocation
) -> Tuple[List[SkillDefinition], List[SkillLoadError]]:
    """Load skills from a directory that either contains SKILL.md or subdirectories."""
    skills: List[SkillDefinition] = []
    errors: List[SkillLoadError] = []
    if not path.exists() or not path.is_dir():
        return skills, errors

    single_skill = path / SKILL_FILE_NAME
    if single_skill.exists():
        skill, error = _load_skill_file(single_skill, location)
        if skill:
            skills.append(skill)
        elif error:
            errors.append(error)
        return skills, errors

    for entry in sorted(path.iterdir()):
        try:
            if not entry.is_dir() and not entry.is_symlink():
                continue
        except OSError:
            continue

        candidate = entry / SKILL_FILE_NAME
        if not candidate.exists():
            continue
        skill, error = _load_skill_file(candidate, location)
        if skill:
            skills.append(skill)
        elif error:
            errors.append(error)
    return skills, errors


def skill_directories(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> List[Tuple[Path, SkillLocation]]:
    """Return the standard skill directories for user and project scopes."""
    home_dir = (home or Path.home()).expanduser()
    project_dir = (project_path or Path.cwd()).resolve()
    return [
        (home_dir / ".ripperdoc" / SKILL_DIR_NAME, SkillLocation.USER),
        (project_dir / ".ripperdoc" / SKILL_DIR_NAME, SkillLocation.PROJECT),
    ]


def load_all_skills(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> SkillLoadResult:
    """Load skills from user and project directories.

    Project skills override user skills with the same name.
    """
    skills_by_name: Dict[str, SkillDefinition] = {}
    errors: List[SkillLoadError] = []

    # Load user first so project overrides take precedence.
    for directory, location in skill_directories(project_path=project_path, home=home):
        loaded, dir_errors = _load_skill_dir(directory, location)
        errors.extend(dir_errors)
        for skill in loaded:
            if skill.name in skills_by_name:
                logger.debug(
                    "[skills] Overriding skill",
                    extra={
                        "skill_name": skill.name,
                        "previous_location": str(skills_by_name[skill.name].location),
                        "new_location": str(location),
                    },
                )
            skills_by_name[skill.name] = skill
    return SkillLoadResult(skills=list(skills_by_name.values()), errors=errors)


def find_skill(
    skill_name: str, project_path: Optional[Path] = None, home: Optional[Path] = None
) -> Optional[SkillDefinition]:
    """Find a skill by name (case-sensitive match)."""
    normalized = skill_name.strip().lstrip("/")
    if not normalized:
        return None
    result = load_all_skills(project_path=project_path, home=home)
    return next((skill for skill in result.skills if skill.name == normalized), None)


def build_skill_summary(skills: Sequence[SkillDefinition]) -> str:
    """Render a concise instruction block listing available skills."""
    if not skills:
        return (
            "# Skills\n"
            "No skills detected. Add SKILL.md under ~/.ripperdoc/skills or ./.ripperdoc/skills "
            "to extend capabilities, then load them with the Skill tool when relevant."
        )
    lines = [
        "# Skills",
        "Skills extend your capabilities with reusable instructions stored in SKILL.md files.",
        'Call the Skill tool with {"skill": "<name>"} to load a skill when it matches the user request.',
        "Available skills:",
    ]
    for skill in skills:
        location = f" ({skill.location.value})" if skill.location else ""
        lines.append(f"- {skill.name}{location}: {skill.description}")
    return "\n".join(lines)


__all__ = [
    "SkillDefinition",
    "SkillLoadError",
    "SkillLoadResult",
    "SkillLocation",
    "SKILL_DIR_NAME",
    "SKILL_FILE_NAME",
    "load_all_skills",
    "find_skill",
    "build_skill_summary",
    "skill_directories",
]
