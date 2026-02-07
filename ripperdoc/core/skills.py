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
- hooks: Hook definitions (same format as hooks.json) scoped to this skill.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from ripperdoc.core.hooks.config import HooksConfig, parse_hooks_config
from ripperdoc.core.plugins import discover_plugins
from ripperdoc.utils.coerce import parse_boolish, parse_optional_int
from ripperdoc.utils.log import get_logger

logger = get_logger()

SKILL_DIR_NAME = "skills"
SKILL_FILE_NAME = "SKILL.md"
SKILL_STATE_FILE_NAME = ".skills_state.json"
_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")


class SkillLocation(str, Enum):
    """Where a skill definition is sourced from."""

    USER = "user"
    PROJECT = "project"
    PLUGIN = "plugin"
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
    hooks: HooksConfig = field(default_factory=HooksConfig)
    plugin_name: Optional[str] = None


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
    path: Path,
    location: SkillLocation,
    *,
    default_name: Optional[str] = None,
    namespace_prefix: Optional[str] = None,
    plugin_name: Optional[str] = None,
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
    resolved_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    if resolved_name is None and default_name:
        resolved_name = default_name.strip()
    if not resolved_name:
        return None, SkillLoadError(path=path, reason='Missing required "name" field')
    if not _SKILL_NAME_RE.match(resolved_name):
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
    hooks = parse_hooks_config(frontmatter.get("hooks"), source=f"skill:{resolved_name}")
    full_name = f"{namespace_prefix}:{resolved_name}" if namespace_prefix else resolved_name

    skill = SkillDefinition(
        name=full_name,
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
        hooks=hooks,
        plugin_name=plugin_name,
    )
    return skill, None


def _load_skill_dir(
    path: Path,
    location: SkillLocation,
    *,
    namespace_prefix: Optional[str] = None,
    plugin_name: Optional[str] = None,
) -> Tuple[List[SkillDefinition], List[SkillLoadError]]:
    """Load skills from a directory that either contains SKILL.md or subdirectories."""
    skills: List[SkillDefinition] = []
    errors: List[SkillLoadError] = []
    if not path.exists() or not path.is_dir():
        return skills, errors

    single_skill = path / SKILL_FILE_NAME
    if single_skill.exists():
        skill, error = _load_skill_file(
            single_skill,
            location,
            default_name=path.name,
            namespace_prefix=namespace_prefix,
            plugin_name=plugin_name,
        )
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
        skill, error = _load_skill_file(
            candidate,
            location,
            default_name=entry.name,
            namespace_prefix=namespace_prefix,
            plugin_name=plugin_name,
        )
        if skill:
            skills.append(skill)
        elif error:
            errors.append(error)
    return skills, errors


def _load_skill_path(
    path: Path,
    location: SkillLocation,
    *,
    namespace_prefix: Optional[str] = None,
    plugin_name: Optional[str] = None,
) -> Tuple[List[SkillDefinition], List[SkillLoadError]]:
    if path.is_file():
        if path.name != SKILL_FILE_NAME:
            return [], []
        skill, error = _load_skill_file(
            path,
            location,
            default_name=path.parent.name,
            namespace_prefix=namespace_prefix,
            plugin_name=plugin_name,
        )
        if skill:
            return [skill], []
        if error:
            return [], [error]
        return [], []
    if path.is_dir():
        return _load_skill_dir(
            path,
            location,
            namespace_prefix=namespace_prefix,
            plugin_name=plugin_name,
        )
    return [], []


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

    plugin_result = discover_plugins(project_path=project_path, home=home)
    for plugin_error in plugin_result.errors:
        errors.append(SkillLoadError(path=plugin_error.path, reason=plugin_error.reason))
    for plugin in plugin_result.plugins:
        for skills_path in plugin.skills_paths:
            loaded, dir_errors = _load_skill_path(
                skills_path,
                SkillLocation.PLUGIN,
                namespace_prefix=plugin.name,
                plugin_name=plugin.name,
            )
            errors.extend(dir_errors)
            for skill in loaded:
                skills_by_name[skill.name] = skill
    return SkillLoadResult(skills=list(skills_by_name.values()), errors=errors)


def _normalize_skill_ref(skill_name: str) -> str:
    return skill_name.strip().lstrip("/")


def _normalize_disabled_skill_names(skill_names: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for raw in skill_names:
        if not isinstance(raw, str):
            continue
        name = _normalize_skill_ref(raw)
        if name and _SKILL_NAME_RE.match(name):
            normalized.add(name)
    return normalized


def _state_file_for_location(
    location: SkillLocation, project_path: Optional[Path] = None, home: Optional[Path] = None
) -> Optional[Path]:
    for directory, loc in skill_directories(project_path=project_path, home=home):
        if loc == location:
            return directory / SKILL_STATE_FILE_NAME
    return None


def _load_disabled_from_state_file(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return set()
    if not isinstance(raw, dict):
        return set()
    raw_values = raw.get("disabled_skills")
    if not isinstance(raw_values, list):
        return set()
    return _normalize_disabled_skill_names(raw_values)


def _save_disabled_to_state_file(path: Optional[Path], disabled_skill_names: set[str]) -> list[str]:
    normalized = sorted(_normalize_disabled_skill_names(disabled_skill_names))
    if path is None:
        return normalized
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"disabled_skills": normalized}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


def get_disabled_skill_names(
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
    *,
    location: Optional[SkillLocation] = None,
) -> set[str]:
    """Return disabled skill names from state files under skills directories."""
    if location is not None:
        state_path = _state_file_for_location(location, project_path=project_path, home=home)
        disabled = _load_disabled_from_state_file(state_path)
        return disabled

    disabled: set[str] = set()
    for _, loc in skill_directories(project_path=project_path, home=home):
        disabled.update(
            get_disabled_skill_names(project_path=project_path, home=home, location=loc)
        )
    return disabled


def save_disabled_skill_names(
    skill_names: Iterable[str],
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
    *,
    location: SkillLocation = SkillLocation.PROJECT,
) -> list[str]:
    """Persist disabled skill names into the state file for a specific scope."""
    state_path = _state_file_for_location(location, project_path=project_path, home=home)
    return _save_disabled_to_state_file(state_path, set(skill_names))


def is_skill_definition_disabled(
    skill: SkillDefinition, project_path: Optional[Path] = None, home: Optional[Path] = None
) -> bool:
    """Check whether a specific skill definition is disabled in its source scope."""
    disabled = get_disabled_skill_names(
        project_path=project_path, home=home, location=skill.location
    )
    return skill.name in disabled


def set_skill_enabled(
    skill: SkillDefinition,
    enabled: bool,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> bool:
    """Enable or disable a skill in the state file for that skill's source scope."""
    if skill.location not in (SkillLocation.USER, SkillLocation.PROJECT):
        return False
    disabled = get_disabled_skill_names(
        project_path=project_path, home=home, location=skill.location
    )
    was_disabled = skill.name in disabled
    if enabled and not was_disabled:
        return False
    if not enabled and was_disabled:
        return False
    if enabled:
        disabled.discard(skill.name)
    else:
        disabled.add(skill.name)
    save_disabled_skill_names(
        disabled, project_path=project_path, home=home, location=skill.location
    )
    return True


def filter_enabled_skills(
    skills: Sequence[SkillDefinition],
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
    disabled_skill_names: Optional[Iterable[str]] = None,
) -> List[SkillDefinition]:
    """Filter out skills disabled in config."""
    if disabled_skill_names is None:
        disabled_by_location: Dict[SkillLocation, set[str]] = {}
        for _, location in skill_directories(project_path=project_path, home=home):
            disabled_by_location[location] = get_disabled_skill_names(
                project_path=project_path, home=home, location=location
            )
        return [
            skill
            for skill in skills
            if skill.name not in disabled_by_location.get(skill.location, set())
        ]

    disabled = {_normalize_skill_ref(name) for name in disabled_skill_names if name}
    return [skill for skill in skills if skill.name not in disabled]


def is_skill_disabled(
    skill_name: str, project_path: Optional[Path] = None, home: Optional[Path] = None
) -> bool:
    """Check whether a skill is disabled in config."""
    skill = find_skill(
        skill_name,
        project_path=project_path,
        home=home,
        include_disabled=True,
    )
    if not skill:
        return False
    return is_skill_definition_disabled(skill, project_path=project_path, home=home)


def find_skill(
    skill_name: str,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
    *,
    include_disabled: bool = False,
) -> Optional[SkillDefinition]:
    """Find a skill by name (case-sensitive match)."""
    normalized = _normalize_skill_ref(skill_name)
    if not normalized:
        return None
    result = load_all_skills(project_path=project_path, home=home)
    skills = (
        result.skills
        if include_disabled
        else filter_enabled_skills(result.skills, project_path=project_path, home=home)
    )
    return next((skill for skill in skills if skill.name == normalized), None)


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
        "IMPORTANT: To use a skill, you MUST call the Skill tool. Do NOT read SKILL.md files directly with Read, Glob, Grep, or Bash commands.",
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
    "SKILL_STATE_FILE_NAME",
    "load_all_skills",
    "find_skill",
    "filter_enabled_skills",
    "get_disabled_skill_names",
    "save_disabled_skill_names",
    "is_skill_disabled",
    "is_skill_definition_disabled",
    "set_skill_enabled",
    "build_skill_summary",
    "skill_directories",
]
