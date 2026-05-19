"""ScheduleCron tools — create, delete, and list scheduled cron jobs."""

from __future__ import annotations

import re
from typing import AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.tools.schedule_cron._cron_store import (
    add_job,
    get_all_jobs,
    job_count,
    list_jobs,
    MAX_JOBS,
    remove_job,
)

logger = get_logger()

CRON_CREATE_NAME = "CronCreate"
CRON_DELETE_NAME = "CronDelete"
CRON_LIST_NAME = "CronList"

_DEFAULT_MAX_AGE_DAYS = 7

# ── Cron expression helpers ──────────────────────────────────────────────

_CRON_PATTERN = re.compile(
    r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$"
)

_FIELD_NAMES = ("minute", "hour", "day of month", "month", "day of week")
_FIELD_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
_DAY_MAP: Dict[int, str] = {
    0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat",
}
_MONTH_MAP: Dict[int, str] = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _parse_cron_field(value: str, low: int, high: int) -> bool:
    """Validate a single cron field against its allowed range."""
    for part in value.split(","):
        step_match = re.fullmatch(r"(.+)/(\d+)", part)
        step = None
        if step_match:
            part = step_match.group(1)
            step = int(step_match.group(2))
            if step < 1:
                return False
        if part == "*":
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)", part)
        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
            if not (low <= lo <= high and low <= hi <= high and lo <= hi):
                return False
            continue
        try:
            v = int(part)
            if not (low <= v <= high):
                return False
        except ValueError:
            return False
    return True


def parse_cron_expression(cron: str) -> bool:
    """Validate a 5-field cron expression."""
    m = _CRON_PATTERN.match(cron.strip())
    if not m:
        return False
    for i, field in enumerate(m.groups()):
        low, high = _FIELD_BOUNDS[i]
        if not _parse_cron_field(field, low, high):
            return False
    return True


def cron_to_human(cron: str) -> str:
    """Convert a cron expression to a human-readable description."""
    m = _CRON_PATTERN.match(cron.strip())
    if not m:
        return cron
    mi, h, dom, mon, dow = m.groups()

    def _fmt(value: str, names: Optional[Dict[int, str]] = None) -> str:
        if value == "*":
            return "every" if names else "every"
        step_match = re.fullmatch(r"\*/(\d+)", value)
        if step_match:
            step = int(step_match.group(1))
            unit = "min" if names is None else ""
            return f"every {step}{unit}"
        if names and value.isdigit():
            v = int(value)
            return names.get(v, value)
        return value

    parts: List[str] = []
    if mi != "0" and mi != "*":
        step = re.fullmatch(r"\*/(\d+)", mi)
        if step:
            parts.append(f"every {step.group(1)} minutes")
        elif mi.isdigit():
            parts.append(f"at minute {mi}")
    if h != "*" and h.isdigit():
        parts.append(f"at {h}:00")

    day_names: Dict[int, str] = {
        1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 0: "Sun",
    }
    if dow != "*" and dow.isdigit():
        parts.append(f"on {day_names.get(int(dow), dow)}")

    if dom != "*" and dom.isdigit():
        parts.append(f"on day {dom}")

    if mon != "*" and mon.isdigit():
        parts.append(f"in {_MONTH_MAP.get(int(mon), mon)}")

    if not parts:
        return f"'{cron}'"
    return " ".join(parts)


# ── CronCreate ───────────────────────────────────────────────────────────

class CronCreateInput(BaseModel):
    cron: str = Field(
        description='Standard 5-field cron expression in local time: "M H DoM Mon DoW" (e.g. "*/5 * * * *" = every 5 minutes, "30 14 28 2 *" = Feb 28 at 2:30pm local once).',
    )
    prompt: str = Field(description="The prompt to enqueue at each fire time.")
    recurring: bool = Field(
        default=True,
        description=(
            f"true (default) = fire on every cron match until deleted or auto-expired"
            f" after {_DEFAULT_MAX_AGE_DAYS} days. false = fire once then auto-delete."
        ),
    )
    durable: bool = Field(
        default=False,
        description=(
            "true = persist to .ripperdoc/scheduled_tasks.json and survive restarts."
            " false (default) = in-memory only, dies when this session ends."
        ),
    )


class CronCreateOutput(BaseModel):
    id: str
    human_schedule: str = Field(serialization_alias="humanSchedule")
    recurring: bool
    durable: bool = Field(default=False)


class CronCreateTool(Tool[CronCreateInput, CronCreateOutput]):
    @property
    def name(self) -> str:
        return CRON_CREATE_NAME

    async def description(self) -> str:
        return "Schedule a recurring or one-shot prompt using cron expressions."

    @property
    def input_schema(self) -> type[CronCreateInput]:
        return CronCreateInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        return (
            "Schedule a prompt to be enqueued at a future time. "
            'Uses standard 5-field cron in local time: "M H DoM Mon DoW". '
            "Set recurring=false for one-shot tasks that auto-delete after firing. "
            "Set durable=true to persist across sessions. "
            "Returns a job ID you can pass to CronDelete."
        )

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[CronCreateInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: CronCreateInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if not parse_cron_expression(input_data.cron):
            return ValidationResult(
                result=False,
                message=f"Invalid cron expression '{input_data.cron}'. Expected 5 fields: M H DoM Mon DoW.",
            )
        if job_count() >= MAX_JOBS:
            return ValidationResult(
                result=False,
                message=f"Too many scheduled jobs (max {MAX_JOBS}). Cancel one first.",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: CronCreateOutput) -> str:
        where = (
            "Persisted to .ripperdoc/scheduled_tasks.json"
            if output.durable
            else "Session-only (not written to disk, dies when Ripperdoc exits)"
        )
        if output.recurring:
            return (
                f"Scheduled recurring job {output.id} ({output.human_schedule}). "
                f"{where}. Auto-expires after {_DEFAULT_MAX_AGE_DAYS} days. "
                f"Use CronDelete to cancel sooner."
            )
        return (
            f"Scheduled one-shot task {output.id} ({output.human_schedule}). "
            f"{where}. It will fire once then auto-delete."
        )

    def render_tool_use_message(self, input_data: CronCreateInput, _verbose: bool = False) -> str:
        return f"Scheduling cron: {input_data.cron}"

    async def call(
        self,
        input_data: CronCreateInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        job_id = add_job(
            cron=input_data.cron,
            prompt=input_data.prompt,
            recurring=input_data.recurring,
            durable=input_data.durable,
        )
        output = CronCreateOutput(
            id=job_id,
            human_schedule=cron_to_human(input_data.cron),
            recurring=input_data.recurring,
            durable=input_data.durable,
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )


# ── CronDelete ───────────────────────────────────────────────────────────

class CronDeleteInput(BaseModel):
    id: str = Field(description="Job ID returned by CronCreate.")


class CronDeleteOutput(BaseModel):
    id: str


class CronDeleteTool(Tool[CronDeleteInput, CronDeleteOutput]):
    @property
    def name(self) -> str:
        return CRON_DELETE_NAME

    async def description(self) -> str:
        return "Cancel a scheduled cron job by ID."

    @property
    def input_schema(self) -> type[CronDeleteInput]:
        return CronDeleteInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        return "Cancel a scheduled cron job by its ID (returned by CronCreate)."

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[CronDeleteInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: CronDeleteInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        all_jobs = get_all_jobs()
        if input_data.id not in all_jobs:
            return ValidationResult(
                result=False,
                message=f"No scheduled job with id '{input_data.id}'",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: CronDeleteOutput) -> str:
        return f"Cancelled job {output.id}."

    def render_tool_use_message(self, input_data: CronDeleteInput, _verbose: bool = False) -> str:
        return f"Cancelling cron job {input_data.id}"

    async def call(
        self,
        input_data: CronDeleteInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        remove_job(input_data.id)
        output = CronDeleteOutput(id=input_data.id)
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )


# ── CronList ─────────────────────────────────────────────────────────────

class CronListInput(BaseModel):
    pass


class CronJobInfo(BaseModel):
    id: str
    cron: str
    human_schedule: str = Field(serialization_alias="humanSchedule")
    prompt: str
    recurring: bool = True
    durable: bool = False


class CronListOutput(BaseModel):
    jobs: List[CronJobInfo] = Field(default_factory=list)


class CronListTool(Tool[CronListInput, CronListOutput]):
    @property
    def name(self) -> str:
        return CRON_LIST_NAME

    async def description(self) -> str:
        return "List all scheduled cron jobs."

    @property
    def input_schema(self) -> type[CronListInput]:
        return CronListInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        return "List all active cron jobs. Use CronCreate to schedule new jobs, CronDelete to cancel them."

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[CronListInput] = None) -> bool:
        return False

    def render_result_for_assistant(self, output: CronListOutput) -> str:
        if not output.jobs:
            return "No scheduled jobs."
        lines: List[str] = []
        for j in output.jobs:
            tag = "recurring" if j.recurring else "one-shot"
            storage = "session-only" if j.durable is False else ""
            prefix = f"{j.id} — {j.human_schedule} ({tag}"
            if storage:
                prefix += f", {storage}"
            prefix += ")"
            prompt_preview = j.prompt[:80] + "..." if len(j.prompt) > 80 else j.prompt
            lines.append(f"{prefix}: {prompt_preview}")
        return "\n".join(lines)

    def render_tool_use_message(self, _input_data: CronListInput, _verbose: bool = False) -> str:
        return "Listing cron jobs"

    async def call(
        self,
        input_data: CronListInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        del input_data
        jobs = [
            CronJobInfo(
                id=j["id"],
                cron=j["cron"],
                human_schedule=cron_to_human(j["cron"]),
                prompt=j["prompt"],
                recurring=j.get("recurring", True),
                durable=j.get("durable", False),
            )
            for j in list_jobs()
        ]
        output = CronListOutput(jobs=jobs)
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
