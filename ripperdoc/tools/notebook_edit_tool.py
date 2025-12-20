"""Notebook edit tool.

Allows performing insert/replace/delete operations on Jupyter notebook cells.
"""

import json
import os
import random
import string
from pathlib import Path
from textwrap import dedent
from typing import AsyncGenerator, List, Optional
from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.file_watch import record_snapshot


logger = get_logger()


def _resolve_path(path_str: str) -> Path:
    """Return an absolute Path, interpreting relative paths from CWD."""
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _generate_cell_id() -> str:
    """Generate a short random cell id."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


NOTEBOOK_EDIT_DESCRIPTION = dedent(
    """\
    Replace, insert, or delete a specific cell in a Jupyter notebook (.ipynb file).
    notebook_path must be an absolute path. cell_id may be a 0-based index or a cell id.
    Use edit_mode=insert to add a new cell after the referenced cell (or at the start if omitted).
    Use edit_mode=delete to delete the referenced cell. Defaults to edit_mode=replace.

    Usage:
    - cell_type: 'code' or 'markdown'. Required for insert; defaults to existing type for replace.
    - new_source: New content for the cell.
    - Edits are applied atomically; failures leave the file unchanged.
    - Code cell replacements clear execution_count and outputs.
    - Only use emojis if explicitly requested; avoid adding emojis otherwise.
    """
)


class NotebookEditInput(BaseModel):
    """Input schema for NotebookEditTool."""

    notebook_path: str = Field(description="Absolute path to the Jupyter notebook file to edit")
    cell_id: Optional[str] = Field(
        default=None,
        description="Cell ID or 0-based index. For insert, omitted means insert at start.",
    )
    new_source: str = Field(description="New source content for the target cell")
    cell_type: Optional[str] = Field(
        default=None,
        description="Cell type: 'code' or 'markdown'. Required for insert.",
    )
    edit_mode: Optional[str] = Field(
        default="replace",
        description="Edit mode: 'replace' (default), 'insert', or 'delete'.",
    )


class NotebookEditOutput(BaseModel):
    """Output from notebook editing."""

    new_source: str
    cell_id: Optional[str] = None
    cell_type: str
    language: str
    edit_mode: str
    error: Optional[str] = None


class NotebookEditTool(Tool[NotebookEditInput, NotebookEditOutput]):
    """Tool for editing Jupyter notebooks."""

    @property
    def name(self) -> str:
        return "NotebookEdit"

    async def description(self) -> str:
        return NOTEBOOK_EDIT_DESCRIPTION

    @property
    def input_schema(self) -> type[NotebookEditInput]:
        return NotebookEditInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Replace a markdown cell by id",
                example={
                    "notebook_path": "/repo/notebooks/analysis.ipynb",
                    "cell_id": "abc123",
                    "new_source": "# Updated overview\\nThis notebook analyzes revenue.",
                    "cell_type": "markdown",
                    "edit_mode": "replace",
                },
            ),
            ToolUseExample(
                description="Insert a new code cell at the beginning",
                example={
                    "notebook_path": "/repo/notebooks/analysis.ipynb",
                    "cell_type": "code",
                    "edit_mode": "insert",
                    "new_source": "import pandas as pd\\nimport numpy as np",
                },
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:
        return NOTEBOOK_EDIT_DESCRIPTION

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[NotebookEditInput] = None) -> bool:
        return True

    async def validate_input(
        self, input_data: NotebookEditInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        path = _resolve_path(input_data.notebook_path)
        resolved_path = str(path.resolve())

        if not path.exists():
            return ValidationResult(
                result=False,
                message="Notebook file does not exist.",
                error_code=1,
            )
        if path.suffix != ".ipynb":
            return ValidationResult(
                result=False,
                message="File must be a Jupyter notebook (.ipynb file). Use Edit for other file types.",
                error_code=2,
            )

        mode = (input_data.edit_mode or "replace").lower()
        if mode not in {"replace", "insert", "delete"}:
            return ValidationResult(
                result=False,
                message="edit_mode must be replace, insert, or delete.",
                error_code=3,
            )
        if mode == "insert" and not input_data.cell_type:
            return ValidationResult(
                result=False,
                message="cell_type is required when using edit_mode=insert.",
                error_code=4,
            )
        if mode != "insert" and not input_data.cell_id:
            return ValidationResult(
                result=False,
                message="cell_id must be specified when using edit_mode=replace or delete.",
                error_code=5,
            )

        # Check if file has been read before editing
        file_state_cache = getattr(context, "file_state_cache", {}) if context else {}
        file_snapshot = file_state_cache.get(resolved_path)

        if not file_snapshot:
            return ValidationResult(
                result=False,
                message="Notebook has not been read yet. Read it first before editing.",
                error_code=6,
            )

        # Check if file has been modified since it was read
        try:
            current_mtime = os.path.getmtime(resolved_path)
            if current_mtime > file_snapshot.timestamp:
                return ValidationResult(
                    result=False,
                    message="Notebook has been modified since read, either by the user or by a linter. "
                    "Read it again before attempting to edit it.",
                    error_code=7,
                )
        except OSError:
            pass  # File mtime check failed, proceed anyway

        # Validate notebook structure and target cell.
        try:
            raw = path.read_text(encoding="utf-8")
            nb_json = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "Failed to parse notebook: %s: %s",
                type(exc).__name__,
                exc,
                extra={"path": str(path)},
            )
            return ValidationResult(
                result=False,
                message="Notebook is not valid JSON.",
                error_code=8,
            )

        cells = nb_json.get("cells", [])
        target_index, _ = self._resolve_cell_index(cells, input_data.cell_id, mode)
        if target_index is None:
            if mode == "insert" and input_data.cell_id is None:
                return ValidationResult(result=True)
            return ValidationResult(
                result=False,
                message=f"Cell '{input_data.cell_id}' not found in notebook.",
                error_code=9,
            )

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: NotebookEditOutput) -> str:
        if output.error:
            return output.error
        action = output.edit_mode or "replace"
        cell_label = output.cell_id or "(new cell)"
        if action == "delete":
            return f"Deleted cell {cell_label}"
        if action == "insert":
            return f"Inserted cell {cell_label}"
        return f"Updated cell {cell_label}"

    def render_tool_use_message(self, input_data: NotebookEditInput, verbose: bool = False) -> str:
        parts = [f"path: {input_data.notebook_path}"]
        if input_data.cell_id:
            parts.append(f"cell_id: {input_data.cell_id}")
        if verbose:
            parts.append(f"mode: {input_data.edit_mode or 'replace'}")
        return ", ".join(parts)

    async def call(
        self, input_data: NotebookEditInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        path = _resolve_path(input_data.notebook_path)
        mode = (input_data.edit_mode or "replace").lower()
        cell_type = (input_data.cell_type or "").lower() or None
        new_source = input_data.new_source
        cell_id = input_data.cell_id

        try:
            raw = path.read_text(encoding="utf-8")
            nb_json = json.loads(raw)
            cells = nb_json.get("cells", [])

            target_index, matched_id = self._resolve_cell_index(cells, cell_id, mode)

            final_mode = mode
            if final_mode == "replace" and target_index is not None and target_index == len(cells):
                final_mode = "insert"
                if not cell_type:
                    cell_type = "code"

            if final_mode == "delete":
                if target_index is None:
                    raise ValueError("Target cell not found for delete.")
                cells.pop(target_index)
            elif final_mode == "insert":
                insert_at = target_index if target_index is not None else 0
                new_id = matched_id or _generate_cell_id()
                new_cell_type = cell_type or "code"
                new_cell = (
                    {
                        "cell_type": "markdown",
                        "id": new_id,
                        "source": new_source,
                        "metadata": {},
                    }
                    if new_cell_type == "markdown"
                    else {
                        "cell_type": "code",
                        "id": new_id,
                        "source": new_source,
                        "metadata": {},
                        "execution_count": None,  # type: ignore[dict-item]
                        "outputs": [],
                    }
                )
                cells.insert(insert_at, new_cell)
                matched_id = new_id
                cell_type = new_cell_type
            else:  # replace
                if target_index is None:
                    raise ValueError("Target cell not found for replace.")
                target_cell = cells[target_index]
                target_cell["source"] = new_source
                if target_cell.get("cell_type") == "code":
                    target_cell["execution_count"] = None
                    target_cell["outputs"] = []
                if cell_type and cell_type != target_cell.get("cell_type"):
                    target_cell["cell_type"] = cell_type
                matched_id = target_cell.get("id") or matched_id
                cell_type = target_cell.get("cell_type", cell_type or "code")

            nb_json["cells"] = cells
            notebook_language = (
                nb_json.get("metadata", {}).get("language_info", {}).get("name", "python")
            )

            path.write_text(json.dumps(nb_json, indent=1), encoding="utf-8")
            # Use resolved absolute path to ensure consistency with validation lookup
            abs_notebook_path = str(path.resolve())
            try:
                record_snapshot(
                    abs_notebook_path,
                    json.dumps(nb_json, indent=1),
                    getattr(context, "file_state_cache", {}),
                )
            except (OSError, IOError, RuntimeError) as exc:
                logger.warning(
                    "[notebook_edit_tool] Failed to record file snapshot: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"file_path": abs_notebook_path},
                )

            output = NotebookEditOutput(
                new_source=new_source,
                cell_type=cell_type or "code",
                language=notebook_language,
                edit_mode=final_mode,
                cell_id=matched_id,
                error=None,
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            # pragma: no cover - error path
            logger.warning(
                "Error editing notebook: %s: %s",
                type(exc).__name__,
                exc,
                extra={"path": input_data.notebook_path},
            )
            output = NotebookEditOutput(
                new_source=new_source,
                cell_type=cell_type or "code",
                language="python",
                edit_mode=mode,
                cell_id=cell_id,
                error=str(exc),
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

    def _resolve_cell_index(
        self, cells: list, cell_id: Optional[str], mode: str
    ) -> tuple[Optional[int], Optional[str]]:
        """Return target index and resolved id."""
        if cell_id is None:
            return (0 if mode == "insert" else None, None)

        # Try numeric index first.
        try:
            idx = int(cell_id)
            if mode == "insert":
                if idx < 0 or idx > len(cells):
                    return None, None
                return min(idx + 1, len(cells)), None
            else:
                if idx < 0 or idx >= len(cells):
                    return None, None
                return idx, None
        except (ValueError, TypeError):
            pass

        for i, cell in enumerate(cells):
            if cell.get("id") == cell_id:
                return (i if mode != "insert" else i + 1, cell.get("id"))
        return None, None
