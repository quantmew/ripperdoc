"""Tests for NotebookEditTool."""

import json
import os
from unittest.mock import MagicMock

from ripperdoc.tools.notebook_edit_tool import NotebookEditInput, NotebookEditTool


class TestNotebookEditValidation:
    """Tests for NotebookEditTool input validation."""

    async def test_validate_input_with_snapshot(self, tmp_path):
        """Validation should succeed with a valid notebook and snapshot."""
        notebook_path = tmp_path / "sample.ipynb"
        notebook = {
            "cells": [
                {
                    "id": "abc123",
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["Hello\n"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

        tool = NotebookEditTool()
        input_data = NotebookEditInput(
            notebook_path=str(notebook_path),
            cell_id="abc123",
            new_source="Updated content",
            edit_mode="replace",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(notebook_path)
        snapshot.content = notebook_path.read_text(encoding="utf-8")

        context = MagicMock()
        context.file_state_cache = {os.path.abspath(str(notebook_path)): snapshot}

        result = await tool.validate_input(input_data, context)
        assert result.result is True
