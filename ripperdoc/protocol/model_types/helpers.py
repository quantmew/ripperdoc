"""Protocol model helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a pydantic model to JSON-serializable dict."""

    return model.model_dump(exclude_none=True, by_alias=True, mode="json")


__all__ = ["model_to_dict"]
