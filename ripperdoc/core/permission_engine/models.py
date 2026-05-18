"""Permission result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ripperdoc.utils.permissions import PermissionDecision


@dataclass
class PermissionResult:
    """Result of a permission check."""

    result: bool
    message: Optional[str] = None
    updated_input: Any = None
    decision: Optional[PermissionDecision] = None


@dataclass
class PermissionPreview:
    """Non-interactive preview of permission evaluation."""

    requires_user_input: bool
    result: Optional[PermissionResult] = None
    decision: Optional[PermissionDecision] = None

