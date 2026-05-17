"""Data models for background shell tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class BackgroundTask:
    """In-memory record of a background shell command."""

    id: str
    command: str
    process: asyncio.subprocess.Process
    start_time: float
    timeout: Optional[float] = None
    end_time: Optional[float] = None
    stdout_chunks: List[str] = field(default_factory=list)
    stderr_chunks: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    killed: bool = False
    timed_out: bool = False
    reader_tasks: List[asyncio.Task] = field(default_factory=list)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    completion_callbacks: List[Callable[["BackgroundTask"], None]] = field(default_factory=list)
    notification_sent: bool = False
