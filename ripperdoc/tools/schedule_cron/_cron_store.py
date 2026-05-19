"""Shared cron job store used by CronCreate, CronDelete, and CronList."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List

from ripperdoc.utils.log import get_logger

logger = get_logger()

MAX_JOBS = 50

# In-memory job store
_jobs: Dict[str, dict] = {}

# Durable storage path
_DURABLE_DIR = Path.home() / ".ripperdoc"
_DURABLE_FILE = _DURABLE_DIR / "scheduled_tasks.json"


def _load_durable_jobs() -> Dict[str, dict]:
    if not _DURABLE_FILE.exists():
        return {}
    try:
        with open(_DURABLE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if v.get("durable", False)}
    except (json.JSONDecodeError, OSError):
        return {}


def save_durable_jobs() -> None:
    durable = {k: v for k, v in _jobs.items() if v.get("durable", False)}
    try:
        _DURABLE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_DURABLE_FILE, "w", encoding="utf-8") as f:
            json.dump(durable, f, indent=2)
    except OSError as exc:
        logger.warning("[cron_store] Failed to save durable jobs: %s", exc)


def get_all_jobs() -> Dict[str, dict]:
    all_jobs = dict(_jobs)
    for job_id, job_data in _load_durable_jobs().items():
        if job_id not in all_jobs:
            all_jobs[job_id] = job_data
    return all_jobs


def list_jobs() -> List[dict]:
    return list(get_all_jobs().values())


def add_job(cron: str, prompt: str, recurring: bool, durable: bool) -> str:
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "id": job_id,
        "cron": cron,
        "prompt": prompt,
        "recurring": recurring,
        "durable": durable,
        "created_at": time.time(),
    }
    if durable:
        save_durable_jobs()
    return job_id


def remove_job(job_id: str) -> bool:
    removed = _jobs.pop(job_id, None)
    # Also remove from durable storage
    durable = _load_durable_jobs()
    if job_id in durable:
        del durable[job_id]
        _DURABLE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_DURABLE_FILE, "w", encoding="utf-8") as f:
                json.dump(durable, f, indent=2)
        except OSError:
            pass
    return removed is not None or job_id is not None


def job_count() -> int:
    return len(get_all_jobs())
