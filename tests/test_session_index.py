"""Tests for session sidecar index behavior."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ripperdoc.utils.messages import create_assistant_message, create_user_message
from ripperdoc.utils.session_history import SessionHistory, list_session_summaries
from ripperdoc.utils.session_index import load_or_build_session_index
from ripperdoc.utils.session_stats import collect_session_stats


def _patch_session_home(monkeypatch, home_path):
    monkeypatch.setattr("ripperdoc.utils.session_history.Path.home", lambda: home_path)
    monkeypatch.setattr("ripperdoc.utils.session_index.Path.home", lambda: home_path)


def test_session_index_incremental_append_and_summary(tmp_path, monkeypatch):
    _patch_session_home(monkeypatch, tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    session = SessionHistory(project, "s1")
    session.append(create_user_message("Need help with ripperdoc indexing performance"))
    session.append(
        create_assistant_message(
            "Done.",
            model="gpt-test",
            input_tokens=120,
            output_tokens=80,
            cache_read_tokens=10,
            cache_creation_tokens=5,
            cost_usd=0.25,
        )
    )
    # Tool-result user messages can carry large tool_use_result payloads; prompt preview
    # should still come from normal user turns.
    session.append(
        create_user_message(
            [{"type": "tool_result", "tool_use_id": "grep1", "text": "large payload omitted"}],
            tool_use_result={"matches": [{"file": "a.py", "content": "x" * 1000}]},
        )
    )

    summaries = list_session_summaries(project)
    assert len(summaries) == 1
    assert summaries[0].session_id == "s1"
    assert summaries[0].message_count == 3
    assert "Need help with ripperdoc" in summaries[0].last_prompt

    index = load_or_build_session_index(project)
    entry = index.sessions["s1"]
    assert entry.total_input_tokens == 120
    assert entry.total_output_tokens == 80
    assert entry.total_cache_read_tokens == 10
    assert entry.total_cache_creation_tokens == 5
    assert entry.total_cost_usd == 0.25
    assert entry.model_usage == {"gpt-test": 1}


def test_session_index_reconciles_external_file_change(tmp_path, monkeypatch):
    _patch_session_home(monkeypatch, tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    session = SessionHistory(project, "s2")
    session.append(create_user_message("First prompt"))
    first_summary = list_session_summaries(project)[0]
    assert first_summary.message_count == 1

    logged_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    raw_entry = {
        "logged_at": logged_at,
        "project_path": str(project.resolve()),
        "payload": create_assistant_message(
            "Reply",
            model="model-x",
            input_tokens=3,
            output_tokens=4,
        ).model_dump(mode="json"),
    }
    with session.path.open("a", encoding="utf-8") as fh:
        json.dump(raw_entry, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")

    # Ensure mtime changes so reconcile detects the update.
    stat = session.path.stat()
    os.utime(session.path, (stat.st_atime, stat.st_mtime + 1))

    refreshed = list_session_summaries(project)[0]
    assert refreshed.message_count == 2

    index = load_or_build_session_index(project)
    entry = index.sessions["s2"]
    assert entry.total_input_tokens == 3
    assert entry.total_output_tokens == 4
    assert entry.model_usage == {"model-x": 1}


def test_collect_session_stats_aggregates_from_index(tmp_path, monkeypatch):
    _patch_session_home(monkeypatch, tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    s1 = SessionHistory(project, "stats-1")
    s1.append(create_user_message("u1"))
    s1.append(
        create_assistant_message(
            "a1",
            model="model-a",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=2,
            cache_creation_tokens=1,
            cost_usd=0.1,
        )
    )

    s2 = SessionHistory(project, "stats-2")
    s2.append(create_user_message("u2"))
    s2.append(
        create_assistant_message(
            "a2",
            model="model-b",
            input_tokens=20,
            output_tokens=10,
            cost_usd=0.2,
        )
    )
    s2.append(
        create_assistant_message(
            "a3",
            model="model-b",
            input_tokens=1,
            output_tokens=2,
            cost_usd=0.05,
        )
    )

    stats = collect_session_stats(project, days=365)
    assert stats.total_sessions == 2
    assert stats.total_messages == 5
    assert stats.total_input_tokens == 31
    assert stats.total_output_tokens == 17
    assert stats.total_cache_read_tokens == 2
    assert stats.total_cache_creation_tokens == 1
    assert stats.total_tokens == 51
    assert stats.total_cost_usd == 0.35
    assert stats.favorite_model == "model-b"
    assert stats.current_streak >= 0
    assert stats.longest_streak >= 1
    assert stats.active_days >= 1
