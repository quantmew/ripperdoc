"""Tests for project storage path helpers."""

from pathlib import Path

from ripperdoc.utils.path_utils import (
    _legacy_sanitize_project_path,
    project_storage_dir,
    sanitize_project_path,
)


def test_project_storage_dir_does_not_migrate_legacy_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "sessions"
    project_path = tmp_path / "project"
    project_path.mkdir(parents=True, exist_ok=True)

    legacy_dir = base_dir / _legacy_sanitize_project_path(project_path)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "old.jsonl").write_text("legacy")

    resolved_dir = project_storage_dir(base_dir, project_path, ensure=False)
    hashed_dir = base_dir / sanitize_project_path(project_path)

    assert resolved_dir == hashed_dir
    assert not hashed_dir.exists()
    assert (legacy_dir / "old.jsonl").read_text() == "legacy"
    assert legacy_dir.exists()


def test_project_storage_dir_does_not_merge_with_legacy_when_both_exist(tmp_path: Path) -> None:
    base_dir = tmp_path / "todos"
    project_path = tmp_path / "project"
    project_path.mkdir(parents=True, exist_ok=True)

    hashed_dir = base_dir / sanitize_project_path(project_path)
    legacy_dir = base_dir / _legacy_sanitize_project_path(project_path)
    hashed_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    (hashed_dir / "keep.txt").write_text("hashed")
    (legacy_dir / "keep.txt").write_text("legacy")
    (legacy_dir / "move.txt").write_text("to-migrate")

    resolved_dir = project_storage_dir(base_dir, project_path, ensure=False)

    assert resolved_dir == hashed_dir
    assert (hashed_dir / "keep.txt").read_text() == "hashed"
    assert not (hashed_dir / "move.txt").exists()
    assert (legacy_dir / "move.txt").read_text() == "to-migrate"
