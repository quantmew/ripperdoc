import os
import tempfile
from pathlib import Path

from ripperdoc.utils.filesystem.temp_paths import (
    RIPPERDOC_TMPDIR_ENV,
    ripperdoc_mkstemp,
    ripperdoc_temporary_directory,
    ripperdoc_tmp_root,
)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def test_tmp_root_defaults_to_system_temp(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(RIPPERDOC_TMPDIR_ENV, raising=False)
    mocked_system_tmp = tmp_path / "system-tmp"
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(mocked_system_tmp))

    root = ripperdoc_tmp_root()
    assert root == mocked_system_tmp / "ripperdoc"
    assert root.exists()
    assert root.is_dir()


def test_tmp_root_honors_env_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom-tmp-root"
    monkeypatch.setenv(RIPPERDOC_TMPDIR_ENV, str(override))

    root = ripperdoc_tmp_root()
    assert root == override / "ripperdoc"
    assert root.exists()
    assert root.is_dir()


def test_temp_wrappers_create_files_under_ripperdoc_tmpdir(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "tmp-root"
    expected_root = base / "ripperdoc"
    monkeypatch.setenv(RIPPERDOC_TMPDIR_ENV, str(base))

    with ripperdoc_temporary_directory(prefix="ripperdoc-test-") as temp_dir:
        assert _is_within(Path(temp_dir), expected_root)

    fd, temp_path = ripperdoc_mkstemp(prefix="ripperdoc-test-", suffix=".txt")
    os.close(fd)
    temp_file = Path(temp_path)
    assert _is_within(temp_file, expected_root)
    temp_file.unlink(missing_ok=True)
