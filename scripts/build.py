#!/usr/bin/env python3
"""Ripperdoc build script.

Package Ripperdoc into a standalone executable with PyInstaller.

Supported build modes:
    --onefile    Build a single executable file
    --dir, -d    Build a directory bundle (default)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def clean_build_dirs(root_dir: Path) -> None:
    """Clean previous build directories."""
    dirs_to_clean = ["build", "dist", "spec"]
    for dir_name in dirs_to_clean:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            print(f"Cleaning directory: {dir_path}")
            shutil.rmtree(dir_path)


def copy_data_files(root_dir: Path, dist_dir: Path) -> None:
    """Copy data files to output directory (for onedir mode).

    PyInstaller onedir layout:
        dist/ripperdoc/
        |- ripperdoc or ripperdoc.exe
        |- _internal/
           |- ripperdoc/
              |- data/
    """
    src_data_dir = root_dir / "ripperdoc" / "data"
    dst_data_dir = dist_dir / "_internal" / "ripperdoc" / "data"

    if not src_data_dir.exists():
        return

    if dst_data_dir.exists() and dst_data_dir.is_file():
        dst_data_dir.unlink()

    dst_data_dir.mkdir(parents=True, exist_ok=True)

    for json_file in src_data_dir.glob("*.json"):
        dst_file = dst_data_dir / json_file.name
        shutil.copy2(json_file, dst_file)
        print(f"Copied data file: {json_file.name} -> {dst_data_dir}")


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def main() -> int:
    """Main build flow."""
    parser = argparse.ArgumentParser(
        description="Build Ripperdoc executable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build.py           # directory mode (default)
    python scripts/build.py --dir     # directory mode
    python scripts/build.py --onefile # one-file mode
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dir",
        "-d",
        action="store_true",
        help="Directory mode: executable + dependencies in one folder",
    )
    mode_group.add_argument(
        "--onefile",
        action="store_true",
        help="One-file mode: build a single executable",
    )
    args = parser.parse_args()

    if args.dir:
        build_mode = "onedir"
    elif args.onefile:
        build_mode = "onefile"
    else:
        build_mode = "onedir"

    root_dir = Path(__file__).parent.parent
    spec_file = root_dir / "ripperdoc.spec"

    if not spec_file.exists():
        print(f"Error: spec file not found: {spec_file}")
        return 1

    try:
        import PyInstaller

        print(f"Using PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Error: PyInstaller is not installed")
        print("Install with: pip install pyinstaller")
        return 1

    os.environ["RIPPERDOC_BUILD_MODE"] = build_mode

    print(f"\n=== Build mode: {build_mode} ===")

    print("\n=== Cleaning old build directories ===")
    clean_build_dirs(root_dir)

    print("\n=== Starting build ===")
    cmd = [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean"]
    exit_code = run_command(cmd, cwd=root_dir)

    if exit_code != 0:
        print(f"\nError: build failed (exit code: {exit_code})")
        return exit_code

    dist_dir = root_dir / "dist"

    if build_mode == "onedir":
        output_dir = dist_dir / "ripperdoc"
        exe_name = "ripperdoc.exe" if sys.platform == "win32" else "ripperdoc"
        exe_file = output_dir / exe_name

        if not exe_file.exists():
            print(f"\nError: build finished but executable not found: {exe_file}")
            print(f"Hint: check output directory: {output_dir}")
            return 1

        print("\n=== Copying data files ===")
        copy_data_files(root_dir, output_dir)

        dir_size = get_dir_size(output_dir)
        size_mb = dir_size / (1024 * 1024)

        print("\n=== Build succeeded ===")
        print(f"Output directory: {output_dir}")
        print(f"Executable: {exe_file}")
        print(f"Directory size: {size_mb:.1f} MB")
        print(f"\nRun: {exe_file} --help")
    else:
        exe_name = "ripperdoc.exe" if sys.platform == "win32" else "ripperdoc"
        exe_file = dist_dir / exe_name

        if not exe_file.exists():
            print(f"\nError: build finished but executable not found: {exe_file}")
            print(f"Hint: check files under: {dist_dir}")
            return 1

        size_mb = exe_file.stat().st_size / (1024 * 1024)
        print("\n=== Build succeeded ===")
        print(f"Executable: {exe_file}")
        print(f"File size: {size_mb:.1f} MB")
        print(f"\nRun: {exe_file} --help")

    return 0


if __name__ == "__main__":
    sys.exit(main())
