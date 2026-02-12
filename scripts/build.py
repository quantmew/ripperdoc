#!/usr/bin/env python3
"""Ripperdoc build script.

Package Ripperdoc into a standalone executable with PyInstaller.

Supports two build modes:
    --onefile    One-file mode (default): package everything into a single executable
    --dir, -d    Directory mode (recommended): executable + dependencies in one directory
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return its exit code."""
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def clean_build_dirs(root_dir: Path) -> None:
    """Clean build directories."""
    dirs_to_clean = ["build", "dist", "spec"]
    for dir_name in dirs_to_clean:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            print(f"清理目录: {dir_path}")
            shutil.rmtree(dir_path)


def copy_data_files(root_dir: Path, dist_dir: Path) -> None:
    """Copy data files to the output directory (for directory mode).

    PyInstaller onedir layout:
        dist/ripperdoc/
        ├── ripperdoc          # executable
        └── _internal/         # dependencies and data
            └── ripperdoc/
                └── data/
    """
    src_data_dir = root_dir / "ripperdoc" / "data"
    # Data files should be placed under _internal because ripperdoc is an executable.
    dst_data_dir = dist_dir / "_internal" / "ripperdoc" / "data"

    if not src_data_dir.exists():
        return

    # If the destination exists as a file, remove it first.
    if dst_data_dir.exists() and dst_data_dir.is_file():
        dst_data_dir.unlink()

    dst_data_dir.mkdir(parents=True, exist_ok=True)

    for json_file in src_data_dir.glob("*.json"):
        dst_file = dst_data_dir / json_file.name
        shutil.copy2(json_file, dst_file)
        print(f"复制数据文件: {json_file.name} -> {dst_data_dir}")


def get_dir_size(path: Path) -> int:
    """Get total size of a directory."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def main() -> int:
    """Main build flow."""
    parser = argparse.ArgumentParser(
        description="构建 Ripperdoc 可执行文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build.py           # one-file mode (default)
    python scripts/build.py --dir     # directory mode (recommended)
    python scripts/build.py --onefile # one-file mode
        """
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dir", "-d",
        action="store_true",
        help="目录模式: 可执行文件 + 依赖在同一目录"
    )
    mode_group.add_argument(
        "--onefile",
        action="store_true",
        help="单文件模式: 所有内容打包成一个可执行文件"
    )
    args = parser.parse_args()

    # Determine build mode.
    if args.dir:
        build_mode = "onedir"
    elif args.onefile:
        build_mode = "onefile"
    else:
        build_mode = "onedir"  # Default to directory mode.

    root_dir = Path(__file__).parent.parent
    spec_file = root_dir / "ripperdoc.spec"

    # Check spec file.
    if not spec_file.exists():
        print(f"错误: 找不到 spec 文件: {spec_file}")
        return 1

    # Check PyInstaller.
    try:
        import PyInstaller
        print(f"使用 PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("错误: 未安装 PyInstaller")
        print("请运行: pip install pyinstaller")
        return 1

    # Set build mode environment variable.
    os.environ["RIPPERDOC_BUILD_MODE"] = build_mode

    print(f"\n=== 构建模式: {build_mode} ===")

    # Clean previous build.
    print("\n=== 清理旧的构建目录 ===")
    clean_build_dirs(root_dir)

    # Run PyInstaller.
    print("\n=== 开始构建 ===")
    cmd = [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean"]
    exit_code = run_command(cmd, cwd=root_dir)

    if exit_code != 0:
        print(f"\n错误: 构建失败 (退出码: {exit_code})")
        return exit_code

    # Show build result.
    dist_dir = root_dir / "dist"

    if build_mode == "onedir":
        output_dir = dist_dir / "ripperdoc"
        exe_file = output_dir / "ripperdoc"

        if not exe_file.exists():
            print(f"\n错误: 构建完成但找不到输出目录: {output_dir}")
            return 1

        # Copy data files.
        print("\n=== 复制数据文件 ===")
        copy_data_files(root_dir, output_dir)

        # Compute directory size.
        dir_size = get_dir_size(output_dir)
        size_mb = dir_size / (1024 * 1024)

        print(f"\n=== 构建成功 ===")
        print(f"输出目录: {output_dir}")
        print(f"可执行文件: {exe_file}")
        print(f"目录大小: {size_mb:.1f} MB")
        print(f"\n可以运行: {exe_file} --help")
        print(f"或进入目录: cd {output_dir}")
    else:
        exe_file = dist_dir / "ripperdoc"

        if not exe_file.exists():
            print(f"\n错误: 构建完成但找不到可执行文件: {exe_file}")
            return 1

        size_mb = exe_file.stat().st_size / (1024 * 1024)
        print(f"\n=== 构建成功 ===")
        print(f"可执行文件: {exe_file}")
        print(f"文件大小: {size_mb:.1f} MB")
        print(f"\n可以运行: {exe_file} --help")

    return 0


if __name__ == "__main__":
    sys.exit(main())
