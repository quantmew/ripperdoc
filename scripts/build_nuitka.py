#!/usr/bin/env python3
"""使用 Nuitka 构建 Ripperdoc

Nuitka 将 Python 编译为 C 代码再编译为机器码，
通常比 PyInstaller 生成更小更快的可执行文件。

安装依赖:
    pip install nuitka
    conda install -c conda-forge gcc_linux-64 gxx_linux-64  # 可选，用于 LTO

使用:
    python scripts/build_nuitka.py              # 基础构建
    python scripts/build_nuitka.py --lto        # 启用 LTO 优化 (需要 conda gcc)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_conda_gcc() -> tuple[bool, str | None]:
    """检查是否安装了 conda gcc"""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return False, None

    gcc_path = Path(conda_prefix) / "bin" / "gcc"
    if gcc_path.exists():
        result = subprocess.run(
            [str(gcc_path), "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return True, str(gcc_path)
    return False, None


def main():
    parser = argparse.ArgumentParser(description="使用 Nuitka 构建 Ripperdoc")
    parser.add_argument("--lto", action="store_true", help="启用链接时优化 (需要 conda gcc)")
    parser.add_argument("--no-onefile", action="store_true", help="使用目录模式而非单文件")
    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent
    main_script = root_dir / "ripperdoc" / "cli" / "cli.py"

    # 检查 gcc 和 LTO 支持
    use_lto = False
    conda_gcc_path = None

    if args.lto:
        has_conda_gcc, gcc_path = check_conda_gcc()
        if has_conda_gcc:
            use_lto = True
            conda_gcc_path = gcc_path
            print(f"使用 conda gcc: {gcc_path}")
        else:
            print("警告: 未找到 conda gcc，LTO 优化已禁用")
            print("安装命令: conda install -c conda-forge gcc_linux-64 gxx_linux-64")

    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",

        # 排除不需要的模块
        "--nofollow-import-to=tkinter",
        "--nofollow-import-to=unittest",
        "--nofollow-import-to=test",
        "--nofollow-import-to=tests",
        "--nofollow-import-to=testing",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=jupyter",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=numpy",
        "--nofollow-import-to=pandas",
        "--nofollow-import-to=scipy",
        "--nofollow-import-to=torch",
        "--nofollow-import-to=PIL",
        "--nofollow-import-to=cv2",

        # 输出
        "--output-dir=dist",
        "--output-filename=ripperdoc",

        # 包含数据文件
        "--include-package-data=ripperdoc",
        "--include-package-data=tiktoken",
        "--include-package-data=tiktoken_ext",
    ]

    # 单文件/目录模式
    if not args.no_onefile:
        cmd.append("--onefile")

    # LTO 优化
    if use_lto:
        cmd.append("--lto=yes")
        # 设置使用 conda gcc
        os.environ["CC"] = str(Path(conda_gcc_path).parent / "gcc")
        os.environ["CXX"] = str(Path(conda_gcc_path).parent / "g++")

    cmd.append(str(main_script))

    print("运行 Nuitka 构建...")
    print(" ".join(cmd))

    # 清理旧的构建
    build_dir = root_dir / "dist" / "cli.build"
    if build_dir.exists():
        print(f"清理旧构建: {build_dir}")
        shutil.rmtree(build_dir)

    result = subprocess.run(cmd, cwd=root_dir)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
