#!/usr/bin/env python3
"""使用 Nuitka 构建 Ripperdoc

Nuitka 将 Python 编译为 C 代码再编译为机器码，
通常比 PyInstaller 生成更小更快的可执行文件。

安装依赖:
    pip install nuitka

使用:
    python scripts/build_nuitka.py              # 基础构建
    python scripts/build_nuitka.py --lto        # 启用 LTO 优化 (需要 gcc)
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def get_current_conda_env() -> str | None:
    """返回当前 conda 环境名，非 conda 环境则返回 None。"""
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    return Path(conda_prefix).name


def maybe_bootstrap_to_conda(args: argparse.Namespace, raw_args: list[str]) -> int | None:
    """必要时自动切换到目标 conda 环境运行当前脚本。"""
    if args.no_conda or args._skip_conda_bootstrap:
        return None

    current_env = get_current_conda_env()
    if current_env == args.conda_env:
        print(f"检测到目标 conda 环境: {current_env}")
        return None

    conda_exe = shutil.which("conda")
    if not conda_exe:
        print("警告: 未找到 conda，回退到当前 Python 继续构建。")
        print("提示: 可使用 --no-conda 显式关闭 conda 引导。")
        return None

    # 避免递归引导
    forwarded_args = [arg for arg in raw_args if arg != "--_skip-conda-bootstrap"]
    forwarded_args.append("--_skip-conda-bootstrap")

    bootstrap_cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(Path(__file__).resolve()),
        *forwarded_args,
    ]
    print(f"切换到 conda 环境构建: {args.conda_env}")
    print(shlex.join(bootstrap_cmd))
    result = subprocess.run(bootstrap_cmd)
    return result.returncode


def main() -> int:
    raw_args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="使用 Nuitka 构建 Ripperdoc")
    parser.add_argument("--lto", action="store_true", help="启用链接时优化 (需要 gcc)")
    parser.add_argument("--no-onefile", action="store_true", help="使用目录模式而非单文件")
    parser.add_argument(
        "--conda-env",
        default="ripperdoc_build",
        help="目标 conda 环境名称 (默认: ripperdoc_build)",
    )
    parser.add_argument("--no-conda", action="store_true", help="禁用 conda 环境引导，使用当前 Python")
    parser.add_argument("--force-conda", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_skip-conda-bootstrap", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.force_conda:
        print("提示: --force-conda 已弃用，conda 构建现在默认支持。")

    bootstrap_exit_code = maybe_bootstrap_to_conda(args, raw_args)
    if bootstrap_exit_code is not None:
        return bootstrap_exit_code

    root_dir = Path(__file__).parent.parent
    main_script = root_dir / "ripperdoc" / "cli" / "cli.py"

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

        # 包含数据文件
        "--include-package-data=ripperdoc",
        "--include-package-data=tiktoken",
        "--include-package-data=tiktoken_ext",
    ]

    if os.environ.get("CONDA_PREFIX"):
        # conda 下禁用 static libpython，避免 setns 等符号链接失败
        cmd.append("--static-libpython=no")

    # 单文件/目录模式
    if not args.no_onefile:
        cmd.append("--output-filename=ripperdoc")
        cmd.append("--onefile")
    else:
        # standalone 目录模式会把可执行文件和包目录放在同级，避免与 ripperdoc/ 目录重名
        cmd.append("--output-filename=ripperdoc_cli")

    # LTO 优化
    if args.lto:
        cmd.append("--lto=yes")

    cmd.append(str(main_script))

    print("运行 Nuitka 构建...")
    print(shlex.join(cmd))

    # 清理旧的构建
    build_dir = root_dir / "dist" / "cli.build"
    if build_dir.exists():
        print(f"清理旧构建: {build_dir}")
        shutil.rmtree(build_dir)

    result = subprocess.run(cmd, cwd=root_dir)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
