#!/usr/bin/env python3
"""使用 Nuitka 构建 Ripperdoc

Nuitka 将 Python 编译为 C 代码再编译为机器码，
通常比 PyInstaller 生成更小更快的可执行文件。

安装依赖:
    pip install nuitka

使用:
    python scripts/build_nuitka.py              # 目录模式构建 (默认)
    python scripts/build_nuitka.py --onefile    # 单文件构建
    python scripts/build_nuitka.py --lto        # 启用 LTO 优化 (需要 gcc)
    python scripts/build_nuitka.py --providers anthropic,openai,gemini  # 按需打包 provider
    python scripts/build_nuitka.py --min-size   # 体积优先（速度可能更慢）
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SLIM_NOFOLLOW_IMPORTS = [
    # Requests 的 pyopenssl 兼容分支会把 cryptography 整套拉入。
    # 对现代 Linux + 标准 ssl 场景通常非必需。
    "requests",
    "urllib3.contrib.pyopenssl",
    "OpenSSL",
    "cryptography",
    # Google ADC 某些分支会引入 cryptography；API Key 直连通常不需要。
    "google.auth.transport.requests",
    "google.auth._agent_identity_utils",
]


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


def strip_elf_binaries(target_files: list[Path]) -> None:
    """Best-effort strip ELF binaries to reduce output size."""
    strip_exe = shutil.which("strip")
    if not strip_exe:
        print("提示: 未找到 strip，跳过二进制瘦身。")
        return

    stripped = 0
    for file_path in target_files:
        if not file_path.exists() or not file_path.is_file():
            continue
        result = subprocess.run(
            [strip_exe, "--strip-unneeded", str(file_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            stripped += 1
    print(f"strip 已处理文件数: {stripped}")


def main() -> int:
    raw_args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="使用 Nuitka 构建 Ripperdoc")
    parser.add_argument("--lto", action="store_true", help="启用链接时优化 (需要 gcc)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--onefile", action="store_true", help="使用单文件模式（默认: 目录模式）")
    # 向后兼容旧参数，保持可用但不在 help 中展示
    mode_group.add_argument("--no-onefile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--conda-env",
        default="ripperdoc_build",
        help="目标 conda 环境名称 (默认: ripperdoc_build)",
    )
    parser.add_argument("--no-conda", action="store_true", help="禁用 conda 环境引导，使用当前 Python")
    parser.add_argument(
        "--slim",
        action="store_true",
        help="瘦身构建：剔除部分可选依赖路径以减小体积（可能影响 Google ADC/Requests 兼容路径）",
    )
    parser.add_argument(
        "--exclude-module",
        action="append",
        default=[],
        help="额外排除模块/包（可重复），等价于增加 --nofollow-import-to=<module>",
    )
    parser.add_argument(
        "--exclude-data-file",
        action="append",
        default=[],
        help="额外排除数据文件模式（可重复），等价于增加 --noinclude-data-files=<pattern>",
    )
    parser.add_argument(
        "--providers",
        default="anthropic,openai,gemini",
        help=(
            "要包含的 provider 客户端（逗号分隔）：anthropic,openai,gemini。"
            "默认 anthropic,openai（更快）；需要 Gemini 时再加 gemini。"
        ),
    )
    parser.add_argument(
        "--strip-binaries",
        action="store_true",
        help="构建成功后对产物执行 strip（减小体积）。",
    )
    parser.add_argument(
        "--min-size",
        action="store_true",
        help=(
            "体积优先预设：启用 --slim、providers=anthropic,openai、--strip-binaries。"
        ),
    )
    parser.add_argument("--force-conda", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_skip-conda-bootstrap", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.force_conda:
        print("提示: --force-conda 已弃用，conda 构建现在默认支持。")

    if args.min_size:
        args.slim = True
        args.providers = "anthropic,openai"
        args.strip_binaries = True
        print("启用体积优先预设: slim + providers=anthropic,openai + strip")

    bootstrap_exit_code = maybe_bootstrap_to_conda(args, raw_args)
    if bootstrap_exit_code is not None:
        return bootstrap_exit_code

    root_dir = Path(__file__).parent.parent
    main_script = root_dir / "ripperdoc" / "cli" / "cli.py"

    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--python-flag=no_docstrings",

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

    # rich 在运行时会动态 import rich._unicode_data.unicode*-*-*。
    # 这些模块名带 '-'，Nuitka 无法通过静态分析自动发现，必须显式 include-module。
    try:
        from rich._unicode_data._versions import VERSIONS as RICH_UNICODE_VERSIONS

        cmd.append("--include-module=rich._unicode_data._versions")
        for version in RICH_UNICODE_VERSIONS:
            module_name = f"rich._unicode_data.unicode{version.replace('.', '-')}"
            cmd.append(f"--include-module={module_name}")
        print(f"已包含 rich unicode 模块: {len(RICH_UNICODE_VERSIONS)} 个")
    except Exception as exc:
        print(f"警告: 无法解析 rich unicode 版本列表，可能导致运行时缺少字符宽度表: {exc}")

    # provider 客户端通过 importlib 动态导入（ripperdoc.core.providers.__init__）。
    # Nuitka 无法静态推断动态模块名，需要显式 include-module。
    provider_module_map = {
        "anthropic": "ripperdoc.core.providers.anthropic",
        "openai": "ripperdoc.core.providers.openai",
        "gemini": "ripperdoc.core.providers.gemini",
    }
    requested_provider_keys = [
        key.strip().lower() for key in (args.providers or "").split(",") if key.strip()
    ]
    unknown_provider_keys = [key for key in requested_provider_keys if key not in provider_module_map]
    if unknown_provider_keys:
        parser.error(
            f"未知 providers: {', '.join(unknown_provider_keys)}. "
            f"可选值: {', '.join(provider_module_map.keys())}"
        )

    selected_provider_keys = requested_provider_keys or ["anthropic", "openai"]
    print(f"包含 provider 模块: {', '.join(selected_provider_keys)}")
    for key in selected_provider_keys:
        module_name = provider_module_map[key]
        cmd.append(f"--include-module={module_name}")

    if os.environ.get("CONDA_PREFIX"):
        # conda 下禁用 static libpython，避免 setns 等符号链接失败
        cmd.append("--static-libpython=no")

    # 单文件/目录模式（默认目录模式）
    if args.onefile:
        cmd.append("--output-filename=ripperdoc")
        cmd.append("--onefile")
    else:
        # standalone 目录模式会把可执行文件和包目录放在同级，避免与 ripperdoc/ 目录重名
        cmd.append("--output-filename=ripperdoc_cli")

    # LTO 优化
    if args.lto:
        cmd.append("--lto=yes")

    nofollow_imports = set(args.exclude_module)
    if args.slim:
        nofollow_imports.update(SLIM_NOFOLLOW_IMPORTS)

    for module_name in sorted(nofollow_imports):
        cmd.append(f"--nofollow-import-to={module_name}")

    for file_pattern in args.exclude_data_file:
        cmd.append(f"--noinclude-data-files={file_pattern}")

    cmd.append(str(main_script))

    print("运行 Nuitka 构建...")
    print(shlex.join(cmd))

    # 清理旧的构建
    build_dir = root_dir / "dist" / "cli.build"
    if build_dir.exists():
        print(f"清理旧构建: {build_dir}")
        shutil.rmtree(build_dir)

    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        return result.returncode

    if args.strip_binaries:
        target_files: list[Path] = []
        if args.onefile:
            target_files.append(root_dir / "dist" / "ripperdoc")
        else:
            dist_dir = root_dir / "dist" / "cli.dist"
            target_files.append(dist_dir / "ripperdoc_cli")
            target_files.extend(dist_dir.rglob("*.so"))
        strip_elf_binaries(target_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())
