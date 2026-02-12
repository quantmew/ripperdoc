#!/usr/bin/env python3
"""Build Ripperdoc with Nuitka.

Nuitka compiles Python to C and then to machine code.
It often produces smaller and faster binaries than PyInstaller.

Install dependency:
    pip install nuitka

Usage:
    python scripts/build_nuitka.py              # directory mode (default)
    python scripts/build_nuitka.py --onefile    # one-file mode
    python scripts/build_nuitka.py --lto        # enable LTO optimization (requires gcc)
    python scripts/build_nuitka.py --clang      # compile with clang (default is gcc)
    python scripts/build_nuitka.py --strip-binaries  # strip binaries after build (best effort)
    python scripts/build_nuitka.py --jobs 4     # limit parallel compile jobs to reduce memory peak
    python scripts/build_nuitka.py --providers anthropic,openai,gemini  # include providers on demand
    python scripts/build_nuitka.py --min-size   # prioritize binary size (may be slower)
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SLIM_NOFOLLOW_IMPORTS = [
    # Requests' pyopenssl compatibility branch can pull in the full cryptography stack.
    # This is usually unnecessary on modern Linux with standard ssl.
    "requests",
    "urllib3.contrib.pyopenssl",
    "OpenSSL",
    "cryptography",
    # Some Google ADC paths can introduce cryptography; direct API key usage usually does not.
    "google.auth.transport.requests",
    "google.auth._agent_identity_utils",
]


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _binary_name(base_name: str) -> str:
    return f"{base_name}.exe" if _is_windows() else base_name


def _append_flag(env: dict[str, str], key: str, flag: str) -> None:
    current = env.get(key, "").strip()
    if flag in current.split():
        return
    env[key] = f"{current} {flag}".strip()


def get_current_conda_env() -> str | None:
    """Return current conda environment name, or None if not in conda."""
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    return Path(conda_prefix).name


def maybe_bootstrap_to_conda(args: argparse.Namespace, raw_args: list[str]) -> int | None:
    """Re-run this script in the target conda environment when needed."""
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

    # Avoid recursive bootstrapping.
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


def strip_binaries(target_files: list[Path]) -> None:
    """Best-effort strip binaries to reduce output size across platforms."""
    strip_exe = shutil.which("strip") or shutil.which("llvm-strip")
    if not strip_exe:
        print("提示: 未找到 strip，跳过二进制瘦身。")
        return

    if _is_windows():
        candidate_templates = [
            ["--strip-unneeded", "{file}"],
            ["-S", "{file}"],
            ["{file}"],
        ]
    elif _is_linux():
        candidate_templates = [
            ["--strip-unneeded", "{file}"],
            ["-s", "{file}"],
            ["{file}"],
        ]
    else:
        # macOS / other UNIX: GNU strip flags may be unavailable, so use safer defaults.
        candidate_templates = [
            ["-x", "{file}"],
            ["-S", "{file}"],
            ["{file}"],
        ]

    stripped = 0
    for file_path in target_files:
        if not file_path.exists() or not file_path.is_file():
            continue
        for template in candidate_templates:
            cmd = [strip_exe, *[arg.format(file=str(file_path)) for arg in template]]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                stripped += 1
                break
    print(f"strip 已处理文件数: {stripped}")


def main() -> int:
    raw_args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="使用 Nuitka 构建 Ripperdoc")
    parser.add_argument("--clang", dest="clang", action="store_true", default=False, help="使用 clang 编译（默认: gcc）")
    parser.add_argument("--no-clang", dest="clang", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--lto", action="store_true", help="启用链接时优化 (需要 gcc)")
    parser.add_argument("--jobs", type=int, default=None, help="并行编译任务数（建议内存紧张时设为 2~8）")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--onefile", action="store_true", help="使用单文件模式（默认: 目录模式）")
    # Backward compatibility for legacy flags; keep available but hidden from help.
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
            "默认 anthropic,openai,gemini"
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

        # Exclude unnecessary modules.
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

        # Output.
        "--output-dir=dist",

        # Include data files.
        "--include-package-data=ripperdoc",
        "--include-package-data=tiktoken",
        "--include-package-data=tiktoken_ext",
    ]

    use_clang = False
    if args.clang:
        if shutil.which("clang"):
            cmd.append("--clang")
            use_clang = True
        else:
            print("警告: 未找到 clang，自动回退到 gcc。")

    # rich dynamically imports rich._unicode_data.unicode*-*-* at runtime.
    # Module names contain '-', so Nuitka cannot discover them via static analysis.
    try:
        from rich._unicode_data._versions import VERSIONS as RICH_UNICODE_VERSIONS

        cmd.append("--include-module=rich._unicode_data._versions")
        for version in RICH_UNICODE_VERSIONS:
            module_name = f"rich._unicode_data.unicode{version.replace('.', '-')}"
            cmd.append(f"--include-module={module_name}")
        print(f"已包含 rich unicode 模块: {len(RICH_UNICODE_VERSIONS)} 个")
    except Exception as exc:
        print(f"警告: 无法解析 rich unicode 版本列表，可能导致运行时缺少字符宽度表: {exc}")

    # Provider clients are imported dynamically via importlib (ripperdoc.core.providers.__init__).
    # Nuitka cannot infer dynamic module names statically, so include-module is required.
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

    if os.environ.get("CONDA_PREFIX") and _is_linux():
        # Disable static libpython on Linux conda to avoid link failures (e.g., setns symbols).
        cmd.append("--static-libpython=no")

    # One-file / directory mode (directory mode by default).
    if args.onefile:
        output_filename = _binary_name("ripperdoc")
        cmd.append("--onefile")
    else:
        # Standalone directory mode outputs binary and package dir at the same level to avoid name collisions.
        output_filename = _binary_name("ripperdoc_cli")
    cmd.append(f"--output-filename={output_filename}")

    # LTO optimization.
    if args.lto:
        cmd.append("--lto=yes")
        if args.jobs and args.jobs > 8:
            print("提示: LTO + 高并发会显著增加内存占用，建议 --jobs 2~8。")

    if args.jobs is not None:
        if args.jobs <= 0:
            parser.error("--jobs 必须是正整数")
        cmd.append(f"--jobs={args.jobs}")

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

    # Clean previous build artifacts.
    build_dir = root_dir / "dist" / "cli.build"
    if build_dir.exists():
        print(f"清理旧构建: {build_dir}")
        shutil.rmtree(build_dir)

    run_env = os.environ.copy()
    if use_clang:
        # Nuitka may forward GCC-style warning flags; clang reports unknown-warning-option.
        # Silence these warnings to keep build logs readable.
        _append_flag(run_env, "CFLAGS", "-Wno-unknown-warning-option")
        _append_flag(run_env, "CXXFLAGS", "-Wno-unknown-warning-option")

    result = subprocess.run(cmd, cwd=root_dir, env=run_env)
    if result.returncode != 0:
        return result.returncode

    if args.strip_binaries:
        target_files: list[Path] = []
        if args.onefile:
            target_files.append(root_dir / "dist" / _binary_name("ripperdoc"))
        else:
            dist_dir = root_dir / "dist" / "cli.dist"
            target_files.append(dist_dir / _binary_name("ripperdoc_cli"))
            target_files.extend(dist_dir.rglob("*.so"))
            target_files.extend(dist_dir.rglob("*.dylib"))
            target_files.extend(dist_dir.rglob("*.dll"))
            target_files.extend(dist_dir.rglob("*.pyd"))
        strip_binaries(target_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())
