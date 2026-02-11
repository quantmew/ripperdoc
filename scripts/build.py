#!/usr/bin/env python3
"""Ripperdoc 构建脚本

使用 PyInstaller 将 Ripperdoc 打包为独立的可执行文件。

支持两种构建模式:
    --onefile    单文件模式 (默认): 所有内容打包成一个可执行文件
    --dir, -d    目录模式 (推荐): 可执行文件 + 依赖文件在同一目录，支持外部数据文件
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """运行命令并返回退出码"""
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def check_upx() -> bool:
    """检查 UPX 是否可用"""
    try:
        result = subprocess.run(["upx", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"UPX 已安装: {version}")
            return True
    except FileNotFoundError:
        pass
    return False


def download_upx(tools_dir: Path) -> Path | None:
    """下载 UPX 压缩工具"""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # 确定 UPX 下载链接
    if system == "linux" and machine == "x86_64":
        url = "https://github.com/upx/upx/releases/download/v4.2.4/upx-4.2.4-amd64_linux.tar.xz"
        extract_dir = "upx-4.2.4-amd64_linux"
        exe_name = "upx"
    elif system == "linux" and machine == "arm64":
        url = "https://github.com/upx/upx/releases/download/v4.2.4/upx-4.2.4-arm64_linux.tar.xz"
        extract_dir = "upx-4.2.4-arm64_linux"
        exe_name = "upx"
    elif system == "darwin":
        url = "https://github.com/upx/upx/releases/download/v4.2.4/upx-4.2.4-darwin_amd64.tar.xz"
        extract_dir = "upx-4.2.4-darwin_amd64"
        exe_name = "upx"
    elif system == "windows":
        url = "https://github.com/upx/upx/releases/download/v4.2.4/upx-4.2.4-win64.zip"
        extract_dir = "upx-4.2.4-win64"
        exe_name = "upx.exe"
    else:
        print(f"不支持的系统: {system} {machine}")
        return None

    tools_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tools_dir / url.split("/")[-1]
    upx_exe = tools_dir / exe_name

    if upx_exe.exists():
        print(f"UPX 已存在: {upx_exe}")
        return upx_exe

    print(f"下载 UPX: {url}")
    try:
        urllib.request.urlretrieve(url, archive_path)
        print(f"解压到: {tools_dir}")

        if url.endswith(".tar.xz"):
            import lzma
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(tools_dir)
            # 移动到工具目录
            extracted_exe = tools_dir / extract_dir / exe_name
            if extracted_exe.exists():
                shutil.move(str(extracted_exe), str(upx_exe))
                # 清理解压目录
                shutil.rmtree(tools_dir / extract_dir)
        elif url.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(tools_dir)
            extracted_exe = tools_dir / extract_dir / exe_name
            if extracted_exe.exists():
                shutil.move(str(extracted_exe), str(upx_exe))
                shutil.rmtree(tools_dir / extract_dir)

        # 清理压缩包
        archive_path.unlink()

        # 设置执行权限
        if system != "windows":
            os.chmod(upx_exe, 0o755)

        print(f"UPX 下载完成: {upx_exe}")
        return upx_exe
    except Exception as e:
        print(f"下载 UPX 失败: {e}")
        return None


def clean_build_dirs(root_dir: Path) -> None:
    """清理构建目录"""
    dirs_to_clean = ["build", "dist", "spec"]
    for dir_name in dirs_to_clean:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            print(f"清理目录: {dir_path}")
            shutil.rmtree(dir_path)


def copy_data_files(root_dir: Path, dist_dir: Path) -> None:
    """复制数据文件到输出目录 (用于目录模式)

    PyInstaller onedir 模式的目录结构:
        dist/ripperdoc/
        ├── ripperdoc          # 可执行文件
        └── _internal/         # 依赖文件和数据
            └── ripperdoc/
                └── data/
    """
    src_data_dir = root_dir / "ripperdoc" / "data"
    # 数据文件应放在 _internal 目录中，因为 ripperdoc 是可执行文件而非目录
    dst_data_dir = dist_dir / "_internal" / "ripperdoc" / "data"

    if not src_data_dir.exists():
        return

    # 如果目标路径是文件而不是目录，先删除它
    if dst_data_dir.exists() and dst_data_dir.is_file():
        dst_data_dir.unlink()

    dst_data_dir.mkdir(parents=True, exist_ok=True)

    for json_file in src_data_dir.glob("*.json"):
        dst_file = dst_data_dir / json_file.name
        shutil.copy2(json_file, dst_file)
        print(f"复制数据文件: {json_file.name} -> {dst_data_dir}")


def get_dir_size(path: Path) -> int:
    """获取目录总大小"""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def main() -> int:
    """主构建流程"""
    parser = argparse.ArgumentParser(
        description="构建 Ripperdoc 可执行文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/build.py           # 单文件模式 (默认)
    python scripts/build.py --dir     # 目录模式 (推荐)
    python scripts/build.py --onefile # 单文件模式
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

    # 确定构建模式
    if args.dir:
        build_mode = "onedir"
    elif args.onefile:
        build_mode = "onefile"
    else:
        build_mode = "onedir"  # 默认使用目录模式

    root_dir = Path(__file__).parent.parent
    spec_file = root_dir / "ripperdoc.spec"
    tools_dir = root_dir / "tools"

    # 检查 spec 文件
    if not spec_file.exists():
        print(f"错误: 找不到 spec 文件: {spec_file}")
        return 1

    # 检查 PyInstaller
    try:
        import PyInstaller
        print(f"使用 PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("错误: 未安装 PyInstaller")
        print("请运行: pip install pyinstaller")
        return 1

    # 设置构建模式环境变量
    os.environ["RIPPERDOC_BUILD_MODE"] = build_mode

    # 检查/下载 UPX
    print(f"\n=== 构建模式: {build_mode} ===")
    print("\n=== 检查 UPX ===")
    upx_path = None
    if not check_upx():
        print("UPX 未安装，正在下载...")
        upx_path = download_upx(tools_dir)
        if upx_path:
            # 将 UPX 路径添加到 PATH
            os.environ["PATH"] = str(tools_dir) + os.pathsep + os.environ.get("PATH", "")
    else:
        upx_path = "upx"  # 使用系统已安装的

    # 清理旧的构建
    print("\n=== 清理旧的构建目录 ===")
    clean_build_dirs(root_dir)

    # 运行 PyInstaller
    print("\n=== 开始构建 ===")
    cmd = [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean"]
    exit_code = run_command(cmd, cwd=root_dir)

    if exit_code != 0:
        print(f"\n错误: 构建失败 (退出码: {exit_code})")
        return exit_code

    # 显示结果
    dist_dir = root_dir / "dist"

    if build_mode == "onedir":
        output_dir = dist_dir / "ripperdoc"
        exe_file = output_dir / "ripperdoc"

        if not exe_file.exists():
            print(f"\n错误: 构建完成但找不到输出目录: {output_dir}")
            return 1

        # 复制数据文件
        print("\n=== 复制数据文件 ===")
        copy_data_files(root_dir, output_dir)

        # 计算目录大小
        dir_size = get_dir_size(output_dir)
        size_mb = dir_size / (1024 * 1024)

        print(f"\n=== 构建成功 ===")
        print(f"输出目录: {output_dir}")
        print(f"可执行文件: {exe_file}")
        print(f"目录大小: {size_mb:.1f} MB")
        if upx_path:
            print(f"UPX 压缩: 已启用")
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
        if upx_path:
            print(f"UPX 压缩: 已启用")
        print(f"\n可以运行: {exe_file} --help")

    return 0


if __name__ == "__main__":
    sys.exit(main())
