# -*- mode: python ; coding: utf-8 -*-

"""
Ripperdoc PyInstaller Spec File

使用方法:
    # 单文件模式 (默认)
    pyinstaller ripperdoc.spec

    # 目录模式 (推荐，支持外部数据文件)
    pyinstaller ripperdoc.spec --onedir

或使用构建脚本:
    python scripts/build.py           # 单文件模式
    python scripts/build.py --dir     # 目录模式
"""

import os
import sys
from pathlib import Path
from importlib.util import find_spec
from PyInstaller.utils.hooks import collect_submodules

# 构建模式: 'onefile' (单文件) 或 'onedir' (目录)
# 可以通过 --onedir 参数覆盖，或在此处修改默认值
BUILD_MODE = os.environ.get("RIPPERDOC_BUILD_MODE", "onedir")

# 项目根目录
ROOT_DIR = Path(SPECPATH).parent

# 数据文件目录
DATA_DIR = ROOT_DIR / "ripperdoc" / "data"

# 收集所有数据文件
datas = []
if DATA_DIR.exists():
    for file_path in DATA_DIR.glob("*.json"):
        datas.append((str(file_path), "ripperdoc/data"))

# 隐藏导入 - PyInstaller 有时无法自动检测这些
hiddenimports = [
    # 核心依赖
    "anthropic",
    "openai",
    "click",
    "rich",
    "textual",
    "pydantic",
    "aiofiles",
    "prompt_toolkit",
    "yaml",
    "mcp",
    "json_repair",
    "tiktoken",
    "charset_normalizer",
    
    # Ripperdoc 内部模块
    "ripperdoc.cli",
    "ripperdoc.core",
    "ripperdoc.tools",
    "ripperdoc.utils",
    "ripperdoc.data",
    
    # LSP 相关
]

# 仅在环境已安装时才加入，避免在CI里出现假阳性 "Hidden import not found" 警告。
_optional_hiddenimports = [
    "google.genai",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
    "pygls.lsp",
    "pygls.protocol",
]

def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False
    except Exception:
        return False


for _module in _optional_hiddenimports:
    if _module_available(_module):
        hiddenimports.append(_module)

# 强制收集 rich 的 unicode 数据子模块，避免打包后动态导入缺失。
hiddenimports.extend(
    [module for module in collect_submodules("rich._unicode_data") if module not in hiddenimports]
)

# Windows ARM 下 strip 会频繁触发文件格式警告（不影响产物），这里统一关闭。
WINDOWS_NO_STRIP = sys.platform == "win32"
STRIP_BINARIES = not WINDOWS_NO_STRIP

block_cipher = None

a = Analysis(
    ["ripperdoc/cli/cli.py"],
    pathex=[str(ROOT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT_DIR / "scripts" / "pyinstaller_hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除测试相关
        "pytest",
        "_pytest",
        "tests",
        # 排除开发工具
        "black",
        "ruff",
        "mypy",
        "pylint",
        "flake8",
        # 排除不需要的标准库
        "tkinter",
        "turtledemo",
        "unittest",
        "pydoc",
        "doctest",
        "lib2to3",
        "xmlrpc",
        "msilib",
        "optparse",
        "poplib",
        "imaplib",
        "nntplib",
        "smtplib",
        "smtpd",
        "telnetlib",
        "ftplib",
        "webbrowser",
        "cgi",
        "cgitb",
        "wsgiref",
        # 排除不需要的第三方库
        "IPython",
        "ipywidgets",
        "jupyter",
        "jupyter_client",
        "jupyter_core",
        "nbconvert",
        "nbformat",
        "notebook",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "tensorflow",
        "keras",
        "sklearn",
        "scikit-learn",
        "PIL",
        "pillow",
        "cv2",
        "opencv",
        "sympy",
        "nose",
        "sphinx",
        "docutils",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 根据构建模式选择不同的打包方式
if BUILD_MODE == "onedir":
    # 目录模式: 可执行文件 + 依赖文件 + 数据文件在同一目录
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="ripperdoc",
        debug=False,
        bootloader_ignore_signals=False,
        strip=STRIP_BINARIES,
        upx=True,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="ripperdoc",
    )
else:
    # 单文件模式: 所有内容打包成一个可执行文件
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="ripperdoc",
        debug=False,
        bootloader_ignore_signals=False,
        strip=STRIP_BINARIES,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,
    )
