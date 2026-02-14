# -*- mode: python ; coding: utf-8 -*-
"""
Ripperdoc PyInstaller Spec File - 目录模式 (onedir)

目录模式比单文件模式:
- 启动更快 (无需解压)
- 更新更容易 (只替换变更文件)
- 分发时用 zip/tar 打包

使用方法:
    pyinstaller ripperdoc_onedir.spec
"""

import os
from pathlib import Path

ROOT_DIR = Path(SPECPATH).parent
DATA_DIR = ROOT_DIR / "ripperdoc" / "data"

datas = []
if DATA_DIR.exists():
    for file_path in DATA_DIR.glob("*.json"):
        datas.append((str(file_path), "ripperdoc/data"))

hiddenimports = [
    "anthropic", "openai", "click", "rich", "textual", "pydantic",
    "aiofiles", "prompt_toolkit", "yaml", "mcp", "json_repair",
    "tiktoken", "tiktoken_ext", "tiktoken_ext.openai_public",
    "charset_normalizer",
    "ripperdoc.cli", "ripperdoc.core", "ripperdoc.tools", "ripperdoc.utils",
    "asyncio",
    "rich._unicode_data", "rich._unicode_data._versions",
    *[f"rich._unicode_data.unicode{i}-{j}-0"
      for i in range(4, 18) for j in range(0, 3)] + ["rich._unicode_data.unicode17-0-0"],
]

a = Analysis(
    ["ripperdoc/cli/cli.py"],
    pathex=[str(ROOT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT_DIR / "scripts" / "pyinstaller_hooks")],
    runtime_hooks=[],
    excludes=[
        "pytest", "_pytest", "tests",
        "black", "ruff", "mypy", "pylint", "flake8",
        "tkinter", "turtledemo", "unittest", "pydoc", "doctest",
        "lib2to3", "xmlrpc",
        "IPython", "ipywidgets", "jupyter",
        "matplotlib", "numpy", "pandas", "scipy", "torch",
        "tensorflow", "keras", "sklearn", "PIL", "cv2",
        "sympy", "nose", "sphinx", "docutils",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # 目录模式关键设置
    name="ripperdoc",
    debug=False,
    strip=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    name="ripperdoc",
)
