#!/usr/bin/env python3
"""使用 Nuitka 构建 Ripperdoc

Nuitka 将 Python 编译为 C 代码再编译为机器码，
通常比 PyInstaller 生成更小更快的可执行文件。

安装: pip install nuitka

使用: python scripts/build_nuitka.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    root_dir = Path(__file__).parent.parent
    main_script = root_dir / "ripperdoc" / "cli" / "cli.py"

    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",  # 独立可执行
        "--onefile",     # 单文件
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

        # 优化选项
        "--lto=yes",  # 链接时优化
        "--disable-console" if sys.platform == "win32" else "--enable-console",

        # 输出
        "--output-dir=dist",
        "--output-filename=ripperdoc",

        # 包含数据文件
        f"--include-package-data=ripperdoc",
        f"--include-package-data=tiktoken",
        f"--include-package-data=tiktoken_ext",

        str(main_script)
    ]

    print("运行 Nuitka 构建...")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=root_dir)


if __name__ == "__main__":
    main()
