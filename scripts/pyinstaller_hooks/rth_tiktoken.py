"""Runtime hook for tiktoken - ensure tiktoken_ext namespace package works in frozen app"""

import sys
import os

# 在 frozen 环境中修复 tiktoken_ext 命名空间包
if getattr(sys, 'frozen', False):
    # 确保 tiktoken_ext 被正确识别为命名空间包
    import tiktoken_ext
    # 设置 __path__ 以便 pkgutil.iter_modules 能找到子模块
    if hasattr(tiktoken_ext, '__path__'):
        tiktoken_ext.__path__ = [p for p in tiktoken_ext.__path__ if os.path.exists(p)]
