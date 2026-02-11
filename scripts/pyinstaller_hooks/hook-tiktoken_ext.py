"""PyInstaller hook for tiktoken_ext namespace package"""

from PyInstaller.utils.hooks import collect_submodules

# tiktoken_ext 是命名空间包，需要收集所有子模块
hiddenimports = collect_submodules("tiktoken_ext")
