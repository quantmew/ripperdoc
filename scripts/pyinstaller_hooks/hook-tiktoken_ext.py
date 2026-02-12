"""PyInstaller hook for tiktoken_ext namespace package"""

from PyInstaller.utils.hooks import collect_submodules

# tiktoken_ext is a namespace package; collect all submodules.
hiddenimports = collect_submodules("tiktoken_ext")
