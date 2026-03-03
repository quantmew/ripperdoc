"""PyInstaller hook for pygls LSP library"""

from PyInstaller.utils.hooks import copy_metadata  # type: ignore[import-untyped]

datas = copy_metadata("pygls")
