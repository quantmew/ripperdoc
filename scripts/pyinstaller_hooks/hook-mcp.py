"""PyInstaller hook for mcp package"""

from PyInstaller.utils.hooks import copy_metadata  # type: ignore[import-untyped]

datas = copy_metadata("mcp")
