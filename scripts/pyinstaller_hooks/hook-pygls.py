"""PyInstaller hook for pygls LSP library"""

from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("pygls")
