"""PyInstaller hook for tiktoken"""

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

# Collect tiktoken data files and metadata.
datas = collect_data_files("tiktoken") + copy_metadata("tiktoken")

# Hidden imports: the tiktoken_ext namespace package is required.
hiddenimports = [
    "tiktoken.registry",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
]
