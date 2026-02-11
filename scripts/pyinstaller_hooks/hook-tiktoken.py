"""PyInstaller hook for tiktoken"""

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

# 收集 tiktoken 的数据文件和元数据
datas = collect_data_files("tiktoken") + copy_metadata("tiktoken")

# 隐藏导入 - 关键是 tiktoken_ext 命名空间包
hiddenimports = [
    "tiktoken.registry",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
]
