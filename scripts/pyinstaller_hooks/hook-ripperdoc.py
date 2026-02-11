"""PyInstaller hook for ripperdoc package"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 收集 ripperdoc 的所有子模块
hiddenimports = collect_submodules("ripperdoc")

# 收集数据文件 (ripperdoc/data/*.json)
# 注意：使用 append 模式，不要覆盖 spec 文件中的 datas
datas = collect_data_files("ripperdoc", include_py_files=False, excludes=["*.py", "*.pyc", "__pycache__"])
