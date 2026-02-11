"""PyInstaller hook for rich library"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 收集 rich 的所有子模块（包括 _unicode_data 下动态加载的模块）
hiddenimports = collect_submodules("rich")

# 收集数据文件
datas = collect_data_files("rich")
