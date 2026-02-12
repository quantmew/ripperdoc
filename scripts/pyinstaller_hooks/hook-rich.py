"""PyInstaller hook for rich library"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all rich submodules, including dynamically loaded modules under _unicode_data.
hiddenimports = collect_submodules("rich")

# Collect data files.
datas = collect_data_files("rich")
