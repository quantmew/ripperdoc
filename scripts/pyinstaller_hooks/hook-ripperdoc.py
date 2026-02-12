"""PyInstaller hook for ripperdoc package"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all ripperdoc submodules.
hiddenimports = collect_submodules("ripperdoc")

# Collect data files (ripperdoc/data/*.json).
# Note: append to datas; do not overwrite datas defined in the spec file.
datas = collect_data_files("ripperdoc", include_py_files=False, excludes=["*.py", "*.pyc", "__pycache__"])
