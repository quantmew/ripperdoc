"""Runtime hook for tiktoken - ensure tiktoken_ext namespace package works in frozen app"""

import sys
import os

# Fix tiktoken_ext namespace package handling in frozen environments.
if getattr(sys, 'frozen', False):
    # Ensure tiktoken_ext is recognized as a namespace package.
    import tiktoken_ext
    # Set __path__ so pkgutil.iter_modules can discover submodules.
    if hasattr(tiktoken_ext, '__path__'):
        tiktoken_ext.__path__ = [p for p in tiktoken_ext.__path__ if os.path.exists(p)]
