"""Compatibility entry point for the current manuscript workflow."""

from pathlib import Path
import os
import runpy

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.chdir(root)
    runpy.run_path(str(root / 'analysis/uniform_revision/make_uniform_hadgem_data.py'), run_name="__main__")
