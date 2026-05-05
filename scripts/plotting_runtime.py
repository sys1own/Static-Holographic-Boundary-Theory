from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.plotting_runtime import managed_figure, plt

__all__ = ["managed_figure", "plt"]
