from __future__ import annotations

from contextlib import contextmanager
import os
import sys

import matplotlib


def _configure_backend() -> None:
    if os.environ.get("MPLBACKEND"):
        return
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    matplotlib.use("Agg")


_configure_backend()

import matplotlib.pyplot as plt


@contextmanager
def managed_figure(*subplots_args, **subplots_kwargs):
    fig, axes = plt.subplots(*subplots_args, **subplots_kwargs)
    try:
        yield fig, axes
    finally:
        plt.close(fig)


__all__ = ["managed_figure", "plt"]
