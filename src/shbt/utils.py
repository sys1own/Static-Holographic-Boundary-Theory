from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


@contextmanager
def managed_figure(*, figsize: tuple[float, float] | None = None) -> Iterator[tuple[Figure, Axes]]:
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


__all__ = ["managed_figure"]
