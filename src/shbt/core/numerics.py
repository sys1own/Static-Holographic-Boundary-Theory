from __future__ import annotations

"""Backward-compatible numeric helper exports.

The numerical helper surface was promoted into ``differential_geometry.py`` so
the geometry and consistency primitives live together. This compatibility
module preserves the original import path.
"""

from shbt.core import differential_geometry as _differential_geometry
from shbt.core.differential_geometry import *  # noqa: F401,F403


__all__ = _differential_geometry.__all__
