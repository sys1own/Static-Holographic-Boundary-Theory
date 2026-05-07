from __future__ import annotations

"""Bootstrap-facing logic exports for rigidity discovery.

The active blind-search implementation continues to live in
``shbt.core.rigidity_landscape``. This module provides the refactored logic
namespace expected by updated tests and downstream imports.
"""

from shbt.core.rigidity_landscape import SymmetrySearcher

__all__ = ["SymmetrySearcher"]
