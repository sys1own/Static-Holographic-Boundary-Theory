from __future__ import annotations

"""Backward-compatible shim for the correspondence engine."""

from shbt.core import correspondence_engine as _correspondence_engine
from shbt.core.correspondence_engine import *  # noqa: F401,F403


__all__ = _correspondence_engine.__all__
