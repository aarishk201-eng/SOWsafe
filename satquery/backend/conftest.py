"""Pytest configuration for the SatQuery backend.

Ensures the backend package root is importable (``main``, ``agents``,
``models``, ``geospatial``, ``evidence`` …) regardless of how pytest is invoked
(full suite from the backend directory, or an explicit subset of test files).
Without this, pytest's default ``prepend`` import mode can fail to place the
backend root on ``sys.path`` when given explicit file arguments, surfacing as
``ModuleNotFoundError: No module named 'agents.registry'``.
"""

import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
