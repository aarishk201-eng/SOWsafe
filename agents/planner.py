"""Root-level export forwarding to satquery.backend.agents.planner."""

import os
import sys

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from satquery.backend.agents.planner import (
    ExecutionPlanner,
    ExecutionGraph,
    STAGE_CATALOG,
    planner,
    TaskStep,
    TaskGraph,
    ExecutionResult,
    build_task_graph,
    execute,
)

__all__ = [
    "ExecutionPlanner",
    "ExecutionGraph",
    "STAGE_CATALOG",
    "planner",
    "TaskStep",
    "TaskGraph",
    "ExecutionResult",
    "build_task_graph",
    "execute",
]

