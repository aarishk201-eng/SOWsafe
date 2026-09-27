"""Root-level export forwarding to satquery.backend.agents."""

import os
import sys

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from satquery.backend.agents import (
    SatQueryRouter,
    query_router,
    route,
    classify_intent,
    TaskType,
    TaskName,
    KNOWN_TASKS,
    VALID_TASKS,
    TASK_TOOLS,
    ExecutionPlanner,
    ExecutionGraph,
    STAGE_CATALOG,
    planner,
    ToolRegistry,
    TaskStep,
    TaskGraph,
    ExecutionResult,
    build_task_graph,
    execute,
)

__all__ = [
    "SatQueryRouter",
    "query_router",
    "route",
    "classify_intent",
    "TaskType",
    "TaskName",
    "KNOWN_TASKS",
    "VALID_TASKS",
    "TASK_TOOLS",
    "ExecutionPlanner",
    "ExecutionGraph",
    "STAGE_CATALOG",
    "planner",
    "ToolRegistry",
    "TaskStep",
    "TaskGraph",
    "ExecutionResult",
    "build_task_graph",
    "execute",
]

