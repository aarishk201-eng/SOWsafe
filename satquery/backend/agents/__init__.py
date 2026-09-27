"""SatQuery multi-agent engine package.

Aggregates the router, planner, registry, and trace builder into a single
public surface so callers can do ``from agents import SatQueryRouter`` etc.
"""

from .router import (
    SatQueryRouter,
    query_router,
    route,
    classify_intent,
    TaskType,
    TaskName,
    KNOWN_TASKS,
    VALID_TASKS,
    TASK_TOOLS,
)
from .planner import (
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
from .registry import ToolRegistry
from .trace import TraceBuilder, AuditTrace

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
    "TraceBuilder",
    "AuditTrace",
]
