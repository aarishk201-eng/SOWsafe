"""Auditable trace object for SatQuery API responses.

Every analytical endpoint creates a TraceBuilder at request entry,
calls log_* helpers as it executes, then calls .build() to obtain
an AuditTrace dict that is returned alongside the answer.

Usage::

    tb = TraceBuilder(endpoint="/vqa", query=question)
    tb.log_validation("GeoTIFF integrity", passed=True, detail="CRS=EPSG:32643")
    tb.log_model("VQA Baseline", version="1.2.0", reason="Single-scene VQA")
    tb.log_evidence("models.vqa.predict", prediction=answer, confidence=0.87)
    return {**existing_response, "trace": tb.build()}
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# AuditTrace — the serialisable trace schema
# ---------------------------------------------------------------------------

class AuditTrace:
    """Immutable snapshot of a single request's audit trail.

    Attributes:
        trace_id:          UUID4 string, unique per request.
        timestamp:         ISO-8601 UTC timestamp of when the trace was built.
        endpoint:          API path that produced this trace (e.g. "/vqa").
        query:             Raw user query string, if any.
        model_selections:  Ordered list of models selected during the request.
        validation_checks: Pre-flight validation checks and their outcomes.
        evidence_sources:  Evidence items consulted to produce the answer.
        execution_steps:   Ordered DAG steps from the execution planner.
        confidence_summary: Overall confidence roll-up.
        status:            "success" or "failed".
    """

    __slots__ = (
        "trace_id",
        "timestamp",
        "endpoint",
        "query",
        "model_selections",
        "validation_checks",
        "evidence_sources",
        "execution_steps",
        "confidence_summary",
        "status",
    )

    def __init__(
        self,
        *,
        trace_id: str,
        timestamp: str,
        endpoint: str,
        query: str,
        model_selections: List[Dict[str, Any]],
        validation_checks: List[Dict[str, Any]],
        evidence_sources: List[Dict[str, Any]],
        execution_steps: List[Dict[str, Any]],
        confidence_summary: Dict[str, Any],
        status: str,
    ) -> None:
        self.trace_id = trace_id
        self.timestamp = timestamp
        self.endpoint = endpoint
        self.query = query
        self.model_selections = model_selections
        self.validation_checks = validation_checks
        self.evidence_sources = evidence_sources
        self.execution_steps = execution_steps
        self.confidence_summary = confidence_summary
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the trace to a plain dict suitable for JSON responses."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "endpoint": self.endpoint,
            "query": self.query,
            "model_selections": list(self.model_selections),
            "validation_checks": list(self.validation_checks),
            "evidence_sources": list(self.evidence_sources),
            "execution_steps": list(self.execution_steps),
            "confidence_summary": dict(self.confidence_summary),
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# TraceBuilder — mutable builder accumulated during a request
# ---------------------------------------------------------------------------

class TraceBuilder:
    """Accumulates audit events during a single request, then builds an AuditTrace.

    Create one instance per request, call log_* helpers as execution proceeds,
    then call .build(status) once at response time.

    Args:
        endpoint: The API route path (e.g. "/vqa").
        query:    Optional natural-language query string for the request.
    """

    def __init__(self, endpoint: str = "", query: str = "") -> None:
        self._trace_id: str = str(uuid.uuid4())
        self._timestamp: str = datetime.now(timezone.utc).isoformat()
        self._endpoint: str = endpoint
        self._query: str = query or ""
        self._model_selections: List[Dict[str, Any]] = []
        self._validation_checks: List[Dict[str, Any]] = []
        self._evidence_sources: List[Dict[str, Any]] = []
        self._execution_steps: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_validation(
        self,
        check: str,
        *,
        passed: bool,
        detail: str = "",
    ) -> "TraceBuilder":
        """Records a single pre-flight validation check result.

        Args:
            check:  Human-readable name of the check (e.g. "GeoTIFF integrity").
            passed: True if the check succeeded; False otherwise.
            detail: Optional extra context (e.g. detected CRS, error message).

        Returns:
            self, enabling method chaining.
        """
        self._validation_checks.append(
            {"check": check, "passed": passed, "detail": detail}
        )
        return self

    def log_model(
        self,
        name: str,
        *,
        version: str = "",
        reason: str = "",
    ) -> "TraceBuilder":
        """Records a model or task that was selected for this request.

        Args:
            name:    Human-readable model/task name (e.g. "VQA Baseline").
            version: Optional semantic version string.
            reason:  Why this model was chosen (e.g. routing reasoning text).

        Returns:
            self, enabling method chaining.
        """
        entry: Dict[str, Any] = {"name": name, "version": version, "reason": reason}
        self._model_selections.append(entry)
        return self

    def log_evidence(
        self,
        source: str,
        *,
        prediction: Any = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TraceBuilder":
        """Records an evidence item consulted during inference.

        Args:
            source:     Dotted module path of the tool/model (e.g. "models.vqa.predict").
            prediction: The value or label the source produced.
            confidence: Float confidence in [0, 1], if available.
            metadata:   Any additional key/value pairs (bbox, change_pct, etc.).

        Returns:
            self, enabling method chaining.
        """
        entry: Dict[str, Any] = {
            "source": source,
            "prediction": prediction,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            **(metadata or {}),
        }
        self._evidence_sources.append(entry)
        return self

    def log_execution_steps(
        self, steps: List[Dict[str, Any]]
    ) -> "TraceBuilder":
        """Records the full ordered execution graph produced by the planner.

        Args:
            steps: List of step dicts from ExecutionGraph.details.

        Returns:
            self, enabling method chaining.
        """
        self._execution_steps = list(steps)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, status: str = "success") -> Dict[str, Any]:
        """Finalises and returns the trace as a plain serialisable dict.

        Args:
            status: Overall outcome — "success" or "failed".

        Returns:
            Dict representation of the AuditTrace, ready for JSON serialisation.
        """
        # Derive overall confidence from logged evidence sources
        confidence_values = [
            e["confidence"]
            for e in self._evidence_sources
            if e.get("confidence") is not None
        ]
        if confidence_values:
            overall = round(sum(confidence_values) / len(confidence_values), 4)
            bucket = (
                "high" if overall >= 0.75
                else "medium" if overall >= 0.50
                else "low"
            )
        else:
            overall = None
            bucket = "unknown"

        confidence_summary: Dict[str, Any] = {
            "overall": overall,
            "bucket": bucket,
            "sources_counted": len(confidence_values),
        }

        trace = AuditTrace(
            trace_id=self._trace_id,
            timestamp=self._timestamp,
            endpoint=self._endpoint,
            query=self._query,
            model_selections=self._model_selections,
            validation_checks=self._validation_checks,
            evidence_sources=self._evidence_sources,
            execution_steps=self._execution_steps,
            confidence_summary=confidence_summary,
            status=status,
        )
        return trace.to_dict()


__all__ = ["AuditTrace", "TraceBuilder"]
