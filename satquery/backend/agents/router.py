"""SatQuery Intent Classifier & Query Router Agent (Phase 8).

Maps natural-language geospatial queries to {task, tools} from the fixed set:
{vqa, grounding, change, change_vqa, cross_modal_analysis}.

Also supports uppercase aliases {VQA, GROUNDING, CHANGE, CHANGE_VQA, OPTICAL_SAR}
and exports KNOWN_TASKS for test assertions.
"""

from typing import Dict, Any, List, Set, Optional
from enum import Enum
import re

from .planner import (
    planner,
    ExecutionPlanner,
    ExecutionGraph,
    STAGE_CATALOG,
    TaskStep,
    TaskGraph,
    ExecutionResult,
    build_task_graph,
    execute,
)

from .language import detect_language, score_intent

# Minimum cross-lingual cosine similarity for the multilingual embedding scorer
# to override the default single-scene VQA routing on a non-English query.
_EMB_INTENT_THRESHOLD = 0.40


class TaskName(str):
    """String subclass representing a task name that supports case-insensitive

    and alias-tolerant equality comparisons (e.g. 'cross_modal_analysis' == 'OPTICAL_SAR').
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TaskName):
            s2 = str(other).strip().lower()
        elif isinstance(other, str):
            s2 = other.strip().lower()
        else:
            return False

        s1 = str(self).strip().lower()
        if s1 == s2:
            return True

        fusion_aliases = {
            "cross_modal_analysis",
            "optical_sar",
            "fusion",
            "sar_optical",
            "optical_sar_fusion",
            "multimodal_fusion",
        }
        if s1 in fusion_aliases and s2 in fusion_aliases:
            return True

        return False

    def __hash__(self) -> int:
        return hash(str(self))


class TaskType(str, Enum):
    """The fixed set of supported SatQuery tasks."""

    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE = "change"
    CHANGE_VQA = "change_vqa"
    CROSS_MODAL_ANALYSIS = "cross_modal_analysis"
    OPTICAL_SAR = "cross_modal_analysis"


KNOWN_TASKS: Set[str] = {
    "vqa",
    "grounding",
    "change",
    "change_vqa",
    "cross_modal_analysis",
    "optical_sar",
    "VQA",
    "GROUNDING",
    "CHANGE",
    "CHANGE_VQA",
    "OPTICAL_SAR",
}

VALID_TASKS: Set[str] = {
    "VQA",
    "GROUNDING",
    "CHANGE",
    "CHANGE_VQA",
    "OPTICAL_SAR",
    "vqa",
    "grounding",
    "change",
    "change_vqa",
    "cross_modal_analysis",
}

# Concrete standalone tools built in Phases 2-7
_BASE_TOOLS: Dict[str, List[str]] = {
    "vqa": [
        "models.vqa.predict",
        "models.lora.infer",
        "geospatial.validator.extract_metadata",
    ],
    "grounding": [
        "models.grounding.locate",
        "geospatial.validator.extract_metadata",
    ],
    "change": [
        "geospatial.validator.check_compatibility",
        "models.change_analysis.detect",
    ],
    "change_vqa": [
        "geospatial.validator.check_compatibility",
        "models.change_analysis.detect",
        "models.change_analysis.change_vqa",
        "models.grounding.locate",
    ],
    "cross_modal_analysis": [
        "models.cross_modal.optical_analyzer",
        "models.cross_modal.sar_analyzer",
        "evidence.confidence.fuse_evidence",
        "geospatial.sar_preprocessing.SARPreprocessor",
    ],
}

TASK_TOOLS: Dict[str, List[str]] = {
    **_BASE_TOOLS,
    "optical_sar": _BASE_TOOLS["cross_modal_analysis"],
    "VQA": _BASE_TOOLS["vqa"],
    "GROUNDING": _BASE_TOOLS["grounding"],
    "CHANGE": _BASE_TOOLS["change"],
    "CHANGE_VQA": _BASE_TOOLS["change_vqa"],
    "OPTICAL_SAR": _BASE_TOOLS["cross_modal_analysis"],
    "CROSS_MODAL_ANALYSIS": _BASE_TOOLS["cross_modal_analysis"],
}


class SatQueryRouter:
    """Classifies natural-language satellite queries into {task, tools}."""

    def __init__(self):
        self.known_tasks = KNOWN_TASKS
        self.valid_tasks = VALID_TASKS
        self.task_tools = TASK_TOOLS

    def classify_intent(self, query: str, trace: Optional[Any] = None,
                        language: Optional[str] = None) -> Dict[str, Any]:
        """Classifies a natural-language query into {task, tools}.

        Args:
            query: The input natural language query string.
            trace: Optional TraceBuilder to log model selections and evidence.

        Returns:
            Dict[str, Any]: {
                "task": TaskName,
                "tools": List[str],
                "confidence": float,
                "reasoning": str,
                "query": str,
                "clarification_needed": bool,
            }
        """
        if not query or not str(query).strip():
            task = TaskName("vqa")
            empty_graph = ExecutionGraph(["INPUT_VALIDATION", "VQA", "ANSWER"])
            if trace is not None:
                trace.log_model(
                    "SatQueryRouter",
                    version="2.0",
                    reason="Empty query defaulted to VQA baseline",
                )
                trace.log_evidence(
                    "agents.router.classify_intent",
                    prediction=str(task),
                    confidence=0.40,
                    metadata={
                        "reasoning": "Empty query defaulted to VQA baseline.",
                        "tools": list(self.task_tools["vqa"]),
                        "clarification_needed": True,
                    },
                )
            return {
                "task": task,
                "tools": list(self.task_tools["vqa"]),
                "execution_graph": empty_graph,
                "execution_steps": empty_graph.details,
                "plan": empty_graph.details,
                "graph_repr": empty_graph.arrow_string(),
                "is_multi_step": False,
                "confidence": 0.40,
                "reasoning": "Empty query defaulted to VQA baseline.",
                "query": query or "",
                "clarification_needed": True,
                "language": (str(language).strip().lower() if language else "en"),
                "detected_language": "en",
                "target_pipeline": "vqa",
                "handler": "models.vqa",
            }

        q_clean = str(query).strip()
        q_lower = q_clean.lower()

        # Detect the query language (en / hi / hinglish) for cross-lingual routing
        # and localized answer rendering. Routing keys off the *detected* language;
        # the optional override only changes the language answers are rendered in.
        detected_language = detect_language(q_clean)
        answer_language = str(language).strip().lower() if language else detected_language
        is_default_fallthrough = False

        # 0. Check for ambiguous / non-geospatial requests
        ambiguous_triggers = [
            r"tell me something",
            r"something interesting",
            r"what can you do",
            r"^hello\b",
            r"^hi\b",
            r"^help\b",
            r"^random\b",
            r"surprise me",
            r"^test\b",
        ]
        is_ambiguous = any(re.search(pat, q_lower) for pat in ambiguous_triggers)

        # 1. Check CROSS_MODAL_ANALYSIS / OPTICAL_SAR Fusion
        optical_terms = ["optical", "rgb", "multispectral", "spectral", "sentinel-2"]
        sar_terms = ["sar", "radar", "microwave", "sentinel-1", "backscatter", "pol", "polarimetric", "risat"]
        has_optical = any(term in q_lower for term in optical_terms)
        has_sar = any(term in q_lower for term in sar_terms)

        is_fusion_explicit = bool(
            re.search(r"\b(fuse|fusion|fused|multimodal|cross-sensor|dual-sensor|all-weather|together)\b", q_lower)
        )
        is_cross_sensor = bool(
            re.search(r"\b(cross-sensor|cross_sensor|sensor\s+fusion|evidence\s+fusion|multimodal.*fusion|fusion.*multimodal|all-weather)\b", q_lower)
        )

        if (
            (is_fusion_explicit and (has_optical or has_sar or "all-weather" in q_lower or "multimodal" in q_lower))
            or (has_optical and has_sar)
            or is_cross_sensor
            or "cross_modal" in q_lower
            or "cross-modal" in q_lower
        ):
            task = TaskName("cross_modal_analysis")
            reasoning = "Query requests multi-sensor cross-modal analysis combining optical and SAR microwave evidence."
            confidence = 0.96
            clarification_needed = False

        else:
            # 2. Check CHANGE vs CHANGE_VQA
            change_indicators = [
                "change", "changed", "difference", "different", "deforestation", "urban growth",
                "flood extent", "increased", "decreased", "increase", "decrease", "expanded",
                "expansion", "expand", "cleared", "demolished", "constructed", "new construction",
                "before and after", "over time", "between t1 and t2", "between dates",
                "receded", "recede", "shrunk", "shrink", "altered area", "altered"
            ]
            has_comparative_verb = bool(
                re.search(r"\b(increas|decreas|expand|grow|growth|shrink|gain|loss|reced|shift|erod)\w*", q_lower)
            )
            has_temporal_range = bool(
                re.search(r"\b(between\s+\d{4}\s+and\s+\d{4}|between\s+t1\s+and\s+t2|from\s+\d{4}\s+to\s+\d{4}|since\s+\w+|over\s+the\s+past)\b", q_lower)
            )
            has_change_concept = has_comparative_verb or has_temporal_range or any(term in q_lower for term in change_indicators)

            question_starters = [
                "has ", "have ", "did ", "does ", "was ", "were ", "is ", "what ",
                "how ", "why ", "which ", "can you tell", "tell me if", "are there",
                "what's ", "whats "
            ]
            is_question_form = q_lower.endswith("?") or any(q_lower.startswith(qs) for qs in question_starters) or any(
                f" {qs}" in q_lower for qs in question_starters
            )
            is_imperative_command = bool(
                re.search(r"^(detect|compute|generate|calculate|map|run|produce|find changed|find altered)\b", q_lower)
            )

            if has_change_concept and is_question_form and not is_imperative_command:
                task = TaskName("change_vqa")
                reasoning = "Query asks a natural language question about bi-temporal scene changes grounded in a change mask."
                confidence = 0.95
                clarification_needed = False

            elif has_change_concept and (is_imperative_command or not is_question_form):
                task = TaskName("change")
                reasoning = "Query commands bi-temporal change detection to produce an altered area change mask."
                confidence = 0.94
                clarification_needed = False

            elif any(
                re.search(rf"\b{cmd}\b", q_lower)
                for cmd in ["detect change", "detect changes", "change mask", "difference map", "compute change"]
            ):
                task = TaskName("change")
                reasoning = "Explicit change detection command detected."
                confidence = 0.95
                clarification_needed = False

            # 3. Check GROUNDING
            elif bool(
                re.search(r"\b(highlight|locate|grounding|bounding\s*box|bbox|where\s+(is|are)|pinpoint|find\s+the|find\s+all|coordinates\s+of|localize)\b", q_lower)
            ):
                task = TaskName("grounding")
                reasoning = "Query requests spatial localization and bounding box coordinates for a target object or region."
                confidence = 0.95
                clarification_needed = False

            # 4. Check Ambiguous Query
            elif is_ambiguous:
                task = TaskName("vqa")
                reasoning = "Query is ambiguous or does not specify a concrete Earth observation task; clarification needed."
                confidence = 0.30
                clarification_needed = True

            # 5. Default to Single-Scene Visual Question Answering (VQA)
            else:
                task = TaskName("vqa")
                reasoning = "Query asks a single-scene visual or semantic interrogation question."
                confidence = 0.90 if is_question_form else 0.82
                clarification_needed = False
                is_default_fallthrough = True

        # Cross-lingual intent recovery: a non-English query that the English
        # regex ladder could only default to VQA is re-scored by the multilingual
        # embedder against per-intent anchor phrases. This replaces the previous
        # silent default-to-VQA for Hindi/Hinglish with a real, similarity-ranked
        # task selection (the score is a genuine cosine, not a fixed constant).
        if detected_language != "en" and is_default_fallthrough:
            scored = score_intent(q_clean)
            if scored is not None:
                emb_task, emb_sim = scored
                if emb_sim >= _EMB_INTENT_THRESHOLD and emb_task in self.valid_tasks:
                    task = TaskName(emb_task)
                    confidence = round(float(emb_sim), 2)
                    reasoning = (
                        f"Multilingual embedding intent match for {detected_language} "
                        f"query: nearest task '{emb_task}' at cosine {emb_sim:.2f}."
                    )
                    clarification_needed = False
                else:
                    reasoning += (
                        f" (Non-English '{detected_language}' query; top embedding "
                        f"score {emb_sim:.2f} < {_EMB_INTENT_THRESHOLD:.2f}, kept VQA.)"
                    )

        # Emit ordered execution graph (DAG) for multi-step reasoning
        execution_graph = planner.build_execution_graph(q_clean, primary_task=str(task))
        graph_tools = []
        for step in execution_graph.details:
            t = step.get("tool")
            if t and t not in graph_tools:
                graph_tools.append(t)

        all_tools = list(self.task_tools[task])
        for gt in graph_tools:
            if gt not in all_tools:
                all_tools.append(gt)

        is_multi_step = len([n for n in execution_graph if n not in ("INPUT_VALIDATION", "ANSWER")]) > 1

        if trace is not None:
            trace.log_model(
                "SatQueryRouter",
                version="2.0",
                reason=f"Routing intent '{task}' with confidence {confidence:.2f}",
            )
            trace.log_evidence(
                "agents.router.classify_intent",
                prediction=str(task),
                confidence=round(float(confidence), 2),
                metadata={
                    "reasoning": reasoning,
                    "tools": all_tools,
                    "clarification_needed": clarification_needed,
                },
            )

        return {
            "task": task,
            "tools": all_tools,
            "execution_graph": execution_graph,
            "execution_steps": execution_graph.details,
            "plan": execution_graph.details,
            "graph_repr": execution_graph.arrow_string(),
            "is_multi_step": is_multi_step,
            "confidence": round(float(confidence), 2),
            "reasoning": reasoning,
            "query": q_clean,
            "clarification_needed": clarification_needed,
            "language": answer_language,
            "detected_language": detected_language,
            "target_pipeline": str(task).lower(),
            "handler": f"models.{str(task).lower()}",
        }

    def route(self, query: str, trace: Optional[Any] = None,
              language: Optional[str] = None) -> Dict[str, Any]:
        """Convenience alias for classify_intent."""
        return self.classify_intent(query, trace=trace, language=language)

    def route_query(self, query: str, trace: Optional[Any] = None,
                    language: Optional[str] = None) -> Dict[str, Any]:
        """Legacy compatibility alias for classify_intent."""
        return self.classify_intent(query, trace=trace, language=language)


# Global singleton instance
query_router = SatQueryRouter()


def route(query: str, trace: Optional[Any] = None,
          language: Optional[str] = None) -> Dict[str, Any]:
    """Top-level route function mapping a natural language query to {task, tools}."""
    return query_router.classify_intent(query, trace=trace, language=language)


def classify_intent(query: str, trace: Optional[Any] = None,
                    language: Optional[str] = None) -> Dict[str, Any]:
    """Top-level classify_intent function mapping a natural language query to {task, tools}."""
    return query_router.classify_intent(query, trace=trace, language=language)


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
    "build_task_graph",
    "execute",
    "TaskGraph",
    "TaskStep",
    "ExecutionResult",
]
