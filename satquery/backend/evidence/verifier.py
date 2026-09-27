"""Verification and Multi-Sensor Evidence Bundling Subsystem.

Combines outputs from independent specialist tools (optical analyzer, SAR analyzer,
change detection, grounding, VQA) into a unified, cross-verified EvidenceBundle per query.

Every specialist tool conforms to the schema:
{
    "prediction": Any,
    "confidence": float,
    "evidence": {
        "bbox": Optional[List[float]],
        "area": Optional[float],
        "mask": Optional[Any],
        ...
    }
}
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np


class EvidenceBundle:
    """Consolidated multi-sensor evidence bundle for a satellite query.

    Combines specialist tool outputs into a unified, verified geospatial claim.
    Conforms to the standard schema:
    {
        "prediction": Any,
        "confidence": float,
        "evidence": {
            "bbox": Optional[List[float]],
            "area": Optional[float],
            "mask": Optional[Any],
            "bboxes": List[List[float]],
            "masks": List[Any],
            "tools": Dict[str, Any],
            "spatial_verification": Dict[str, Any],
            ...
        }
    }
    """

    def __init__(
        self,
        query: str,
        prediction: Any,
        confidence: float,
        evidence: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
        verified: bool = True,
        conflicts: Optional[List[Dict[str, Any]]] = None,
        summary: str = "",
        sources: Optional[List[Dict[str, Any]]] = None,
    ):
        self.query = str(query)
        self.prediction = prediction
        self.confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)
        self.evidence = dict(evidence or {})
        self.tools = dict(tools or {})
        self.verified = bool(verified)
        self.conflicts = list(conflicts or [])
        self.summary = str(summary)
        self.sources = list(sources or [])

        # Ensure standard keys exist in evidence
        if "bbox" not in self.evidence:
            self.evidence["bbox"] = None
        if "area" not in self.evidence:
            self.evidence["area"] = None
        if "mask" not in self.evidence:
            self.evidence["mask"] = None
        if "sources" not in self.evidence:
            self.evidence["sources"] = self.sources

    def __getitem__(self, item: str) -> Any:
        if item == "sources":
            return self.sources
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.evidence:
            return self.evidence[item]
        raise KeyError(item)

    def __contains__(self, item: str) -> bool:
        if item == "sources":
            return True
        return hasattr(self, item) or item in self.evidence

    def get(self, key: str, default: Any = None) -> Any:
        if key == "sources":
            return self.sources
        if hasattr(self, key):
            return getattr(self, key)
        return self.evidence.get(key, default)

    def keys(self):
        standard_keys = {
            "query",
            "prediction",
            "confidence",
            "evidence",
            "tools",
            "verified",
            "conflicts",
            "summary",
            "sources",
        }
        return standard_keys

    def to_dict(self) -> Dict[str, Any]:
        """Serializes bundle to a standard dictionary conforming to the specialist schema."""
        return {
            "query": self.query,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "tools": self.tools,
            "verified": self.verified,
            "conflicts": self.conflicts,
            "summary": self.summary,
            "sources": self.sources,
        }

    def __repr__(self) -> str:
        return (
            f"<EvidenceBundle query={self.query!r} prediction={self.prediction!r} "
            f"confidence={self.confidence:.2f} verified={self.verified}>"
        )


class EvidenceVerifier:
    """Verifies geospatial detections against spatial constraints and physical priors,

    and combines specialist tool outputs into a unified EvidenceBundle.
    """

    def __init__(
        self,
        max_scene_coverage_ratio: float = 0.85,
        default_image_width: int = 512,
        default_image_height: int = 512,
        min_confidence_threshold: float = 0.50,
    ):
        self.max_scene_coverage_ratio = max_scene_coverage_ratio
        self.default_image_width = default_image_width
        self.default_image_height = default_image_height
        self.min_confidence_threshold = min_confidence_threshold

    def verify_bounding_boxes(
        self, bboxes: List[List[float]], image_width: int, image_height: int
    ) -> Dict[str, Any]:
        """Validates that candidate detections fall within image coordinates."""
        valid_boxes = []
        invalid_boxes = []

        for bbox in bboxes:
            if not bbox or len(bbox) != 4:
                invalid_boxes.append({"bbox": bbox, "reason": "Expected 4 coordinates [x1, y1, x2, y2]"})
                continue

            x1, y1, x2, y2 = bbox
            if 0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height:
                # Check maximum scene coverage ratio
                box_area = (x2 - x1) * (y2 - y1)
                scene_area = image_width * image_height
                if scene_area > 0 and (box_area / scene_area) > self.max_scene_coverage_ratio:
                    invalid_boxes.append({"bbox": bbox, "reason": f"Box exceeds max scene coverage ratio ({self.max_scene_coverage_ratio})"})
                else:
                    valid_boxes.append(bbox)
            else:
                invalid_boxes.append({"bbox": bbox, "reason": "Out of image bounds or inverted coordinates"})

        return {
            "all_valid": len(invalid_boxes) == 0,
            "valid_count": len(valid_boxes),
            "invalid_count": len(invalid_boxes),
            "valid_boxes": valid_boxes,
            "invalid_boxes": invalid_boxes,
        }

    def verify_tool_output(
        self,
        tool_output: Any,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validates that a specialist tool adheres to the structured schema:

        {prediction, confidence, evidence: {bbox?, area?, mask?}}
        """
        errors = []
        if tool_output is None:
            return {"valid": False, "errors": ["tool_output is None"], "schema_compliant": False}

        # Handle dict or objects supporting to_dict or dict-like indexing
        if hasattr(tool_output, "to_dict") and not isinstance(tool_output, dict):
            data = tool_output.to_dict()
        elif isinstance(tool_output, dict):
            data = tool_output
        else:
            try:
                data = {
                    "prediction": tool_output["prediction"],
                    "confidence": tool_output["confidence"],
                    "evidence": tool_output["evidence"],
                }
            except Exception as e:
                return {"valid": False, "errors": [f"Cannot parse tool output into schema: {e}"], "schema_compliant": False}

        # 1. Prediction check
        if "prediction" not in data:
            errors.append("Missing required field 'prediction'.")

        # 2. Confidence check
        if "confidence" not in data:
            errors.append("Missing required field 'confidence'.")
        else:
            try:
                conf = float(data["confidence"])
                if not (0.0 <= conf <= 1.0):
                    errors.append(f"Confidence {conf} out of valid range [0.0, 1.0].")
            except (ValueError, TypeError):
                errors.append(f"Confidence '{data['confidence']}' is not a numeric float.")

        # 3. Evidence check
        if "evidence" not in data or not isinstance(data["evidence"], dict):
            errors.append("Missing or invalid 'evidence' dict field.")
        else:
            ev = data["evidence"]
            # Bounding box check
            bbox = ev.get("bbox")
            if bbox is not None:
                w = image_width or self.default_image_width
                h = image_height or self.default_image_height
                verif = self.verify_bounding_boxes([bbox], w, h)
                if not verif["all_valid"]:
                    errors.append(f"Invalid bbox {bbox}: {verif['invalid_boxes'][0].get('reason')}")

            # Area check
            area = ev.get("area")
            if area is not None:
                try:
                    a_val = float(area)
                    if a_val < 0:
                        errors.append(f"Area {a_val} cannot be negative.")
                except (ValueError, TypeError):
                    errors.append(f"Area '{area}' is not a valid number.")

        return {
            "valid": len(errors) == 0,
            "schema_compliant": len(errors) == 0,
            "errors": errors,
            "normalized_output": data,
        }

    def create_evidence_bundle(
        self,
        query: Any = None,
        tool_outputs: Optional[Union[List[Any], Dict[str, Any]]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        **kwargs,
    ) -> EvidenceBundle:
        """Combines specialist tool outputs into a single evidence bundle per query.

        Performs:
        1. Multi-source raw evidence schema validation against {prediction, confidence, evidence: {bbox?, area?, mask?}}.
           Rejects silently-malformed tool outputs (raising ValueError) before entering fusion.
        2. Preservation of per-source attribution in bundle["sources"].
        3. Spatial bounding box and mask integrity checks.
        4. Cross-modal Dempster-Shafer fusion and contradiction/conflict detection.
        5. Factual narrative synthesis.
        """
        # Handle polymorphic argument ordering: combine_evidence([tool1, tool2]) vs create_evidence_bundle("query", [...])
        if tool_outputs is None:
            if isinstance(query, (list, tuple, set, np.ndarray)):
                tool_outputs = query # type: ignore
                query_str = kwargs.pop("query_text", None) or kwargs.pop("query", None) or "Consolidated multi-specialist query"
            elif isinstance(query, dict) and (
                "prediction" in query or any(isinstance(v, dict) and "prediction" in v for v in query.values())
            ):
                tool_outputs = query
                query_str = kwargs.pop("query_text", None) or kwargs.pop("query", None) or "Consolidated multi-specialist query"
            elif isinstance(query, str):
                query_str = query
                tool_outputs = []
            else:
                query_str = "Consolidated multi-specialist query"
                tool_outputs = query or []
        else:
            if isinstance(query, str):
                query_str = query
            else:
                query_str = str(kwargs.pop("query_text", None) or kwargs.pop("query", None) or "Consolidated multi-specialist query")

        w = image_width or self.default_image_width
        h = image_height or self.default_image_height

        # Normalize incoming raw items into an ordered list of (name, raw_item)
        raw_items: List[Tuple[Optional[str], Any]] = []

        metadata_keys = {"compatibility", "metadata", "context", "status", "completed_steps", "failed_at"}
        if isinstance(tool_outputs, dict):
            for k, v in tool_outputs.items():
                if k in metadata_keys:
                    continue
                raw_items.append((k, v))
        elif isinstance(tool_outputs, (list, tuple, set)):
            for item in tool_outputs:
                raw_items.append((None, item))

        for k, v in kwargs.items():
            if k not in ("image_width", "image_height", "min_confidence_threshold") and k not in metadata_keys and (
                isinstance(v, dict) or hasattr(v, "prediction") or hasattr(v, "keys")
            ):
                raw_items.append((k, v))

        if not raw_items:
            # Fallback baseline bundle for empty input
            return EvidenceBundle(
                query=query_str,
                prediction="No specialist evidence provided",
                confidence=0.0,
                evidence={"bbox": None, "area": None, "mask": None, "tools": {}},
                tools={},
                verified=False,
                summary=f"No specialist tool evidence available for query: '{query_str}'.",
                sources=[],
            )

        # PASS GATE: Schema validation passes for EVERY tool's raw output before it enters fusion.
        # Reject silently-malformed tool outputs rather than passing them through!
        validated_sources: List[Dict[str, Any]] = []
        valid_tools: Dict[str, Dict[str, Any]] = {}

        for idx, (explicit_name, raw_output) in enumerate(raw_items):
            # 1. Output must not be None
            if raw_output is None:
                raise ValueError(
                    f"Silently-malformed tool output rejected: raw tool output at index {idx} is None."
                )

            # 2. Output must have .keys() or be a dictionary
            if not isinstance(raw_output, dict) and not hasattr(raw_output, "keys"):
                raise ValueError(
                    f"Silently-malformed tool output rejected: item at index {idx} ({type(raw_output)}) "
                    f"does not provide dictionary keys."
                )

            keys_present = set(raw_output.keys())
            required_keys = {"prediction", "confidence", "evidence"}
            if not required_keys.issubset(keys_present):
                missing = sorted(list(required_keys - keys_present))
                raise ValueError(
                    f"Silently-malformed tool output rejected: missing required schema keys {missing} in tool output."
                )

            # 3. Confidence must be numeric and in range [0.0, 1.0]
            conf_raw = raw_output["confidence"]
            try:
                conf_val = float(conf_raw)
                if conf_val < 0.0 or conf_val > 1.0 or np.isnan(conf_val):
                    raise ValueError(
                        f"Silently-malformed tool output rejected: confidence {conf_raw} is out of bounds [0.0, 1.0]."
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Silently-malformed tool output rejected: confidence '{conf_raw}' is not a valid number: {e}"
                )

            # 4. Evidence must be a mapping/dict
            ev_raw = raw_output["evidence"]
            if ev_raw is None or not (isinstance(ev_raw, dict) or hasattr(ev_raw, "keys")):
                raise ValueError(
                    f"Silently-malformed tool output rejected: 'evidence' field must be a dictionary/mapping, got {type(ev_raw)}."
                )

            # 5. Verify against spatial bounds if bbox is specified
            check = self.verify_tool_output(raw_output, w, h)
            if not check.get("valid", True):
                raise ValueError(
                    f"Silently-malformed tool output rejected: {', '.join(check.get('errors', []))}"
                )

            # Determine source_tool name
            source_tool_name = None
            if hasattr(raw_output, "source_tool") and getattr(raw_output, "source_tool"):
                source_tool_name = str(getattr(raw_output, "source_tool"))
            elif isinstance(raw_output, dict) and raw_output.get("source_tool"):
                source_tool_name = str(raw_output["source_tool"])
            elif explicit_name:
                source_tool_name = str(explicit_name)
            elif isinstance(raw_output, dict) and raw_output.get("sensor"):
                source_tool_name = str(raw_output["sensor"])
            elif isinstance(raw_output, dict) and raw_output.get("tool"):
                source_tool_name = str(raw_output["tool"])
            elif hasattr(raw_output, "to_dict"):
                d = raw_output.to_dict() # type: ignore
                source_tool_name = d.get("source_tool") or d.get("sensor") or d.get("tool")

            if not source_tool_name:
                # Infer from characteristic clues
                if isinstance(raw_output, str) or (isinstance(raw_output, dict) and "question" in ev_raw):
                    source_tool_name = "vqa"
                elif hasattr(raw_output, "change_detected") or (isinstance(raw_output, dict) and "change_percentage" in raw_output):
                    source_tool_name = "change_detection"
                elif isinstance(raw_output, dict) and "phrase" in ev_raw:
                    source_tool_name = "grounding"
                elif isinstance(raw_output, dict) and "mean_vv_db" in ev_raw:
                    source_tool_name = "sar_analyzer"
                elif isinstance(raw_output, dict) and "ndvi" in ev_raw:
                    source_tool_name = "optical_analyzer"
                else:
                    source_tool_name = f"specialist_{idx + 1}"

            # Construct normalized source dictionary preserving attribution
            if hasattr(raw_output, "to_dict") and not isinstance(raw_output, dict):
                norm_dict = raw_output.to_dict()
            elif isinstance(raw_output, dict):
                norm_dict = dict(raw_output)
            else:
                norm_dict = {
                    "prediction": raw_output["prediction"],
                    "confidence": conf_val,
                    "evidence": dict(ev_raw),
                }

            norm_dict["source_tool"] = source_tool_name
            validated_sources.append(norm_dict)

            # Deduplicate tool naming for fusion map
            map_key = source_tool_name
            if map_key in valid_tools:
                map_key = f"{map_key}_{idx + 1}"
            valid_tools[map_key] = norm_dict

        # 2. Extract spatial elements (bounding boxes, areas, masks)
        all_bboxes: List[List[float]] = []
        all_areas: List[float] = []
        all_masks: List[Any] = []

        for name, data in valid_tools.items():
            ev = data.get("evidence", {}) or {}
            bx = ev.get("bbox") or data.get("bbox")
            if bx:
                all_bboxes.append(bx)

            ar = ev.get("area")
            if ar is not None:
                try:
                    all_areas.append(float(ar))
                except (ValueError, TypeError):
                    pass

            mk = ev.get("mask")
            if mk is not None:
                all_masks.append(mk)

        # Spatial bounds verification
        spatial_verif = self.verify_bounding_boxes(all_bboxes, w, h) if all_bboxes else {"all_valid": True, "valid_boxes": []}
        valid_bboxes = spatial_verif.get("valid_boxes", [])

        # Consensus BBox: primary detection or bounding envelope
        consensus_bbox = None
        if valid_bboxes:
            if len(valid_bboxes) == 1:
                consensus_bbox = valid_bboxes[0]
            else:
                # Tightest bounding envelope across verified boxes
                x1 = min(b[0] for b in valid_bboxes)
                y1 = min(b[1] for b in valid_bboxes)
                x2 = max(b[2] for b in valid_bboxes)
                y2 = max(b[3] for b in valid_bboxes)
                consensus_bbox = [int(x1), int(y1), int(x2), int(y2)]

        # Consensus Area
        consensus_area = None
        if all_areas:
            consensus_area = round(float(np.mean(all_areas)), 2)
        elif consensus_bbox:
            consensus_area = float((consensus_bbox[2] - consensus_bbox[0]) * (consensus_bbox[3] - consensus_bbox[1]))

        # Consensus Mask
        consensus_mask = all_masks[0] if all_masks else None

        # 3. Consensus Prediction & Epistemic Conflict Quantification
        # Check optical and SAR multi-sensor pair first
        has_optical = "optical" in valid_tools
        has_sar = "sar" in valid_tools
        conflicts: List[Dict[str, Any]] = []

        if has_optical and has_sar:
            from .confidence import fuse_evidence
            fused = fuse_evidence(valid_tools["optical"], valid_tools["sar"])
            consensus_prediction = fused["prediction"]
            consensus_confidence = fused["confidence"]
            agreement = fused.get("agreement", "moderate")
            conflict_score = fused.get("conflict", 0.0)

            if agreement == "low" or conflict_score > 0.35:
                conflicts.append({
                    "type": "epistemic_contradiction",
                    "sources": ["optical", "sar"],
                    "conflict_mass": conflict_score,
                    "detail": fused.get("explanation", "Optical and SAR sensors contradict."),
                })
        else:
            # Aggregate generic specialist tool predictions
            preds = [str(d.get("prediction", "")).strip().lower() for d in valid_tools.values() if d.get("prediction")]
            confs = [float(d.get("confidence", 0.5)) for d in valid_tools.values()]

            # Determine dominant prediction
            if preds:
                unique_preds = set(preds)
                if len(unique_preds) == 1:
                    consensus_prediction = list(unique_preds)[0]
                    # Corroboration boost
                    combined_conf = 1.0
                    for c in confs:
                        combined_conf *= (1.0 - float(np.clip(c, 0.0, 0.99)))
                    consensus_confidence = round(1.0 - combined_conf, 4)
                    agreement = "high"
                    conflict_score = 0.0
                else:
                    # Contradiction between tools
                    consensus_prediction = preds[0]
                    # Penalize confidence on disagreement
                    consensus_confidence = round(float(np.mean(confs)) * 0.45, 4)
                    agreement = "low"
                    conflict_score = 0.55
                    conflicts.append({
                        "type": "tool_disagreement",
                        "predictions": list(unique_preds),
                        "conflict_mass": conflict_score,
                        "detail": f"Specialist tools returned conflicting predictions: {unique_preds}",
                    })
            else:
                consensus_prediction = "inconclusive"
                consensus_confidence = 0.40
                agreement = "low"
                conflict_score = 0.0

        # Overall verification flag
        verified = bool(spatial_verif.get("all_valid", True) and len(conflicts) == 0 and consensus_confidence >= self.min_confidence_threshold)

        # Synthesize factual explanation
        summary_parts = [
            f"Consolidated evidence for query '{query_str}':",
            f"Prediction: '{consensus_prediction}' (Confidence: {consensus_confidence:.2f}).",
            f"Tools bundled: {list(valid_tools.keys())}.",
        ]
        if consensus_bbox:
            summary_parts.append(f"Localized bounding box: {consensus_bbox}.")
        if consensus_area:
            summary_parts.append(f"Affected area: {consensus_area:.1f} px.")
        if conflicts:
            summary_parts.append(f"Warning: Epistemic contradiction detected ({conflicts[0]['detail']}).")

        summary = " ".join(summary_parts)

        # Build bundled evidence payload
        bundle_evidence_dict = {
            "bbox": consensus_bbox,
            "area": consensus_area,
            "mask": consensus_mask,
            "bboxes": valid_bboxes,
            "masks": all_masks,
            "tools": valid_tools,
            "agreement": agreement,
            "conflict": conflict_score,
            "spatial_verification": spatial_verif,
            "sources": validated_sources,
        }

        return EvidenceBundle(
            query=query_str,
            prediction=consensus_prediction,
            confidence=consensus_confidence,
            evidence=bundle_evidence_dict,
            tools=valid_tools,
            verified=verified,
            conflicts=conflicts,
            summary=summary,
            sources=validated_sources,
        )

    def combine_evidence(self, *args, **kwargs) -> EvidenceBundle:
        """Alias for create_evidence_bundle."""
        return self.create_evidence_bundle(*args, **kwargs)

    def bundle_evidence(self, *args, **kwargs) -> EvidenceBundle:
        """Alias for create_evidence_bundle."""
        return self.create_evidence_bundle(*args, **kwargs)


# Global default verifier instance
evidence_verifier = EvidenceVerifier()


def bundle_evidence(
    query_or_tools: Any = None,
    tool_outputs: Optional[Union[List[Any], Dict[str, Any]]] = None,
    query: Optional[str] = None,
    **kwargs,
) -> EvidenceBundle:
    """Convenience function to combine specialist tool outputs into one evidence bundle per query."""
    if query is not None:
        q = query
        tools = tool_outputs if tool_outputs is not None else query_or_tools
    else:
        q = query_or_tools
        tools = tool_outputs
    return evidence_verifier.create_evidence_bundle(query=q, tool_outputs=tools, **kwargs)


def combine_evidence(
    query_or_tools: Any = None,
    tool_outputs: Optional[Union[List[Any], Dict[str, Any]]] = None,
    query: Optional[str] = None,
    **kwargs,
) -> EvidenceBundle:
    """Convenience function alias to combine specialist tool outputs into an EvidenceBundle.

    Accepts:
        combine_evidence([vqa_output, grounding_output, change_output])
        combine_evidence("Detect urban change", [opt, sar])
        combine_evidence(tool_outputs={"optical": opt, "sar": sar})
        combine_evidence(query="...", tool_outputs={...})
    """
    if query is not None:
        q = query
        tools = tool_outputs if tool_outputs is not None else query_or_tools
    else:
        q = query_or_tools
        tools = tool_outputs
    return evidence_verifier.create_evidence_bundle(query=q, tool_outputs=tools, **kwargs)


def verify_and_bundle(
    query_or_tools: Any = None,
    tool_outputs: Optional[Union[List[Any], Dict[str, Any]]] = None,
    query: Optional[str] = None,
    **kwargs,
) -> EvidenceBundle:
    """Validates specialist tool outputs and creates an evidence bundle."""
    if query is not None:
        q = query
        tools = tool_outputs if tool_outputs is not None else query_or_tools
    else:
        q = query_or_tools
        tools = tool_outputs
    return evidence_verifier.create_evidence_bundle(query=q, tool_outputs=tools, **kwargs)


__all__ = [
    "EvidenceBundle",
    "EvidenceVerifier",
    "evidence_verifier",
    "bundle_evidence",
    "combine_evidence",
    "verify_and_bundle",
]
