"""Remote Sensing Vision-Language Model (VLM) Prompt Engineering Module.

Provides specialized prompt templates and guidance for general VLMs (LLaVA, Qwen-VL,
InternVL) applied to top-down, nadir Earth Observation and aerial imagery.
"""

from typing import Optional, Dict, Any

RS_VQA_SYSTEM_PROMPT = """You are an expert Earth Observation and Satellite Imagery Analyst.
You are inspecting top-down, nadir or near-nadir aerial/satellite imagery.

CRITICAL INTERPRETATION GUIDELINES:
1. VIEWPOINT & PERSPECTIVE: The camera is positioned overhead pointing downwards towards the ground. Do NOT assume ground-level or horizontal photography perspectives.
2. SHADOWS & ELEVATION: Shadows are cast by vertical relief (buildings, trees, industrial towers, terrain) relative to sun azimuth and elevation.
3. SCALE & RESOLUTION: Ground Sampling Distance (GSD) means small objects (vehicles, vessels, containers) appear as compact geometric clusters.
4. LAND COVER:
   - Water bodies: Absorptive, dark to deep blue/cyan/black, smooth specular texture.
   - Vegetation: High absorption in red, high reflectance in near-infrared; textured canopy for forests, uniform texture for agriculture.
   - Built-up/Urban: High geometric structure, rectilinear footprints, asphalt/concrete roads, bright metallic roofs.
   - Bare Soil/Sand: High uniform reflectance, arid landscape texture.
5. CONCISE ANSWER: Formulate a direct, objective, and precise answer based strictly on the visible geospatial features."""


def build_rs_vqa_prompt(question: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Constructs a context-enriched prompt for general VLMs analyzing satellite scenes."""
    prompt_parts = [RS_VQA_SYSTEM_PROMPT, "\n---"]

    if metadata:
        meta_lines = []
        if "resolution" in metadata:
            meta_lines.append(f"Spatial Resolution (GSD): {metadata['resolution']} meters/pixel")
        if "crs" in metadata:
            meta_lines.append(f"Coordinate Reference System: {metadata['crs']}")
        if "band_count" in metadata:
            meta_lines.append(f"Number of Spectral Bands: {metadata['band_count']}")
        if meta_lines:
            prompt_parts.append("Scene Metadata:\n" + "\n".join(meta_lines) + "\n---")

    prompt_parts.append(f"Question: {question.strip()}")
    prompt_parts.append("Answer:")

    return "\n\n".join(prompt_parts)
