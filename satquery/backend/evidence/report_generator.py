import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

OUTPUT_DIR = Path("out")
REPORTS_DIR = OUTPUT_DIR / "reports"

def generate_execution_report(task_query: str, trace_summary: list, result: Dict[str, Any]) -> str:
    """
    Generates a structured execution report and saves it to the reports directory.
    Returns the path to the generated report.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    report_id = f"report_{uuid.uuid4().hex[:8]}"
    report_path = REPORTS_DIR / f"{report_id}.json"
    
    report = {
        "report_id": report_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "task_query": task_query,
        "execution_summary": {
            "steps": trace_summary,
            "models_used": list(set(step.get("model") for step in trace_summary if "model" in step))
        },
        "results": result,
        "confidence_scores": result.get("confidence", "N/A"),
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    return str(report_path)
