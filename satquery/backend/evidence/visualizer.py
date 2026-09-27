import os
import uuid
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw

OUTPUT_DIR = Path("out")
VISUAL_DIR = OUTPUT_DIR / "visual"

def overlay_bounding_boxes(image_path: str, boxes: List[Tuple[int, int, int, int]], label: str = "") -> str:
    """
    Draws bounding boxes on an image for grounding visualization.
    boxes should be a list of [xmin, ymin, xmax, ymax].
    Returns the path to the generated visual evidence.
    """
    os.makedirs(VISUAL_DIR, exist_ok=True)
    
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    for box in boxes:
        draw.rectangle(box, outline="red", width=3)
        if label:
            draw.text((box[0], max(0, box[1]-10)), label, fill="red")
            
    vis_id = f"vis_{uuid.uuid4().hex[:8]}.png"
    out_path = VISUAL_DIR / vis_id
    img.save(out_path)
    
    return str(out_path)
