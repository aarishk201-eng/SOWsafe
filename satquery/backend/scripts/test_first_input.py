import os
import sys
import rasterio
from PIL import Image
import numpy as np

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__main__.__file__ if '__main__' in locals() else __file__))))

from models.grounding import locate
from evidence.report_generator import generate_execution_report
from evidence.visualizer import overlay_bounding_boxes

def convert_tif_to_png(tif_path, png_path):
    """Converts a potentially multi-band GeoTIFF to an RGB PNG for visualization and model input."""
    with rasterio.open(tif_path) as src:
        # Read the first 3 bands (assumed RGB for this example)
        # Handle cases with fewer bands gracefully
        bands = min(src.count, 3)
        img_array = np.dstack([src.read(i) for i in range(1, bands + 1)])
        
        # Normalize to 0-255 uint8 if it's not already
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(float)
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = 255 * (img_array - img_min) / (img_max - img_min)
            img_array = img_array.astype(np.uint8)
            
        # If it's a single band, convert to RGB
        if bands == 1:
            img_array = np.dstack([img_array, img_array, img_array])
            
        img = Image.fromarray(img_array)
        img.save(png_path)
        return png_path

def main():
    print("Setting up first input and output...")
    
    input_tif = "sample_scene.tif"
    if not os.path.exists(input_tif):
        print(f"Error: {input_tif} not found in the current directory.")
        return
        
    # Convert TIF to a friendly PNG format for the vision model
    input_png = "out/sample_scene.png"
    print(f"Converting {input_tif} to {input_png} for the vision model...")
    os.makedirs("out", exist_ok=True)
    convert_tif_to_png(input_tif, input_png)
    
    query = "forest area"
    print(f"\nQuerying the Grounding Model: '{query}'")
    
    # 1. Run Grounding
    result = locate(input_png, query)
    print(f"Grounding Result: {result}")
    
    # 2. Generate Visual Evidence
    if result.get("bbox"):
        vis_path = overlay_bounding_boxes(input_png, [result["bbox"]], label=query)
        print(f"Visual evidence generated at: {vis_path}")
    else:
        print("No bounding box found, skipping visualization.")
        
    # 3. Generate Execution Report
    trace = [
        {"step": 1, "action": "parse_query", "query": query},
        {"step": 2, "action": "load_image", "file": input_png},
        {"step": 3, "action": "grounding", "model": "google/owlvit-base-patch32"}
    ]
    
    report_path = generate_execution_report(query, trace, result)
    print(f"Execution report generated at: {report_path}")
    print("\nSetup complete!")

if __name__ == "__main__":
    main()
