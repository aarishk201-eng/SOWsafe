import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

def create_geotiff(filename, image_data, crs="EPSG:32633", transform=None):
    if transform is None:
        # Default transform (Origin at 500000, 4600000, 10m resolution)
        transform = from_origin(500000, 4600000, 10, 10)
    
    height, width, bands = image_data.shape
    
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=image_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(image_data[:, :, b], b + 1)

def main():
    os.makedirs("pairs", exist_ok=True)
    
    # Base image dimensions
    width, height = 256, 256
    
    # ----------------------------------------------------
    # Pair 1: Identical Scenes (No change)
    # ----------------------------------------------------
    img_base = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)
    # Add some green-ish hue for vegetation
    img_base[:, :, 1] = np.clip(img_base[:, :, 1] + 30, 0, 255)
    
    create_geotiff("pairs/pair1_t1_base.tif", img_base)
    create_geotiff("pairs/pair1_t2_identical.tif", img_base)
    
    # ----------------------------------------------------
    # Pair 2: Clear cut / Vegetation change
    # ----------------------------------------------------
    img_cleared = img_base.copy()
    # Create a brown patch in the center
    # Brownish: Red > Green > Blue
    img_cleared[100:150, 100:150, 0] = 139  # R
    img_cleared[100:150, 100:150, 1] = 69   # G
    img_cleared[100:150, 100:150, 2] = 19   # B
    
    create_geotiff("pairs/pair2_t1_forest.tif", img_base)
    create_geotiff("pairs/pair2_t2_cleared.tif", img_cleared)
    
    # ----------------------------------------------------
    # Pair 3: Building construction
    # ----------------------------------------------------
    img_built = img_base.copy()
    # Create a bright white/gray building block top-right
    img_built[30:80, 180:230, :] = 220
    
    create_geotiff("pairs/pair3_t1_empty.tif", img_base)
    create_geotiff("pairs/pair3_t2_building.tif", img_built)
    
    # ----------------------------------------------------
    # Pair 4: Incompatible scenes (Wrong CRS / No overlap)
    # ----------------------------------------------------
    # Different CRS for the second image
    create_geotiff("pairs/pair4_t1_epsg32633.tif", img_base, crs="EPSG:32633")
    create_geotiff("pairs/pair4_t2_epsg4326.tif", img_base, crs="EPSG:4326")
    
    print("Successfully generated sample GeoTIFF test pairs in 'sample_data/pairs/'")

if __name__ == "__main__":
    main()
