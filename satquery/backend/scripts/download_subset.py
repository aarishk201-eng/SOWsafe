import os
from datasets import load_dataset
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_origin

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'subset'))
os.makedirs(out_dir, exist_ok=True)

print("Fetching dataset...")
ds = load_dataset('sshh12/sentinel-2-rgb-captioned', split='train', streaming=True)

count = 0
for item in ds:
    img = item['image'] # PIL image
    img_np = np.array(img)
    
    # Save as TIF (mocking Sentinel 2)
    # The dataset has RGB (3 bands)
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        out_path = os.path.join(out_dir, f"sample_{count}.tif")
        
        # Create a mock transform
        transform = from_origin(0, 0, 10, 10)
        
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=c,
            dtype=img_np.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            for band in range(c):
                dst.write(img_np[:, :, band], band + 1)
        
        count += 1
        if count % 10 == 0:
            print(f"Downloaded {count}/100")
        if count >= 100:
            break

print(f"Successfully downloaded 100 images to {out_dir}")
