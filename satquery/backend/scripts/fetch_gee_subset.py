import os
import sys
import pandas as pd
import requests
import json
import time

try:
    import ee
except ImportError:
    print("Error: earthengine-api is not installed. Run: pip install earthengine-api")
    sys.exit(1)

def initialize_gee():
    try:
        ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print("\n" + "="*80)
        print("Earth Engine Authentication Required!")
        print("Please run the following command in your terminal to authenticate:")
        print("    earthengine authenticate")
        print("\nAfter authenticating, run this script again.")
        print("="*80 + "\n")
        sys.exit(1)

def fetch_gee_subset(parquet_path, output_dir, sample_size=500):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # We need rows with valid lat/lon
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    valid_df = df.dropna(subset=['latitude', 'longitude'])
    
    print(f"Found {len(valid_df)} valid rows with coordinates.")
    
    # Stratified sampling to get diverse images
    # We sample by country and season
    stratified = valid_df.groupby(['country', 'season']).apply(lambda x: x.sample(n=min(len(x), max(1, sample_size // 40)), random_state=42)).reset_index(drop=True)
    
    # Take exactly sample_size if we got too many
    if len(stratified) > sample_size:
        stratified = stratified.sample(n=sample_size, random_state=42).reset_index(drop=True)
    elif len(stratified) < sample_size:
        # If too few, just sample randomly from the rest
        remaining = sample_size - len(stratified)
        additional = valid_df[~valid_df['patch_id'].isin(stratified['patch_id'])].sample(n=remaining, random_state=42)
        stratified = pd.concat([stratified, additional], ignore_index=True)
        
    print(f"Selected {len(stratified)} unique patches for download.")
    
    # We also want to save a subset parquet for the DataLoader to use so it doesn't fail on missing files
    subset_parquet_path = os.path.join(output_dir, "subset.parquet")
    stratified.to_parquet(subset_parquet_path)
    print(f"Saved subset metadata to {subset_parquet_path}")

    # Start fetching
    success_count = 0
    
    for i, row in stratified.iterrows():
        patch_id = row['patch_id']
        lat, lon = row['latitude'], row['longitude']
        tif_path = os.path.join(output_dir, f"{patch_id}.tif")
        
        if os.path.exists(tif_path):
            print(f"[{i+1}/{sample_size}] Skipping {patch_id} (already downloaded)")
            success_count += 1
            continue
            
        print(f"[{i+1}/{sample_size}] Fetching {patch_id} at ({lat:.4f}, {lon:.4f})...", end="", flush=True)
        
        try:
            # Create a 1.2km x 1.2km bounding box (600m buffer from center)
            point = ee.Geometry.Point([lon, lat])
            buffer = point.buffer(600).bounds()
            
            # Query Sentinel-2 Surface Reflectance
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(buffer) \
                .filterDate('2017-01-01', '2018-12-31') \
                .sort('CLOUDY_PIXEL_PERCENTAGE')
            
            # Get the least cloudy image
            image = collection.first()
            
            # Select RGB bands and convert to 8-bit (divide by 10000 reflectance scaling, then multiply by 255)
            # Clip to the bounding box
            rgb = image.select(['B4', 'B3', 'B2']).clip(buffer)
            # Earth Engine optical surface reflectance is typically scaled 0-10000
            # To get a nice looking 8-bit image for VLMs:
            rgb_8bit = rgb.divide(3000).clamp(0, 1).multiply(255).uint8()
            
            url = rgb_8bit.getDownloadURL({
                'scale': 10,
                'crs': 'EPSG:4326',
                'region': buffer,
                'format': 'GEO_TIFF'
            })
            
            response = requests.get(url)
            if response.status_code == 200:
                with open(tif_path, 'wb') as f:
                    f.write(response.content)
                print(" OK")
                success_count += 1
            else:
                print(f" Failed HTTP {response.status_code}")
                
        except Exception as e:
            print(f" Error: {e}")
            
        # Small delay to avoid API rate limits
        time.sleep(1.0)
        
    print(f"\nFinished fetching. Successfully downloaded {success_count}/{sample_size} tiles.")

if __name__ == "__main__":
    initialize_gee()
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet = os.path.join(backend_dir, "BigEarthNet.txt", "BigEarthNet.txt.parquet")
    out_dir = os.path.join(backend_dir, "fixtures", "gee_subset")
    fetch_gee_subset(parquet, out_dir, sample_size=500)
