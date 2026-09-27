import os
import sys
import pandas as pd
import numpy as np
import time

try:
    import pystac_client
    import planetary_computer
    import rioxarray
    import rasterio
    from pyproj import CRS
    from rasterio.warp import transform_bounds
    from rasterio.transform import from_bounds
except ImportError:
    print("Dependencies not met.")
    sys.exit(1)

def bbox_for_coordinate(lat, lon, size_km=1.2):
    # Roughly 1.2km buffered bounding box around lat, lon
    buf_lat = (size_km / 2.0) / 111.0
    buf_lon = (size_km / 2.0) / (111.0 * np.cos(np.radians(lat)))
    return [lon - buf_lon, lat - buf_lat, lon + buf_lon, lat + buf_lat]

def fetch_stac_bench_subset(csv_path, parquet_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load coordinates
    coords_df = pd.read_csv(csv_path)
    print(f"Loaded {len(coords_df)} unique coordinate pairs from bench split.")
    
    # Load original dataset to get patch_ids corresponding to these coords
    df = pd.read_parquet(parquet_path)
    bench_df = df[df["split"] == "bench"].copy()
    
    # STAC Catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    success_count = 0
    missing_coords = []
    
    # Group patch IDs by coordinate
    coord_to_patches = {}
    for _, row in bench_df.iterrows():
        coord = (round(float(row['latitude']), 6), round(float(row['longitude']), 6))
        if coord not in coord_to_patches:
            coord_to_patches[coord] = []
        coord_to_patches[coord].append(row['patch_id'])
    
    for i, row in coords_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        coord_key = (round(float(lat), 6), round(float(lon), 6))
        patch_ids = coord_to_patches.get(coord_key, [])
        
        # If all patches for this coordinate are already downloaded, skip
        all_exist = all(os.path.exists(os.path.join(output_dir, f"{p}.tif")) for p in patch_ids)
        if all_exist and len(patch_ids) > 0:
            success_count += 1
            print(f"[{i+1}/{len(coords_df)}] Skipping ({lat:.4f}, {lon:.4f}) - already downloaded {len(patch_ids)} patches.")
            continue
            
        print(f"[{i+1}/{len(coords_df)}] Fetching STAC for ({lat:.4f}, {lon:.4f}) with {len(patch_ids)} patches... ", end="", flush=True)
        
        bbox = bbox_for_coordinate(lat, lon, size_km=1.2)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2017-01-01/2018-12-31",
            query={"eo:cloud_cover": {"lt": 30}},
        )
        items = list(search.items())
        
        if not items:
            print("NO RESULTS (No cloud-free tile found in window).")
            missing_coords.append(coord_key)
            continue
            
        # Sort by cloud cover
        items.sort(key=lambda item: item.properties["eo:cloud_cover"])
        item = items[0]
        
        url_b4 = item.assets["B04"].href
        url_b3 = item.assets["B03"].href
        url_b2 = item.assets["B02"].href
        
        try:
            profile = {}
            proj_bounds = (0.0, 0.0, 0.0, 0.0)
            arr = np.zeros((3, 120, 120), dtype=np.uint8)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif"):
                for b_idx, url in enumerate([url_b4, url_b3, url_b2]):
                    with rasterio.open(url) as src:
                        if not profile:
                            profile = dict(src.profile)
                        proj_bounds = tuple(transform_bounds('EPSG:4326', src.crs, bbox[0], bbox[1], bbox[2], bbox[3]))
                        window = src.window(*proj_bounds)
                        data = src.read(1, window=window, boundless=True, fill_value=0, out_shape=(120, 120))
                        arr[b_idx] = np.clip((data / 3000.0) * 255.0, 0, 255).astype(np.uint8)
            
            profile.update({
                'driver': 'GTiff',
                'count': 3,
                'dtype': 'uint8',
                'width': 120,
                'height': 120,
                'transform': from_bounds(float(proj_bounds[0]), float(proj_bounds[1]), float(proj_bounds[2]), float(proj_bounds[3]), 120, 120)
            })
            
            # Save for ALL patches matching this coordinate
            for patch_id in patch_ids:
                tif_path = os.path.join(output_dir, f"{patch_id}.tif")
                # Attach cloud cover metadata in rasterio tags
                with rasterio.open(tif_path, 'w', **{str(k): v for k, v in profile.items()}) as dst:
                    dst.write(arr)
                    dst.update_tags(
                        eo_cloud_cover=item.properties["eo:cloud_cover"],
                        center_lat=lat,
                        center_lon=lon,
                        stac_id=item.id
                    )
            print("OK")
            success_count += 1
        except Exception as e:
            print(f"Error: {e}")
            missing_coords.append(coord_key)
            
    print(f"\nFinished fetching. Successfully downloaded tiles for {success_count}/{len(coords_df)} coordinates.")
    if missing_coords:
        print(f"Missing coordinates: {len(missing_coords)}")

if __name__ == "__main__":
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(backend_dir, "bench_split_coordinates.csv")
    parquet = os.path.join(backend_dir, "BigEarthNet.txt", "BigEarthNet.txt.parquet")
    out_dir = os.path.join(backend_dir, "fixtures", "gee_subset")
    fetch_stac_bench_subset(csv_path, parquet, out_dir)
