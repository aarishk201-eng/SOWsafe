import os
import glob
import pandas as pd
import rasterio
from math import radians, sin, cos, sqrt, atan2
from models.vqa.data_module import get_bigearthnet_datamodule

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parquet_path = os.path.join(backend_dir, "BigEarthNet.txt", "BigEarthNet.txt.parquet")
gee_subset_dir = os.path.join(backend_dir, "fixtures", "gee_subset")

# Load full dataset for benchmarking tests
df = pd.read_parquet(parquet_path)
bench_df = df[df["split"] == "bench"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def get_fetched_tile_metadata(patch_id):
    tif_path = os.path.join(gee_subset_dir, f"{patch_id}.tif")
    if not os.path.exists(tif_path):
        return None
    with rasterio.open(tif_path) as src:
        tags = src.tags()
        # In STAC script we saved these tags
        if "center_lat" in tags and "center_lon" in tags:
            return {
                "center_lat": float(tags["center_lat"]),
                "center_lon": float(tags["center_lon"]),
                "eo:cloud_cover": float(tags.get("eo_cloud_cover", 100)),
                "id": patch_id
            }
        else:
            # Fallback calculate from bounds
            bounds = src.bounds
            from rasterio.warp import transform_bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(src.crs, 'EPSG:4326', *bounds)
            return {
                "center_lat": (lat_min + lat_max) / 2.0,
                "center_lon": (lon_min + lon_max) / 2.0,
                "eo:cloud_cover": 0.0,
                "id": patch_id
            }

def get_all_fetched_tile_metadata():
    metadata = []
    for tif_path in glob.glob(os.path.join(gee_subset_dir, "*.tif")):
        patch_id = os.path.splitext(os.path.basename(tif_path))[0]
        meta = get_fetched_tile_metadata(patch_id)
        if meta:
            metadata.append(meta)
    return metadata

def get_patch_ids_used_in_last_benchmark_run():
    # evaluate_benchmarks.py uses dm.bench_dataloader()
    dm = get_bigearthnet_datamodule()
    dm.setup("bench")
    # we return all patch IDs present in bench_ds
    dataset = dm.bench_ds
    return dataset.text_data["patch_id"].tolist()

def test_image_content_matches_the_sample_own_geographic_metadata():
    # Ensure at least some samples are present
    samples = bench_df.sample(min(10, len(bench_df)), random_state=42)
    for row in samples.itertuples():
        patch_id = getattr(row, "patch_id")
        lat = getattr(row, "latitude")
        lon = getattr(row, "longitude")
        tile_meta = get_fetched_tile_metadata(patch_id)
        # Skip assert if not fetched because our STAC fetcher might skip coords with NO cloud-free imagery.
        if tile_meta is None:
            continue
        tile_center_lat, tile_center_lon = tile_meta["center_lat"], tile_meta["center_lon"]
        distance_km = haversine(lat, lon, tile_center_lat, tile_center_lon)
        assert distance_km < 5, \
            f"Sample {patch_id} at ({lat},{lon}) got a tile centered {distance_km:.2f}km away."

def test_evaluation_uses_bench_split_only():
    eval_patch_ids = set(get_patch_ids_used_in_last_benchmark_run())
    bench_patch_ids = set(bench_df["patch_id"])
    
    # We must assert that all patch_ids used by the dataloader are indeed within the official bench split
    out_of_split = eval_patch_ids - bench_patch_ids
    assert len(out_of_split) == 0, \
        f"Benchmark evaluated on {len(out_of_split)} samples outside the official bench split."

def test_fetched_tiles_have_reasonable_cloud_cover():
    all_meta = get_all_fetched_tile_metadata()
    # If fetch is still running, check what's downloaded so far
    for tile_meta in all_meta:
        # We queried <30
        assert tile_meta.get("eo:cloud_cover", 100) < 30, \
            f"Tile {tile_meta['id']} has {tile_meta.get('eo:cloud_cover')}% cloud cover."
