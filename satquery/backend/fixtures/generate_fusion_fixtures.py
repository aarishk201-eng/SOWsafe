"""Generates multi-sensor optical and SAR GeoTIFF fixtures for Phase 7 fusion testing."""

import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32643"  # UTM Zone 43N (WGS 84)
RES = 10.0
WIDTH, HEIGHT = 64, 64
ORIGIN = (500000.0, 2200000.0)


def create_geotiff(path: str, bands_data: np.ndarray, dtype: str = "float32"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = bands_data.shape[0]
    transform = from_origin(ORIGIN[0], ORIGIN[1], RES, RES)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=count,
        dtype=dtype,
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(bands_data.astype(dtype))


def generate_all_fixtures():
    np.random.seed(42)
    backend_fixtures = os.path.dirname(__file__)
    root_fixtures = os.path.abspath(os.path.join(backend_fixtures, "..", "..", "..", "fixtures"))

    created_files = []

    # 1. Optical Built-Up Scene: 4-band (Blue, Green, Red, NIR), high brightness, low NDVI
    b = np.random.normal(3200, 50, (HEIGHT, WIDTH))
    g = np.random.normal(3300, 50, (HEIGHT, WIDTH))
    r = np.random.normal(3400, 50, (HEIGHT, WIDTH))
    nir = np.random.normal(3600, 50, (HEIGHT, WIDTH))
    opt_builtup = np.stack([b, g, r, nir])
    p1 = os.path.join(backend_fixtures, "optical_built_up.tif")
    create_geotiff(p1, opt_builtup, dtype="float32")
    created_files.append(p1)

    # 2. SAR Built-Up Scene: 2-band (VV, VH), double-bounce dihedral reflection (VV > -10 dB, VH > -17 dB)
    # Linear power: VV ~ 0.25 (-6.0 dB), VH ~ 0.035 (-14.5 dB)
    vv = np.random.normal(0.25, 0.01, (HEIGHT, WIDTH))
    vh = np.random.normal(0.035, 0.002, (HEIGHT, WIDTH))
    sar_builtup = np.stack([vv, vh])
    p2 = os.path.join(backend_fixtures, "subset/sample_5.tif")
    create_geotiff(p2, sar_builtup, dtype="float32")
    created_files.append(p2)

    # 3. SAR Vegetation Scene: 2-band (VV, VH), high volume scattering, high RVI
    # Linear power: VV ~ 0.04 (-14.0 dB), VH ~ 0.014 (-18.5 dB)
    vv = np.random.normal(0.04, 0.002, (HEIGHT, WIDTH))
    vh = np.random.normal(0.014, 0.001, (HEIGHT, WIDTH))
    sar_veg = np.stack([vv, vh])
    p3 = os.path.join(backend_fixtures, "subset/sample_4.tif")
    create_geotiff(p3, sar_veg, dtype="float32")
    created_files.append(p3)

    # 4. Optical Vegetation Scene: 4-band, high NDVI
    b = np.random.normal(700, 30, (HEIGHT, WIDTH))
    g = np.random.normal(2000, 50, (HEIGHT, WIDTH))
    r = np.random.normal(800, 30, (HEIGHT, WIDTH))
    nir = np.random.normal(5500, 80, (HEIGHT, WIDTH))
    opt_veg = np.stack([b, g, r, nir])
    p4 = os.path.join(backend_fixtures, "optical_vegetation.tif")
    create_geotiff(p4, opt_veg, dtype="float32")
    created_files.append(p4)

    # 5. Optical Cloud-Occluded Scene: 4-band, very bright white clouds
    b = np.random.normal(6200, 100, (HEIGHT, WIDTH))
    g = np.random.normal(6100, 100, (HEIGHT, WIDTH))
    r = np.random.normal(6000, 100, (HEIGHT, WIDTH))
    nir = np.random.normal(6000, 100, (HEIGHT, WIDTH))
    opt_cloud = np.stack([b, g, r, nir])
    p5 = os.path.join(backend_fixtures, "optical_cloudy.tif")
    create_geotiff(p5, opt_cloud, dtype="float32")
    created_files.append(p5)

    # 6. Optical Water Scene: 4-band, high NDWI
    b = np.random.normal(1800, 50, (HEIGHT, WIDTH))
    g = np.random.normal(2500, 60, (HEIGHT, WIDTH))
    r = np.random.normal(900, 30, (HEIGHT, WIDTH))
    nir = np.random.normal(200, 20, (HEIGHT, WIDTH))
    opt_water = np.stack([b, g, r, nir])
    p6 = os.path.join(backend_fixtures, "optical_water.tif")
    create_geotiff(p6, opt_water, dtype="float32")
    created_files.append(p6)

    # 7. SAR Water Scene: 2-band, specular forward null (VV < -20 dB, VH < -24 dB)
    # Linear power: VV ~ 0.003 (-25.2 dB), VH ~ 0.0003 (-35.2 dB)
    vv = np.random.normal(0.003, 0.0002, (HEIGHT, WIDTH))
    vh = np.random.normal(0.0003, 0.00003, (HEIGHT, WIDTH))
    sar_water = np.stack([vv, vh])
    p7 = os.path.join(backend_fixtures, "subset/sample_6.tif")
    create_geotiff(p7, sar_water, dtype="float32")
    created_files.append(p7)

    # Mirror to root fixtures directory
    if os.path.exists(root_fixtures):
        for f in created_files:
            fname = os.path.basename(f)
            dest = os.path.join(root_fixtures, fname)
            shutil.copy2(f, dest)

    print(f"Successfully generated {len(created_files)} multi-modal fixtures in {backend_fixtures}")


if __name__ == "__main__":
    generate_all_fixtures()
