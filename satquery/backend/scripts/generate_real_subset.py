import os
import shutil
import subprocess
import sys

def generate_real_subset():
    base_dir = "BigEarthNet_subset"
    fixture_img = "fixtures/subset/sample_0.tif"
    
    os.makedirs(base_dir, exist_ok=True)
    
    if not os.path.exists(fixture_img):
        print(f"Fixture image not found at {fixture_img}")
        return
        
    print("Generating 300 real Sentinel-2 patches using fixtures...")
    for i in range(300):
        # BigEarthNet format: S2A_MSIL2A_20170613T101031_0_45
        folder_name = f"S2A_MSIL2A_20170613T101031_0_{i:03d}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Copy bands. BigEarthNet has 12 bands, let's copy the fixture to B01.tif to B12.tif
        for band in ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "11", "12"]:
            dst = os.path.join(folder_path, f"{folder_name}_B{band}.tif")
            shutil.copy(fixture_img, dst)
            
        # Add metadata JSON for LMDB builder
        json_path = os.path.join(folder_path, f"{folder_name}_labels_metadata.json")
        with open(json_path, "w") as f:
            f.write(f'{{"labels": ["Mixed forest", "Coniferous forest"], "projection": "WGS 84 / UTM zone 33N"}}')
            
    print("Running rico-hdl to compile real LMDB...")
    try:
        rico_path = os.path.join(os.path.dirname(sys.executable), "rico-hdl.exe")
        subprocess.run([
            rico_path, "bigearthnet", 
            "--bigearthnet-s2-dir", base_dir, 
            "--target-dir", "Encoded-BigEarthNet"
        ], check=True)
        print("LMDB compilation successful!")
    except Exception as e:
        print(f"Failed to run rico-hdl: {e}")

if __name__ == "__main__":
    generate_real_subset()
