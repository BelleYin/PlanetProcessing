import glob
import geopandas as gpd
import shutil
import os
from cloud_clear.planetscope import PlanetScope
from cloud_clear.rapideye import RapidEye  # This now uses the enhanced version
from compositing import create_median_composite

# Define directories
base_dir = "/Users/belle/Desktop/Planet"
output_base_dir = "/Users/belle/Desktop/Project/Output"  # Output folder in the Project directory
aoi = gpd.read_file('/Users/belle/Desktop/Planet/aotea/aotea.shp').to_crs('EPSG:2193')

# Create Output folder and subfolders if they don't exist
os.makedirs(os.path.join(output_base_dir, "RapidEye"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "PlanetScope"), exist_ok=True)

# Get a list of folders
folders = glob.glob(f"{base_dir}/*")
print(f"Found {len(folders)} folders in {base_dir}")

# Dictionary to store processed files by year
processed_files_by_year = {}

for folder in folders:
    print(f"\nProcessing folder: {folder}")
    
    if 'psscene' in folder.lower():
        # PlanetScope processing (unchanged)
        file_list = glob.glob(f"{folder}/PSScene/*.tif")
        print(f"Found {len(file_list)} .tif files in {folder}/PSScene")
        processor = PlanetScope(
            tmp_dir=f"{folder}/tmp",
            output_dir=os.path.join(output_base_dir, "PlanetScope"),
            aoi=aoi
        )
    elif 'reorthotile' in folder.lower():
        # RapidEye processing with enhanced masking
        file_list = glob.glob(f"{folder}/REOrthoTile/*.tif")
        print(f"Found {len(file_list)} .tif files in {folder}/REOrthoTile")
        processor = RapidEye(  # Uses the enhanced version
            tmp_dir=f"{folder}/tmp",
            output_dir=os.path.join(output_base_dir, "RapidEye"),
            aoi=aoi
        )
    else:
        print(f"Skipping unsupported folder: {folder}")
        continue

    for file in file_list:
        print(f"\nProcessing file: {file}")
        
        if 'Analytic_SR' in file:
            udm_file = file.replace('Analytic_SR', 'udm')
        elif 'AnalyticMS_SR_8b_harmonized' in file:
            udm_file = file.replace('AnalyticMS_SR_8b_harmonized', 'udm2')
        else:
            print(f"Unsupported file naming convention: {file}. Skipping.")
            continue

        if not os.path.exists(udm_file):
            print(f"UDM file not found: {udm_file}. Skipping.")
            continue

        print(f"Found UDM file: {udm_file}")

        analytic_clipped_path = processor.reproject_and_clip(file, 'analytic')
        udm_clipped_path = processor.reproject_and_clip(udm_file, 'udm')

        if not processor.check_file_properties(analytic_clipped_path, udm_clipped_path):
            print(f"Files not aligned: {analytic_clipped_path}, {udm_clipped_path}. Skipping.")
            continue

        cleaned_path = processor.apply_udm_mask(udm_clipped_path, analytic_clipped_path)
        print(f"Processed file: {cleaned_path}")

        # Group files by year
        year = file.split('_')[1]  # Adjust based on your file naming convention
        print(f"Grouping file {file} under year: {year}")
        processed_files_by_year.setdefault(year, []).append(cleaned_path)

    # Delete tmp directory
    if os.path.exists(processor.tmp_dir):
        shutil.rmtree(processor.tmp_dir)
    print(f"Temporary files deleted for {folder}.")

# Print files grouped by year
print("\nFiles grouped by year:")
for year, files in processed_files_by_year.items():
    print(f"Year: {year}, Files: {len(files)}")

def create_composites(output_dir):
    """
    Creates median composites for RapidEye and PlanetScope images from cleaned files
    
    Args:
        output_dir: Where the cleaned files are stored and composites will be saved
                   (e.g., "/Users/belle/Desktop/Project/Output")
    """
    # Create composites directory if it doesn't exist
    composites_dir = os.path.join(output_dir, "composites")
    os.makedirs(composites_dir, exist_ok=True)

    # 1. Find all RapidEye (REOrthoTile) cleaned files
    re_files = glob.glob(os.path.join(output_dir, "RapidEye", "*_cleaned.tif"))
    print(f"Found {len(re_files)} RapidEye cleaned files")
    
    # 2. Find all PlanetScope (PSScene) cleaned files
    ps_files = glob.glob(os.path.join(output_dir, "PlanetScope", "*_cleaned.tif"))
    print(f"Found {len(ps_files)} PlanetScope cleaned files")

    # 3. Create RapidEye composite
    if re_files:
        re_output = os.path.join(composites_dir, "REOrthoTile_median_composite.tif")
        print(f"\nCreating RapidEye median composite from {len(re_files)} images...")
        if create_median_composite(re_files, re_output):
            print(f"Successfully created RapidEye composite at {re_output}")
        else:
            print("Failed to create RapidEye composite")
    else:
        print("\nNo RapidEye cleaned files found")

    # 4. Create PlanetScope composite
    if ps_files:
        ps_output = os.path.join(composites_dir, "PSScene_median_composite.tif")
        print(f"\nCreating PlanetScope median composite from {len(ps_files)} images...")
        if create_median_composite(ps_files, ps_output):
            print(f"Successfully created PlanetScope composite at {ps_output}")
        else:
            print("Failed to create PlanetScope composite")
    else:
        print("\nNo PlanetScope cleaned files found")

if __name__ == "__main__":
    # Set your output directory where cleaned files are stored
    output_directory = "/Users/belle/Desktop/Project/Output"
    create_composites(output_directory)
