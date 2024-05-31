import os
import logging
from argparse import ArgumentParser
import shutil
import sqlite3

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--color_path", "-c", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# Function to run a system command and check for errors
def run_command(command):
    exit_code = os.system(command)
    if exit_code != 0:
        logging.error(f"Command failed with code {exit_code}. Exiting. Command: {command}")
        exit(exit_code)

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    # Feature extraction
    feat_extraction_cmd = (colmap_command + " feature_extractor "
        "--database_path " + args.source_path + "/distorted/database.db "
        "--image_path " + args.source_path + "/input "
        "--ImageReader.single_camera 1 "
        "--ImageReader.camera_model " + args.camera + " "
        "--SiftExtraction.use_gpu " + str(use_gpu))
    run_command(feat_extraction_cmd)

    # Feature matching
    feat_matching_cmd = (colmap_command + " exhaustive_matcher "
        "--database_path " + args.source_path + "/distorted/database.db "
        "--SiftMatching.use_gpu " + str(use_gpu))
    run_command(feat_matching_cmd)

    # Bundle adjustment
    mapper_cmd = (colmap_command + " mapper "
        "--database_path " + args.source_path + "/distorted/database.db "
        "--image_path " + args.source_path + "/input "
        "--output_path " + args.source_path + "/distorted/sparse "
        "--Mapper.ba_global_function_tolerance=0.000001")
    run_command(mapper_cmd)

# Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter "
    "--image_path " + args.source_path + "/input "
    "--input_path " + args.source_path + "/distorted/sparse/0 "
    "--output_path " + args.source_path + " "
    "--output_type COLMAP")
run_command(img_undist_cmd)

# Move sparse files
sparse_files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
for file in sparse_files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

# Resize images if required
if args.resize:
    print("Copying and resizing...")

    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)

    image_files = os.listdir(args.source_path + "/images")
    for file in image_files:
        source_file = os.path.join(args.source_path, "images", file)

        for size, folder in zip([50, 25, 12.5], ["images_2", "images_4", "images_8"]):
            destination_file = os.path.join(args.source_path, folder, file)
            shutil.copy2(source_file, destination_file)
            run_command(magick_command + f" mogrify -resize {size}% " + destination_file)

print("Done.")

# Overwrite color information from color_path
def update_image_colors(database_path, color_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Get image_id and image_name from images table
    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()
    
    for image_id, name in images:
        color_image_path = os.path.join(color_path, name)
        if os.path.exists(color_image_path):
            # Update the path in the database (assuming you need to change the path)
            cursor.execute("UPDATE images SET name = ? WHERE image_id = ?", (color_image_path, image_id))
    
    conn.commit()
    conn.close()

update_image_colors(os.path.join(args.source_path, "distorted/database.db"), args.color_path)
