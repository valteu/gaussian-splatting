import os
import logging
from argparse import ArgumentParser
import shutil
import sqlite3
import struct
from PIL import Image
import numpy as np

# Argument parser
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

# Function to read points3D.bin in a memory-efficient way
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_bin(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_line = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = point_line[0]
            xyz = np.array(point_line[1:4])
            rgb = np.array(point_line[4:7])
            error = point_line[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            points3D[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            }
    return points3D

# Function to write points3D.bin in chunks
def write_points3D_bin(file_path, points3D):
    print('writing')
    with open(file_path, "wb") as f:
        f.write(struct.pack("Q", len(points3D)))
        for point_id, data in points3D.items():
            f.write(struct.pack("Q", point_id))
            f.write(struct.pack("ddd", *data["xyz"]))
            f.write(struct.pack("BBB", *data["rgb"]))
            f.write(struct.pack("d", data["error"]))
            f.write(struct.pack("Q", len(data["track"]) // 2))
            f.write(struct.pack("ii" * (len(data["track"]) // 2), *data["track"]))

# Function to update colors in points3D
def update_point_colors(points3D, color_path, image_names):
    image_cache = {}
    missing_image_ids = set()
    for point_id, data in points3D.items():
        for i in range(0, len(data["track"]), 2):
            image_id = data["track"][i]
            if image_id not in image_names:
                missing_image_ids.add(image_id)
                continue
            image_name = image_names[image_id]
            if image_name not in image_cache:
                image_cache[image_name] = Image.open(os.path.join(color_path + 'input', image_name))
            img = image_cache[image_name]
            x, y = data["xyz"][:2]
            color = img.getpixel((int(x), int(y)))
            data["rgb"] = color[:3]  # Assuming color is (R, G, B, A)

    if missing_image_ids:
        logging.warning(f"Missing image IDs: {missing_image_ids}")
    return points3D

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
    color_path = os.path.join(color_path, 'input')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Get image_id and image_name from images table
    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()

    image_names = {}
    for image_id, name in images:
        name = name.split('/')[-1]
        color_image_path = os.path.join(color_path, name)
        print(color_image_path)
        if os.path.exists(color_image_path):
            # Update the path in the database (assuming you need to change the path)
            cursor.execute("UPDATE images SET name = ? WHERE image_id = ?", (color_image_path, image_id))
            image_names[image_id] = name
        else:
            print(f"Image {name} not found in {color_path}")
            print(os.listdir(color_path))
 
    conn.commit()
    conn.close()
    return image_names


image_names = update_image_colors(os.path.join(args.source_path, "distorted/database.db"), args.color_path)

# Print image names from database
print(f"Image names from database: {image_names}")

# Read points3D
points3D = read_points3D_bin(os.path.join(args.source_path, "sparse/0/points3D.bin"))

# Print first few points to debug
print("First few points from points3D:")
for point_id, data in list(points3D.items())[:5]:
    print(f"Point ID: {point_id}, Track: {data['track']}")

# Update point colors
points3D = update_point_colors(points3D, args.color_path, image_names)

# Write updated points3D
write_points3D_bin(os.path.join(args.source_path, "sparse/0/points3D.bin"), points3D)

