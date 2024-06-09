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
parser.add_argument("--dataset_path", "-d", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

logging.basicConfig(level=logging.DEBUG)

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
            track_flat = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            track = [(track_flat[i], track_flat[i + 1]) for i in range(0, len(track_flat), 2)]
            points3D[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "tracks": track  # Ensure the key is 'tracks'
            }
    return points3D

# Function to write points3D.bin in chunks
def write_points3D_bin(file_path, points3D):
    with open(file_path, "wb") as f:
        f.write(struct.pack("Q", len(points3D)))
        for point_id, data in points3D.items():
            f.write(struct.pack("Q", point_id))
            f.write(struct.pack("ddd", *data["xyz"]))
            f.write(struct.pack("BBB", *data["rgb"]))
            f.write(struct.pack("d", data["error"]))
            f.write(struct.pack("Q", len(data["tracks"])))
            for track in data["tracks"]:
                f.write(struct.pack("ii", *track))

# Function to extract color from an image
def extract_color_from_image(image_path, coordinates):
    img = Image.open(image_path)
    x, y = coordinates
    color = img.getpixel((x, y))
    return color

# Function to update colors in points3D
def update_point_colors(points3D, color_path, image_names):
    image_id_to_name = {v: k for k, v in image_names.items()}
    missing_image_ids = set()

    for point_id, point in points3D.items():
        if 'tracks' not in point:
            continue

        for track in point['tracks']:
            image_id = track[0]
            feature_id = track[1]  # Assuming track stores image_id and feature_id
            if image_id in image_id_to_name:
                image_name = image_id_to_name[image_id]
                image_path = os.path.join(color_path, image_name)
                if os.path.exists(image_path):
                    color = extract_color_from_image(image_path, (feature_id % img.width, feature_id // img.width))
                    point['rgb'] = color
                    break
                else:
                    missing_image_ids.add(image_id)
            else:
                missing_image_ids.add(image_id)

    logging.debug(f"Missing image IDs: {missing_image_ids}")

    # Print a few sample points for debugging
    for i, (point_id, point) in enumerate(points3D.items()):
        logging.debug(f"Point {point_id}: Color {point.get('rgb', 'No color assigned')}")
        if i >= 10:
            break

    return points3D

# Function to update image paths in the database
def update_image_colors(database_path, color_path):
    color_path = os.path.join(color_path, 'input')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()

    image_names = {}
    for image_id, name in images:
        name = name.split('/')[-1]
        color_image_path = os.path.join(color_path, name)
        if os.path.exists(color_image_path):
            cursor.execute("UPDATE images SET name = ? WHERE image_id = ?", (color_image_path, image_id))
            image_names[image_id] = name
        else:
            print(f"Image {name} not found in {color_path}")

    conn.commit()
    conn.close()
    return image_names

def write_ply(points3D, file_path):
    with open(file_path, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points3D)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for point_id, data in points3D.items():
            xyz = data["xyz"]
            rgb = data["rgb"]
            ply_file.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")

# Read points3D and update colors

if not args.skip_matching:
    os.makedirs(args.dataset_path + "/distorted/sparse", exist_ok=True)

    # Feature extraction
    feat_extraction_cmd = (colmap_command + " feature_extractor "
        "--database_path " + args.dataset_path + "/distorted/database.db "
        "--image_path " + os.path.join(args.dataset_path, "structure/input") + " "
        "--ImageReader.single_camera 1 "
        "--ImageReader.camera_model " + args.camera + " "
        "--SiftExtraction.use_gpu " + str(use_gpu))
    run_command(feat_extraction_cmd)

    # Feature matching
    feat_matching_cmd = (colmap_command + " exhaustive_matcher "
        "--database_path " + args.dataset_path + "/distorted/database.db "
        "--SiftMatching.use_gpu " + str(use_gpu))
    run_command(feat_matching_cmd)

    # Bundle adjustment
    mapper_cmd = (colmap_command + " mapper "
        "--database_path " + args.dataset_path + "/distorted/database.db "
        "--image_path " + os.path.join(args.dataset_path, "structure/input") + " "
        "--output_path " + args.dataset_path + "/distorted/sparse "
        "--Mapper.ba_global_function_tolerance=0.000001")
    run_command(mapper_cmd)

# Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter "
    "--image_path " + os.path.join(args.dataset_path, "structure/input") + " "
    "--input_path " + args.dataset_path + "/distorted/sparse/0 "
    "--output_path " + args.dataset_path + " "
    "--output_type COLMAP")
run_command(img_undist_cmd)

# Move sparse files
sparse_files = os.listdir(args.dataset_path + "/sparse")
os.makedirs(args.dataset_path + "/sparse/0", exist_ok=True)
for file in sparse_files:
    if file == '0':
        continue
    source_file = os.path.join(args.dataset_path, "sparse", file)
    destination_file = os.path.join(args.dataset_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

# Resize images if required
if args.resize:
    print("Copying and resizing...")

    os.makedirs(args.dataset_path + "/images_2", exist_ok=True)
    os.makedirs(args.dataset_path + "/images_4", exist_ok=True)
    os.makedirs(args.dataset_path + "/images_8", exist_ok=True)

    image_files = os.listdir(os.path.join(args.dataset_path, "color/input"))
    for image_file in image_files:
        img = Image.open(os.path.join(args.dataset_path, "color/input", image_file))

        # Resize and save the images
        img.resize((img.width // 2, img.height // 2)).save(os.path.join(args.dataset_path, "images_2", image_file))
        img.resize((img.width // 4, img.height // 4)).save(os.path.join(args.dataset_path, "images_4", image_file))
        img.resize((img.width // 8, img.height // 8)).save(os.path.join(args.dataset_path, "images_8", image_file))

# Update database and points3D colors
image_names = update_image_colors(os.path.join(args.dataset_path, "distorted/database.db"), os.path.join(args.dataset_path, "color"))
points3D = read_points3D_bin(os.path.join(args.dataset_path, "sparse/0/points3D.bin"))
points3D = update_point_colors(points3D, os.path.join(args.dataset_path, "color"), image_names)
write_points3D_bin(os.path.join(args.dataset_path, "sparse/0/points3D.bin"), points3D)

# Write PLY file
write_ply(points3D, os.path.join(args.dataset_path, "sparse/0/colmap_points.ply"))

# Replace the structure/input and /images directories with the color/input images
shutil.rmtree(os.path.join(args.dataset_path, "structure/input"))
shutil.rmtree(os.path.join(args.dataset_path, "images"))
shutil.copytree(os.path.join(args.dataset_path, "color/input"), os.path.join(args.dataset_path, "structure/input"))
shutil.copytree(os.path.join(args.dataset_path, "color/input"), os.path.join(args.dataset_path, "images"))
