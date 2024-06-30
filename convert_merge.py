import os
import logging
from argparse import ArgumentParser
import shutil
import sqlite3
import struct
from PIL import Image
import numpy as np
import cv2

# Argument parser
parser = ArgumentParser("Colmap converter and merger")
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
                "tracks": track
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

# Function to update image colors in points3D
def update_image_colors(database_path, color_paths):
    print(color_paths)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images';")
    if cursor.fetchone() is None:
        logging.error(f"Table 'images' does not exist in {database_path}.")
        conn.close()
        exit(1)

    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()

    image_names = {}
    for image_id, name in images:
        name = name.split('/')[-1]
        found = False
        for color_path in color_paths:
            color_image_path = os.path.join(color_path, name)
            if os.path.exists(color_image_path):
                cursor.execute("UPDATE images SET name = ? WHERE image_id = ?", (color_image_path, image_id))
                image_names[image_id] = name
                found = True
                break
        if not found:
            logging.warning(f"Image {name} not found in provided color paths")

    conn.commit()
    conn.close()
    return image_names

def is_valid_keypoint(pt):
    try:
        x, y, size, angle, response, octave = map(float, pt)
        size = max(size, 1.0)
        angle = angle if angle >= 0 else 0
        response = response if response >= 0 else 0
        octave = int(octave)
        return True
    except (ValueError, TypeError):
        return False

def extract_color_from_image(image_path, coordinates):
    img = Image.open(image_path)
    x, y = coordinates
    color = img.getpixel((x, y))
    return color

def update_point_colors(points3D, color_paths, image_names):
    image_id_to_name = {v: k for k, v in image_names.items()}
    missing_image_ids = set()

    for point_id, point in points3D.items():
        if 'tracks' not in point:
            continue

        for track in point['tracks']:
            image_id = track[0]
            feature_id = track[1]
            if image_id in image_id_to_name:
                image_name = image_id_to_name[image_id]
                for color_path in color_paths:
                    image_path = os.path.join(color_path, image_name)
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        color = extract_color_from_image(image_path, (feature_id % img.width, feature_id // img.width))
                        point['rgb'] = color
                        break
                else:
                    missing_image_ids.add(image_id)
            else:
                missing_image_ids.add(image_id)

    logging.debug(f"Missing image IDs: {missing_image_ids}")

    for i, (point_id, point) in enumerate(points3D.items()):
        logging.debug(f"Point {point_id}: Color {point.get('rgb', 'No color assigned')}")
        if i >= 10:
            break

    return points3D

def merge_points3D(points3D_1, points3D_2):
    merged_points3D = points3D_1.copy()
    offset = max(merged_points3D.keys(), default=0) + 1

    for point_id, data in points3D_2.items():
        merged_points3D[offset + point_id] = data

    return merged_points3D

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

# Ensure necessary directories exist
os.makedirs(args.dataset_path + "/distorted", exist_ok=True)
os.makedirs(args.dataset_path + "/distorted/sparse_color", exist_ok=True)

# Run convert.py on both the color and structure datasets
def run_convert(dataset_part):
    command = f"python3 convert.py --source_path={os.path.join(args.dataset_path, dataset_part)}"
    run_command(command)


def estimate_relative_transform(matches):
    if not matches:
        logging.error("No matches found.")
        exit(1)

    for match in matches:
        keypoints1 = match['keypoints1']
        keypoints2 = match['keypoints2']

        # Estimate the essential matrix from the matched keypoints
        essential_matrix, _ = cv2.findEssentialMat(np.array(keypoints1), np.array(keypoints2), focal=1.0, pp=(0, 0))

        # Decompose the essential matrix to get the relative rotation and translation
        _, R, t, _ = cv2.recoverPose(essential_matrix, keypoints1, keypoints2)

        # Construct the relative transformation matrix
        relative_transform = np.eye(4)
        relative_transform[:3, :3] = R
        relative_transform[:3, 3] = t.squeeze()

        return relative_transform

    logging.error("No valid matches found to estimate relative transform.")
    exit(1)


def save_camera_poses(poses, file_path):
    with open(file_path, 'w') as f:
        for pose in poses:
            np.savetxt(f, pose)


def get_images_and_features_from_db(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()

    image_features = {}
    for image_id, image_name in images:
        cursor.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id,))
        keypoints_blob = cursor.fetchone()
        if keypoints_blob is None:
            continue

        keypoints = cv2.KeyPoint_convert(cv2.KeyPoint().from_bytes(keypoints_blob[0]))
        image_features[image_name] = keypoints

    conn.close()
    return image_features

def extract_points(points3D):
    points = []
    for data in points3D.values():
        points.append(data["xyz"])
    return np.array(points)

def perform_icp(source_points, target_points, max_iterations=50, tolerance=1e-6):
    # Create an ICP object
    icp = cv2.ppf_match_3d_ICP(max_iterations, tolerance)
    
    # Perform ICP alignment
    retval, residual, transform = icp.registerModelToScene(source_points, target_points)
    
    if retval:
        return transform
    else:
        logging.error("ICP alignment failed.")
        exit(1)

def apply_transformation(points3D, transform):
    transformed_points3D = {}
    for point_id, data in points3D.items():
        xyz = np.append(data["xyz"], 1)  # Convert to homogeneous coordinates
        transformed_xyz = transform.dot(xyz)[:3]  # Apply transformation
        transformed_points3D[point_id] = {
            "xyz": transformed_xyz,
            "rgb": data["rgb"],
            "error": data["error"],
            "tracks": data["tracks"]
        }
    return transformed_points3D

if __name__ == "__main__":
    # Run convert.py on color and structure datasets
    # run_convert("color")
    # run_convert("structure")

    # Update image colors in points3D
    database_path = os.path.join(args.dataset_path, "structure", "distorted", "database.db")
    color_paths = [os.path.join(args.dataset_path, "color"), os.path.join(args.dataset_path, "structure")]

    color_img_paths = [c + "/input" for c in color_paths]
    image_names = update_image_colors(database_path, color_img_paths)

    # Read and merge points3D from color and structure models
    points3D_color = read_points3D_bin(os.path.join(args.dataset_path, "color/sparse/0/points3D.bin"))
    points3D_structure = read_points3D_bin(os.path.join(args.dataset_path, "structure/sparse/0/points3D.bin"))

    # Extract points from both point clouds
    color_points = extract_points(points3D_color)
    structure_points = extract_points(points3D_structure)

    # Perform ICP alignment
    transform = perform_icp(structure_points, color_points)

    # Apply transformation to points3D_structure
    transformed_points3D_structure = apply_transformation(points3D_structure, transform)

    # Merge the aligned point clouds
    merged_points3D = merge_points3D(points3D_color, transformed_points3D_structure)
    merged_points3D = update_point_colors(merged_points3D, color_paths, image_names)

    # Write merged points3D to binary file and PLY file
    write_points3D_bin(os.path.join(args.dataset_path, "distorted/sparse_color/points3D.bin"), merged_points3D)
    write_ply(merged_points3D, os.path.join(args.dataset_path, "distorted/sparse_color/points3D.ply"))

    logging.info("Conversion and merging completed.")
