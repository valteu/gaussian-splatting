import os
import cv2
import struct

def read_points3D_bin(file_path):
    points3D = {}
    with open(file_path, "rb") as f:
        while True:
            binary_data = f.read(44)  # Updated to 44 bytes
            if len(binary_data) == 0:
                break
            if len(binary_data) != 44:
                print(f"Warning: Expected 44 bytes, but got {len(binary_data)} bytes")
                print(f"Data: {binary_data.hex()}")
                continue

            try:
                # Use little-endian format
                point_id, x, y, z, r, g, b, error, track_length = struct.unpack('<Q3d3BfI', binary_data)
            except struct.error as e:
                print(f"Unpack error: {e}")
                print(f"Data: {binary_data.hex()}")
                continue

            track = []
            for _ in range(track_length):
                track_data = f.read(8)  # 2 * 4 (uint32)
                if len(track_data) != 8:
                    print(f"Warning: Expected 8 bytes, but got {len(track_data)} bytes for track data")
                    print(f"Track Data: {track_data.hex()}")
                    continue
                try:
                    image_id, point2D_idx = struct.unpack('<II', track_data)
                except struct.error as e:
                    print(f"Track unpack error: {e}")
                    print(f"Track Data: {track_data.hex()}")
                    continue
                track.append((image_id, point2D_idx))

            points3D[point_id] = {
                'coords': (x, y, z),
                'color': (r, g, b),
                'error': error,
                'track_length': track_length,
                'track': track
            }
    return points3D

def write_points3D_bin(file_path, points3D):
    with open(file_path, "wb") as f:
        for point_id, point_data in points3D.items():
            x, y, z = point_data['coords']
            r, g, b = point_data['color']
            error = point_data['error']
            track_length = point_data['track_length']
            binary_data = struct.pack('<Q3d3BfI', point_id, x, y, z, r, g, b, error, track_length)
            f.write(binary_data)
            for image_id, keypoint_id in point_data['track']:
                f.write(struct.pack('<II', image_id, keypoint_id))

# Paths to the input directories and files
dir_name = os.path.dirname(__file__)
structure_path = os.path.join(dir_name, "./datasets/train/input")
color_path = os.path.join(dir_name, "./datasets/train/color")
undistorted_images_path = os.path.join(dir_name, "./datasets/train/images")
points3D_file = os.path.join(dir_name, "./datasets/train/distorted/sparse/0/points3D.bin")

# Load the 3D points
points3D = read_points3D_bin(points3D_file)

# Iterate over the 3D points to update their color values
for point_id, point_data in points3D.items():
    track = point_data['track']
    total_r, total_g, total_b = 0, 0, 0
    count = 0

    for image_id, keypoint_id in track:
        # Load the undistorted image
        image_name = f"{image_id:04d}.png"  # Adjust based on your naming convention
        undistorted_image_path = os.path.join(undistorted_images_path, image_name)
        undistorted_image = cv2.imread(undistorted_image_path)
        if undistorted_image is None:
            continue

        # Read keypoints from the COLMAP database
        keypoints_file = os.path.join(structure_path, f"keypoints/{image_id}.txt")
        with open(keypoints_file, 'r') as kp_file:
            keypoints = [line.strip().split() for line in kp_file]
            kp_x, kp_y = map(float, keypoints[keypoint_id][:2])

        # Extract the color value from the undistorted image
        color = undistorted_image[int(kp_y), int(kp_x)]
        total_r += color[2]  # OpenCV uses BGR format
        total_g += color[1]
        total_b += color[0]
        count += 1

    if count > 0:
        avg_r = int(total_r / count)
        avg_g = int(total_g / count)
        avg_b = int(total_b / count)
        points3D[point_id]['color'] = (avg_r, avg_g, avg_b)

# Write the updated 3D points back to points3D.bin
write_points3D_bin(points3D_file, points3D)
