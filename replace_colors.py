import numpy as np
import cv2
import os
from plyfile import PlyData, PlyElement
import argparse
from read_write_model import read_model  # Assumes you have COLMAP's read_write_model.py available

def read_ply(file_path):
    """Read the PLY file and extract vertex data."""
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    return vertex_data, points

def project_points(points, camera_params, image_params):
    """Project 3D points to 2D image coordinates using camera parameters."""
    projected_points = {}
    for image_id, image in image_params.items():
        R = cv2.Rodrigues(image.qvec)[0]  # Rotation matrix
        t = image.tvec.reshape((3, 1))  # Translation vector
        K = np.array([[camera_params[image.camera_id].params[0], 0, camera_params[image.camera_id].params[2]],
                      [0, camera_params[image.camera_id].params[1], camera_params[image.camera_id].params[3]],
                      [0, 0, 1]])  # Intrinsic matrix

        P = K @ np.hstack((R, t))  # Projection matrix

        points_h = np.vstack((points.T, np.ones((1, points.shape[0]))))  # Homogeneous coordinates
        points_2d_h = P @ points_h  # Projected points in homogeneous coordinates
        points_2d = points_2d_h[:2, :] / points_2d_h[2, :]  # Normalize to get 2D coordinates

        projected_points[image_id] = points_2d.T

    return projected_points

def replace_colors_with_new_dataset(points, projected_points, image_folder, image_params):
    """Replace the colors of the 3D points with colors from a new dataset."""
    new_colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    counts = np.zeros((points.shape[0],), dtype=int)

    for image_id, points_2d in projected_points.items():
        image_path = os.path.join(image_folder, image_params[image_id].name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        h, w, _ = image.shape
        valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        for i, (x, y) in enumerate(points_2d[valid_points]):
            color = image[int(y), int(x)]
            new_colors[valid_points][i] += color
            counts[valid_points][i] += 1

    new_colors = (new_colors / counts[:, None]).astype(np.uint8)
    return new_colors

def save_ply(file_path, vertex_data):
    """Save the updated PLY file."""
    new_ply_data = PlyData([PlyElement.describe(vertex_data, 'vertex')], text=True)
    new_ply_data.write(file_path)

def main(original_ply_path, new_image_folder, colmap_model_folder, output_ply_path):
    # Step 1: Read the original PLY file
    vertex_data, points = read_ply(original_ply_path)

    # Step 2: Read COLMAP model
    cameras, images, points3D = read_model(colmap_model_folder, ext='.bin')

    # Step 3: Project 3D points to 2D image coordinates
    projected_points = project_points(points, cameras, images)

    # Step 4: Replace colors with the new dataset
    new_colors = replace_colors_with_new_dataset(points, projected_points, new_image_folder, images)

    # Step 5: Update vertex data with new colors
    vertex_data['red'] = new_colors[:, 0]
    vertex_data['green'] = new_colors[:, 1]
    vertex_data['blue'] = new_colors[:, 2]

    # Step 6: Save the updated PLY file
    save_ply(output_ply_path, vertex_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace colors in a PLY file with colors from a different dataset.')
    parser.add_argument('--original_ply', '-o', required=True, help='Path to the original PLY file')
    parser.add_argument('--new_images', '-n', required=True, help='Path to the new dataset images')
    parser.add_argument('--colmap_model', '-m', required=True, help='Path to the COLMAP model folder')
    parser.add_argument('--output_ply', '-p', required=True, help='Path to the output PLY file')

    args = parser.parse_args()

    main(args.original_ply, args.new_images, args.colmap_model, args.output_ply)

