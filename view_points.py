import struct

def read_points3D_bin(file_path):
    points3D = {}
    with open(file_path, "rb") as f:
        while True:
            binary_data = f.read(43)  # Intentionally read the unexpected length
            if len(binary_data) == 0:
                break
            if len(binary_data) != 44:
                print(f"Warning: Expected 44 bytes, but got {len(binary_data)} bytes")
                print(f"Data: {binary_data.hex()}")
                continue

            try:
                # Attempt to read using little-endian format
                point_id, x, y, z, r, g, b, error, track_length = struct.unpack('<Q3d3Bfi', binary_data)
            except struct.error:
                # Attempt to read using big-endian format
                point_id, x, y, z, r, g, b, error, track_length = struct.unpack('>Q3d3Bfi', binary_data)

            print(f"Read point ID {point_id} with track length {track_length}")

            track = []
            for _ in range(track_length):
                track_data = f.read(8)  # 2 * 4 (uint32)
                if len(track_data) != 8:
                    print(f"Warning: Expected 8 bytes, but got {len(track_data)} bytes for track data")
                    continue
                try:
                    image_id, point2D_idx = struct.unpack('<II', track_data)
                except struct.error:
                    image_id, point2D_idx = struct.unpack('>II', track_data)
                track.append((image_id, point2D_idx))

            points3D[point_id] = {
                'coords': (x, y, z),
                'color': (r, g, b),
                'error': error,
                'track_length': track_length,
                'track': track
            }
    return points3D

# Path to the points3D.bin file
points3D_file = "./datasets/train/distorted/sparse/0/points3D.bin"

# Load the 3D points
points3D = read_points3D_bin(points3D_file)

# Print the 3D points
for point_id, point_data in points3D.items():
    print(f"Point ID: {point_id}")
    print(f"  Coordinates: {point_data['coords']}")
    print(f"  Color: {point_data['color']}")
    print(f"  Error: {point_data['error']}")
    print(f"  Track Length: {point_data['track_length']}")
    print(f"  Track: {point_data['track']}")
    print()
