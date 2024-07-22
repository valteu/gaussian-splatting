import os
import subprocess
import shutil

# Path to the datasets directory
datasets_path = 'datasets'

# Output directory path
output_root = 'output'


src_output_dir = '/workspace/output'
dest_output_dir = '/dhc/home/valentin.teutschbein/output'


def copy_output_directory(src, dest):
    try:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"Successfully copied {src} to {dest}")
    except Exception as e:
        print(f"An error occurred while copying {src} to {dest}: {e}")


# Perform the copy operation
copy_output_directory('/dhc/home/valentin.teutschbein/datasets', '/workspace/datasets')
# Process each directory inside the datasets folder
for filename in os.listdir(datasets_path):
    dir_path = os.path.join(datasets_path, filename)
    
    # Check if the path is a directory
    if os.path.isdir(dir_path):
        try:
            # Convert using Xvfb (uncomment if needed)
            # print(f"Processing: {filename} - convert")
            # subprocess.run(['python3', 'convert.py', '-s', dir_path], check=True)
            
            # Train
            output_dir = os.path.join(output_root, filename)
            print(f"Processing: {filename} - train")
            subprocess.run(['python3', 'train.py', '-s', dir_path, '--eval', '-m', output_dir], check=True)
            
            # Copy the output_dir/input.ply to the dir_path/input.ply
            src_file = os.path.join(output_dir, 'input.ply')
            dest_file = os.path.join(dir_path, 'input.ply')
            print(f"Copying: {src_file} to {dest_file}")
            shutil.copy(src_file, dest_file)
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {filename}: {e}")
        except FileNotFoundError as e:
            print(f"File not found error while processing {filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

# Copy the output directory to the specified location on the server
# Paths
src_output_dir = '/workspace/output'
dest_output_dir = '/dhc/home/valentin.teutschbein/output'

# Perform the copy operation
copy_output_directory(src_output_dir, dest_output_dir)
