import os
import subprocess
import shutil

# Path to the datasets directory
datasets_path = 'datasets_colmap_2'

# Iterate over each directory inside the datasets folder
for filename in os.listdir(datasets_path):
    dir_path = os.path.join(datasets_path, filename)
    
    # Check if the path is a directory
    if os.path.isdir(dir_path):
        try:
            # Convert using Xvfb
            # print(f"Processing: {filename} - convert")
            # subprocess.run(['python3', 'convert.py', '-s', dir_path], check=True)
            
            # Train
            output_dir = os.path.join('output', filename)
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
