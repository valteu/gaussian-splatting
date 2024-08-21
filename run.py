import os
import subprocess
import shutil

# Define paths
source_dir = '/dhc/home/valentin.teutschbein/degrade_data'
dest_dir = './datasets'
result_dir = '/dhc/home/valentin.teutschbein/degrade_data_results'

# Step 1: Copy the source directory to the datasets directory
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
print(f"Copied {source_dir} to {dest_dir}")

# Step 2: Iterate over each directory inside the datasets folder
for filename in os.listdir(dest_dir):
    dir_path = os.path.join(dest_dir, filename)
    
    # Check if the path is a directory
    if os.path.isdir(dir_path):
        try:
            # Convert using Xvfb
            # print(f"Processing: {filename} - convert")
            subprocess.run(['python3', './convert_optimal_params.py', '-s', dir_path], check=True)
            
            # Train
            print(f"Processing: {dir_path} - train")
            subprocess.run(['python3', './create_pointclouds.py', '-s', dir_path, '--eval', '-m', dir_path], check=True)
            
            # Render
            # print(f"Processing: {filename} - render")
            # subprocess.run(['python3', 'render.py', '-m', output_dir], check=True)
            
            # Metrics
            # print(f"Processing: {filename} - metrics")
            # subprocess.run(['python', 'metrics.py', '-m', output_dir], check=True)
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {filename}: {e}")

# Step 3: Copy the datasets directory to the results directory
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

shutil.copytree(dest_dir, result_dir, dirs_exist_ok=True)
print(f"Copied {dest_dir} to {result_dir}")
