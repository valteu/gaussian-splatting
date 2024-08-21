import os
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Define paths
source_dir = '/dhc/home/valentin.teutschbein/degrade_data'
dest_dir = './datasets'
result_dir = '/dhc/home/valentin.teutschbein/degrade_data_results'

# Step 1: Copy the source directory to the datasets directory
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
print(f"Copied {source_dir} to {dest_dir}")

# Function to process each directory
def process_directory(dir_path):
    try:
        # Convert using Xvfb
        # subprocess.run(['python3', './convert_optimal_params.py', '-s', dir_path], check=True)
        
        # Train
        print(f"Processing: {dir_path} - train")
        subprocess.run(['python3', './create_pointclouds.py', '-s', dir_path, '--eval', '-m', dir_path], check=True)
        
        # Render (optional, uncomment if needed)
        # subprocess.run(['python3', 'render.py', '-m', dir_path], check=True)
        
        # Metrics (optional, uncomment if needed)
        # subprocess.run(['python', 'metrics.py', '-m', dir_path], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing {dir_path}: {e}")

# Get the number of available CPUs
num_cpus = multiprocessing.cpu_count()

# Step 2: Process directories in parallel
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = []
    for filename in os.listdir(dest_dir):
        dir_path = os.path.join(dest_dir, filename)
        
        # Check if the path is a directory
        if os.path.isdir(dir_path):
            futures.append(executor.submit(process_directory, dir_path))
    
    # Wait for all threads to complete
    for future in as_completed(futures):
        future.result()  # Raise any exceptions that occurred during processing

# Step 3: Copy the datasets directory to the results directory
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

shutil.copytree(dest_dir, result_dir, dirs_exist_ok=True)
print(f"Copied {dest_dir} to {result_dir}")
