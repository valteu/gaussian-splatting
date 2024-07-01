import os
import subprocess

# Path to the datasets directory
datasets_path = 'datasets_git'

# Iterate over each directory inside the datasets folder
for filename in os.listdir(datasets_path):
    dir_path = os.path.join(datasets_path, filename)
    
    # Check if the path is a directory
    if os.path.isdir(dir_path):
        try:
            # Set the environment variable for offscreen rendering
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            
            # Convert using Xvfb
            print(f"Processing: {filename} - convert")
            subprocess.run(['xvfb-run', '-a', 'python3', 'convert.py', '-s', dir_path], check=True, env=env)
            
            # Train
            output_dir = os.path.join('output', filename)
            print(f"Processing: {filename} - train")
            subprocess.run(['xvfb-run', '-a', 'python3', 'train.py', '-s', dir_path, '--eval', '-m', output_dir], check=True, env=env)
            
            # Render
            print(f"Processing: {filename} - render")
            subprocess.run(['xvfb-run', '-a', 'python3', 'render.py', '-m', output_dir], check=True, env=env)
            
            # Metrics
            print(f"Processing: {filename} - metrics")
            subprocess.run(['xvfb-run', '-a', 'python', 'metrics.py', '-m', output_dir], check=True, env=env)
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {filename}: {e}")
