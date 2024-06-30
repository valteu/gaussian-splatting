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
            # Convert
            print(f"Processing: {filename} - convert")
            subprocess.run(['python3', 'convert.py', '-s', dir_path], check=True)
            
            # Train
            output_dir = os.path.join('output', filename)
            print(f"Processing: {filename} - train")
            subprocess.run(['python3', 'train.py', '-s', dir_path, '--eval', '-m', output_dir], check=True)
            
            # Render
            print(f"Processing: {filename} - render")
            subprocess.run(['python3', 'render.py', '-m', output_dir], check=True)
            
            # Metrics
            print(f"Processing: {filename} - metrics")
            subprocess.run(['python', 'metrics.py', '-m', output_dir], check=True)
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {filename}: {e}")
