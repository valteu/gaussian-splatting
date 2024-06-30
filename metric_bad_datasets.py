import os
import subprocess

# Define the directory containing the files
directory = 'output'

# Get the list of files in the directory
files = os.listdir(directory)

# Iterate over each file in the directory
for file in files:
    file_path = os.path.join(directory, file)
    
    # Execute the render command
    # Execute the metrics command
    metrics_command = f'python metrics.py -m {file_path}'
    subprocess.run(metrics_command, shell=True)
