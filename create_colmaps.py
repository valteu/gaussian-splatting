import os
import shutil
import subprocess

# Define the root directory of your datasets
root_dir = "/path/to/your/datasets"

# Define the command to call the convert.py script
command = ["python", "/home/valteu/study/mml/gaussian-splatting/convert.py", "--resize"]

# Iterate over all directories and files in the root directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Check if the current directory is a leaf directory (i.e., it has no subdirectories)
    if not dirnames:
        # Construct the path to the input directory for the current dataset
        input_dir = os.path.join(dirpath, "input")
        
        # Create the input directory if it doesn't exist
        os.makedirs(input_dir, exist_ok=True)
        
        # Copy all files in the current directory to the input directory
        for filename in filenames:
            # Construct the full path of the current file
            file_path = os.path.join(dirpath, filename)
            
            # Copy the current file to the input directory
            shutil.copy(file_path, input_dir)
        
        # Call the convert.py script with the -s option set to the current input directory
        subprocess.run(command + ["-s", input_dir])
