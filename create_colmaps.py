import os
import shutil
import subprocess

root_dir = "/home/valteu/study/mml/gaussian-splatting/datasets"

command = ["python", "/home/valteu/study/mml/gaussian-splatting/convert.py", "--resize"]

for dir_name in os.listdir(root_dir):
    # Construct the full path of the current directory
    dir_path = os.path.join(root_dir, dir_name)
    
    # Check if the current path is a directory
    if os.path.isdir(dir_path):
        try:
            subprocess.run(command + ["-s", dir_path])
        except Exception as e:
            print(f"Error: {e}")
            continue
