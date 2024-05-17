import os
import shutil
import subprocess

root_dir = "/home/valteu/study/mml/gaussian-splatting/output/"

command = ["python", "/home/valteu/study/mml/gaussian-splatting/metrics.py"]

for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)
    
    if os.path.isdir(dir_path):
        try:
            print( command + ["-s", dir_path] )
            subprocess.run(command + ["--model_paths", dir_path])
        except Exception as e:
            print(f"Error: {e}")
            continue
