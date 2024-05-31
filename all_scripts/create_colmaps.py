import os
import subprocess


def create_colmaps():
    root_dir = "/home/valteu/study/mml/gaussian-splatting/datasets"

    command = ["python", "/home/valteu/study/mml/gaussian-splatting/convert.py", "--resize"]

    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            try:
                subprocess.run(command + ["-s", dir_path])
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == "__main__":
    create_colmaps()
