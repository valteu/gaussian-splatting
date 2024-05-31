import os
import subprocess

def train_colmaps():
    root_dir = "/home/valteu/study/mml/gaussian-splatting/datasets"

    command = ["python", "/home/valteu/study/mml/gaussian-splatting/train.py", "--eval", "--test_iterations", "-1", "--densify_grad_threshold", "0.002"]

    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            try:
                print(command + ["-s", dir_path])
                subprocess.run(command + ["-s", dir_path])
            except Exception as e:
                print(f"Error: {e}")
                continue

if __name__ == "__main__":
    train_colmaps()
