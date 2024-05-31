import os
import subprocess


def metric_all():
    root_dir = "/home/valteu/study/mml/gaussian-splatting/output/"
    command = ["python", "/home/valteu/study/mml/gaussian-splatting/metrics.py"]
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        
        if os.path.isdir(dir_path):
            try:
                print( command + ["-m", dir_path] )
                subprocess.run(command + ["-m", dir_path])
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == "__main__":
    metric_all()
