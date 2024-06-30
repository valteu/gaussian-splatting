import os
import subprocess

# Define the directory containing the files
directory = 'datasets/new_preprocessing'

# Get the list of files in the directory
files = os.listdir(directory)

# Iterate over each file in the directory
for file in files:
    file_path = os.path.join(directory, file)
    
    convert_command = f'python3 convert.py -s {file_path}'
    subprocess.run(convert_command, shell=True)
    
    train_command = f'python3 train.py -s {file_path} --eval --test_iterations -1'
    subprocess.run(train_command, shell=True)
