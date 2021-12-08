from pathlib import Path
import os

def get_data_dir(dir_name): 
    return Path.joinpath(Path().resolve(), dir_name)

def get_data_files(data_dir):
    return os.listdir(data_dir)