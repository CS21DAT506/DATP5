from pathlib import Path
import os
import json
import numpy as np
import pickle
from os import sep                                                            
import datetime

def load_nn_data(json_path):
    data = None
    with open(json_path) as file:
        data = json.load(file)
    return np.array(data["input"]), np.array(data["output"])

def join_path_strs(*path_strs):                                               
    res = ""                                                                  
    for path_str in path_strs:                                                
        res += sep + path_str                                                 
    return res     

def load_byte_file(path):
    with open(path, "rb") as filename:
        return pickle.load(filename)

def ensure_dir_exists(path_to_dir):
    Path(path_to_dir).mkdir(parents=True, exist_ok=True)

def get_dir(dir_name): 
    return Path.joinpath(Path().resolve(), dir_name)

def get_data_files(data_dir):
    return os.listdir(data_dir)

def add_rhombic_layer(layers):
    if (len(layers) % 2 == 1):
        return np.insert(layers, 1 + len(layers) // 2, np.max(layers) * 2)
    else:
        return np.insert(layers, 1 + len(layers) // 2, np.max(layers) // 2)

def add_decreasing_layer(layers):
    return np.insert(layers * 2, len(layers), 8)

def get_updated_layers(factor, layer_nums, is_funnel):
    """Returns tuple of new factor and updated layer_nums"""
    if (factor >= 4):
        return 1, (add_decreasing_layer(layer_nums) if is_funnel else add_rhombic_layer(layer_nums))
    else:
        return factor * 2**(1/4), layer_nums

def file_name_from_layers(layer_nums):
    out = "nn_grav_vec"
    for layer_size in layer_nums:
        out += "_" + str(int(layer_size))
    return out

def save_history_as_byte_file(hist_folder, hist):
    if not os.path.exists(hist_folder):
        os.mkdir(hist_folder)
    
    file_path = join_path_strs(hist_folder, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    with open(file_path, 'wb') as file:
        pickle.dump(hist.history, file)