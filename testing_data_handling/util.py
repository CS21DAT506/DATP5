from pathlib import Path
import os
import numpy as np
import json
import util as Util
from pathlib import Path
from progress.bar import IncrementalBar
from data_plotting import *
from scipy import stats

def get_data_dir(dir_name): 
    return Path.joinpath(Path().resolve(), dir_name)

def get_data_files(data_dir):
    return os.listdir(data_dir)

def cap_vector_length(vec, max_length):
    length = np.linalg.norm(vec)
    if length > max_length:
        return vec / length * max_length
    return vec

def mean_confidence_interval(data, confidence=0.95):
    """Stolen from stack-overflow"""
    a = np.array(data, dtype=np.float32)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)

    return m, h

def Sci_Formatter(x,lim):
    """Stolen from stack overflow"""
    if x == 0:
        return 0
    return '{0:.2f}e{1:.0f}'.format(np.sign(x)*10**(-np.floor(np.log10(abs(x)))+np.log10(abs(x))),np.floor(np.log10(abs(x))))

def compute_label(model_name):
    if model_name == "gcpd":
        return "GCPD"

    is_grav_vec = "nn_grav_vec" in model_name

    numbers = list(map(int, model_name.split("_")[3 if is_grav_vec else 2:]))

    is_rhombic = numbers[0] < numbers[1]

    postfix = (("R" if is_rhombic else "F") if is_grav_vec else "S")
    return str(np.sum(numbers)) + postfix

def save_json(content, path):
    with open(path, "w") as file:
            jsonstr = json.dumps(content, indent=4)
            file.write(jsonstr)
    print("File saved")

def load_json(path):
    with open(path, "r") as file:
        return json.loads(file.read())

def compute_statistics(statistics, raw_data) :
    for key in raw_data.keys():
            m, h = Util.mean_confidence_interval(raw_data[key])
            statistics[key + "_m"] = float(m)
            statistics[key + "_h"] = float(h)