from pathlib import Path
import os
import numpy as np
import json
import util as Util
from pathlib import Path
from progress.bar import IncrementalBar
from data_plotting import *
from scipy import stats
from constants import *

def get_data_dir(dir_name): 
    return Path.joinpath(Path().resolve(), dir_name)

def get_data_files(data_dir):
    return os.listdir(data_dir)

def cap_vector_length(vec, max_length):
    length = np.linalg.norm(vec)
    if length > max_length:
        return vec / length * max_length
    return vec

def geometric_mean_confidence_interval(data, confidence=0.95):
    log_data = np.log(data)
    m, h = mean_confidence_interval(log_data, confidence)
    return np.exp(m), np.exp(h)

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
    return f"{len(numbers)}:{np.sum(numbers)}{postfix}"

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

def get_fuel(agent_acceleration):
    agent_fuel = 0
    for acc in agent_acceleration:
        agent_fuel += cap_acceleration_length(acc)
    return agent_fuel

def cap_acceleration_length(vec):
    return min(MAX_ACCELERATION, np.linalg.norm(vec))

def reach_target_stats(metrics, extracted_data, dist_to_target):
    agent_fuel = 0
    target_reached = 0
    grav_lengths = []
    for i in range(len(dist_to_target)):
        agent_fuel += Util.cap_acceleration_length(metrics["agent_acceleration"][i])
        grav_lengths.append(np.linalg.norm(metrics["grav_acceleration"][i]))
        if dist_to_target[i] < dist_to_target[0] * DISTANCE_ERROR_MARGIN:
            extracted_data["time_to_5p_to_target"].append(i * TIME_STEP_SIZE)
            extracted_data["fuel_to_5p_to_target"].append(agent_fuel)
            extracted_data["grav_length_reach_target"].extend(grav_lengths)
            target_reached = 1
            break
    return agent_fuel, target_reached, grav_lengths

def handle_loss(metrics, extracted_data):
    for ii in range(len(metrics["agent_acceleration"])):
        model_to_gcpd_diff = np.array(metrics["agent_acceleration"][ii]) - np.array(metrics["gcpd_acceleration"][ii])
        capped_model_to_gcpd_diff = np.array(Util.cap_vector_length(metrics["agent_acceleration"][ii], 10)) - np.array(metrics["gcpd_acceleration"][ii])

        for value in capped_model_to_gcpd_diff[:2]:
            extracted_data["capped_acc_se"].append(value**2)
            extracted_data["capped_acc_ae"].append(abs(value))

        for value in model_to_gcpd_diff[:2]:
            extracted_data["acc_se"].append(value**2)
            extracted_data["acc_ae"].append(abs(value))