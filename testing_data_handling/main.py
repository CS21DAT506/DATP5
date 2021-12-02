import file_util as fu
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import json 
from matplotlib import cm
import file_util as FileHandler
from pathlib import Path
from progresbar import *
from scipy import stats
import pandas as pd


def extract_data():
    num_of_environments = 1000

    data_dir = FileHandler.get_data_dir("testing_data_handling/testing_data")

    data = [dir for dir in FileHandler.get_data_files(data_dir) if "nn_grav" in dir ]

    processed_data = {}

    for model in data:
        
        extracted_data = {
            "agent_time": [],
            "gcpd_time": [],
            "overhead_time": [],
            "fuel": [],
            "end_cost": [],
            "time_to_5p_to_target": [],
            "fuel_to_5p_to_target": [],
            "collisions": [],
            "reaches_target": [],
            "stays_at_target": [],
            "fuel_for_at_target_at_end": []
        }

        for i in range(num_of_environments):
              
            folder = Path.joinpath(data_dir, model)

            archive_path = str(folder) + "/archive_" + str(i) + ".json"

            metrics = None
            with open(archive_path, "r") as file:
                json_file = file.read()
                metrics = json.loads(json_file)

            extracted_data["collisions"].append(int(metrics["collision"]))

            if not metrics["collision"]:
                for key in ["agent_time", "gcpd_time", "overhead_time"]:
                    extracted_data[key].extend(metrics[key])

                agent_fuel = 0
                for acc in metrics["agent_acceleration"]:
                    agent_fuel += min(10, np.linalg.norm(acc))
                extracted_data["fuel"].append(agent_fuel)

                dist_to_target = metrics["dist_to_target"]

                at_target_at_end = dist_to_target[-1] < dist_to_target[0] * 0.05
                extracted_data["stays_at_target"].append(1 if at_target_at_end else 0)
                if at_target_at_end :
                    extracted_data["fuel_for_at_target_at_end"].append(agent_fuel)

                agent_fuel = 0
                target_reached = 0
                for i in range(len(dist_to_target)):
                    agent_fuel += min(10, np.linalg.norm(metrics["agent_acceleration"][i]))
                    if dist_to_target[i] < dist_to_target[0] * 0.05:
                        extracted_data["time_to_5p_to_target"].append(i * 0.01)
                        extracted_data["fuel_to_5p_to_target"].append(agent_fuel)
                        target_reached = 1
                        break
                extracted_data["reaches_target"].append(target_reached)

                extracted_data["end_cost"].append(dist_to_target[-1])


            bar.next()
        print( f"\t{model} " + " " * (30 - len(model)) + "has been loaded")
        resetBar()

        processed_data[model] = {}

        for key in extracted_data.keys():
            m, h = mean_confidence_interval(extracted_data[key])
            processed_data[model][key + "_m"] = float(m)
            processed_data[model][key + "_h"] = float(h)

    with open("testing_data_handling/extracted_data.json", "w") as file:
            jsonstr = json.dumps(processed_data, indent=4)
            file.write(jsonstr)

def mean_confidence_interval(data, confidence=0.95):
    """Stolen from stack-overflow"""
    a = np.array(data, dtype=np.float32)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)

    return m, h

def plot_data_2d(data_key, title, label = "y", min = 0, max = 1000, save_plot = False):

    data = None
    with open("testing_data_handling/extracted_data.json", "r") as file:
                json_file = file.read()
                data = json.loads(json_file)

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))
    plot_design = ["-o", "-d"]

    mean = [model[data_key + "_m"] for model in data.values()]
    h = [model[data_key + "_h"] for model in data.values()]

    x = range(len(data))
    
    labels = [np.sum(list(map(int, s.split("_")[3:]))) for s in data.keys()]

    data = pd.DataFrame({
        "labels": labels,
        "h": h,
        "mean": mean
    })

    sorted_data = data.sort_values(by=["labels"])

    plt.bar(x, sorted_data["mean"], yerr=sorted_data["h"], color=COLOR, capsize=4)

    plt.xticks(x, sorted_data["labels"])
    plt.ylim(min, max)
    plt.ylabel(label)
    plt.xlabel("Model")
    plt.title(title)

    if(save_plot):
        plt.savefig("plots/" + title.replace(" ", "_") + ".png")
    else:
        plt.show()

def fuel_to_5p_to_target_plot(save_plot = False):
    plot_data_2d("fuel_to_5p_to_target", title="Fuel to reach target", label = "Fuel", min = 18000, max = 22000, save_plot=save_plot)

def fuel_plot(save_plot = False):
    plot_data_2d("fuel", title="Fuel", label = "Fuel", min = 25000, max = 35000, save_plot=save_plot)

def reaches_target_plot(save_plot = False):
    plot_data_2d("reaches_target", title="Rate of reaching target", label = "Rate", max = 1, save_plot=save_plot)

def collisions_plot(save_plot = False):
    plot_data_2d("collisions", title = "Rate of collision", label="Rate", max = 1, save_plot=save_plot)

def agent_time_plot(save_plot = False):
    plot_data_2d("agent_time", title="Time of neural network computations", label = "Time [s]", max = 0.001, save_plot=save_plot)

def gcpd_time_plot(save_plot = False):
    plot_data_2d("gcpd_time", title="Time of GCPD computations", label = "Time [s]", max = 0.001, save_plot=save_plot)

def overhead_time_plot(save_plot = False):
    plot_data_2d("overhead_time", title="Time of overhead computations", label = "Time [s]", max = 0.001, save_plot=save_plot)

def end_cost_plot(save_plot = False):
    plot_data_2d("end_cost", title="Cost given as final distance to target", label = "Cost", max = 40000, save_plot=save_plot)

def time_to_5p_to_target_plot(save_plot = False):
    plot_data_2d("time_to_5p_to_target", title="Time to reach target", label = "Time", max = 25, save_plot=save_plot)

def stays_at_target_plot(save_plot = False):
    plot_data_2d("stays_at_target", title="Rate of staying at Target", label = "Rate", max = 1, save_plot=save_plot)

def fuel_for_at_target_at_end_plot(save_plot = False):
    plot_data_2d("fuel_for_at_target_at_end", title="Fuel to reach and stay at target", label = "Fuel", min = 20000, max = 30000, save_plot=save_plot)

def acc_squared_error_plot(save_plot = False):
    plot_data_2d("acc_se", title="Mean squared error", label = "Error", max = 5, save_plot=save_plot)

def acc_absolute_error_plot(save_plot = False):
    plot_data_2d("acc_ae", title="Mean absolute error", label = "Error", max = 5, save_plot=save_plot)
   
def capped_acc_squared_error_plot(save_plot = False):
    plot_data_2d("capped_acc_se", title="Mean squared error of capped acceleration", label = "Error", max = 5, save_plot=save_plot)

def capped_acc_absolute_error_plot(save_plot = False):
    plot_data_2d("capped_acc_ae", title="Mean absolute error of capped acceleration", label = "Error", max = 5, save_plot=save_plot)
    
def grav_length_plot(save_plot = False):
    plot_data_2d("grav_length", title="Length of gravitatial vector", label = "Gravitational acceleration", max = 5, save_plot=save_plot)

def grav_length_reach_target_plot(save_plot = False):
    plot_data_2d("grav_length_reach_target", title="Length of gravitational vector for reaching target", label = "Gravitational acceleration", max = 5, save_plot=save_plot)

def grav_length_stay_at_target_plot(save_plot = False):
    plot_data_2d("grav_length_stay_at_target", title="Length of gravitational vector for staying at target", label = "Gravitational acceleration", max = 5, save_plot=save_plot)

def max_grav_length_plot(save_plot = False):
    plot_data_2d("max_grav", title="Maximum gravitational acceleration", label = "Gravitational acceleration", max = 10, save_plot=save_plot)

def max_grav_length_stay_at_target_plot(save_plot = False):
    plot_data_2d("max_grav_stays_at_target", title="Maximum gravitational acceleration for staying at target", label = "Gravitational acceleration", max = 10, save_plot=save_plot)

if __name__ == "__main__":
    #extract_data()
    max_grav_length_stay_at_target_plot()

    ...
