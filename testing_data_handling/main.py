import file_util as fu
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import json 
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
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
                if at_target_at_end:
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
    with open("extracted_data.json", "r") as file:
                json_file = file.read()
                data = json.loads(json_file)

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))
    plot_design = ["-o", "-d"]

    mean = [model[data_key + "_m"] for model in data.values()]
    h = [model[data_key + "_h"] for model in data.values()]

    x = range(len(data))
    
    labels = [compute_label(s) for s in data.keys()]

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

    if(max < 1):
        plt.gca().get_yaxis().set_major_formatter(FuncFormatter(Sci_Formatter)) #plt.LogFormatter(10, labelOnlyBase = False))

    if(save_plot):
        plt.savefig("plots/" + title.replace(" ", "_") + ".png")
        plt.clf()
    else:
        plt.show()

def Sci_Formatter(x,lim):
    """Stolen from stack overflow"""
    if x == 0:
        return 0
    return '{0:.2f}e{1:.0f}'.format(np.sign(x)*10**(-np.floor(np.log10(abs(x)))+np.log10(abs(x))),np.floor(np.log10(abs(x))))

def compute_label(model_name):
    is_grav_vec = "nn_grav_vec" in model_name

    numbers = list(map(int, model_name.split("_")[3 if is_grav_vec else 2:]))

    is_rhombic = numbers[0] < numbers[1]

    postfix = (("R" if is_rhombic else "F") if is_grav_vec else "S")
    return str(np.sum(numbers)) + postfix

plot_setups = {
    "fuel_to_5p_to_target":         {"title":"Fuel to reach target",                                        "label": "Fuel", "min": 16000, "max": 35000},
    "fuel":                         {"title":"Fuel",                                                        "label": "Fuel", "min": 16000, "max": 35000},
    "reaches_target":               {"title":"Likelihood of reaching target",                               "label": "Likelihood", "min": 0, "max": 1},
    "collisions":                   {"title":"Likelihood of collision",                                     "label": "Likelihood", "min": 0,  "max": 1},
    "agent_time":                   {"title":"Time of neural network computations",                         "label": "Time [s]", "min": 0,  "max": 0.002},
    "gcpd_time":                    {"title":"Time of GCPD computations",                                   "label": "Time [s]", "min": 0,  "max": 0.002},
    "overhead_time":                {"title":"Time of overhead computations",                               "label": "Time [s]", "min": 0,  "max": 0.002},
    "end_cost":                     {"title":"Cost given as final distance to target",                      "label": "Cost", "min": 0,  "max": 21000},
    "time_to_5p_to_target":         {"title":"Time to reach target",                                        "label": "Time", "min": 0,  "max": 25},
    "stays_at_target":              {"title":"Likelihood of staying at Target",                             "label": "Likelihood", "min": 0,  "max": 1},
    "fuel_for_at_target_at_end":    {"title":"Fuel to reach and stay at target",                            "label": "Fuel", "min": 16000, "max": 35000},
    "acc_se":                       {"title":"Mean squared error",                                          "label": "Error", "min": 0,  "max": 20000},
    "acc_ae":                       {"title":"Mean absolute error",                                         "label": "Error", "min": 0,  "max": 7},
    "capped_acc_se":                {"title":"Mean squared error of capped acceleration",                   "label": "Error", "min": 0,  "max": 22},
    "capped_acc_ae":                {"title":"Mean absolute error of capped acceleration",                  "label": "Error", "min": 0,  "max": 7},
    "grav_length":                  {"title":"Length of gravitatial vector",                                "label": "Gravitational acceleration", "min": 0,  "max": 40},
    "grav_length_reach_target":     {"title":"Length of gravitational vector for reaching target",          "label": "Gravitational acceleration", "min": 0,  "max": 40},
    "grav_length_stay_at_target":   {"title":"Length of gravitational vector for staying at target",        "label": "Gravitational acceleration", "min": 0,  "max": 40},
    "max_grav":                     {"title":"Maximum gravitational acceleration",                          "label": "Gravitational acceleration", "min": 0,  "max": 11000},
    "max_grav_stays_at_target":     {"title":"Maximum gravitational acceleration for staying at target",    "label": "Gravitational acceleration", "min": 0,  "max": 2500}
}

def plot_with_formatting(data_key, save_plot = False):
    setup = plot_setups[data_key]
    plot_data_2d(data_key, setup["title"], setup["label"], setup["min"], setup["max"], save_plot=save_plot)

def plot_stacked_bars():
    data = None
    with open("extracted_data.json", "r") as file:
                json_file = file.read()
                data = json.loads(json_file)

    COLOR = cm.rainbow(np.linspace(0, 1, 5))
    plot_design = ["-o", "-d"]

    collisions = [model["collisions_m"] for model in data.values()] #P(Collision)
    reaches_target  = [model["reaches_target_m"]  * (1 - model["collisions_m"]) for model in data.values()] #P(reach \/ stay | !Collision) * (1 - P(Collision))
    stays_at_target = [model["stays_at_target_m"] * (1 - model["collisions_m"]) for model in data.values()] #P(stay | Collision) * (1 - P(Collision))

    reaches_not_stays_at_target = [reaches_target[i] for i in range(len(collisions))]
    does_not_reach = [1.0 - collisions[i] - reaches_target[i] + reaches_not_stays_at_target[i] for i in range(len(collisions))]
    collisions = [collisions[i] + does_not_reach[i] for i in range(len(collisions))]


    labels = [compute_label(s) for s in data.keys()]

    data = pd.DataFrame({
        "labels": labels,
        "collisions": collisions,
        "reaches_not_stays": reaches_not_stays_at_target,
        "stays": stays_at_target,
        "not_reach": does_not_reach
    })

    sorted_data = data.sort_values(by=["labels"])

    fig, ax = plt.subplots()
    width = 0.6

    # print(sorted_data["reaches_not_stays"])
    # print(sorted_data["stays"])
    # print([0] * 10)

    ax.bar(sorted_data["labels"], sorted_data["collisions"],        width, label="Collides",                         color=COLOR[3])
    ax.bar(sorted_data["labels"], sorted_data["not_reach"],         width, label="Does not reach target",            color=COLOR[2])
    ax.bar(sorted_data["labels"], sorted_data["reaches_not_stays"], width, label="Reaches target but does not stay", color=COLOR[1])
    ax.bar(sorted_data["labels"], sorted_data["stays"],             width, label="Stays at target",                  color=COLOR[0])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])

    ax.set_ylabel("Likelihood")
    ax.set_title("Outcomes and their likelihood")
    ax.legend(loc='center left', bbox_to_anchor=(0.2, -0.3))
    plt.show()

if __name__ == "__main__":
    #extract_data()
    plot_stacked_bars()
    # for key in plot_setups.keys():
    #     plot_with_formatting(key, save_plot=True)

    ...
