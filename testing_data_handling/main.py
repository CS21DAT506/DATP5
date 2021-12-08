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
from progress.bar import IncrementalBar
from scipy import stats
import pandas as pd

def extract_additional_data():
    data_dir = FileHandler.get_data_dir("testing_data")

    data = FileHandler.get_data_files(data_dir)

    processed_data = {}

    for model in data:
        extracted_data = {
            "min_cost": [],
            "min_start_cost_ratio": [],
            "ends_further_than_start": []
        }

        folder = Path.joinpath(data_dir, model)

        num_of_environments = int(len(FileHandler.get_data_files(folder))/2)

        bar = IncrementalBar('Files loaded: ', max=num_of_environments//10, suffix='%(percent)d%%')

        for i in range(num_of_environments):

            archive_path = str(folder) + "/archive_" + str(i) + ".json"

            metrics = None
            with open(archive_path, "r") as file:
                json_file = file.read()
                metrics = json.loads(json_file)

            if not metrics["collision"]:
                dist_to_target = metrics["dist_to_target"]
                min_dist = min(dist_to_target)
                extracted_data["min_cost"].append(min_dist)
                extracted_data["min_start_cost_ratio"].append(min_dist / dist_to_target[0])
                extracted_data["ends_further_than_start"].append(1 if dist_to_target[-1] > dist_to_target[0] else 0)

            if i % 10 == 0: 
                bar.next()
        print( f"{model} " + " " * (30 - len(model)) + "\n Processesing complete")

        processed_data[model] = {}

        for key in extracted_data.keys():
            m, h = mean_confidence_interval(extracted_data[key])
            processed_data[model][key + "_m"] = float(m)
            processed_data[model][key + "_h"] = float(h)

    with open("data/additional_data.json", "w") as file:
            jsonstr = json.dumps(processed_data, indent=4)
            file.write(jsonstr)
    print("File saved")

def extract_data():
    data_dir = FileHandler.get_data_dir("testing_data")

    data = FileHandler.get_data_files(data_dir)

    for model in data:
        extracted_data = {
            "agent_time": [],
            "gcpd_time": [],
            "overhead_time": [],
            "fuel": [],
            "end_cost": [],
            "min_cost": [],
            "time_to_5p_to_target": [],
            "fuel_to_5p_to_target": [],
            "collisions": [],
            "reaches_target": [],
            "stays_at_target": [],
            "fuel_for_at_target_at_end": [],
            "acc_se": [],
            "acc_ae": [],
            "capped_acc_se": [],
            "capped_acc_ae": [],
            "grav_length": [],
            "grav_length_reach_target": [],
            "grav_length_stay_at_target": [],
            "max_grav": [],
            "max_grav_stays_at_target": []
        }

        folder = Path.joinpath(data_dir, model)

        num_of_environments = int(len(FileHandler.get_data_files(folder))/2)

        bar = IncrementalBar('Files loaded: ', max=num_of_environments//10, suffix='%(percent)d%%')

        for i in range(num_of_environments):

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

                grav_lengths = []
                for vec in metrics["grav_acceleration"]:
                    grav_lengths.append(np.linalg.norm(vec))
                extracted_data["grav_length"].extend(grav_lengths)

                max_grav = max(grav_lengths)

                extracted_data["max_grav"].append(max_grav)

                if at_target_at_end :
                    extracted_data["fuel_for_at_target_at_end"].append(agent_fuel)
                    extracted_data["grav_length_stay_at_target"].extend(grav_lengths)
                    extracted_data["max_grav_stays_at_target"].append(max_grav)

                agent_fuel = 0
                target_reached = 0
                grav_lengths = []
                for ii in range(len(dist_to_target)):
                    agent_fuel += min(10, np.linalg.norm(metrics["agent_acceleration"][ii]))
                    grav_lengths.append(np.linalg.norm(metrics["grav_acceleration"][ii]))
                    if dist_to_target[ii] < dist_to_target[0] * 0.05:
                        extracted_data["time_to_5p_to_target"].append(ii * 0.01)
                        extracted_data["fuel_to_5p_to_target"].append(agent_fuel)
                        extracted_data["grav_length_reach_target"].extend(grav_lengths)
                        target_reached = 1
                        break
                extracted_data["reaches_target"].append(target_reached)

                extracted_data["end_cost"].append(dist_to_target[-1])
                extracted_data["min_cost"].append(min(dist_to_target))

                for ii in range(len(metrics["agent_acceleration"])):
                    model_to_gcpd_diff = np.array(metrics["agent_acceleration"][ii]) - np.array(metrics["gcpd_acceleration"][ii])
                    capped_model_to_gcpd_diff = np.array(cap_vector_length(metrics["agent_acceleration"][ii], 10)) - np.array(metrics["gcpd_acceleration"][ii])

                    for value in capped_model_to_gcpd_diff[:2]:
                        extracted_data["capped_acc_se"].append(value**2)
                        extracted_data["capped_acc_ae"].append(abs(value))

                    for value in model_to_gcpd_diff[:2]:
                        extracted_data["acc_se"].append(value**2)
                        extracted_data["acc_ae"].append(abs(value))

                ...

            if i % 10 == 0: 
                bar.next()
        print( f"{model} " + " " * (30 - len(model)) + "\n Processesing complete")

        processed_data = {}

        for key in extracted_data.keys():
            m, h = mean_confidence_interval(extracted_data[key])
            processed_data[key + "_m"] = float(m)
            processed_data[key + "_h"] = float(h)

        with open("data/extracted_data_" + model + ".json", "w") as file:
                jsonstr = json.dumps(processed_data, indent=4)
                file.write(jsonstr)
        print("File saved")

def abs_vector_pr_dim(vec):
    sum = 0
    for value in vec:
        sum += abs(value)
    return sum

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

def plot_data_2d(data_key, title, label = "y", min = 0, max = 1000, unwanted_models = [], save_plot = False):

    data = None
    with open("extracted_data.json", "r") as file:
                json_file = file.read()
                data = json.loads(json_file)

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))
    plot_design = ["-o", "-d"]

    mean = [model[data_key + "_m"] for model in data.values()]
    h = [model[data_key + "_h"] for model in data.values()]
   
    labels = [compute_label(s) for s in data.keys()]

    data = pd.DataFrame({
        "labels": labels,
        "h": h,
        "mean": mean
    })

    sorted_data = data.sort_values(by=["labels"])
    labels = sorted_data["labels"]

    sorted_data = sorted_data.set_index("labels").drop(unwanted_models, axis = 0)
    
    if len(unwanted_models) > 0 :
        labels = [x for x in labels if x not in unwanted_models]
    
    x = range(len(labels))

    plt.bar(x, sorted_data["mean"], yerr=sorted_data["h"], color=COLOR, capsize=4)

    fig = plt.subplot()

    box = fig.get_position()
    fig.set_position([box.x0 + box.width * 0.02, box.y0, box.width, box.height])

    plt.xticks(x, labels)
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

def time_plot(save_plot = False):

    data = None
    with open("extracted_data.json", "r") as file:
                json_file = file.read()
                data = json.loads(json_file)

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))
    plot_design = ["-o", "-d"]

    mean = [model["agent_time_m"] for model in data.values()]
    h = [model["agent_time_h"] for model in data.values()]
   
    labels = [compute_label(s) for s in data.keys()]

    gcpd_index = labels.index("GCPD")
    mean[gcpd_index] = data["gcpd"]["gcpd_time_m"]
    h[gcpd_index] = data["gcpd"]["gcpd_time_h"]

    data = pd.DataFrame({
        "labels": labels,
        "h": h,
        "mean": mean
    })

    sorted_data = data.sort_values(by=["labels"])
    labels = sorted_data["labels"]
    
    x = range(0, len(labels))

    plt.bar(x, sorted_data["mean"], yerr=sorted_data["h"], color=COLOR, capsize=4)

    title = "Time of computations"

    plt.xticks(x, labels)
    plt.ylim(0, 0.002)
    plt.ylabel("Time [s]")
    plt.xlabel("Model")
    plt.title(title)
    
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(Sci_Formatter)) #plt.LogFormatter(10, labelOnlyBase = False))

    fig = plt.subplot()

    box = fig.get_position()
    fig.set_position([box.x0 + box.width * 0.02, box.y0, box.width * 1.08, box.height])

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
    if model_name == "gcpd":
        return "GCPD"

    is_grav_vec = "nn_grav_vec" in model_name

    numbers = list(map(int, model_name.split("_")[3 if is_grav_vec else 2:]))

    is_rhombic = numbers[0] < numbers[1]

    postfix = (("R" if is_rhombic else "F") if is_grav_vec else "S")
    return str(np.sum(numbers)) + postfix

plot_setups = {
    "Fuel to reach target":                                     {"data_key":"fuel_to_5p_to_target",         "label": "Fuel", "min": 16000, "max": 32000},
    "Fuel":                                                     {"data_key":"fuel",                         "label": "Fuel", "min": 16000, "max": 32000},
    "Fuel to reach and stay at target":                         {"data_key":"fuel_for_at_target_at_end",    "label": "Fuel", "min": 16000, "max": 32000},
    "Likelihood of reaching target":                            {"data_key":"reaches_target",               "label": "Likelihood", "min": 0, "max": 1},
    "Likelihood of collision":                                  {"data_key":"collisions",                   "label": "Likelihood", "min": 0,  "max": 1},
    "Likelihood of staying at Target":                          {"data_key":"stays_at_target",              "label": "Likelihood", "min": 0,  "max": 1},
    "Time of neural network computations":                      {"data_key":"agent_time",                   "label": "Time [s]", "min": 0,  "max": 0.002, "unwanted_models": ["GCPD"]},
    "Time of GCPD computations":                                {"data_key":"gcpd_time",                    "label": "Time [s]", "min": 0,  "max": 0.002},
    "Time of overhead computations":                            {"data_key":"overhead_time",                "label": "Time [s]", "min": 0,  "max": 0.002},
    "Cost given as final distance to target":                   {"data_key":"end_cost",                     "label": "Cost", "min": 0,  "max": 151000},
    "Cost given as final distance to target (508R removed)":    {"data_key":"end_cost",                     "label": "Cost", "min": 0,  "max": 25000,   "unwanted_models": ["508R"]},
    "Cost given as minimum distance to target":                 {"data_key":"min_cost",                     "label": "Cost", "min": 0,  "max": 150},
    "Time to reach target":                                     {"data_key":"time_to_5p_to_target",         "label": "Time", "min": 0,  "max": 25},
    "Mean squared error (168S removed)":                        {"data_key":"acc_se",                       "label": "Error", "min": 0,  "max": 200,    "unwanted_models": ["GCPD", "168S"]},
    "Mean squared error":                                       {"data_key":"acc_se",                       "label": "Error", "min": 0,  "max": 20000,  "unwanted_models": ["GCPD"]},
    "Mean absolute error":                                      {"data_key":"acc_ae",                       "label": "Error", "min": 0,  "max": 7,      "unwanted_models": ["GCPD"]},
    "Mean squared error of capped acceleration":                {"data_key":"capped_acc_se",                "label": "Error", "min": 0,  "max": 22,     "unwanted_models": ["GCPD"]},
    "Mean absolute error of capped acceleration":               {"data_key":"capped_acc_ae",                "label": "Error", "min": 0,  "max": 7,      "unwanted_models": ["GCPD"]},
    "Length of gravitatial vector":                             {"data_key":"grav_length",                  "label": "Gravitational acceleration", "min": 0,  "max": 40},
    "Length of gravitational vector for reaching target":       {"data_key":"grav_length_reach_target",     "label": "Gravitational acceleration", "min": 0,  "max": 40},
    "Length of gravitational vector for staying at target":     {"data_key":"grav_length_stay_at_target",   "label": "Gravitational acceleration", "min": 0,  "max": 40},
    "Maximum gravitational acceleration":                       {"data_key":"max_grav",                     "label": "Gravitational acceleration", "min": 0,  "max": 11000},
    "Maximum gravitational acceleration for staying at target": {"data_key":"max_grav_stays_at_target",     "label": "Gravitational acceleration", "min": 0,  "max": 2500}
}

def plot_with_formatting(title, save_plot = False):
    setup = plot_setups[title]
    plot_data_2d(setup["data_key"], title, setup["label"], setup["min"], setup["max"], 
                 unwanted_models= setup["unwanted_models"] if "unwanted_models" in setup else [], 
                 save_plot=save_plot)

def plot_stacked_bars(save_plot=False):
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
    does_not_reach = [1.0 - collisions[i] for i in range(len(collisions))]
    collisions = [1.0 for i in range(len(collisions))]


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
    ax.set_title("Outcomes and their likelihoods")
    ax.legend(loc='center left', bbox_to_anchor=(0.2, -0.3))
    
    if(save_plot):
        plt.savefig("plots/Outcomes.png")
        plt.clf()
    else:
        plt.show()

def printList(list):  
        for value in list:
            string_value = "{:.2f}".format(value) if type(value) is float else value
            print(string_value + ", ", end="")
        print("")

if __name__ == "__main__":
    #extract_data()
    #extract_additional_data()
    # plot_stacked_bars(save_plot=True)
    # for key in plot_setups.keys():
    #     plot_with_formatting(key, save_plot=True)
    time_plot(save_plot=True)

    ...
