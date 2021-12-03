import file_util as fu
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import json 
from matplotlib import cm
import file_util as FileHandler
from pathlib import Path
from progress.bar import IncrementalBar
from scipy import stats
import pandas as pd

def extract_data():
    data_dir = FileHandler.get_data_dir("testing_data")

    data = [dir for dir in FileHandler.get_data_files(data_dir)]

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

        bar = IncrementalBar('File loaded: ', max=num_of_environments, suffix='%(percent)d%%')

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
                for i in range(len(dist_to_target)):
                    agent_fuel += min(10, np.linalg.norm(metrics["agent_acceleration"][i]))
                    grav_lengths.append(np.linalg.norm(metrics["grav_acceleration"][i]))
                    if dist_to_target[i] < dist_to_target[0] * 0.05:
                        extracted_data["time_to_5p_to_target"].append(i * 0.01)
                        extracted_data["fuel_to_5p_to_target"].append(agent_fuel)
                        extracted_data["grav_length_reach_target"].extend(grav_lengths)
                        target_reached = 1
                        break
                extracted_data["reaches_target"].append(target_reached)

                extracted_data["end_cost"].append(dist_to_target[-1])


                for i in range(len(metrics["agent_acceleration"])):
                    model_to_gcpd_diff = np.array(metrics["agent_acceleration"][i]) - np.array(metrics["gcpd_acceleration"][i])
                    capped_model_to_gcpd_diff = np.array(cap_vector_length(metrics["agent_acceleration"][i], 10)) - np.array(metrics["gcpd_acceleration"][i])

                    for value in capped_model_to_gcpd_diff[:2]:
                        extracted_data["capped_acc_se"].append(value**2)
                        extracted_data["capped_acc_ae"].append(abs(value))

                    for value in model_to_gcpd_diff[:2]:
                        extracted_data["acc_se"].append(value**2)
                        extracted_data["acc_ae"].append(abs(value))

                ...

            bar.next()
        print( f"\t{model} " + " " * (30 - len(model)) + "has been loaded")

        processed_data[model] = {}

        for key in extracted_data.keys():
            m, h = mean_confidence_interval(extracted_data[key])
            processed_data[model][key + "_m"] = float(m)
            processed_data[model][key + "_h"] = float(h)

    with open("extracted_data.json", "w") as file:
            jsonstr = json.dumps(processed_data, indent=4)
            file.write(jsonstr)

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

def plot_data_2d(data_key, title = "", label = "y", min = 0, max = 1000):

    data = None
    with open("extracted_data.json", "r") as file:
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

    plt.show()

def fuel_to_5p_to_target_plot():
    plot_data_2d("fuel_to_5p_to_target", title="Fuel to reach target", label = "Fuel", min = 17000, max = 21000)

def fuel_plot():
    plot_data_2d("fuel", title="Fuel", label = "Fuel", min = 25000, max = 30000)

def reaches_target_plot():
    plot_data_2d("reaches_target", title="Rate of reaching target", label = "Rate", min = 0, max = 1)

def collisions_plot():
    plot_data_2d("collisions", title="Rate of collision", label = "Rate", min = 0.0, max = 1)

def agent_time_plot():
    plot_data_2d("agent_time", title="Time of neural network computations", label = "Time", min = 0, max = 0.002)

def gcpd_time_plot():
    plot_data_2d("gcpd_time", title="Time of GCPD computations", label = "Time", min = 0, max = 0.0002)

def overhead_time_plot():
    plot_data_2d("overhead_time", title="Time of overhead computations", label = "Time", min = 0, max = 0.0002)

def end_cost_plot():
    plot_data_2d("end_cost", title="Cost given as final distance to target", label = "Cost", min = 0, max = 30000)

def time_to_5p_to_target_plot():
    plot_data_2d("time_to_5p_to_target", title="Time to reach target", label = "Time", min = 20, max = 22)

def stays_at_target_plot():
    plot_data_2d("stays_at_target", title="Rate of staying at target", label = "Rate", min = 0, max = 1)

def fuel_for_at_target_at_end_plot():
    plot_data_2d("fuel_for_at_target_at_end", title="Fuel to reach and stay at target", label = "Fuel", min = 20000, max = 30000)

def abs_loss_plot():
    plot_data_2d("acc_abs_loss", title="Loss between model and GCPD (mae)", label = "Loss", min = 0, max = 5)

def square_loss_plot():
    plot_data_2d("acc_diff", title="Loss between model and GCPD (mse)", label = "Loss", min = 0, max = 5)

if __name__ == "__main__":
    #extract_data()
    # fuel_to_5p_to_target_plot()
    # fuel_plot()
    # reaches_target_plot()
    # collisions_plot()
    # agent_time_plot()
    # gcpd_time_plot()
    # overhead_time_plot()
    end_cost_plot()
    # time_to_5p_to_target_plot()
    # stays_at_target_plot()
    #fuel_for_at_target_at_end_plot()
    #abs_loss_plot()
    # square_loss_plot()

    ...
