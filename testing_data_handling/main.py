
import numpy as np
import json
import util as Util
from pathlib import Path
from progress.bar import IncrementalBar
from data_plotting import *
from plot_setups import plot_setups

def extract_additional_data():
    data_dir = Util.get_data_dir("testing_data")

    data = Util.get_data_files(data_dir)

    processed_data = {}

    for model in data:
        extracted_data = {
            "min_cost": [],
            "min_start_cost_ratio": [],
            "ends_further_than_start": []
        }

        folder = Path.joinpath(data_dir, model)

        num_of_environments = int(len(Util.get_data_files(folder))/2)
        bar = IncrementalBar('Files loaded: ', max=num_of_environments//10, suffix='%(percent)d%%')

        for i in range(num_of_environments):
            metrics = Util.load_json(str(folder) + "/archive_" + str(i) + ".json")

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

        Util.compute_statistics(processed_data[model], extracted_data)

    Util.save_json(processed_data, "data/additional_data.json")

def extract_data():
    data_dir = Util.get_data_dir("testing_data")

    data = Util.get_data_files(data_dir)

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

        num_of_environments = int(len(Util.get_data_files(folder))/2)
        bar = IncrementalBar('Files loaded: ', max=num_of_environments//10, suffix='%(percent)d%%')

        for i in range(num_of_environments):
            metrics = Util.load_json(str(folder) + "/archive_" + str(i) + ".json")

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
                    capped_model_to_gcpd_diff = np.array(Util.cap_vector_length(metrics["agent_acceleration"][ii], 10)) - np.array(metrics["gcpd_acceleration"][ii])

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

        Util.compute_statistics(processed_data, extracted_data)
        Util.save_json(processed_data, "data/extracted_data_" + model + ".json")

def merge_data(path1, path2, save_path):
    data1 = Util.load_json(path1)
    data2 = Util.load_json(path2)

    for model_keys in data2.keys():
        for data_keys in data2[model_keys]:
            data1[model_keys][data_keys] = data2[model_keys][data_keys] 

    Util.save_json(data1, save_path)

def plot_all(save_plots):
    plot_stacked_bars(save_plots)
    for title in plot_setups.keys():
        plot_with_formatting(plot_setups, title, save_plots)
    time_plot(save_plots)

if __name__ == "__main__":
    #merge_data("data/extracted_data.json", "data/additional_data.json", "data/merged_data.json")
    #extract_data()
    #extract_additional_data()
    #plot_all(save_plots=True)
    plot_all_508_cost()
    ...
