
import numpy as np
import json
from constants import *
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
            metrics = Util.load_json(f"{folder}/archive_{i}.json")

            if not metrics["collision"]:
                dist_to_target = metrics["dist_to_target"]
                min_dist = min(dist_to_target)
                extracted_data["min_cost"].append(min_dist)
                extracted_data["min_start_cost_ratio"].append(min_dist / dist_to_target[0])
                extracted_data["ends_further_than_start"].append(1 if dist_to_target[-1] > dist_to_target[0] else 0)

            if i % 10 == 0: 
                bar.next()
        print(f"{model}{' ' * (30 - len(model))}\n Processesing complete")

        processed_data[model] = {}

        Util.compute_statistics(processed_data[model], extracted_data)

    Util.save_json(processed_data, "data/additional_data.json")

def extract_cost_data():
    data_dir = Util.get_data_dir("testing_data")
    data = Util.get_data_files(data_dir)
    all_costs = {}

    model_508 = "nn_grav_vec_63_127_255_63"
    # costs_for_508 = list_cost(model_508, data_dir)
    # all_costs[model_508] = costs_for_508
    # Util.save_json(costs_for_508, "cost_508_data.json")

    for model in data:
        if not (model is model_508):
            all_costs[model] = list_cost(model, data_dir)
            print(f"{model}{' ' * (30 - len(model))}\n Processesing complete")

    Util.save_json(all_costs, "all_costs_data.json")

def list_cost(model, data_dir):
    costs =  []

    folder = Path.joinpath(data_dir, model)

    num_of_environments = int(len(Util.get_data_files(folder))/2)
    bar = IncrementalBar('Files loaded: ', max=num_of_environments//10, suffix='%(percent)d%%')

    for i in range(num_of_environments):
        metrics = Util.load_json(f"{folder}/archive_{i}.json")

        if not metrics["collision"]:
            costs.append(metrics["dist_to_target"][-1])

        if i % 10 == 0: 
            bar.next()
    return costs

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

                agent_fuel = Util.get_fuel(metrics["agent_acceleration"])
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

                agent_fuel, target_reached, grav_lengths = Util.reach_target_stats(metrics, extracted_data, dist_to_target)
                extracted_data["reaches_target"].append(target_reached)

                extracted_data["end_cost"].append(dist_to_target[-1])
                extracted_data["min_cost"].append(min(dist_to_target))

                Util.handle_loss(metrics, extracted_data)

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
    #extract_cost_data()
    #plot_outlierless_cost()
    #plot_stacked_bars(save_plot=True)
    #time_plot()

    #merge_data("data/extracted_data.json", "data/additional_data.json", "data/merged_data.json")
    #extract_data()
    #extract_additional_data()
    plot_all(save_plots=True)
