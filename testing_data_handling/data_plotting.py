import matplotlib.pyplot as plt
import numpy as np
import json 
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
import pandas as pd
import util as Util


def plot_with_formatting(plot_setups, title, save_plot = False):
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


    labels = [Util.compute_label(s) for s in data.keys()]

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

def plot_data_2d(data_key, title, label = "y", min = 0, max = 1000, unwanted_models = [], save_plot = False):

    data = None
    with open("extracted_data.json", "r") as file:
                json_file = file.read()
                data = json.loads(json_file)

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))
    plot_design = ["-o", "-d"]

    mean = [model[data_key + "_m"] for model in data.values()]
    h = [model[data_key + "_h"] for model in data.values()]
   
    labels = [Util.compute_label(s) for s in data.keys()]

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
        plt.gca().get_yaxis().set_major_formatter(FuncFormatter(Util.Sci_Formatter)) #plt.LogFormatter(10, labelOnlyBase = False))

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
   
    labels = [Util.compute_label(s) for s in data.keys()]

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
    
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(Util.Sci_Formatter)) #plt.LogFormatter(10, labelOnlyBase = False))

    fig = plt.subplot()

    box = fig.get_position()
    fig.set_position([box.x0 + box.width * 0.02, box.y0, box.width * 1.08, box.height])

    if(save_plot):
        plt.savefig("plots/" + title.replace(" ", "_") + ".png")
        plt.clf()
    else:
        plt.show()