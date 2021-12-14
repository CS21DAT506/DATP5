import matplotlib.pyplot as plt
import numpy as np
import json 
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
import pandas as pd
import util as Util
from pathlib import Path
from progress.bar import IncrementalBar
import util as Util
import matplotlib.ticker as ticker

def plot_all_508_cost():
    costs = Util.load_json("cost_508_data.json")
    
    amounts = [0] * 10
    accumulated = [0] * 10
    sum_accumulated = 0

    print(len(costs))

    for cost in costs:
        index = 1 + int(np.floor(np.log10(cost)))
        if index < 0:
            raise Exception("index too low")
        amounts[index] += 1
        accumulated[index] += cost
        sum_accumulated += cost

    print((accumulated[8] / sum_accumulated) * 100)

    x = range(-1, 9)
    simple_plot(x, amounts, "Amount", "$\lfloor Log_{10}(Cost) \\rfloor$", "Amount of simulations with cost of each order of magnitude", max = 3500)
    simple_plot(x, accumulated, "Accumulated Cost", "$\lfloor Log_{10}(Cost) \\rfloor$", "Accumulated cost within each order of magnitude", max = 500000000)
    
def simple_plot(x, y, ylabel, xlabel, title, min=0, max=1000):
    COLOR = cm.rainbow(np.linspace(0, 1, len(x)))

    x = [ str(value) for value in x]

    plt.bar(x, y, color=COLOR, capsize=4)

    fig = plt.subplot()

    box = fig.get_position()
    fig.set_position([box.x0 + box.width * 0.02, box.y0, box.width, box.height])

    plt.ylim(min, max)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)


    plt.show()

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

    plt.xticks(sorted_data["labels"], rotation = 30)

    ax.set_ylabel("Likelihood")
    ax.set_title("Outcomes and their likelihoods")
    ax.legend(loc='center left', bbox_to_anchor=(0.2, -0.4))
    
    if(save_plot):
        plt.savefig("plots/Outcomes.png")
        plt.clf()
    else:
        plt.show()

def plot_data_2d(data_key, title, label = "y", min = 0, max = 1000, unwanted_models = [], save_plot = False):

    data = Util.load_json("extracted_data.json")

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
    fig.set_position([box.x0 + box.width * 0.02, box.y0 + box.height * 0.1, box.width, box.height * 0.95])

    plt.xticks(x, labels, rotation = 30)
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

def plot_geometric_mean_cost(save_plot = False):

    data = Util.load_json("all_costs_data.json")

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))
    means = []
    errors=[]

    for model_costs in data.values():
        m, h = Util.geometric_mean_confidence_interval(model_costs)
        means.append(m)
        errors.append([m*h,m/h])
   
    labels = [Util.compute_label(s) for s in data.keys()]

    #Pandas is used to sort items by labels
    data = pd.DataFrame({
        "labels": labels,
        "h": errors,
        "mean": means
    })

    sorted_data = data.sort_values(by=["labels"])
    labels = sorted_data["labels"]
    
    x = range(len(labels))

    plt.bar(x, sorted_data["mean"], color=COLOR, capsize=4)

    fig = plt.subplot()

    box = fig.get_position()
    fig.set_position([box.x0 + box.width * 0.02, box.y0, box.width, box.height])

    title = "Geometric mean of cost given as final distance to target"

    plt.xticks(x, labels)
    plt.ylim(0, 100)
    plt.ylabel("Geometric mean of cost")
    plt.xlabel("Model")
    plt.title(title)

    if(save_plot):
        plt.savefig("plots/" + title.replace(" ", "_") + ".png")
        plt.clf()
    else:
        plt.show()

def plot_outlierless_cost(save_plot = False):

    data = Util.load_json("all_costs_data.json")

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))
    means = []
    erros = []

    sum = 0
    outlier_sum = 0

    for model, model_costs in data.items():
        non_outliers = [x for x in model_costs if np.log10(x) < 7]
        diff = len(model_costs) - len(non_outliers)
        m, h = Util.mean_confidence_interval(non_outliers)
        means.append(m)
        erros.append(h)

        conf_int_str = f"{int(m)} +- {int(h)}"
        print(f"{model}:{' ' * (30 - len(model))} {conf_int_str}{' ' * (15 - len(conf_int_str))} | {diff} outliers removed")

        sum_for_model = np.sum(model_costs)
        sum += sum_for_model
        outlier_sum +=  sum_for_model - np.sum(non_outliers)
   
    print(f"{outlier_sum / sum * 100}")

    labels = [Util.compute_label(s) for s in data.keys()]

    data = pd.DataFrame({
        "labels": labels,
        "h": erros,
        "mean": means
    })

    sorted_data = data.sort_values(by=["labels"])
    labels = sorted_data["labels"]
    
    x = range(len(labels))

    plt.bar(x, sorted_data["mean"], yerr=sorted_data["h"], color=COLOR, capsize=4)

    fig = plt.subplot()

    box = fig.get_position()
    fig.set_position([box.x0 + box.width * 0.02, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    title = "Cost given as final distance to target (costs over $10^7$ removed)"

    plt.xticks(x, labels, rotation = 30)
    plt.ylim(0, 6000)
    plt.ylabel("Cost")
    plt.xlabel("Model")
    plt.title(title)

    if(save_plot):
        plt.savefig("plots/" + title.replace(" ", "_") + ".png")
        plt.clf()
    else:
        plt.show()

def time_plot(save_plot = False):

    data = Util.load_json("extracted_data.json")

    COLOR = cm.rainbow(np.linspace(0, 1, len(data)))

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

    plt.xticks(x, labels, rotation = 30)
    plt.ylim(0, 0.002)
    plt.ylabel("Time [s]")
    plt.xlabel("Model")
    plt.title(title)
    
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(Util.Sci_Formatter)) #plt.LogFormatter(10, labelOnlyBase = False))

    fig = plt.subplot()

    box = fig.get_position()
    fig.set_position([box.x0 + box.width * 0.02, box.y0 + box.width * 0.05, box.width * 1.08, box.height])

    if(save_plot):
        plt.savefig("plots/" + title.replace(" ", "_") + ".png")
        plt.clf()
    else:
        plt.show()