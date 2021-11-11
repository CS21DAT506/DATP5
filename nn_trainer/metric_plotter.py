import file_util as fu
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import json 
import matplotlib.cm as cm


def plot(metrics, first_axis, second_axis):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axe = fig.gca(projection='3d')

    #for i in range(metric_length):
    axe.plot(metrics[first_axis], metrics[second_axis], metrics["last_loss"], "o", color=np.array((1,0,0,1)), label="last")
    axe.plot(metrics[first_axis], metrics[second_axis], metrics["min_loss"], "o", color=np.array((0,0,1,1)), label="min")

    axe.set_xlabel(format(first_axis))
    axe.set_ylabel(format(second_axis))
    axe.set_zlabel("Loss")

    axe.legend()

    plt.show()

    ...

def loss_plot():    
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axe = fig.gca(projection='3d')

    files = fu.get_data_files("finalists")

    dicts = []

    for loss_file in files:
        with open("finalists\\" + loss_file, "rb") as file:
            json_file = file.read()
            dicts.append(json.loads(json_file))

    COLOR = cm.rainbow(np.linspace(0, 1, len(dicts)))
    for i in range(len(dicts)):
        loss = dicts[i]["loss"]
        indices = range(len(loss))
        axe.plot([i for _ in loss], indices, loss, "o", label=files[i], color=COLOR[i], markersize = 1)

    axe.set_xlabel("Model")
    axe.set_ylabel("Epoch")
    axe.set_zlabel("Loss")
    axe.set_zlim3d(0, 3)

    axe.legend()

    plt.show()

    ...

def format(string):
    new_string = string.replace("_", " ")
    return new_string[0:1].upper() + new_string[1:]


if __name__ == '__main__':
    loss_plot()
    # jsonstr = None
    # with open("model_metrics.json", "rb") as file:
    #     json_file = file.read()
    #     jsonstr = json.loads(json_file)
    # ...
    # plot(jsonstr, "layer_count", "max_layer_size") #"max_layer_size" layer_count