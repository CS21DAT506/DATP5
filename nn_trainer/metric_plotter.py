import file_util as fu
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import json 
import matplotlib.cm as cm


def plot(file_name, first_axis, second_axis, val=False):
    metrics = None
    with open(file_name, "rb") as file:
        json_file = file.read()
        metrics = json.loads(json_file)
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axe = fig.gca(projection='3d')

    #for i in range(metric_length):
    axe.plot(metrics[first_axis], metrics[second_axis], metrics["last_loss"], "o", color=np.array((1,0,0,1)), label="last")
    axe.plot(metrics[first_axis], metrics[second_axis], metrics["min_loss"], "o", color=np.array((0,0,1,1)), label="min")
    if val: 
        axe.plot(metrics[first_axis], metrics[second_axis], metrics["last_val_loss"], "o", color=np.array((0,1,0,1)), label="val")

    axe.set_xlabel(format(first_axis))
    axe.set_ylabel(format(second_axis))
    axe.set_zlabel("Loss")

    axe.legend()

    plt.show()

    ...

def loss_plot(file_folder, smooth=False):    
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axe = fig.gca(projection='3d')

    files = fu.get_data_files(file_folder)

    dicts = []

    for loss_file in files:
        with open(file_folder + "\\" + loss_file, "rb") as file:
            json_file = file.read()
            dicts.append(json.loads(json_file))

    dicts_len = len(dicts)
    COLOR = cm.rainbow(np.linspace(0, 1, dicts_len * 2))
    for i in range(dicts_len):
        loss = dicts[i]["loss"]
        val_loss = dicts[i]["val_loss"]
        loss_indices = range(len(loss))
        val_loss_indices = range(len(val_loss))
        smooth_loss = [(loss[i] + loss[(i - 1) if i != 0 else i]) / 2 for i in loss_indices]
        smooth_val_loss = [(val_loss[i] + val_loss[(i - 1) if i != 0 else i]) / 2 for i in val_loss_indices]
        axe.plot([2 * i for _ in loss], loss_indices , smooth_loss if smooth else loss, "o", label="loss " + str(i), color=COLOR[i], markersize = 1)
        axe.plot([2 * i + 0.5 for _ in val_loss], val_loss_indices, smooth_val_loss if smooth else val_loss, "o", label="validation loss " + str(i), color=COLOR[i + dicts_len], markersize = 1)

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
    #loss_plot("rhomb_val_finalists", smooth=False)
    plot("fun_w_val_grav_vec_metrics.json", "layer_count", "max_layer_size", val=True) #"max_layer_size" layer_count epochs