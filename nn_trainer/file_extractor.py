import pickle
import file_util as fu
import numpy as np
from pathlib import Path
import json 

def extract(): 

    data_dir = fu.get_data_dir("saved_models")

    data = [dir for dir in fu.get_data_files(data_dir) if "nn_grav" in dir ]

    layer_sizes = [string.split('_')[3:] for string in data ]

    layer_count = [len(layer) for layer in layer_sizes]
    max_layer_size = [max([int(size) for size in model]) for model in layer_sizes]
    last_loss = []
    min_loss = []
    epochs = []
    all_losses = []
    all_val_losses = []
    last_val_loss = []

    for model in data:
        hist_files = fu.get_data_files(Path.joinpath(data_dir, model + "\\history"))

        for file in hist_files:
            hist = None
            with open(Path.joinpath(data_dir, model + "\\history\\" + file), "rb") as filename:
                hist = pickle.load(filename)
            losses = hist["loss"]
            val_losses = hist["val_loss"]
            last_val_loss.append(val_losses[-1])
            all_losses.append(losses)
            all_val_losses.append(val_losses)
            last_loss.append(losses[-1])
            min_loss.append(min(losses))
            epochs.append(len(losses))


    for i in range(len(epochs)):
        print(str(i) + " | 0.4 > " + str(last_val_loss[i]))
        if last_val_loss[i] < 0.42:
            print(str(i) )
            config = ""
            for size in layer_sizes[i]:
               config = config + str(size) + "_"
        
            losses = {
                "loss": all_losses[i],
                "val_loss": all_val_losses[i]
            }

            with open("fun_w_val_finalists\\" + config + ".json", "w") as file:
                jsonstr = json.dumps(losses, indent=4)
                file.write(jsonstr)

    # metrics = {
    #     "last_val_loss": last_val_loss,
    #     "layer_count": layer_count,
    #     "max_layer_size": max_layer_size,
    #     "last_loss": last_loss,
    #     "min_loss": min_loss,
    #     "epochs": epochs
    # }

    # with open("val_rhomb_model_metrics.json", "w") as file:
    #     jsonstr = json.dumps(metrics, indent=4)
    #     file.write(jsonstr)

    



    ...

    # for file_index in range(len(training_data)):
    #     file = data[file_index]

    #     path_to_json_file = str( Path.joinpath( data_dir, file ) )
    #     print(f" data: {path_to_json_file}")
    #     print(f"Training file: {file_index}/{len(training_data)}", end="\r")


    #     X, y = nn_util.load_nn_data(path_to_json_file)


if __name__ == '__main__':
    extract()

