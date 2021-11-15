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

    for model in data:
        hist_files = fu.get_data_files(Path.joinpath(data_dir, model + "\\history"))

        for file in hist_files:
            hist = None
            with open(Path.joinpath(data_dir, model + "\\history\\" + file), "rb") as filename:
                hist = pickle.load(filename)
            hist = hist["loss"]
            all_losses.append(hist)
            last_loss.append(hist[-1])
            min_loss.append(min(hist))
            epochs.append(len(hist))


    for i in range(len(epochs)):
        if last_loss[i] < 1.0:
            config = ""
            for size in layer_sizes[i]:
               config = config + str(size) + "_"
        
            losses = {
                "loss": all_losses[i]
            }

            with open("dec_finalists\\" + config + ".json", "w") as file:
                jsonstr = json.dumps(losses, indent=4)
                file.write(jsonstr)

    # metrics = {
    #     "layer_count": layer_count,
    #     "max_layer_size": max_layer_size,
    #     "last_loss": last_loss,
    #     "min_loss": min_loss,
    #     "epochs": epochs
    # }

    # with open("dec_model_metrics.json", "w") as file:
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

