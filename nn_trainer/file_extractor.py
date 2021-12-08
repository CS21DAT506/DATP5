import util as Util
from pathlib import Path
import json 

def extract(path, save_metrics, save_epochs_for_best_models): 

    data_dir = Util.get_dir(path)

    data = [dir for dir in Util.get_data_files(data_dir) if "nn_grav" in dir ]

    layer_sizes = [string.split('_')[3:] for string in data ]

    metrics = {
        "last_val_loss": [],
        "layer_count": [len(layer) for layer in layer_sizes],
        "max_layer_size": [max([int(size) for size in model]) for model in layer_sizes],
        "last_loss": [],
        "min_loss": [],
        "epochs": []
    }

    all_losses = []
    all_val_losses = []

    for model in data:
        hist_files = Util.get_data_files(Path.joinpath(data_dir, model + "\\history"))

        for file in hist_files:
            hist = Util.load_byte_file(Path.joinpath(data_dir, model + "\\history\\" + file))
            
            losses = hist["loss"]
            val_losses = hist["val_loss"]

            metrics["last_val_loss"].append(val_losses[-1])
            metrics["last_loss"].append(losses[-1])
            metrics["min_loss"].append(min(losses))
            metrics["epochs"].append(len(losses))

            all_losses.append(losses)
            all_val_losses.append(val_losses)

    if save_epochs_for_best_models:
        save_epochs_for_best_models(len(metrics["epochs"]), metrics["last_val_loss"], layer_sizes, all_losses, all_val_losses)

    if save_metrics :
        save_metrics(path, metrics)

def save_metrics(path, metrics):
    with open(path + "_metrics.json", "w") as file:
        jsonstr = json.dumps(metrics, indent=4)
        file.write(jsonstr)

def save_epochs_for_best_models(num_of_epochs, last_val_loss, layer_sizes, all_losses, all_val_losses):
    for i in range(num_of_epochs):
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

if __name__ == '__main__':
    extract(path = "fun_w_val_grav_vec", save_metrics = True, save_epochs_for_best_models=False)

