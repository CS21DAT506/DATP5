from datetime import datetime
import tensorflow as tf
import tensorflow.keras.layers as layers
from TFTrainer import TFTrainer
import nn_util
from pathlib import Path
import os
import time
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle


def get_data_dir(dir_name): 
    return Path.joinpath(Path().resolve(), dir_name)

def get_data_files(data_dir):
    return os.listdir(data_dir)

def train_nn_from_layers(settings, training_settings, layer_nums, X, y):
    tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()

    for layer_size in layer_nums:
        model.add(layers.Dense(layer_size, activation=activations.relu))

    
    # model.add(layers.Dense(32, activation=activations.relu))
    # model.add(layers.Dense(8, activation=activations.relu))
    model.add(layers.Dense(2))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3, decay=1e-3 / 200),  # learning_rate=0.1
                  loss=losses.MAE,
                  #   metrics=[tf.keras.metrics.Accuracy()]
                  )

    trainer = TFTrainer(model, settings["model_save_name"])

    #data_dir = get_data_dir(settings["data_dir_name"])
    # start_time = time.time()

    if (settings["do_early_stopping"]):
        es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=settings["early_stopping_patience"])
        trainer.cp_callbacks.append(es)

    # hist = None
    # hist_folder = None
    # if (settings["data_in_single_file"]):
    hist = trainer.fit(X, y, batch_size=training_settings["bacth_size"], epochs=training_settings["epochs"], verbose=training_settings["verbose"], validation_split=0.3)
    hist_folder = trainer.save_path + "\\" + settings["model_save_name"] + "\\history"

    #print("Done Training!")
    # print(f"Time spent: { str( timedelta( seconds=time.time()-start_time ) ) }")

    if (settings["do_save_model"]):
        trainer.save_model()
    if (not os.path.exists(hist_folder)):
        os.mkdir(hist_folder)
    with open(hist_folder + "\\" + datetime.now().strftime("%Y_%m_%d_%H_%M"), 'wb') as file:
        pickle.dump(hist.history, file)


def file_name_from_layers(layer_nums):
    out = "nn_grav_vec"
    for layer_size in layer_nums:
        out += "_" + str(int(layer_size))
    return out

def add_layer(layers):
    if (len(layers) % 2 == 1):
        return np.insert(layers, 1 + len(layers) // 2, np.max(layers) * 2)
    else:
        return np.insert(layers, 1 + len(layers) // 2, np.max(layers) // 2)

def add_decreasing_layer(layers):
    return np.insert(layers * 2, len(layers), 8)

if __name__ == '__main__':
    layer_nums = np.array([32, 16, 8])
    counter = 1
    factor = 1
    settings = {
            "data_in_single_file": True,
            "data_dir_name": "data",
            "data_file_name": "gravity_vector_data.json",
            "model_save_name": file_name_from_layers(layer_nums),
            "do_evaluation": True,
            "do_early_stopping": False,
            "do_save_model": True,
            "evaluation_size": 30,
            "early_stopping_patience": 500,
            "shuffle_data_in_batch": False,
    }
    training_settings = {
        "bacth_size": 32000,
        "epochs": 6000,
        "verbose": 0,
    }

    data_dir = get_data_dir(settings["data_dir_name"])
    path_to_json_file = Path.joinpath(data_dir, settings["data_file_name"])
    print("Loading data from", path_to_json_file, "...")
    X, y = nn_util.load_nn_data(path_to_json_file)
    print("Data retrieved!")
    X = np.array(X)
    y = np.array(y)
    print("Ready!")

    while len(layer_nums) < 10:
        actual_layers = np.floor(layer_nums * factor)
        settings["model_save_name"] = file_name_from_layers(actual_layers)
        print(f"Run {counter}/{70} in progress...", end="\r")
        train_nn_from_layers(settings, training_settings, actual_layers, X, y)
        print(" "*30, f"\rRuns trained: {counter}/{70}")
        counter += 1
        if (factor >= 4):
            layer_nums = add_decreasing_layer(layer_nums)
            factor = 1
        else:
            factor *= 2 ** (1/4)
        


    # else:
    #     if (settings["do_evaluation"]):
    #         data = get_data_files(data_dir)
    #         evaluation_data = data[:settings["evaluation_size"]]
    #         training_data = data[settings["evaluation_size"]:]
    #         evaluate_every = np.floor(len(training_data)/settings["evaluation_size"])

    #     for file_index in range(len(training_data)):
    #         file = data[file_index]

    #         path_to_json_file = str( Path.joinpath( data_dir, file ) )
    #         # print(f" data: {path_to_json_file}")
    #         print(f"Training file: {file_index}/{len(training_data)}", end="\r")


    #         X, y = nn_util.load_nn_data(path_to_json_file)

    #         if (settings["shuffle_data_in_batch"]):
    #             shuffled_indexes = [i for i in range(len(X))]

    #             X = [X[i] for i in shuffled_indexes]
    #             y = [y[i] for i in shuffled_indexes]

    #         trainer.fit(X, y, batch_size=training_settings["bacth_size"], epochs=training_settings["epochs"], verbose=training_settings["verbose"])
            
    #         if (settings["do_evaluation"]):
    #             if file_index != 0 and file_index % (len(training_data)//settings["evaluation_size"]) == 0:
    #                 eval_index = np.floor(settings["evaluation_size"]*(file_index/len(training_data)))
                    
    #                 file = evaluation_data[eval_index]
    #                 path_to_json_file = str( Path.joinpath( data_dir, file ) )
    #                 eval_X, eval_y = nn_util.load_nn_data(path_to_json_file, 17, 2)

    #                 print(f"eval {eval_index}/{settings['evaluation_size']}:", end="\n")
    #                 trainer.evaluate(eval_X, eval_y, batch_size=training_settings["bacth_size"])
                    
