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


def get_data_dir(dir_name): 
    return Path.joinpath(Path().resolve(), dir_name)

def get_data_files(data_dir):
    return os.listdir(data_dir)

if __name__ == '__main__':
    settings = {
        "data_in_single_file": True,
        "data_dir_name": "data",
        "data_file_name": "gravity_vector_data.json",
        "model_save_name": "no_planet_128_32_8",
        "do_evaluation": True,
        "do_early_stopping": True,
        "do_save_model": True,
        "evaluation_size": 30,
        "early_stopping_patience": 50,
        "shuffle_data_in_batch": False,
    }
    training_settings = {
        "bacth_size": 32000,
        "epochs": 10000,
        "verbose": 2,
    }

    model = tf.keras.models.Sequential()

    model.add(layers.Dense(128, activation=activations.relu))
    model.add(layers.Dense(32, activation=activations.relu))
    model.add(layers.Dense(8, activation=activations.relu))
    model.add(layers.Dense(2))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3, decay=1e-3 / 200),  # learning_rate=0.1
                  loss=losses.MAE,
                  #   metrics=[tf.keras.metrics.Accuracy()]
                  )

    trainer = TFTrainer(model, settings["model_save_name"])

    data_dir = get_data_dir(settings["data_dir_name"])
    # start_time = time.time()

    if (settings["do_early_stopping"]):
        es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=settings["early_stopping_patience"])
        trainer.cp_callbacks.append(es)

    if (settings["data_in_single_file"]):
        path_to_json_file = Path.joinpath(data_dir, settings["data_file_name"])
        print("Loading data from", path_to_json_file, "...")
        X, y = nn_util.load_nn_data(path_to_json_file)
        print("Data retrieved!")
        trainer.fit(X, y, batch_size=training_settings["bacth_size"], epochs=training_settings["epochs"], verbose=training_settings["verbose"])
    else:
        if (settings["do_evaluation"]):
            data = get_data_files(data_dir)
            evaluation_data = data[:settings["evaluation_size"]]
            training_data = data[settings["evaluation_size"]:]
            evaluate_every = np.floor(len(training_data)/settings["evaluation_size"])

        for file_index in range(len(training_data)):
            file = data[file_index]

            path_to_json_file = str( Path.joinpath( data_dir, file ) )
            # print(f" data: {path_to_json_file}")
            print(f"Training file: {file_index}/{len(training_data)}", end="\r")


            X, y = nn_util.load_nn_data(path_to_json_file)

            if (settings["shuffle_data_in_batch"]):
                shuffled_indexes = [i for i in range(len(X))]

                X = [X[i] for i in shuffled_indexes]
                y = [y[i] for i in shuffled_indexes]

            trainer.fit(X, y, batch_size=training_settings["bacth_size"], epochs=training_settings["epochs"], verbose=training_settings["verbose"])
            
            if (settings["do_evaluation"]):
                if file_index != 0 and file_index % (len(training_data)//settings["evaluation_size"]) == 0:
                    eval_index = np.floor(settings["evaluation_size"]*(file_index/len(training_data)))
                    
                    file = evaluation_data[eval_index]
                    path_to_json_file = str( Path.joinpath( data_dir, file ) )
                    eval_X, eval_y = nn_util.load_nn_data(path_to_json_file, 17, 2)

                    print(f"eval {eval_index}/{settings['evaluation_size']}:", end="\n")
                    trainer.evaluate(eval_X, eval_y, batch_size=training_settings["bacth_size"])
                    

    print("Done Training!")
    # print(f"Time spent: { str( timedelta( seconds=time.time()-start_time ) ) }")

    if (settings["do_save_model"]):
        trainer.save_model()
