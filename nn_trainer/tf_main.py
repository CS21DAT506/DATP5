from datetime import datetime
import tensorflow as tf
import tensorflow.keras.layers as layers
from TFTrainer import TFTrainer
import util as Util
from pathlib import Path
import os
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle
from settings.settings_access import settings

def train_nn_from_layers(settings, layer_nums, X, y, model_save_name):
    tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()

    for layer_size in layer_nums:
        model.add(layers.Dense(layer_size, activation=activations.relu))

    model.add(layers.Dense(2))

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3, decay=1e-3 / 200), loss=losses.MAE)

    trainer = TFTrainer(model, model_save_name)

    if settings.do_early_stopping:
        es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=settings.early_stopping_patience)
        trainer.cp_callbacks.append(es)

    hist = trainer.fit(X, y, batch_size=settings.batch_size, epochs=settings.epochs, verbose=settings.verbose, validation_split=0.3)
    hist_folder = trainer.save_path_str + "\\" + model_save_name + "\\history"

    if settings.do_save_model:
        trainer.save_model()
    if not os.path.exists(hist_folder):
        os.mkdir(hist_folder)
    
    #Save history as byte file with pickle 
    with open(hist_folder + "\\" + datetime.now().strftime("%Y_%m_%d_%H_%M"), 'wb') as file:
        pickle.dump(hist.history, file)

def train_multiple():
    data_dir = Util.get_dir(settings.data_dir_name)
    path_to_json_file = Path.joinpath(data_dir, settings.data_file_name)

    print("Loading data from", path_to_json_file, "...")
    x, y = Util.load_nn_data(path_to_json_file)
    print("Ready!")

    layer_nums = np.array(settings.initial_layers)
    counter = 1
    factor = 1
    while len(layer_nums) < settings.max_layers:
        actual_layers = np.floor(layer_nums * factor)

        print(f"Run {counter} in progress...", end="\r")
        model_save_name = Util.file_name_from_layers(layer_nums)

        train_nn_from_layers(settings, actual_layers, x, y, model_save_name)
        print(" "*30, f"\rRuns trained: {counter}")

        counter += 1
        factor, layer_nums = Util.get_updated_layers(factor, layer_nums, settings.is_funnel_shaped)

if __name__ == '__main__':
    train_multiple()
