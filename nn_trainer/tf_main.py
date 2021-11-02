from operator import mod
from numpy.core.arrayprint import format_float_scientific
import tensorflow as tf
from tensorflow.keras import callbacks
import tensorflow.keras.layers as layers
from tensorflow.python.eager.context import graph_mode
from TFTrainer import TFTrainer
import nn_util
from pathlib import Path
import os
import numpy as np
import math
import time
from datetime import timedelta
from plotting import plot_funcs, plot_nn_funcs
from smoketest import get_smoketest_data_points, generate_unseen_data_points, linear, parabola, sinus
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def get_data_dir(): 
    return Path.joinpath(Path().resolve(), "data")

def get_data_files(data_dir):
    return os.listdir(data_dir)

def create_model():
    model = tf.keras.models.Sequential()
    # model.add(layers.Dense(1))

    # model.add(layers.Dense(10, activation=activations.relu, input_shape=(1,)))
    # model.add(layers.Dense(10, activation=activations.relu))
    # model.add(layers.Dense(10, activation=activations.relu))

    model.add(layers.Dense(128, activation=activations.relu, input_shape=(1,)))
    model.add(layers.Dense(32, activation=activations.relu))
    model.add(layers.Dense(8, activation=activations.relu))

    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.Adam(lr=1e-3, decay=1e-3 / 200), # learning_rate=0.1
                  loss=losses.MAE,
                  #   metrics=[tf.keras.metrics.Accuracy()]
                 )
    return model

if __name__ == '__main__':

    model = create_model()
    model.summary()
    
    model_name = "sanity_training_1_64_1_relu"
    trainer = TFTrainer(model, model_name)

    start_time = time.time()

    # X, y = get_smoketest_data_points(start=-1000, stop=1000, num=4001)
    # X, y = get_smoketest_data_points(num=1201)
    X, y = get_smoketest_data_points(func=parabola, start=0, stop=10, num=10000)

    # Patient early stopping
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=200)

    trainer.cp_callbacks.append(es)
    trainer.fit(X, y, batch_size=10, epochs=10000000, verbose=1)
    
    print("Done Training!")
    print(f"Time spent: { str( timedelta( seconds=time.time()-start_time ) ) }")

    # Evaluate the modelf or unseen data!
    # X_new, y_new = generate_unseen_data_points(start=1000.5, stop=1030.5)
    X_new, y_new = generate_unseen_data_points(func=parabola, start=11, stop=20, num=9)
    trainer.evaluate(X_new, y_new, batch_size=32)


    plot_nn_funcs(parabola, "Actual parabola", trainer.predict, "nn model", start=0, stop=30)
    trainer.save_model()


