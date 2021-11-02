from numpy.core.arrayprint import format_float_scientific
import tensorflow as tf
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


def get_data_dir(): 
    return Path.joinpath(Path().resolve(), "data")

def get_data_files(data_dir):
    return os.listdir(data_dir)

if __name__ == '__main__':
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(17))
    model.add(layers.Dense(17, activation="sigmoid"))
    model.add(layers.Dense(17, activation="relu"))
    model.add(layers.Dense(2))
    
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                #   metrics=[tf.keras.metrics.Accuracy()]
                  )


    data_dir = get_data_dir()
    data = get_data_files(data_dir)
    model_name = "full_training_17_17_sigmoid"
    trainer = TFTrainer(model, model_name)
    # trainer = TFTrainer.load_model("full_training_32_32/2021_10_27_13_43")

    evaluation_size = 30
    evaluation_data = data[:evaluation_size]
    training_data = data[evaluation_size:]
    
    evaluate_every = np.floor(len(training_data)/evaluation_size)

    start_time = time.time()
    for file_index in range(len(training_data)):
        file = data[file_index]

        path_to_json_file = str( Path.joinpath( data_dir, file ) )
        # print(f" data: {path_to_json_file}")
        print(f"Training file: {file_index}/{len(training_data)}", end="\r")


        X, y = nn_util.load_nn_data(path_to_json_file, 17, 2)

        shuffled_indexes = [i for i in range(len(X))]
        
        X = [X[i] for i in shuffled_indexes]
        y = [y[i] for i in shuffled_indexes]


        trainer.fit(X, y, batch_size=32, epochs=15, verbose=0)
        
        if file_index != 0 and file_index % (len(training_data)//evaluation_size) == 0:
            eval_index = math.floor(evaluation_size*(file_index/len(training_data)))
            
            file = evaluation_data[eval_index]
            path_to_json_file = str( Path.joinpath( data_dir, file ) )
            eval_X, eval_y = nn_util.load_nn_data(path_to_json_file, 17, 2)

            print(f"eval {eval_index}/{evaluation_size}:", end="\n")
            trainer.evaluate(eval_X, eval_y, batch_size=32)
    
    print("Done Training!")
    print(f"Time spent: { str( timedelta( seconds=time.time()-start_time ) ) }")
    trainer.evaluate(X, y, batch_size=32)
    trainer.save_model()
    
        # trainer2 = TFTrainer.load_model_from_checkpoint(model, model_name)
        # print(trainer2)

    # trainer.save_model()
    # print(predict_y, y[0])

