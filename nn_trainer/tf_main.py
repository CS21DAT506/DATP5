import tensorflow as tf
import tensorflow.keras.layers as layers

from TFTrainer import TFTrainer
import nn_util
from pathlib import Path
import os

def get_data_dir(): 
    return Path.joinpath(Path().resolve(), "data")

def get_data_files(data_dir):
    return os.listdir(data_dir)

if __name__ == '__main__':
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(17, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2))
    
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy']
                  )


    data_dir = get_data_dir()
    data = get_data_files(data_dir)

    for file in data:

        path_to_json_file = str( Path.joinpath( data_dir, file ) )
        print(f" data: {path_to_json_file}")

        X, y = nn_util.load_nn_data(path_to_json_file, 17, 2)

        model_name = "model_4"
        trainer = TFTrainer(model, model_name)
        trainer.setup_checkpoint_save()

        trainer.fit(X, y, epochs=5, batch_size=32, verbose=1)

        # trainer2 = TFTrainer.load_model_from_checkpoint(model, model_name)
        # print(trainer2)

    # trainer.save_model()
    # print(predict_y, y[0])

