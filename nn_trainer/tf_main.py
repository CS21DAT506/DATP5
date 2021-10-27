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
    model.add(layers.Dense(17))
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
    model_name = "full_training_nn_256"
    trainer = TFTrainer(model, model_name)

    for file_index in range(len(data)):
        file = data[file_index]

        path_to_json_file = str( Path.joinpath( data_dir, file ) )
        print(f" data: {path_to_json_file}")
        print(f"Training file: {file_index}/{len(data)}")

        X, y = nn_util.load_nn_data(path_to_json_file, 17, 2)

    trainer.fit(X, y, batch_size=32, epochs=5, verbose=0)
    trainer.save_model()

        # trainer2 = TFTrainer.load_model_from_checkpoint(model, model_name)
        # print(trainer2)

    # trainer.save_model()
    # print(predict_y, y[0])

