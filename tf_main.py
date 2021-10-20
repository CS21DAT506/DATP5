import tensorflow as tf
import tensorflow.keras.layers as layers

from TFTrainer import TFTrainer
import nn_util

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

    X, y = nn_util.load_nn_data("data.json", 17, 2)

    trainer = TFTrainer(model, "", "first_model")

    trainer.fit(X, y, epochs=5, batch_size=32)
    # print(predict_y, y[0])

