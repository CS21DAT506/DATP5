import tensorflow as tf
from tensorflow.keras.layers import Dense
from pathlib import Path

import os
import datetime

from tensorflow.python.ops.gen_batch_ops import batch

class TFTrainer:
    def __init__(self, model, check_point_path, save_name, save_path="saved_models", batch_size=32):
        self.check_point_path = str(Path().resolve()) + "\\" + check_point_path
        self.save_path = str(Path().resolve()) + "\\" + save_path
        self.save_name = save_name
        self.batch_size = batch_size
        self.model = model
        # self.cp_callback = self.setup__checkpoint_save(model, check_point_path)


    # def setup_checkpoint_save(self):
    #     pass

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, x, y, *args, **kwargs):
        # return self.model.train_on_batch(x, y)
        return self.model.fit(x, y, *args, **kwargs)
        # return self.model.fit(x, y, verbose=verbose, steps_per_epoch=len(x), epochs=5)
        # return self.model.fit(x, y, verbose=verbose, batch_size=self.batch_size)
        # return self.model.fit(x, y, args, kwargs, verbose=verbose)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)




    def save_model(self):
        model_path = self.save_path + "\\" +self.save_name
        total_save_path = model_path + "\\" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

        paths = [self.save_path, model_path, total_save_path]

        for path in paths:
            if (not os.path.exists(path)):
                os.mkdir(path)        

        # return self.model.save(total_save_path)
        return self.model.save(total_save_path)

    # @classmethod
    # def load_weights(cls, model, check_point_path, save_dir="saved_models", batch_size=32):
    #     cls = TFTrainer(model, check_point_path, save_dir, batch_size)
    #     cls.load_weights(check_point_path)
    #     return cls

    @classmethod
    def load_model(cls, checkpoint_path, save_name, save_dir="saved_models", batch_size=32):
        """
        save_name is of the format {name_of_model}\{timestamp_of_safe}
            timestamp is of format {year_month_data_24HourFormat_minutes}
        """
        save_path = save_dir + "\\" + save_name
        model = tf.keras.models.load_model(save_path)

        cls = TFTrainer(model, checkpoint_path, save_name, save_dir, batch_size)
        return cls





if __name__ == "__main__":
    
    model = tf.keras.models.Sequential()

    model.add(Dense(4, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    trainer = TFTrainer(model, "data", "model_3")    
    trainer.predict([[[0,1,2,3]]])
    trainer.save_model()

    # trainer = TFTrainer.load_model("cp_path",r"model_1\2021_10_20_11_29")
    print(trainer)
