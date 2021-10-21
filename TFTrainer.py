import tensorflow as tf
from tensorflow.keras.layers import Dense
from pathlib import Path

import os
import datetime

class TFTrainer:
    def __init__(self, model, save_name, save_path="saved_models", batch_size=32):
        self.save_path = str(Path().resolve()) + "\\" + save_path
        self.model_save_path = self.save_path + "\\" + save_name
        self.check_point_path = self.model_save_path + "\\checkpoints\\"
        # self.save_name = save_name
        self.model_save_path_Exist = False

        self.batch_size = batch_size
        self.model = model
        self.cp_callbacks = []


    def setup_checkpoint_save(self, verbose=0):
        self._setup_model_save_path()

        if (not os.path.exists(self.check_point_path)):
            os.mkdir(self.check_point_path)

        self.cp_callbacks.append(
            # tf.keras.callbacks.ModelCheckpoint(self.check_point_path,save_weights_only=True,verbose=verbose)
            tf.keras.callbacks.ModelCheckpoint(self.check_point_path,verbose=verbose)
        )

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, x, y, *args, **kwargs):
        return self.model.fit(x, y, *args, callbacks=self.cp_callbacks, **kwargs)



    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)


    def _setup_model_save_path(self):
        if self.model_save_path_Exist:
            return
        else:
            paths = [self.save_path, self.model_save_path]

        for path in paths:
            if (not os.path.exists(path)):
                os.mkdir(path)
        self.model_save_path_Exist = True

    def save_model(self):
        self._setup_model_save_path()

        save_instance_path = self.model_save_path + "\\" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

        if (not os.path.exists(save_instance_path)):
            os.mkdir(save_instance_path)        

        return self.model.save(save_instance_path)

    def load_weights(self):
        self.model.load_weights(self.check_point_path)

    @classmethod
    def load_model_from_checkpoint(cls, model, save_name, save_dir="saved_models", batch_size=32):
        cls = TFTrainer(model, save_name, save_dir, batch_size)
        cls.load_weights()
        return cls

    @classmethod
    def load_model(cls, save_name, save_dir="saved_models", batch_size=32):
        """
        save_name is of the format {name_of_model}\{timestamp_of_safe}
            timestamp is of format {year_month_data_24HourFormat_minutes}
        """
        save_path = save_dir + "\\" + save_name
        model = tf.keras.models.load_model(save_path)

        cls = TFTrainer(model,  save_name, save_dir, batch_size)
        return cls
    


if __name__ == "__main__":
    
        # model = tf.keras.models.Sequential()

        # model.add(Dense(4, activation="relu"))
        # model.add(Dense(64, activation="relu"))
        # model.add(Dense(64, activation="relu"))
        # model.add(Dense(2, activation="softmax"))

        # model.compile(optimizer='adam',
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])

        # trainer = TFTrainer(model, "model_3")    
        # trainer.predict([[[0,1,2,3]]])
        # trainer.save_model()

    trainer = TFTrainer.load_model(r"model_4\2021_10_21_10_41")
