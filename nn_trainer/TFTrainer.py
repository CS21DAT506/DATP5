import tensorflow as tf
from pathlib import Path
import datetime
import util as Util

class TFTrainer:
    def __init__(self, model, save_name, save_path_str="saved_models", batch_size=32, eager=True):
        self._initialize_paths(save_name, save_path_str)
        self.model_save_path_Exist = False

        if not eager:
            tf.compat.v1.disable_eager_execution()

        self.batch_size = batch_size
        self.model = model
        self.cp_callbacks = []

    def _initialize_paths(self, save_name, save_path_str):
        save_path = Util.get_dir(save_path_str)
        self.model_save_path = Path.joinpath( save_path, save_name )

        self.save_path_str = str(save_path)
        self.model_save_path_str = str( self.model_save_path )

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
            paths = [self.save_path_str, self.model_save_path_str]

        for path in paths:
            Util.ensure_dir_exists(path)
        self.model_save_path_Exist = True

    def save_model(self):
        self._setup_model_save_path()

        save_instance_path = str( Path.joinpath( self.model_save_path, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") ) )
        Util.ensure_dir_exists(save_instance_path)

        return self.model.save(save_instance_path)
