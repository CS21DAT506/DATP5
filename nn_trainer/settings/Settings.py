import confuse
from pathlib import Path

dir_path = Path(__file__).resolve().parent
app_name = dir_path.parent.name

class Settings():

    def parse_config(self):
        self._config_obj = confuse.Configuration(app_name, __name__)
        abs_path_to_file = str ( Path.joinpath(dir_path, "config.yaml" ) )
        self._config_obj.set_file( abs_path_to_file )

    def _get_entry(self, name):
         return self._config_obj[name].get()

    def _get_training_settings(self, name):
        return self._config_obj["training_settings"][name].get()

    @property
    def data_dir_name(self):
         return self._get_entry('data_dir_name')

    @property
    def data_file_name(self):
        return self._get_entry('data_file_name')

    @property
    def do_early_stopping(self):
        return self._get_entry('do_early_stopping')

    @property
    def do_save_model(self):
        return self._get_entry('do_save_model')

    @property
    def early_stopping_patience(self):
        return self._get_entry('early_stopping_patience')

    @property
    def max_layers(self):
        return self._get_entry('max_layers')

    @property
    def is_funnel_shaped(self):
        return self._get_entry('is_funnel_shaped')

    @property
    def initial_layers(self):
        return self._get_entry('initial_layers')

    @property
    def batch_size(self):
        return self._get_training_settings('batch_size')

    @property
    def epochs(self):
        return self._get_training_settings('epochs')

    @property
    def verbose(self):
        return self._get_training_settings('verbose')


