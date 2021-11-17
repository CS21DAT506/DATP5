import confuse
from pathlib import Path

dir_path = Path(__file__).resolve().parent
app_name = dir_path.parent.name

class Settings():

    def parse_config(self):
        self._config_obj = confuse.Configuration(app_name, __name__)
        abs_path_to_file = str ( Path.joinpath(dir_path, "config.yaml" ) )
        self._config_obj.set_file( abs_path_to_file )

    def _get_sim_entry(self, name):
         return self._config_obj['sim'][name].get()

    def _get_data_entry(self, name):
         return self._config_obj['data'][name].get()

    def _get_logging_entry(self, name):
         return self._config_obj['logging'][name].get()

    def _get_run_entry(self, name):
         return self._get_sim_entry('run')[name]

    def _get_file_exts_entry(self, name):
         return self._get_data_entry('file_extensions')[name]

    def _get_config_generation_entry(self, name):
         return self._get_sim_entry('config_generation')[name]

    def _get_bodies_entry(self, name):
         return self._get_sim_entry('bodies')[name]

    def _get_agent_entry(self, name):
        return self._get_bodies_entry('agent')[name]

    def _get_planets_entry(self, name):
        return self._get_bodies_entry('planets')[name]

    def _get_scale_policy_entry(self, name):
        return self._get_agent_entry('scale_policy')[name]

    def _get_info_str_entry(self, name):
         return self._get_logging_entry('info_str')[name]

    @property
    def execution_mode(self):
        return self._get_run_entry('execution_mode')

    @property
    def batch_size(self):
        return self._get_run_entry('batch_size')

    @property
    def info_str_separator(self):
        return self._get_info_str_entry("separator")

    @property
    def data_dir_name(self):
        return self._get_data_entry("dir_name")

    @property
    def data_dir_analytical_agent(self):
        return self._get_data_entry("dir_analytical_agent")

    @property
    def data_dir_gcpd_agent(self):
        return self._get_data_entry("dir_gcpd_agent")

    @property
    def data_dir_nn_agent(self):
        return self._get_data_entry("dir_nn_agent")

    @property
    def data_dir_bin(self):
        return self._get_data_entry("dir_bin")

    @property
    def data_dir_json(self):
        return self._get_data_entry("dir_json")

    @property
    def max_pos_radius(self):
        return self._get_config_generation_entry("max_pos_radius")

    @property
    def max_vel_radius(self):
        return self._get_config_generation_entry("max_vel_radius")

    @property
    def min_dist_to_target(self):
        return self._get_config_generation_entry("min_dist_to_target")

    @property
    def target_pos(self):
        return self._get_agent_entry("target_pos")

    @property
    def max_acceleration(self):
        return self._get_agent_entry("max_acceleration")

    @property
    def sim_time(self):
        return self._get_sim_entry("sim_time")

    @property
    def num_of_data_points(self):
        return self._get_sim_entry("num_of_data_points")

    @property
    def num_of_iterations(self):
        return self._get_sim_entry("num_of_iterations")

    @property
    def num_of_planets(self):
        return self._get_config_generation_entry("num_of_planets")

    @property
    def agent_index(self):
        return self._get_agent_entry("index")

    @property
    def agent_mass(self):
        return self._get_agent_entry("mass")

    @property
    def agent_radius(self):
        return self._get_agent_entry("radius")

    @property
    def use_fixed_setup_index(self):
        return self._get_bodies_entry("use_fixed_setup")

    @property
    def planets_mass(self):
        return float( self._get_planets_entry("mass") )

    @property
    def planets_radius(self):
        return self._get_planets_entry("radius")  

    @property
    def bin_file_ext(self):
        return self._get_file_exts_entry("bin")

    @property
    def json_file_ext(self):
        return self._get_file_exts_entry("json")

    @property
    def preferred_value(self):
        return self._get_scale_policy_entry('preferred_value')

    @property
    def invalid_value(self):
        return self._get_scale_policy_entry('invalid_value')

    @property
    def agent_type(self):
        return self._get_agent_entry('agent_type')

    @property
    def nn_model_path(self):
        return self._get_agent_entry('nn_model_path')
    
    @property
    def write_data_to_files(self):
        return self._get_data_entry('write_data_to_files')
