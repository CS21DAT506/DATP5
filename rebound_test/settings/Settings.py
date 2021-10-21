import confuse
import os 
import pathlib
from pathlib import Path

dir_path = Path(__file__).resolve().parent
app_name = dir_path.parent.name

class Settings():

    def parse_config(self):
        self.config_obj = confuse.Configuration(app_name, __name__)
        abs_path_to_file = str ( Path.joinpath(dir_path, "config.yaml" ) )
        self.config_obj.set_file( abs_path_to_file )

    def _get_sim_entry(self, name):
         return self.config_obj['sim'][name].get()

    def _get_data_entry(self, name):
         return self.config_obj['data'][name].get()

    def _get_logging_entry(self, name):
         return self.config_obj['logging'][name].get()

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

    def get_do_infinite_run(self):
        return self._get_run_entry('do_infinite_run')

    def get_batch_size(self):
        return self._get_run_entry('batch_size')

    def get_info_str_separator(self):
        return self._get_info_str_entry("separator")

    def get_data_dir_name(self):
        return self._get_data_entry("dir_name")

    def get_dir_analytical_agent(self):
        return self._get_data_entry("dir_analytical_agent")

    def get_dir_gcpd_agent(self):
        return self._get_data_entry("dir_gcpd_agent")

    def get_dir_bin(self):
        return self._get_data_entry("dir_bin")

    def get_dir_json(self):
        return self._get_data_entry("dir_json")

    def get_max_pos_radius(self):
        return self._get_config_generation_entry("max_pos_radius")

    def get_max_vel_radius(self):
        return self._get_config_generation_entry("max_vel_radius")

    def get_min_dist_to_target(self):
        return self._get_config_generation_entry("min_dist_to_target")

    def get_target_pos(self):
        return self._get_agent_entry("target_pos")

    def get_max_acceleration(self):
        return self._get_agent_entry("max_acceleration")

    def get_sim_time(self):
        return self._get_sim_entry("sim_time")

    def get_num_of_data_points(self):
        return self._get_sim_entry("num_of_data_points")

    def get_num_of_iterations(self):
        return self._get_sim_entry("num_of_iterations")

    def get_num_of_planets(self):
        return self._get_config_generation_entry("num_of_planets")

    def get_agent_index(self):
        return self._get_agent_entry("index")

    def get_agent_mass(self):
        return self._get_agent_entry("mass")

    def get_agent_radius(self):
        return self._get_agent_entry("radius")

    def get_use_fixed_setup_index(self):
        return self._get_bodies_entry("use_fixed_setup")

    def get_mass_for_planets(self):
        return float( self._get_planets_entry("mass") )

    def get_radius_for_planets(self):
        return self._get_planets_entry("radius")  

    def get_bin_file_ext(self):
        return self._get_file_exts_entry("bin")

    def get_json_file_ext(self):
        return self._get_file_exts_entry("json")

    def get_preferred_value(self):
        return self._get_scale_policy_entry('preferred_value')

    def get_invalid_value(self):
        return self._get_scale_policy_entry('invalid_value')

    def get_agent_type(self):
        return self._get_agent_entry('agent_type')