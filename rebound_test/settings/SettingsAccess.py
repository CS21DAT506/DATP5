from typing import Set
from settings.Settings import Settings

class SettingsAccess():
    def __init__(self):
        settings = Settings()
        settings.parse_config()

        self.do_infinite_run = settings.get_do_infinite_run()
        self.batch_size = settings.get_batch_size()

        self.sim_time = settings.get_sim_time()
        self.num_of_data_points = settings.get_num_of_data_points()
        self.num_of_iterations = settings.get_num_of_iterations()

        self.num_of_planets = settings.get_num_of_planets()
        self.max_pos_radius = settings.get_max_pos_radius()
        self.max_vel_radius = settings.get_max_vel_radius()
        self.min_dist_to_target = settings.get_min_dist_to_target()

        self.target_pos = settings.get_target_pos()
        self.agent_type = settings.get_agent_type()
        self.nn_model_path = settings.get_nn_model_path()
        self.agent_index = settings.get_agent_index()
        self.max_acceleration = settings.get_max_acceleration()
        self.agent_mass = settings.get_agent_mass()
        self.agent_radius = settings.get_agent_radius()
        self.preferred_value = settings.get_preferred_value()
        self.invalid_value = settings.get_invalid_value()

        self.planets_mass = settings.get_mass_for_planets()
        self.planets_radius = settings.get_radius_for_planets()

        self.data_dir_name = settings.get_data_dir_name()
        self.data_dir_analytical_agent = settings.get_dir_analytical_agent()
        self.data_dir_gcpd_agent = settings.get_dir_gcpd_agent()
        self.data_dir_bin = settings.get_dir_bin()
        self.data_dir_json = settings.get_dir_json()
        self.bin_file_ext = settings.get_bin_file_ext() 
        self.json_file_ext = settings.get_json_file_ext()

        self.info_str_separator = settings.get_info_str_separator()

settings = SettingsAccess()
