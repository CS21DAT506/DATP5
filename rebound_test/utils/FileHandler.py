from agent.AgentType import AgentType
from exceptions.InvalidAgentType import InvalidAgentType
from settings.SettingsAccess import settings
import datetime
import pathlib
from pathlib import Path


class FileHandler():
    def __init__(self, agent_type):
        self.agent_type = agent_type

        self.cwd_path = pathlib.Path().resolve()
        self.path_to_data_dir = Path.joinpath(self.cwd_path, settings.data_dir_name)

        self.path_to_analytical = Path.joinpath(self.path_to_data_dir, settings.data_dir_analytical_agent)
        self.path_to_analytical_bin = Path.joinpath(self.path_to_analytical, settings.data_dir_bin)
        self.path_to_analytical_json = Path.joinpath(self.path_to_analytical, settings.data_dir_json)

        self.path_to_gcpd = Path.joinpath(self.path_to_data_dir, settings.data_dir_gcpd_agent)
        self.path_to_gcpd_bin = Path.joinpath(self.path_to_gcpd, settings.data_dir_bin)
        self.path_to_gcpd_json = Path.joinpath(self.path_to_gcpd, settings.data_dir_json)

        self.path_to_nn = Path.joinpath(self.path_to_data_dir, settings.data_dir_nn_agent)
        self.path_to_nn_bin = Path.joinpath(self.path_to_nn, settings.data_dir_bin)

        self._ensure_data_dir_exists()
        self.file_name = self.get_timestamp_str()

    def _ensure_data_dir_exists(self):
        Path(str(self.path_to_data_dir)).mkdir(parents=True, exist_ok=True)

    def _ensure_analytical_dir_exists(self):
        Path(str(self.path_to_analytical)).mkdir(parents=True, exist_ok=True)
        Path(str(self.path_to_analytical_bin)).mkdir(parents=True, exist_ok=True)
        Path(str(self.path_to_analytical_json)).mkdir(parents=True, exist_ok=True)

    def _ensure_gcpd_dir_exists(self):
        Path(str(self.path_to_gcpd)).mkdir(parents=True, exist_ok=True)
        Path(str(self.path_to_gcpd_bin)).mkdir(parents=True, exist_ok=True)
        Path(str(self.path_to_gcpd_json)).mkdir(parents=True, exist_ok=True)

    def _ensure_nn_dir_exists(self):
        Path(str(self.path_to_nn)).mkdir(parents=True, exist_ok=True)
        Path(str(self.path_to_nn_bin)).mkdir(parents=True, exist_ok=True)

    def _get_path_to_agent_dir(self):
        if (self.agent_type == AgentType.ANALYTICAL.value):
            self._ensure_analytical_dir_exists()
            path_to_dir = Path.joinpath(self.path_to_data_dir, settings.data_dir_analytical_agent)
        elif(self.agent_type == AgentType.GCPD.value):
            self._ensure_gcpd_dir_exists()
            path_to_dir = Path.joinpath(self.path_to_data_dir, settings.data_dir_gcpd_agent)
        elif(self.agent_type == AgentType.NN.value):
            self._ensure_nn_dir_exists()
            path_to_dir = Path.joinpath(self.path_to_data_dir, settings.data_dir_nn_agent)
        else:
            raise InvalidAgentType("The given agent type is not recognized. Please provide a valid agent type.")
        return path_to_dir

    def _get_path_based_on(self, path_to_agent_dir, extension):
        if (extension == settings.bin_file_ext):
            path_to_dir = Path.joinpath(path_to_agent_dir, settings.data_dir_bin)
        elif(extension == settings.json_file_ext):
            path_to_dir = Path.joinpath(path_to_agent_dir, settings.data_dir_json)
        return path_to_dir if path_to_dir is not None else path_to_agent_dir

    def _get_path_to_dir(self, extension):
        path_to_dir = self._get_path_to_agent_dir()
        path_to_dir = self._get_path_based_on(path_to_dir, extension)
        return path_to_dir 

    def get_abs_path_of_file(self, extension, file_name=None):
        path_to_dir = self._get_path_to_dir(extension)
        if file_name is None:
            file_name = self.file_name
        return str( Path.joinpath(path_to_dir, file_name + extension ) )

    def get_timestamp_str(self):
        return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def _write_to_file(self, file_name, file_extension, data):
        abs_path_to_file = self.get_abs_path_of_file(file_extension, file_name)
        f_handle = open(abs_path_to_file , 'w')
        f_handle.write(data)
        f_handle.close()

    def write_to_file(self, file_name, file_extension, json):
        self._write_to_file(file_name, file_extension, json)

    def write_to_file(self, file_extension, json):
        self._write_to_file(self.file_name, file_extension, json)