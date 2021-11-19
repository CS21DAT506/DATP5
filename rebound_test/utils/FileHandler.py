from agent.AgentType import AgentType
from exceptions.InvalidAgentType import InvalidAgentType
from settings.settings_access import settings
import pathlib
from pathlib import Path
from utils.BaseFileHandler import BaseFileHandler
import datetime

class FileHandler(BaseFileHandler):
    def __init__(self, agent_type):
        super().__init__()
        self.agent_type = agent_type

        self.path_to_data_dir = Path.joinpath(self.project_dir, settings.data_dir_name)

        self.path_to_analytical = Path.joinpath(self.path_to_data_dir, settings.data_dir_analytical_agent)
        self.path_to_analytical_bin = Path.joinpath(self.path_to_analytical, settings.data_dir_bin)
        self.path_to_analytical_json = Path.joinpath(self.path_to_analytical, settings.data_dir_json)

        self.path_to_gcpd = Path.joinpath(self.path_to_data_dir, settings.data_dir_gcpd_agent)
        self.path_to_gcpd_bin = Path.joinpath(self.path_to_gcpd, settings.data_dir_bin)
        self.path_to_gcpd_json = Path.joinpath(self.path_to_gcpd, settings.data_dir_json)

        self.path_to_nn = Path.joinpath(self.path_to_data_dir, settings.data_dir_nn_agent)
        self.path_to_nn_bin = Path.joinpath(self.path_to_nn, settings.data_dir_bin)

        self._ensure_data_dir_exists()
        self.default_file_name = self.get_timestamp_str()

    def _ensure_data_dir_exists(self):
        self.ensure_dir_exists(self.path_to_data_dir)

    def _ensure_analytical_dir_exists(self):
        self.ensure_dir_exists(self.path_to_analytical)
        self.ensure_dir_exists(self.path_to_analytical_bin)
        self.ensure_dir_exists(self.path_to_analytical_json)

    def _ensure_gcpd_dir_exists(self):
        self.ensure_dir_exists(self.path_to_gcpd)
        self.ensure_dir_exists(self.path_to_gcpd_bin)
        self.ensure_dir_exists(self.path_to_gcpd_json)

    def _ensure_nn_dir_exists(self):
        self.ensure_dir_exists(self.path_to_nn)
        self.ensure_dir_exists(self.path_to_nn_bin)

    def _get_path_to_agent_dir(self):
        if (self.agent_type == AgentType.ANALYTICAL.value):
            self._ensure_analytical_dir_exists()
            path_to_dir = Path.joinpath(self.path_to_data_dir, settings.data_dir_analytical_agent)
        elif(self.agent_type == AgentType.GCPD.value):
            self._ensure_gcpd_dir_exists()
            path_to_dir = Path.joinpath(self.path_to_data_dir, settings.data_dir_gcpd_agent)
        elif(self.agent_type == AgentType.NN.value
          or self.agent_type == AgentType.NN_NOP.value
          or self.agent_type == AgentType.NN_GRAV.value):
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

    def _write_to_file(self, file_name, file_extension, data):
        relative_path = self._get_path_to_dir(file_extension)
        abs_path_to_file = self.get_abs_path(relative_path, file_name, file_extension)
        self.write(abs_path_to_file, data)

    def get_timestamp_str(self):
        return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def get_default_file_path(self, extension, file_name=None):
        path_to_dir = self._get_path_to_dir(extension) # relative path
        file_name = self.default_file_name if file_name is None else file_name
        return str( self.join(path_to_dir, file_name + extension) )

    def write_to_file(self, file_extension, json, file_name=None):
        file_name = self.default_file_name if file_name is None else file_name
        self._write_to_file(file_name, file_extension, json)

    def write_to_file(self, file_extension, json):
        self._write_to_file(self.default_file_name, file_extension, json)
