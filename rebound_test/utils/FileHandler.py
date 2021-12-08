from agent.AgentType import AgentType
from exceptions.InvalidAgentType import InvalidAgentType
from settings.settings_access import settings
from pathlib import Path
from utils.BaseFileHandler import BaseFileHandler
import datetime
import os

class FileHandler(BaseFileHandler):
    def __init__(self, agent_type):
        super().__init__()
        self.agent_type = agent_type

        self.path_to_data_dir = self.join(self.project_dir, settings.data_dir_name)
        self._ensure_data_dir_exists()

        self.agent_path = self.join(self.path_to_data_dir, settings.agent_type)
        self.agent_path_bin = self.join(self.agent_path, settings.data_dir_bin)
        if (agent_type == AgentType.ANALYTICAL.value or agent_type == AgentType.GCPD.value):
            self.agent_path_json = self.join(self.agent_path, settings.data_dir_json)
        self._ensure_agent_dirs_exists()

        self.default_file_name = self.get_timestamp_str()

    def get_default_file_path(self, extension, file_name=None):
        path_to_dir = self._get_path_to_dir(extension) # relative path
        file_name = self.default_file_name if file_name is None else file_name
        return str( self.join(path_to_dir, file_name + extension) )

    def write_to_file(self, file_extension, json, file_name=None):
        file_name = self.default_file_name if file_name is None else file_name
        self._write_to_file(file_name, file_extension, json)

    def get_timestamp_str(self):
        return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    @staticmethod
    def ensure_dir_exists(path):
        Path(str(path)).mkdir(parents=True, exist_ok=True)

    def _ensure_analytical_dir_exists(self):
        Path(str(self.path_to_analytical)).mkdir(parents=True, exist_ok=True)
        Path(str(self.path_to_analytical_bin)).mkdir(parents=True, exist_ok=True)
        Path(str(self.path_to_analytical_json)).mkdir(parents=True, exist_ok=True)

    def _ensure_data_dir_exists(self):
        self.ensure_dir_exists(self.path_to_data_dir)

    def _ensure_agent_dirs_exists(self):
        self.ensure_dir_exists(self.agent_path)
        self.ensure_dir_exists(self.agent_path_bin)
        if (self.agent_type == AgentType.ANALYTICAL.value 
         or self.agent_type == AgentType.GCPD.value):
            self.ensure_dir_exists(self.agent_path_json)

    def _get_path_to_dir(self, extension):
        path_to_dir = self._get_path_based_on(self.agent_path, extension)
        return path_to_dir

    def _get_path_based_on(self, path_to_agent_dir, extension):
        if (extension == settings.bin_file_ext):
            path_to_dir = Path.joinpath(path_to_agent_dir, settings.data_dir_bin)
        elif(extension == settings.json_file_ext):
            path_to_dir = Path.joinpath(path_to_agent_dir, settings.data_dir_json)
        return path_to_dir if path_to_dir is not None else path_to_agent_dir

    def _write_to_file(self, file_name, file_extension, data):
        relative_path = self._get_path_to_dir(file_extension)
        abs_path_to_file = self.get_abs_path(relative_path, file_name, file_extension)
        self.write(abs_path_to_file, data)


    def write_to_file(self, file_extension, json):
        self._write_to_file(self.file_name, file_extension, json)

    
    @staticmethod
    def get_data_dir(dir_name): 
        return Path.joinpath(Path().resolve(), dir_name)

    @staticmethod
    def get_data_files(data_dir):
        return os.listdir(data_dir)
