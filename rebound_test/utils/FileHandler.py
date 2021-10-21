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
        self.path_to_analytical = Path.joinpath(self.path_to_data_dir, settings.analytical_agent_dir)
        self.path_to_gcpd_dir = Path.joinpath(self.path_to_data_dir, settings.gcpd_agent_dir)

        self._ensure_data_dir_exists()

    def _ensure_data_dir_exists(self):
        Path(str(self.path_to_data_dir)).mkdir(parents=True, exist_ok=True)

    def _ensure_analytical_dir_exists(self):
        Path(str(self.path_to_analytical)).mkdir(parents=True, exist_ok=True)

    def _ensure_gcpd_dir_exists(self):
        Path(str(self.path_to_gcpd_dir)).mkdir(parents=True, exist_ok=True)

    def get_abs_path_of_file(self, file_name, extension):
        if (self.agent_type == AgentType.ANALYTICAL.value):
            self._ensure_analytical_dir_exists()
            path_to_dir = Path.joinpath(self.path_to_data_dir, settings.analytical_agent_dir)
        elif(self.agent_type == AgentType.GCDP.value):
            self._ensure_gcpd_dir_exists()
            path_to_dir = Path.joinpath(self.path_to_data_dir, settings.gcpd_agent_dir)
        else:
            raise InvalidAgentType("The given agent type is not recognized. Please provide a valid agent type.")
        return str( Path.joinpath(path_to_dir, file_name + extension ) )

    def get_timestamp_str(self):
        return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def write_to_file(self, file_name, file_extension, json):
        abs_path_to_file = self.get_abs_path_of_file(file_name, file_extension)
        f_handle = open(abs_path_to_file , 'w')
        f_handle.write(json)
        f_handle.close()

