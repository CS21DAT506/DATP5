import datetime
import pathlib
from pathlib import Path

from settings.settings import DATA_DIRECTORY

cwd_path = pathlib.Path().resolve()
path_to_data_dir = Path.joinpath(cwd_path, DATA_DIRECTORY )

def get_timestamp_str():
    return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.bin'

def get_data_path_with_file():
    return str( Path.joinpath(path_to_data_dir, get_timestamp_str()) )
