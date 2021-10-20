from settings.SettingsAccess import settings
import datetime
import pathlib
from pathlib import Path

cwd_path = pathlib.Path().resolve()
path_to_data_dir = Path.joinpath(cwd_path, settings.data_dir_name )

def get_timestamp_str():
    return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

def get_abs_path_of_file(file_name, extension):
    return str( Path.joinpath(path_to_data_dir, file_name + extension ) )

def write_to_file(file_name, file_extension, json):
    f_handle = open( get_abs_path_of_file(file_name, file_extension), 'w')
    f_handle.write(json)
    f_handle.close()
