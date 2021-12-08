import pathlib
from pathlib import Path

class BaseFileHandler():
    def __init__(self):
        self.cwd_path = Path().resolve()
        self.project_dir = Path(__file__).parent.parent

    def ensure_dir_exists(self, path_to_dir):
        """
        Parameters
        ----------
        path_to_dir: Union[Path, str] 
            A path
        """
        Path( str(path_to_dir) ).mkdir(parents=True, exist_ok=True)

    def join(self, abs_path, file_path):
        """
        Gets absolute path to 'relative_path_to_dir/file_name.extension'. See Path.joinpath for more.

        Parameters
        ----------
        Path abs_path: Path
        str file_path: Name of the file which resides in relative_path_to_dir
        
        Returns
        -------
        Path
            joined path
        """
        return Path.joinpath(abs_path, file_path)

    def get_abs_path(self, relative_path_to_dir, file_name, extension=""):
        """
        Gets absolute path to 'relative_path_to_dir/file_name.extension'

        Parameters
        ----------
        relative_path_to_dir: Path 
            A path relative to project dir which should contain file_name
        file_name: str
            Name of the file which resides in relative_path_to_dir
        extension: str
            File extension
        
        Returns
        -------
        str
            absolute path to file
        """
        return str( self.join(relative_path_to_dir, file_name + extension ) )

    def write(self, abs_path_to_file, data):
        """
        Writes data to location abs_path_to_file

        Parameters
        ----------
        abs_path_to_file: str 
            Location where data will be written to
        data: 
            Data which will be written  
        """
        f_handle = open(abs_path_to_file , 'w')
        f_handle.write(data)
        f_handle.close()


if __name__ == "__main__":
    bfh = BaseFileHandler()
    print(f"cwd_path: { str(bfh.cwd_path) }")
    print(f"project_dir: { str(bfh.project_dir) }")

