import os
import shutil

def move_file(source_path, destination_path):
    """
    Move a file from source path to destination path.
    If the destination is on a different disk, it will copy and then delete the source file.
    
    Args:
        source_path (str): The current path of the file
        destination_path (str): The new path where the file should be moved to
    """
    return shutil.move(source_path, destination_path)
