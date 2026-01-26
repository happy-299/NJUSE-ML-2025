import os
import os.path

def count_files_in_directory(directory_path):
    return len([name for name in os.listdir(directory_path) 
               if os.path.isfile(os.path.join(directory_path, name))])
