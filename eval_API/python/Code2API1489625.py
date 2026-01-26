import os

def get_pythonpath_directories():
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    return user_paths
