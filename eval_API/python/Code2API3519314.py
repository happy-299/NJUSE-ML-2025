from numpy import genfromtxt

def read_csv_to_array(file_path, delimiter=','):
    return genfromtxt(file_path, delimiter=delimiter)
