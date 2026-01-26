import numpy

def remove_nan_values(arr):
    return arr[~numpy.isnan(arr)]
