import numpy as np

def create_and_fill_array(rows, cols, values):
    # Create zero array with specified shape
    a = np.zeros(shape=(rows, cols))
    
    # Fill array with values
    for i in range(rows):
        a[i] = values[i]
    
    return a
