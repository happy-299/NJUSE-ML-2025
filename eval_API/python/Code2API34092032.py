import pandas as pd

def read_csv_with_headers(file_path, 
                         column_names=["Sequence", "Start", "End", "Coverage"],
                         separator='\t'):
    return pd.read_csv(file_path, 
                      sep=separator, 
                      names=column_names)
