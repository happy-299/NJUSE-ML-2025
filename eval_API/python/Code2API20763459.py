import pandas as pd

def create_dataframe_from_array(data):
    return pd.DataFrame(data=data[1:,1:],    # values
                       index=data[1:,0],      # 1st column as index
                       columns=data[0,1:])    # 1st row as the column names
