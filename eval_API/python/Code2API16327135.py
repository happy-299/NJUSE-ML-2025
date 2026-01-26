import numpy as np
import pandas as pd

def add_empty_columns(df, column_names, fill_value=""):
    """
    Add empty columns to a DataFrame
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_names (list): List of column names to add
    fill_value (str or np.nan): Value to fill the columns with, defaults to empty string
    
    Returns:
    pd.DataFrame: DataFrame with new empty columns
    """
    df_copy = df.copy()
    for col in column_names:
        df_copy[col] = fill_value
    return df_copy
