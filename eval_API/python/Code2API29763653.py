import pandas as pd

def select_all_except_column(df, column_name):
    return df.loc[:, df.columns != column_name]
