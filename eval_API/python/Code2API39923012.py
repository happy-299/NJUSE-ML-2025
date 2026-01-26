import pandas as pd

def group_by_fruit_name_sum(df):
    return df.groupby(['Fruit', 'Name'])['Number'].sum()
