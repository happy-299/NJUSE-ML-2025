def get_index_name(df):
    return df.index.name

def set_index_name(df, new_name):
    df.index.name = new_name
