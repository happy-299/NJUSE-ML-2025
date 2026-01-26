def drop_nan_rows_by_column(df, column):
    return df[df[column].notna()]
