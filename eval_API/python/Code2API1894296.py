import ast

def convert_string_to_list(string_list):
    # Convert string to list using ast.literal_eval
    list_data = ast.literal_eval(string_list)
    # Strip whitespace from each element
    return [n.strip() for n in list_data]
