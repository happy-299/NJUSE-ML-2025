def count_shared_items(dict1, dict2):
    shared_items = {k: dict1[k] for k in dict1 if k in dict2 and dict1[k] == dict2[k]}
    return len(shared_items)
