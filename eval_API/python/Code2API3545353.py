def print_dict_key_value(dictionary):
    result = []
    for key in dictionary:
        print("key: %s , value: %s" % (key, dictionary[key]))
        result.append((key, dictionary[key]))
    return result
