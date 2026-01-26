import json

def get_json_value(json_str, key):
    data = json.loads(json_str)
    return data[key]
