import json

def build_json_object(key, value):
    data = {}
    data[key] = value
    json_data = json.dumps(data)
    return json_data
