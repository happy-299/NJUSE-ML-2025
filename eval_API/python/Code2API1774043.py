import yaml

def parse_yaml_file(file_path):
    try:
        with open(file_path, "r") as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise exc
