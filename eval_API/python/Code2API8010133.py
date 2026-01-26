def read_file_lines(file_path, callback):
    with open(file_path) as f:
        for line in f:
            callback(line)
