def delete_line_from_file(file_path, line_to_delete):
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open(file_path, "w") as f:
        for line in lines:
            if line.strip("\n") != line_to_delete:
                f.write(line)
