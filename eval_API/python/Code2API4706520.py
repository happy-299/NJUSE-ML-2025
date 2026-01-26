def append_to_file(file_path, text):
    with open(file_path, "a") as myfile:
        myfile.write(text)
