def string_to_binary(text):
    return ' '.join(format(ord(x), 'b') for x in text)
