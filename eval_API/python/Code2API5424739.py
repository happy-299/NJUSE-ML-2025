def is_valid_integer(user_input):
    try:
        val = int(user_input)
        return True
    except ValueError:
        return False
