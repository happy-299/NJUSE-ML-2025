def contains_any_extension(url_string, extensions):
    return any(ext in url_string for ext in extensions)
