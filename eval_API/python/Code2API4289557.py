def extract_integers(text):
    return [int(s) for s in text.split() if s.isdigit()]
