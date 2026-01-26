import datetime

def convert_seconds_to_hms(seconds):
    return str(datetime.timedelta(seconds=seconds))
