import datetime

def get_current_time_components():
    now = datetime.datetime.now()
    return (now.year, now.month, now.day, now.hour, now.minute, now.second)
