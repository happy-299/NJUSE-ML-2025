import datetime

def get_time_difference(first_time, later_time):
    difference = later_time - first_time
    seconds_in_day = 24 * 60 * 60
    return divmod(difference.days * seconds_in_day + difference.seconds, 60)
