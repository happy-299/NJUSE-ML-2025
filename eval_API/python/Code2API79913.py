import time

def convert_local_to_utc(local_time_str, format_str="%Y-%m-%d %H:%M:%S"):
    return time.strftime(format_str, 
                        time.gmtime(time.mktime(time.strptime(local_time_str, 
                                                             format_str))))
