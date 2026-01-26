from datetime import datetime, timedelta

def subtract_days_from_today(days_to_subtract):
    return datetime.today() - timedelta(days=days_to_subtract)
