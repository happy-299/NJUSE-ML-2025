from datetime import date

def calculate_days_between_dates(year1, month1, day1, year2, month2, day2):
    d0 = date(year1, month1, day1)
    d1 = date(year2, month2, day2)
    delta = d1 - d0
    return delta.days
