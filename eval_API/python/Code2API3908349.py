from dateutil import parser

def parse_iso8601_datetime(date_string):
    return parser.parse(date_string)
