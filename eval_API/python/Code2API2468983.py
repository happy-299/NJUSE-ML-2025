import psutil

def get_system_usage():
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    ram_used_percent = memory.percent
    ram_available_percent = memory.available * 100 / memory.total
    
    return {
        'cpu_percent': cpu_usage,
        'ram_used_percent': ram_used_percent,
        'ram_available_percent': ram_available_percent
    }
