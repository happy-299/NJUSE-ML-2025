import torch

def check_gpu_info():
    gpu_info = {
        'is_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    return gpu_info
