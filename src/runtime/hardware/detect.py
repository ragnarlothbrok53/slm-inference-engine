import platform
import torch
import psutil


def detect_hardware():
    info = {
        "os": platform.system(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "accelerator": "cpu",
    }

    if torch.backends.mps.is_available():
        info["accelerator"] = "mps"

    if torch.cuda.is_available():
        info["accelerator"] = "cuda"

    return info
