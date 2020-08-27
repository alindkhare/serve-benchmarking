import platform
import psutil
import json


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def dump_system_information(filepath):
    system_information = dict(platform.uname()._asdict())

    pustil_information = {
        "physical_cores": psutil.cpu_count(logical=False),
        "virtual_cores": psutil.cpu_count(logical=True),
        "RAM": get_size(psutil.virtual_memory().total),
    }
    platform_information = {
        **system_information,
        "stats": pustil_information,
    }
    with open(filepath, "w") as fp:
        json.dump(platform_information, fp)
