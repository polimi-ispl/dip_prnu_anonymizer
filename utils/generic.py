import numpy as np
import os
from socket import gethostname
from typing import Union
from pathlib import Path
import json
from argparse import Namespace
import random
import string
from PIL import Image
from GPUtil import getFirstAvailable, getGPUs


def ten_digit(number):
    return int(np.floor(np.log10(number)) + 1)


def int2str(in_content, digit_number):
    in_content = int(in_content)
    return str(in_content).zfill(ten_digit(digit_number))


def random_code(n=6):
    return ''.join([random.choice(string.ascii_letters + string.digits)
                    for _ in range(int(n))])


def machine_name():
    return gethostname()


def read_image(filename):
    return np.asarray(Image.open(filename))


def sec2time(seconds: float) -> str:
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    timestamp = '%dh:%dm:%ds' % (h, m, s)
    return timestamp


def time2sec(timestamp: str) -> int:
    h, m, s = timestamp.split(":")
    h = int(h.replace("h", "")) * 3600
    m = int(m.replace("m", "")) * 60
    s = int(s.replace("s", ""))
    return h + m + s


def read_args(filename: Union[str, Path]) -> Namespace:
    args = Namespace()
    with open(filename, 'r') as fp:
        args.__dict__.update(json.load(fp))
    return args


def write_args(filename: Union[str, Path], args: Namespace, indent: int = 2) -> None:
    with open(filename, 'w') as fp:
        json.dump(args.__dict__, fp, indent=indent)


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)
    """
    if id is None:
        # CPU only
        print('GPU not selected')
    else:
        try:
            device = id if id != -1 else getFirstAvailable(order='memory')[0]  # -1 for automatic choice
        except RuntimeError:
            print('WARNING! No GPU available, switching to CPU')
            return
        try:
            name = getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = getFirstAvailable(order='memory')[0]
            name = getGPUs()[device].name

        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def get_gpu_name(id: int) -> str:
    name = getGPUs()[id].name
    return '%s (%d)' % (name, id)
