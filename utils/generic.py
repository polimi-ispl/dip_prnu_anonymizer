"""
@Author: Francesco Picetti - francesco.picetti@polimi.it
"""
import numpy as np
import os
from socket import gethostname

import random
import string
from PIL import Image
from GPUtil import getFirstAvailable, getGPUs
from .processing import normalize


def set_data_path(host='polimi'):
    if host == 'polimi':
        data_path = '/nas/home/fpicetti/geophysics/datasets'
    elif host == 'cineca':
        data_path = '/gpfs/scratch/usera06ptm/a06ptm04/fpicetti/datasets'
    else:
        raise ValueError('host name not recognized!')
    return data_path


def log10plot(in_content):
    return np.log10(np.asarray(in_content) / in_content[0])


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


def idle_cpu_count(mincpu=1):
    # the load is computed over the last 1 minute
    idle = int(os.cpu_count() - np.floor(os.getloadavg()[0]))
    return max(mincpu, idle)


def plot2pgf(temp, filename, folder='./'):
    """
    :param temp:        list of equally-long data
    :param filename:    filename without extension nor path
    :param folder:      folder where to save
    """
    if len(temp) == 1:  # if used as plt.plot(y) without the abscissa axis
        temp = [list(range(len(temp[0]))), temp[0]]

    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(os.path.join(folder, filename + '.txt'), np.asarray(temp).T,
               fmt="%f", encoding='ascii')


def save_image(in_content, filename, clim=(None, None), folder='./'):
    """
    Save a gray-scale PNG image of the 2D content
    
    :param in_content:  2D np.ndarray
    :param filename:    name of the output file (without extension)
    :param clim:        tuple for color clipping (as done in matplotlib.pyplot.imshow)
    :param folder:      output directory
    :return:
    """
    if clim[0] and clim[1] is not None:
        in_content = np.clip(in_content, clim[0], clim[1])
        in_content = normalize(in_content, in_min=clim[0], in_max=clim[1])[0]
    else:
        in_content = normalize(in_content)[0]
    out = Image.fromarray(((in_content + 1) / 2 * 255).astype(np.uint8))

    if not os.path.exists(folder):
        os.makedirs(folder)
    out.save(os.path.join(folder, filename + '.png'))


def sec2time(seconds):
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    timestamp = '%dh:%dm:%ds' % (h, m, s)
    return timestamp


def nextpow2(x):
    return int(2 ** np.ceil(np.log2(x)))


def spectrum(in_content, nfft=None):
    """
    Spectrum computed along the first dimension and then averaged along the other dimensions
    
    :param in_content:  numpy.ndarray
    :param nfft:        number of frequency bin for FFT computation (default nexpow2)
    :return: averaged spectrum
    """
    if nfft is None:
        nfft = nextpow2(in_content.shape[0])
    return np.mean(np.fft.fft(in_content, nfft, axis=0),
                   axis=tuple([i for i in range(1, len(in_content.shape))]))


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)
    """
    if id is None:
        # CPU only
        print('GPU not selected')
    else:
        try:
            device = id if id is not -1 else getFirstAvailable(order='memory')[0]  # -1 for automatic choice
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
