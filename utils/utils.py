"""
@Author: Francesco Picetti - francesco.picetti@polimi.it
"""
import numpy as np
import os
from socket import gethostname

import torch
from skimage.measure import compare_ssim
import random
import string
from PIL import Image
from GPUtil import getFirstAvailable, getGPUs
from termcolor import colored

from . import prnu
from . import pytorch_ssim


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


def clim(in_content, ratio, zero_mean=True):
    """
    Compute the lower-bound and upper-bound `clim` tuple as a percentage
    of the content dynamic range.
    
    :param in_content:  np.ndarray
    :param ratio:       float, percentage for the dynamic range (default 1.)
    :param zero_mean:   bool, use symmetric bounds (default True)
    :return: clim tuple (as required by matplotlib.pyplot.imshow)
    """
    if zero_mean:
        max_abs_value = np.max(np.abs(in_content))
        return -ratio * max_abs_value, ratio * max_abs_value
    else:
        return ratio * in_content.min(), ratio * in_content.max()


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


def mse(target, output):
    return np.mean((target - output) ** 2)


def snr(target, output):
    """
    Compute SNR between the target and the reconstructed images
    
    :param target:  numpy array of reference
    :param output:  numpy array we have produced
    :return: SNR in dB
    """
    if target.shape != output.shape:
        raise ValueError('There is something wrong with the dimensions!')
    return 20 * np.log10(np.linalg.norm(target) / np.linalg.norm(target - output))


def clip_normalize_power(x, mymin, mymax, p):
    """
    Preprocessing function to be applied to migrated images in the C2F scenario
    
    :param x:       data to be processed
    :param mymin:   min value for clipping
    :param mymax:   max value for clipping
    :param p:       exponent for the power function
    :return:
    """
    x = np.clip(x, a_min=mymin, a_max=mymax)
    x, _, _ = normalize(x)
    x = np.sign(x) * np.power(np.abs(x), p)
    return x


def clip_normalize_power_inverse(x, mymin, mymax, p):
    """
    Inverse preprocessing function to be applied to output images in the C2F scenario
    :param x: data to be processed
    :param mymin: min value used for clipping
    :param mymax: max value used for clipping
    :param p: exponent for the power function (to be inverted)
    :return:
    """
    x = np.sign(x) * np.power(np.abs(x), 1 / p)
    x = denormalize(x, mymin, mymax)
    return x


def normalize(x, in_min=None, in_max=None, zero_mean=True):
    if in_min is None and in_max is None:
        in_min = np.min(x)
        in_max = np.max(x)
    if np.isclose(in_min, in_max):
        raise ValueError("Error! Input minimum and maximum are too close!")
    x = (x - in_min) / (in_max - in_min)
    if zero_mean:
        x = x * 2 - 1
    return x, in_min, in_max


def denormalize(x, in_min, in_max):
    """
    Denormalize data.
    :param x: ndarray, normalized data
    :param in_min: float, the minimum value
    :param in_max: float, the maximum value
    :return: denormalized data in [in_min, in_max]
    """
    if x.min() == 0.:
        return x * (in_max - in_min) + in_min
    else:
        return (x + 1) * (in_max - in_min) / 2 + in_min


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


def compute_l1_weight(A, c, use_mean=False):
    if use_mean:
        return c * abs(A).mean()
    else:
        return c * abs(A).max()


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)
    """
    if id is None:
        # CPU only
        print(colored('GPU not selected', 'yellow'))
    else:
        try:
            device = id if id is not -1 else getFirstAvailable(order='memory')[0]  # -1 for automatic choice
        except RuntimeError:
            print(colored('WARNING! No GPU available, switching to CPU', 'yellow'))
            return
        try:
            name = getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most '
                  'available one.')
            device = getFirstAvailable(order='memory')[0]
            name = getGPUs()[device].name

        print(colored('GPU selected: %d - %s' % (device, name), 'yellow'))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def dotproduct_test(A):
    """
    Perform the dot product test for checking the adjointness of a linear
    operator. Consider the following linear system:
        y = A x
    The dot product test is defined as verifying that:
        y • A x = x • A' y
    
    :param A:   scipy.sparse.linalg.LinearOperator to be tested
    :return:    boolean, whether the test is passed or not
    """
    np.random.seed(118)
    x = np.random.rand(A.shape[1])
    y = np.random.rand(A.shape[0])
    return np.isclose(np.dot(A.matvec(x), y), np.dot(A.rmatvec(y), x))


def add_prnu(img, k, weight=0.01, to_each_channel=True):
    if to_each_channel:
        return img * (1 + weight * k.repeat(1, img.shape[1], 1, 1))
    else:
        raise NotImplementedError("For now the PRNU is added to each channel")


def psnr(img1: torch.Tensor or np.ndarray, img2: torch.Tensor or np.ndarray, color_channel: int = -1) -> torch.Tensor or \
                                                                                                         np.ndarray:
    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        return 10 * np.log10(255 ** 2 / np.mean((img1 - img2) ** 2))
    else:
        if color_channel != -1:
            img1 = img1.permute(0, 2, 3, 1)
            img2 = img2.permute(0, 2, 3, 1)
        return 10 * torch.log10(255 ** 2 / torch.mean((img1 - img2).view(-1) ** 2))


def ncc(k1: torch.Tensor or np.ndarray, k2: torch.Tensor or np.ndarray) -> float:
    if isinstance(k1, np.ndarray) and isinstance(k2, np.ndarray):
        return np.dot(k1.ravel(), k2.ravel()) / (np.linalg.norm(k1) * np.linalg.norm(k2))
    else:
        k1 = k1.view(-1, 1)
        k2 = k2.view(-1, 1)
        k1_norm = torch.norm(k1, 2)
        k2_norm = torch.norm(k2, 2)
        _ncc = torch.sum(k1 * k2)
        _ncc = _ncc / (k1_norm * k2_norm + np.finfo(float).eps)
        return _ncc


def ssim(img1: torch.Tensor or np.ndarray, img2: torch.Tensor or np.ndarray) -> torch.Tensor or np.ndarray:
    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        return compare_ssim(img1, img2, multichannel=True if img1.shape[-1] == 3 else False)
    else:
        return pytorch_ssim.ssim(img1, img2)


def torch2numpy(in_content):
    assert isinstance(in_content, torch.Tensor), "ERROR! in_content has to be a torch.Tensor object"
    return in_content.cpu().detach().numpy()


def numpy2torch(in_content, dtype=torch.cuda.FloatTensor):
    assert isinstance(in_content, np.ndarray), "ERROR! in_content has to be a numpy.ndarray object"
    return torch.from_numpy(in_content).type(dtype)


def float2png(in_content):
    return np.clip((255 * in_content), 0, 255).astype(np.uint8)


def png2float(in_content):
    return in_content.astype(np.float32) / 255.


def rgb2gray(in_content: np.ndarray or torch.Tensor, color_channel: int = -1):
    if isinstance(in_content, np.ndarray):
        return prnu.rgb2gray(in_content)  # TODO handle the color channel

    rgb2gray_vector = torch.Tensor([0.29893602, 0.58704307, 0.11402090]).type(in_content.dtype).to(in_content.device)

    ndim = len(in_content.shape)

    if color_channel != -1:
        in_content = in_content.permute(0, 2, 3, 1)

    if ndim == 3:
        im_gray = in_content.clone()
    elif in_content.shape[-1] == 1:
        im_gray = in_content.clone()
    elif in_content.shape[-1] == 3:
        w, h = in_content.shape[1:3]
        im = in_content.reshape(w * h, 3)
        im_gray = (im @ rgb2gray_vector).reshape(w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray[None, None, :, :]


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]
