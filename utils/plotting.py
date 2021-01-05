import os
import numpy as np


def log10plot(in_content):
    return np.log10(np.asarray(in_content) / in_content[0])


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
