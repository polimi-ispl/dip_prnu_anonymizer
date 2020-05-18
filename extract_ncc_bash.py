# extract NCC from anonymized images

import os
import numpy as np
from scipy.io import loadmat
import h5py
from utils import prnu
import time
import argparse
from multiprocessing import Pool

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, required=True)
    config, _ = parser.parse_known_args()

    output_folder = os.path.join('/nas/home/fpicetti/dip_prnu_anonymizer/results', config.input_path.split('/')[-2])
    # output folder (if smandelli)
    # output_folder = os.path.join('/nas/home/smandelli/Pycharm_projects/dip_prnu_anonymizer/results/', config.input_path.split('/')[-2])
    os.makedirs(output_folder, exist_ok=True)

    # load the device PRNU
    dev = '_'.join(config.input_path.split('/')[-1].split('_')[:3])
    prnu_root = '/nas/home/smandelli/Pycharm_projects/dip_prnu_anonymizer/PRNU_ZM_W'
    prnu_4ncc = loadmat(os.path.join(prnu_root, 'prnuZM_W_{}.mat'.format(dev)))['prnu']

    # open the pool
    with Pool() as pool:

        # load images
        with h5py.File(config.input_path, "r") as f:

            a_group_key = list(f.keys())[0]
            # Get the data
            data = list(f[a_group_key])

        # print('loaded file {}'.format(config.input_path))

        data_uint8 = [np.clip((255 * data[i]), 0, 255).astype(np.uint8) for i in range(len(data))]

        # time0 = time.time()
        noises = prnu.noise_extract_multiple(data_uint8, pool)
        # time1 = time.time() - time0
        # print('time: {} seconds'.format(time1))

        ncc_list = []
        # compute the ncc of the images
        for i in range(len(noises)):

            ncc_list.append(np.sum(noises[i] * prnu_4ncc)/(np.sqrt(np.sum(noises[i] ** 2)) * np.sqrt(np.sum(prnu_4ncc ** 2))))

        # save NCC results
        np.save(os.path.join(output_folder, '{}_ncc.npy'.format(config.input_path.split('/')[-1].split('.')[0])), ncc_list)

    return 0


if __name__ == '__main__':
    main()
