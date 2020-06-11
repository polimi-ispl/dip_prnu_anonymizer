# extract NCC from anonymized images, considering squared blocks of various size

import os
import numpy as np
from scipy.io import loadmat
import h5py
from utils import prnu
import time
import argparse
from multiprocessing import Pool, cpu_count
from python_patch_extractor.PatchExtractor import PatchExtractor
from tqdm import tqdm
import gc


def ncc_pool(args):
    """
    Compute the NCC between Image and PRNU:
    args[0] = PRNU
    args[1] = grayscale Image
    args[2] = Noise extracted from Image
    All arguments are np.ndarray with the same dimensions
    """
    ncc_arr_pool = np.zeros((args[0].shape[0]), )
    for n_idx, n_b in enumerate(range(args[0].shape[0])):
        ncc_arr_pool[n_idx] = np.sum(args[2][n_b] *
                                         (args[1][n_b] * args[0][n_b])) / (np.sqrt(np.sum(args[2][n_b] ** 2)) *
                                                                           np.sqrt(np.sum((args[1][n_b] * args[0][n_b]) ** 2)))

    return ncc_arr_pool


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, required=True, help='path to the hdf5 file')
    config, _ = parser.parse_known_args()

    # dimensions of the squared blocks
    block_sizes = [32, 64, 128, 256, 512]

    # load the device PRNU
    dev = '_'.join(config.input_path.split('/')[-1].split('_')[:3])
    # root on CINECA
    prnu_root = '/gpfs/scratch/usera06ptm/a06ptm04/fpicetti/dip_prnu_anonymizer/dataset600/{}'.format(dev)
    # root on NAS
    # prnu_root = '/nas/home/fpicetti/dip_prnu_anonymizer/dataset600/{}'.format(dev)
    prnu_4ncc = loadmat(os.path.join(prnu_root, 'prnuZM_W.mat'))['prnu']

    # open the pool
    with Pool() as pool:

        with h5py.File(config.input_path, "r+") as f:
            data_uint8 = list(f['all_outputs'])

            if len(list(f.keys())) > 1:
                # noises have already been extracted
                noises = list(f['noises'])
            else:
                # noise extraction
                noises = prnu.noise_extract_multiple(data_uint8, pool)
                f.create_dataset('noises', data=noises, dtype='float32')

        # path to already saved results
        npy_path = config.input_path.split('.')[0] + '.npy'
        # load results
        results_run = np.load(npy_path, allow_pickle=True).tolist()

        # check if NCCs have already been extracted
        if results_run['history']['ncc_w'] == []:

            # save data in gray-scale
            data_uint8_gray = []
            for n_iter in range(len(data_uint8)):
                data_uint8_gray.append(prnu.rgb2gray(data_uint8[n_iter]))

            # initialize the list where to save results
            ncc_list = []
            # loop over the block sizes
            for b in block_sizes:

                # extract blocks from the PRNU
                pe = PatchExtractor(dim=(b, b), stride=(b, b))
                prnu_4ncc_blocks = pe.extract(prnu_4ncc).reshape((-1,) + pe.dim[0:])

                # extract blocks from the noises and from images
                pe1 = PatchExtractor(dim=(1, b, b), stride=(1, b, b))

                noises_blocks = pe1.extract(np.asarray(noises))
                noises_blocks = noises_blocks.reshape((len(noises), noises_blocks.shape[1] *
                                                       noises_blocks.shape[2], ) + pe1.dim[1:])

                images_blocks = pe1.extract(np.asarray(data_uint8_gray))
                images_blocks = images_blocks.reshape(
                    (len(noises), images_blocks.shape[1] * images_blocks.shape[2],) + pe1.dim[1:])

                args_list = []
                for idx in range(len(noises)):
                    args_list += [(prnu_4ncc_blocks, images_blocks[idx], noises_blocks[idx])]

                # this can help saving RAM
                del images_blocks
                del noises_blocks

                # extract the NCCs considering block-size = b
                ncc_block = []
                for batch_idx0 in tqdm(np.arange(start=0, step=cpu_count(), stop=len(noises)), disable='' == '',
                                       desc=('' + ' (2/2)'), dynamic_ncols=True):
                    ncc_map = pool.map(ncc_pool, args_list[batch_idx0:batch_idx0 + cpu_count()])
                    ncc_block += ncc_map
                    del ncc_map

                # update the list containing NCCs
                ncc_list += [np.asarray(ncc_block)]

            # save NCC results
            results_run['history']['ncc_w'] = ncc_list
            np.save(npy_path, results_run)

    return 0


if __name__ == '__main__':
    main()
