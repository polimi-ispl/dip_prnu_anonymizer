# extract NCC from anonymized images, considering squared blocks of various size

import os
import numpy as np
import h5py
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import utils as u


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run', type=str, required=True,
                        help='path to the run file')
    parser.add_argument('--block_size', type=int, required=True,
                        help='Square block size for the post-processing computation')
    args = parser.parse_args()

    # open the pool
    with Pool() as pool:
        # path to already saved results
        run_name, _ = os.path.splitext(args.run)

        # load run parameters
        run_dict = np.load(run_name + '.npy', allow_pickle=True).item()

        # load the DIP output images
        with h5py.File(run_name + '.hdf5', "r+") as f:
            images_uint8 = list(f['all_outputs'])

            if len(list(f.keys())) > 1:
                # noises have already been extracted
                noises = list(f['noises'])
            else:
                # noise extraction
                noises = u.prnu.noise_extract_multiple(images_uint8, pool)
                f.create_dataset('noises', data=noises, dtype='float32')

            # go grayscale
            images_uint8 = [u.prnu.rgb2gray(_) for _ in images_uint8]
            n_iter = len(images_uint8)

        # extract blocks
        pe = u.PatchExtractor(dim=(1, args.block_size, args.block_size), stride=(1, args.block_size, args.block_size))

        prnu_blocks = pe.extract(run_dict['prnu4ncc'][None]).reshape((-1,) + pe.dim[1:])

        noises_blocks = pe.extract(np.asarray(noises)).reshape((n_iter,) + prnu_blocks.shape)

        images_blocks = pe.extract(np.asarray(images_uint8)).reshape((n_iter,) + prnu_blocks.shape)

        # now we have images_blocks and noises_blocks with shape (epochs, nblock, *block_shape)
        # prnu_blocks has shape (nblock, *block_shape)
        pool_args = [(prnu_blocks, images_blocks[_], noises_blocks[_]) for _ in range(n_iter)]
        del prnu_blocks, noises_blocks, images_blocks

        # compute NCCs
        ncc_block = []
        for batch_idx0 in tqdm(np.arange(start=0, step=cpu_count(), stop=len(pool_args)), disable='' == '',
                               desc=('' + ' (2/2)'), dynamic_ncols=True):
            ncc_map = pool.map(ncc_pool, pool_args[batch_idx0:batch_idx0 + cpu_count()])
            ncc_block += ncc_map
            del ncc_map

        # save NCCs into the same run .npy file
        run_dict['history'][f"ncc_block{args.block_size}"] = np.asarray(ncc_block)
        np.save(run_name + '.npy', run_dict)
