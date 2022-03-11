# select the generated image that minimizes the NCC (above a certain threshold)

import os
import numpy as np
import h5py
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import utils as u
from termcolor import colored
from main_2_blocks import ncc_pool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run', type=str, required=True,
                        help='path to the run file, without extension')
    parser.add_argument('--psnr_thresh', type=float, required=True,
                        help='PSNR threshold for selecting the images')

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

            images_uint8 = [u.prnu.rgb2gray(_) for _ in images_uint8]
            n_iter = len(images_uint8)

            # compute NCCs
            pool_args = [(run_dict["prnu4ncc"][None], images_uint8[_][None], noises[_][None]) for _ in range(n_iter)]
                        # TODO if run_dict["history"]["psnr"][_] >= args.psnr_thresh]

            nccs = []
            for batch_idx0 in tqdm(np.arange(start=0, step=cpu_count(), stop=len(pool_args)), disable='' == '',
                                   desc=('' + ' (2/2)'), dynamic_ncols=True):
                ncc_map = pool.map(ncc_pool, pool_args[batch_idx0:batch_idx0 + cpu_count()])
                nccs += ncc_map
                del ncc_map

    nccs = np.asarray(nccs).squeeze()
    psnr = np.asarray(run_dict["history"]["psnr"]).squeeze()
    idx = np.argmin(np.where(psnr >= args.psnr_thresh, np.abs(nccs), np.inf))

    # save NCCs into the same run .npy file
    naive = {'image': images_uint8[idx],
             'psnr': psnr[idx],
             'ncc': nccs[idx]}
    np.save(run_name + '_naive.npy', naive)

    print(colored(f"Naive image saved to {run_name}\n\tPSNR = %+2.2f dB\n\tNCC  = %+.6f" % (psnr[idx], nccs[idx]), "yellow"))
