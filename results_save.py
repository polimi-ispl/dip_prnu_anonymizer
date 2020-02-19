# view output results from DEEP PRNU ANONYMIZER:

import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from scipy.io import loadmat

# devices:
devices = ['Nikon_D200_0', 'Nikon_D200_1', 'Nikon_D70_0', 'Nikon_D70_1', 'Nikon_D70s_0', 'Nikon_D70s_1']

# results folder:
result_folder = '/nas/home/fpicetti/dip_prnu_anonymizer/results/gamma0.1'

# output folder
output_folder = '/nas/home/fpicetti/dip_prnu_anonymizer/results/_evaluation/gamma0.1'

psnr_min_vec = [37, 38, 39, 40]

os.makedirs(output_folder, exist_ok=True)

# load negative values.
negative_ncc_path = '/nas/home/smandelli/ispg-greeneyes-data/icip2017_ImageAnonymizer/online_code/NCC_others.mat'
ncc_from_sara = loadmat(negative_ncc_path)['ncc']
# estimate the standard deviation of the negative distribution
std_neg_ncc = np.std(ncc_from_sara.ravel())

for dev_idx, dev in enumerate(devices):

    # extract only npy files coming from this device.
    list_npy_dev = sorted(glob(os.path.join(result_folder, '{}*_run.npy'.format(dev))))

    anonymized_img_dev_params = np.zeros((len(psnr_min_vec), len(list_npy_dev), 2))

    for r, npy_path in enumerate(list_npy_dev):

        # load file
        result = np.load(npy_path, allow_pickle=True).tolist()

        # load psnr and ncc
        psnr_list = result['history']['psnr']
        idx_ascending_psnr = np.argsort(psnr_list)
        psnr_array = np.asarray(psnr_list)
        psnr_ascending = psnr_array[idx_ascending_psnr]   # psnr in ascending order.

        ncc_list = result['history']['ncc_w']
        ncc_ascending_psnr = np.asarray(ncc_list)[idx_ascending_psnr]  # ncc with the same order of psnr.

        # loop over psnr.
        for p, psnr_min in enumerate(psnr_min_vec):

            psnr_idx_ok = np.where(psnr_ascending >= np.min([np.max(psnr_ascending), psnr_min]))[0]

            ####################### limit the ncc to the +- 3sigma interval of negative values #########################
            # ncc_idx_ok = np.where((ncc_ascending_psnr >= -3*std_neg_ncc) & (ncc_ascending_psnr <= 3*std_neg_ncc))[0]
            #
            # # intersection between the two sets.
            # overall_idx_ok = np.where(np.in1d(ncc_idx_ok, psnr_idx_ok) == True)[0]
            #
            # if overall_idx_ok.size == 0:
            #    overall_idx_ok = psnr_idx_ok

            ############################################################################################################
            overall_idx_ok = psnr_idx_ok

            # select the index related to the minimum of ncc
            ncc_psnr_ok_idx = ncc_ascending_psnr[overall_idx_ok].argmin()
            ncc_psnr_ok = ncc_ascending_psnr[overall_idx_ok[ncc_psnr_ok_idx]]
            psnr_ok = psnr_ascending[psnr_idx_ok[ncc_psnr_ok_idx]]

            anonymized_img_dev_params[p, r, :] = [psnr_ok, ncc_psnr_ok]

    # save results per device.
    np.save(os.path.join(output_folder, 'anonymized_params_{}.npy'.format(dev)), anonymized_img_dev_params)
