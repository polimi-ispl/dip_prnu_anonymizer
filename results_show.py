# show output results from DEEP PRNU ANONYMIZER:

import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_curve

# devices:
devices = ['Nikon_D200_0', 'Nikon_D200_1', 'Nikon_D70_0', 'Nikon_D70_1', 'Nikon_D70s_0', 'Nikon_D70s_1']

# result folder
output_folder = '/nas/home/fpicetti/dip_prnu_anonymizer/results/_evaluation/gamma0.1'

psnr_min_vec = [37, 38, 39, 40]

negative_ncc_path = '/nas/home/smandelli/ispg-greeneyes-data/icip2017_ImageAnonymizer/online_code/NCC_others.mat'
ncc_from_sara = loadmat(negative_ncc_path)['ncc']

for p, psnr_min in enumerate(psnr_min_vec):

    thr_values = []
    psnr_vec = []

    for dev_idx, dev in enumerate(devices):

        # load DEEP PRIOR ANONYMIZATION results
        anonymized_img_dev_params = np.load(os.path.join(output_folder, 'anonymized_params_{}.npy'.format(dev)))
        # size = (len(psnr_min_vec), len(list_npy_dev), [psnr_value, ncc_value]))

        psnr_vec += [anonymized_img_dev_params[p, :, 0]]

        # load NEGATIVE NCC VALUES (from Sara, Matlab values): (500 values of ncc).
        negative_ncc_values = ncc_from_sara[dev_idx, np.setdiff1d(np.arange(0, 6), dev_idx), :].ravel()

        values_concat = np.concatenate((negative_ncc_values, anonymized_img_dev_params[p, :, 1].ravel()))

        # update thr value
        thr_values += [values_concat]

    thr_values_arr = np.sort(np.unique(np.concatenate(thr_values).ravel()))

    tn = np.zeros((len(devices), thr_values_arr.shape[0]))
    fn = np.zeros((len(devices), thr_values_arr.shape[0]))
    tp = np.zeros((len(devices), thr_values_arr.shape[0]))
    fp = np.zeros((len(devices), thr_values_arr.shape[0]))

    for dev_idx, dev in enumerate(devices):

        anonymized_img_dev_params = np.load(os.path.join(output_folder, 'anonymized_params_{}.npy'.format(dev)))
        tp_set = anonymized_img_dev_params[p, :, 1].ravel()
        tn_set = ncc_from_sara[dev_idx, np.setdiff1d(np.arange(0, 6), dev_idx), :].ravel()

        for t, thr in enumerate(thr_values_arr):

            tn[dev_idx, t] = np.sum(tn_set <= thr)
            fn[dev_idx, t] = np.sum(tp_set <= thr)
            tp[dev_idx, t] = np.sum(tp_set > thr)
            fp[dev_idx, t] = np.sum(tn_set > thr)

    # ROC CURVE OF EAH DEVICE BY SKLEARN
    #     fpr, tpr, _ = roc_curve(np.concatenate([np.ones_like(tp_set), np.zeros_like(tn_set)]),
    #                             np.concatenate([tp_set, tn_set]))
    #
    #     plt.plot(fpr, tpr)
    # plt.show()

    tp_rate = np.mean(tp / (tp + fn), axis=0)
    fp_rate = np.mean(fp / (fp + tn), axis=0)

    # final ROC curve, averaged over devices
    plt.figure(figsize=(5, 5))
    plt.plot(fp_rate, tp_rate, lw=2)
    plt.xlabel('FPR'), plt.ylabel('TPR')
    plt.show()

    tp_sort = np.sort(tp_rate)
    fp_sort = np.sort(fp_rate)

    tpr_001_ind = np.where(fp_sort >= 0.01)[0][0]
    tpr_001 = tp_sort[tpr_001_ind]

    print('min PSNR = {}, AUC = {}'.format(psnr_min, np.trapz(tp_sort, fp_sort)))
    print('min PSNR = {}, TPR@0.01 = {}'.format(psnr_min, tpr_001))
    print('min PSNR = {}, Median PSNR [dB] = {}'.format(psnr_min, np.median(np.concatenate(psnr_vec))))
    print('min PSNR = {}, Mean PSNR [dB] = {}'.format(psnr_min, np.mean(np.concatenate(psnr_vec))))

    print('0')
