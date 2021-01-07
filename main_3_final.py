# Generate the final anonymized image

import numpy as np
import h5py
import argparse
from utils.patch_extractor import PatchExtractor
from multiprocessing import Pool
from PIL import Image


def sort_nccs_block(ncc_block, psnr_idx_ok):

    """
    Sort the NCCs associated with a certain block position
    ncc_block = array containing the NCCs associated with the block position, for all the iterations with PSNR >= tauPsnr
    psnr_idx_ok = array containing the iterations indices for which PSNR >= tauPsnr
    """

    # substitute nan with inf
    ncc_block[np.isnan(ncc_block)] = np.inf

    # first, consider negative nccs
    nccs_negative_idx = np.flip(np.argsort(ncc_block[ncc_block < 0]))
    idx_negative = [np.where(ncc_block == ncc_block[ncc_block < 0][nccs_negative_idx[i]])[0][0] for i in
                range(len(nccs_negative_idx))]

    # then, include also positive nccs
    nccs_positive_idx = np.argsort(ncc_block[ncc_block >= 0])
    idx_positive = [np.where(ncc_block == ncc_block[ncc_block >= 0][nccs_positive_idx[i]])[0][0] for i in
                    range(len(nccs_positive_idx))]

    # order the NCC indices
    if idx_negative == []:
        nccs_ordered_idx = np.argsort(np.abs(ncc_block))
    elif idx_positive == []:
        nccs_ordered_idx = np.array(idx_negative)
    else:
        nccs_ordered_idx = np.concatenate(
            (idx_negative, idx_positive))

    # ordered NCC indices associated with the block position b
    nccs_idx_ordered_b = psnr_idx_ok[nccs_ordered_idx.astype('int')]

    return nccs_idx_ordered_b


def sort_nccs_block_call(args):

    ncc_block = args['ncc_block']
    psnr_idx_ok = args['psnr_idx_ok']

    return sort_nccs_block(ncc_block, psnr_idx_ok)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True,
                        help='path to the run file')
    parser.add_argument('--block', type=int, required=True,
                        help='Square block size for the post-processing computation')
    parser.add_argument('--firstL', type=int, required=True,
                        help='first L blocks to be averaged in the post-processing computation')
    parser.add_argument('--tauPsnr', type=int, required=True,
                        help='PSNR threshold for selecting the images')
    config, _ = parser.parse_known_args()

    # load h5py file
    with h5py.File(config.run + '.hdf5', "r") as f:

        a_group_key = list(f.keys())[0]
        # Get the images
        data = list(f[a_group_key])

    # load PSNRs and NCCs
    run_dict = np.load(config.run + '.npy', allow_pickle=True).tolist()
    psnrs = run_dict['history']['psnr']
    nccs = run_dict['history'][f"ncc_block{config.block}"]

    # select only image indices with PSNR >= tauPsnr
    psnr_idx_ok = np.where(psnrs >= np.min([np.max(psnrs), config.tauPsnr]))[0]

    ############################################ Extract blocks from images ############################################

    pe = PatchExtractor(dim=(1, config.block, config.block), stride=(1, config.block, config.block))

    # initialize the array containing the image color blocks
    images_color_blocks = np.zeros((len(data), (data[0].shape[0] // config.block) ** 2, config.block, config.block, 3),
                                   dtype='uint8')
    # loop over the three color channels
    for ch in range(3):
        images_blocks = pe.extract(np.asarray(data)[:, :, :, ch])
        patch_n = images_blocks.shape[1]
        images_blocks = images_blocks.reshape(
            (len(data), images_blocks.shape[1] * images_blocks.shape[2],) + pe.dim[1:])
        images_color_blocks[:, :, :, :, ch] = images_blocks

    del images_blocks

    ###################################### Order the NCCs for each block position ######################################

    # consider only NCCs with acceptable PSNR
    nccs_ok = nccs[psnr_idx_ok]
    # size of "nccs_ok" = M (i.e., number of iterations with PSNR>=tauPsnr) x number of extracted blocks per image

    args_list = []
    # loop over the number of extracted blocks per image (i.e., loop over the possible block positions)
    for b_idx in range(nccs_ok.shape[1]):
        args = {}
        args['ncc_block'] = nccs_ok[:, b_idx]
        args['psnr_idx_ok'] = psnr_idx_ok
        args_list += [args]

    # Open pooling
    with Pool() as pool:

        nccs_best_idx_ordered = pool.map(sort_nccs_block_call, args_list)

    nccs_best_idx_ordered = np.array(nccs_best_idx_ordered).T  # transpose it for further operations

    ####################################### Generate the final anonymized image ########################################

    # define a new Patch extractor for image reconstruction
    pe1 = PatchExtractor(dim=(1, config.block, config.block), stride=(1, config.block, config.block))
    # aux variable (needed for image reconstruction)
    aux = pe1.extract(data[0][:, :, 0].reshape((1, 512, 512)))

    # average at most firstL blocks
    if nccs_best_idx_ordered.shape[0] < config.firstL:
        nccs_firstL_idx = nccs_best_idx_ordered
    else:
        nccs_firstL_idx = nccs_best_idx_ordered[:config.firstL, :]

    img_firstL = []
    best_blocks = np.zeros((nccs_firstL_idx.shape[0], images_color_blocks.shape[1], images_color_blocks.shape[2],
                            images_color_blocks.shape[3], images_color_blocks.shape[-1]), dtype='uint8')
    for n_it in range(nccs_firstL_idx.shape[0]):
        for bb in range(images_color_blocks.shape[1]):
            best_blocks[n_it, bb] = images_color_blocks[nccs_firstL_idx[n_it, bb], bb, :]

        # reconstruct the image associated to the block position
        img_rec = np.zeros((512, 512, 3))
        for ch in range(3):
            img_rec[:, :, ch] = pe1.reconstruct(
                best_blocks[n_it, :, :, :, ch].reshape(
                    (1, patch_n, patch_n, 1, config.block, config.block))).squeeze()
        img_firstL.append(img_rec)

    del best_blocks

    # average results
    avg_firstL = np.mean(img_firstL, axis=0)
    # final anonymized image (uint8)
    anonymized_img = np.clip(avg_firstL, 0, 255).astype(np.uint8)

    # save the resulting image:
    final_image = Image.fromarray(anonymized_img)
    image_name = config.run + '.png'
    final_image.save(image_name, "PNG")

    return 0


if __name__ == '__main__':
    main()
