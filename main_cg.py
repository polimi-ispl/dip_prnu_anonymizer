from __future__ import print_function
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import h5py
from scipy.io import loadmat
import pylops
from scipy.sparse.linalg import cg
import utils as u
from PIL import Image

import matplotlib.pyplot as plt
from argparse import ArgumentParser
from time import time
from termcolor import colored
from glob import glob
import json
from inspect import currentframe


np.random.seed(0)


def _post_processing(img, in_min, in_max):
    img = u.normalize(img, zero_mean=False)[0]
    img = u.denormalize(img, in_min=in_min, in_max=in_max)
    # img = u.float2uint8(img)
    return img


def _compute_metrics(img1, img2, k):
    psnr = u.psnr(img1, img2)
    ncc = u.ncc(k * u.float2uint8(u.rgb2gray(img2)), u.prnu.extract_single(img2, sigma=3.))
    return psnr, ncc


class CG_Inversion:
    def __init__(self, args, outpath, img_shape=(512, 512, 3)):
        self.args = args
        self.outpath = outpath
        self.img_shape = img_shape
        self.out_list = []
        self.elapsed = 0
        self.iiter = 0
        self.history = {'loss': [], 'psnr': [], 'ssim': [], 'nccw': []}

        # data
        self.operator = None
        self.imgpath = None
        self.image_name = None
        self.img = None
        self.prnu_injection = None
        self.prnu_4ncc = None
        self.out_img = None

    def load_image(self, image_path):
        self.imgpath = image_path
        self.image_name = self.imgpath.split('.')[-2].split('/')[-1]

        _ext = os.path.splitext(self.imgpath)[-1].lower()

        if _ext == '.npy':
            self.img = np.load(self.imgpath)
        elif _ext in ['.png', '.jpeg', '.jpg']:
            self.img = u.read_image(self.imgpath)
        else:
            raise ValueError('Invalid image file extension: it has to be npy, png or jpg')

        self.img = u.crop_center(self.img, self.img_shape[0], self.img_shape[1])

        if self.img.shape != self.img_shape:
            raise ValueError('The loaded image shape has to be', self.img_shape)

    def load_prnu(self, device_path):

        policy = self.args.prnu

        # clean PRNU to be added to the output
        if policy == 'clean':
            self.prnu_injection = loadmat(os.path.join(device_path, 'prnu.mat'))['prnu']
        elif policy == 'wiener':
            self.prnu_injection = loadmat(os.path.join(device_path, 'prnu_wiener.mat'))['prnu']
        elif policy == 'extract':
            assert self.img is not None, 'No image has been loaded'
            self.prnu_injection = u.prnu.extract_single(self.img, sigma=3.)
        else:
            raise ValueError('PRNU policy has to be clean, wiener or extract')

        if self.prnu_injection.shape != self.img_shape[:2]:
            raise ValueError('The loaded clean PRNU shape has to be', self.img_shape[:2])

        # filtered PRNU for computing the NCC
        self.prnu_4ncc = loadmat(os.path.join(device_path, 'prnu_wiener.mat'))['prnu']
        if self.prnu_4ncc.shape != self.img_shape[:2]:
            raise ValueError('The loaded filtered PRNU shape has to be', self.img_shape[:2])

        self._build_operator()

    def _build_operator(self):
        k = np.repeat(np.expand_dims(self.prnu_injection, -1), 3, axis=-1).ravel()
        self.operator = pylops.Identity(k.size) + self.args.gamma * pylops.Diagonal(k)

    def _callback(self, output_image):
        frame = currentframe().f_back
        mse = np.linalg.norm(frame.f_locals['resid']) ** 2
        self.history['loss'].append(mse)

        output_image = _post_processing(output_image.reshape(self.img_shape),
                                        in_min=self.img.min(), in_max=self.img.max())
        self.out_list.append(output_image)

        psnr, ncc = _compute_metrics(self.img, output_image, self.prnu_4ncc)
        self.history['psnr'].append(psnr)
        self.history['nccw'].append(ncc)

        msg = "\tPicture %s, \tIter %s, Obj=%.2e, PSNR=%2.2f dB, NCC=%+.6f" \
              % (self.imgpath.split('/')[-1],
                 str(self.iiter + 1).zfill(u.ten_digit(self.args.epochs)),
                 self.history['loss'][-1], psnr, ncc)
        print(colored(msg, 'yellow'), '\r', end='')

        self.iiter += 1

    def optimize(self):
        start = time()
        self.cg_out = cg(self.operator, self.img.ravel().astype(np.float),
                         maxiter=self.args.epochs,
                         callback=self._callback if not self.args.disable_callback else None)[0]
        self.cg_out = u.float2uint8(self.cg_out).reshape(self.img_shape)
        # self.out_img = _post_processing(self.cg_out, in_min=self.img.min(), in_max=self.img.max())
        self.out_img = self.cg_out
        self.elapsed = time() - start

    def save_result(self, save_images=False):
        # mydict = {
        #     'server': u.machine_name(),
        #     'device': 'cpu',
        #     'elapsed time': u.sec2time(self.elapsed),
        #     'history': self.history,
        #     'args': self.args,
        #     'prnu': self.prnu_injection,
        #     'prnu4ncc': self.prnu_4ncc,
        #     'image': self.img,
        #     'anonymized': self.out_img,
        #     'cg_out': self.cg_out,
        # }
        outname = self.image_name.split('/')[-1]# + '_run'
        # np.save(os.path.join(self.outpath, outname + '.npy'), mydict)
        img = Image.fromarray(self.out_img)
        img.save(os.path.join(self.outpath, outname + '.png'))

        if save_images:
            with h5py.File(os.path.join(self.outpath, outname + '.hdf5'), 'w') as f:
                dset = f.create_dataset("all_outputs", data=np.asarray(self.out_list))

        if self.args.plot:
            clim = (0, 255)
            fig, axs = plt.subplots(1, 3, figsize=(19, 6))
            axs[0].imshow(u.float2uint8(self.img), clim=clim)
            axs[0].set_title("Input image, γ=%.e, PRNU %s" % (self.args.gamma, self.args.prnu))

            axs[1].imshow(self.cg_out, clim=clim)
            axs[1].set_title("CG output, PSNR=%.2f, NCC=%.6f"
                             % _compute_metrics(self.img, self.cg_out, self.prnu_4ncc))

            axs[2].imshow(self.out_img, clim=clim)
            axs[2].set_title("post-processed, PSNR=%.2f, NCC=%.6f"
                             % _compute_metrics(self.img, self.out_img, self.prnu_4ncc))

            fig.tight_layout(pad=.5)
            plt.savefig(os.path.join(self.outpath, outname + '.pdf'))
            plt.show()

    def reset(self):
        # print('')
        self.iiter = 0
        self.history = {'loss': [], 'psnr': [], 'ssim': [], 'nccw': []}
        self.out_list = []


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--device', nargs='+', type=str, required=False, default='all',
                        help='Device name')
    parser.add_argument('--dataset', type=str, required=False, default='dresdenPNG',
                        help='Dataset to be used')
    parser.add_argument('--pics_idx', nargs='+', type=int, required=False,
                        help='indeces of the first and last pictures to be processed'
                             '(e.g. 10, 15 to process images from the 10th to the 15th)')
    parser.add_argument('--pics_IDs', nargs='+', type=str, required=False,
                        help='5-long code of the picture to be loaded')
    parser.add_argument('--disable_callback', action='store_true', default=False,
                        help='CG will not save the iteration results')
    parser.add_argument('--outpath', type=str, required=False, default='debug',
                        help='Run name in ./results/')
    parser.add_argument('--save_outputs', action='store_true', default=False,
                        help='Save every network output to disk in a hdf5 file.')
    parser.add_argument('--prnu', type=str, default='clean', required=False,
                        choices=['clean', 'wiener', 'extract'],
                        help='Which PRNU to inject: clean, wiener or extracted from the picture')
    parser.add_argument('--epochs', '-e', type=int, required=False, default=1000,
                        help='Number of CG iterations')
    parser.add_argument('--gamma', type=float, required=False, default=0.01,
                        help='Fixed gamma parameter.')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Save a pdf file with the three images.')

    return parser.parse_args()


def main():
    args = _parse_args()

    # create output folder
    outpath = os.path.join('./results/', args.outpath)
    os.makedirs(outpath, exist_ok=True)
    print(colored('Saving to %s' % outpath, 'yellow'))
    with open(os.path.join(outpath, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=4)

    problem = CG_Inversion(args, outpath)

    if args.device == 'all':
        device_list = glob(os.path.join(args.dataset, '*'))
    elif isinstance(args.device, list):
        device_list = [os.path.join(args.dataset, d) for d in args.device]
    elif isinstance(args.device, str):
        device_list = [os.path.join(args.dataset, args.device)]
    device_list = sorted(device_list)

    pics_idx = args.pics_idx if args.pics_idx is not None else [0, None]  # all the pictures

    for device in device_list:  # ./dataset/device
        print(colored('Device %s' % device.split('/')[-1], 'yellow'))

        if args.prnu != 'extract':
            problem.load_prnu(device)

        if 'png' in args.dataset.lower():
            _ext = 'png'
        else:
            _ext = 'jpg'
        if 'float' in device:
            _ext = 'npy'

        if args.pics_IDs is not None:  # load specified image
            pic_list = [os.path.join(device, device.split('/')[-1] + '_%s.%s' % (_, _ext)) for _ in args.pics_IDs]
        else:
            pic_list = sorted(glob(os.path.join(device, '*.%s' % _ext)))
            pic_list = pic_list[pics_idx[0]:pics_idx[-1]]

        for picpath in pic_list:

            problem.load_image(picpath)

            if args.prnu == 'extract':
                problem.load_prnu(device)

            # cg_out = cg(problem.operator, problem.img.ravel(), maxiter=args.epochs)[0].reshape(problem.img_shape)  # float
            # cg_psnr, cg_ncc = _compute_metrics(problem.img, u.float2uint8(cg_out), problem.prnu_4ncc)
            #
            # cg_norm_out = _post_processing(cg_out, problem.img.min(), problem.img.max())
            # cg_norm_out_uint8 = _post_processing(u.float2uint8(cg_norm_out), problem.img.min(), problem.img.max())
            #
            # _compute_metrics(problem.img, cg_norm_out_uint8, problem.prnu_4ncc)
            # _compute_metrics(problem.img, cg_norm_out, problem.prnu_4ncc)
            #
            # fig, axs = plt.subplots(1, 3, figsize=(19, 6))
            # axs[0].imshow(u.float2uint8(problem.img), clim=(0, 255))
            # axs[0].set_title("Input for CG, γ=%.e, PRNU %s" % (problem.args.gamma, args.prnu))
            # axs[1].imshow(cg_out, clim=(0, 255))
            # axs[1].set_title("PSNR=%.2f, NCC=%.6f" % (cg_psnr, cg_ncc))
            # axs[2].imshow(cg_norm_out, clim=(0, 255))
            # axs[2].set_title("normalized, PSNR=%.2f, NCC=%.6f" % (cg_norm_psnr, cg_norm_ncc))
            # fig.tight_layout(pad=.5)
            # plt.show()

            problem.optimize()
            problem.save_result(save_images=args.save_outputs)
            problem.reset()

    print(colored('Anonymization done!', 'yellow'))


if __name__ == '__main__':
    main()
