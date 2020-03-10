from __future__ import print_function
import warnings

warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
import matplotlib.image as mpimg
from scipy.io import loadmat

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import architectures as a
from utils.common_utils import *
from utils import utils as u

from perceptual_loss import PerceptualLoss

from argparse import ArgumentParser
from collections import namedtuple
from time import time
from termcolor import colored
from glob import glob
import json

# this is defined here because of pickle
History = namedtuple("History", ['loss', 'psnr', 'ssim', 'ncc_w', 'ncc_d', 'vgg19'])


class Training:
    def __init__(self, args, dtype, outpath, img_shape=(512, 512, 3)):
        self.args = args
        self.dtype = dtype
        self.outpath = outpath
        self.img_shape = img_shape

        # losses
        self.l2dist = torch.nn.MSELoss().type(self.dtype)
        self.ssim = a.SSIMLoss().type(self.dtype)
        # self.kldiv = torch.nn.KLDivLoss().type(self.dtype)

        # training parameters
        self.history = History._make([list() for _ in History._fields])
        self.iiter = 0
        self.saving_interval = 0
        self.psnr_max = 0

        # data
        self.imgpath = None
        self.image_name = None
        self.img = None
        self.img_tensor = None
        self.prnu_clean = None
        self.prnu_clean_tensor = None
        self.prnu_4ncc = None
        self.prnu_4ncc_tensor = None
        self.out_img = None

        # build input tensors
        self.input_tensor = None
        self.input_tensor_old = None
        self.additional_noise_tensor = None
        self._build_input()

        # build network
        self.net = None
        self.parameters = None
        self.num_params = None
        self._build_model()
        if self.args.dncnn > 0. or self.args.nccd:
            self.DnCNN = a.DnCNN().to(self.input_tensor.device)
        if self.args.perc > 0.:
            self.vgg19loss = PerceptualLoss(
                input_range='sigmoid',  # tanh
                net_type='vgg19_pytorch_modified',  # original vgg19 with slope for LeakyReLU 0.02
                preprocessing_type='corresponding',
                matching_loss='L1',
                match=[{'matching_type': 'features', 'layers': self.args.perc_layers}],
                average_loss=True,
                extractor=None
            ).type(self.dtype)

    def _build_input(self):
        self.input_tensor = get_noise(self.args.input_depth, 'noise', self.img_shape[:2],
                                      noise_type=self.args.noise_dist, var=self.args.noise_std).type(dtype)
        self.input_tensor_old = self.input_tensor.detach().clone()
        self.additional_noise_tensor = self.input_tensor.detach().clone()

    def _build_model(self):
        if self.args.network == 'unet':
            self.net = a.UNet(num_input_channels=self.args.input_depth,
                              num_output_channels=self.img_shape[-1],
                              filters=self.args.filters,
                              more_layers=1,  # default is 0
                              concat_x=False,
                              upsample_mode=self.args.upsample,  # default is nearest
                              activation=self.args.activation,
                              pad=self.args.pad,  # default is zero
                              norm_layer=torch.nn.InstanceNorm2d,
                              need_sigmoid=self.args.need_sigmoid,
                              need_bias=True
                              ).type(self.dtype)
        elif self.args.network == 'skip':
            self.net = a.Skip(num_input_channels=self.args.input_depth,
                              num_output_channels=self.img_shape[-1],
                              num_channels_down=self.args.filters,
                              num_channels_up=self.args.filters,
                              num_channels_skip=self.args.skip,
                              upsample_mode=self.args.upsample,  # default is bilinear
                              need_sigmoid=self.args.need_sigmoid,
                              need_bias=True,
                              pad=self.args.pad,  # default is reflection, but Fantong uses zero
                              act_fun=self.args.activation  # default is LeakyReLU
                              ).type(self.dtype)
        else:
            raise ValueError('ERROR! The network has to be either unet or skip')
        self.parameters = get_params('net', self.net, self.input_tensor)
        self.num_params = sum(np.prod(list(p.size())) for p in self.net.parameters())

    def load_image(self, image_path):
        self.imgpath = image_path
        self.image_name = self.imgpath.split('.')[-2].split('/')[-1]
        # if PNG, imread normalizes to 0, 1
        # if JPEG, we must do it by hand
        self.img = u.crop_center(mpimg.imread(image_path), self.img_shape[0], self.img_shape[1])
        if self.imgpath.split('.')[-1] in ['JPEG', 'JPG', 'jpeg', 'jpg']:
            self.img = u.normalize(self.img, 0, 255, False)[0]

        if self.img.shape != self.img_shape:
            raise ValueError('The loaded image shape has to be', self.img_shape)
        self.img_tensor = u.numpy2torch(np.swapaxes(self.img, 2, 0)[np.newaxis])

    def load_prnu(self, device_path):
        # clean PRNU to be added to the output
        self.prnu_clean = loadmat(os.path.join(device_path, 'prnu%s.mat' % ('_comp' if self.args.compressed else
                                                                            '')))[ 'prnu']
        if self.prnu_clean.shape != self.img_shape[:2]:
            raise ValueError('The loaded clean PRNU shape has to be', self.img_shape[:2])

        # filtered PRNU for computing the NCC
        self.prnu_4ncc = loadmat(os.path.join(device_path, 'prnuZM_W%s.mat' % ('_comp' if self.args.compressed else '')))[
            'prnu']
        if self.prnu_4ncc.shape != self.img_shape[:2]:
            raise ValueError('The loaded filtered PRNU shape has to be', self.img_shape[:2])

        # create relative tensors
        self.prnu_clean_tensor = u.numpy2torch(self.prnu_clean[np.newaxis, np.newaxis])
        self.prnu_4ncc_tensor = u.numpy2torch(self.prnu_4ncc[np.newaxis, np.newaxis])

    def _optimization_loop(self):
        if self.args.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) == 4]:
                n += n.detach().clone().normal_() * n.std() / 50

        input_tensor = self.input_tensor_old
        if self.args.reg_noise_std > 0:
            input_tensor = self.input_tensor_old + (self.additional_noise_tensor.normal_() * self.args.reg_noise_std)

        output_tensor = self.net(input_tensor)

        if self.args.dncnn > 0. or self.args.nccd:
            w = self.DnCNN(u.rgb2gray(output_tensor, 1))

        if self.args.gamma == 0.:  # MSE between reference image and output image
            mse = self.l2dist(output_tensor, self.img_tensor)
        else:  # MSE between reference image and output image with true PRNU (weighted by gamma)
            mse = self.l2dist(u.add_prnu(output_tensor, self.prnu_clean_tensor, weight=self.args.gamma), self.img_tensor)

        dncnn_loss = u.ncc(self.prnu_clean_tensor * u.rgb2gray(output_tensor, 1), w) if self.args.dncnn else 0.
        ssim_loss = (1 - self.ssim(output_tensor, self.img_tensor)) if self.args.ssim else 0.
        perc_loss = self.vgg19loss(input=output_tensor, target=self.img_tensor) if self.args.perc else 0.

        total_loss = mse + self.args.ssim * ssim_loss + self.args.dncnn * dncnn_loss + self.args.perc * perc_loss
        total_loss.backward()

        self.history.loss.append(total_loss.item())
        self.history.psnr.append(u.psnr(output_tensor * 255, self.img_tensor * 255, 1).item())
        self.history.ssim.append(u.ssim(self.img_tensor, output_tensor).item())
        msg = "\tPicture %s,\tIter %s, Loss = %.2e, MSE=%.2e"\
              % (self.imgpath.split('/')[-1], str(self.iiter + 1).zfill(u.ten_digit(self.args.epochs)),
                 self.history.loss[-1], mse.item())

        if self.args.nccd:  # compute also the final NCC with DnCNN
            self.history.ncc_d.append(u.ncc(self.prnu_4ncc_tensor * u.rgb2gray(output_tensor, 1), w).item())
            msg += ', NCC_d = %+.4f' % self.history.ncc_d[-1]

        if self.args.perc:
            self.history.vgg19.append(perc_loss.item())
            msg += ', VGG19 = %.2e' % self.history.vgg19[-1]

        # display also evaluation metrics
        out_img = np.swapaxes(u.torch2numpy(output_tensor).squeeze(), 0, -1)
        self.history.ncc_w.append(u.ncc(self.prnu_4ncc * u.float2png(u.prnu.rgb2gray(out_img)),
                                        u.prnu.extract_single(u.float2png(out_img), sigma=3)))
        msg += ', PSNR = %2.2f dB, SSIM = %.4f, NCC_w = %+.4f'\
               % (self.history.psnr[-1], self.history.ssim[-1], self.history.ncc_w[-1])

        print(colored(msg, 'yellow'), '\r', end='')

        # save if the PSNR is increasing (above a threshold) and only every tot iterations
        if self.psnr_max < self.history.psnr[-1]:
            self.psnr_max = self.history.psnr[-1]
            if self.args.save_png_every and \
                    self.psnr_max > self.args.psnr_min and \
                    self.iiter > 0 \
                    and self.saving_interval >= self.args.save_png_every:
                self.out_img = out_img
                outname = self.image_name + \
                          '_i' + str(self.iiter).zfill(u.ten_digit(self.args.epochs)) + \
                          '_psnr%.2f_nccw%.6f.png' % (self.history.psnr[-1], self.history.ncc_w[-1])
                Image.fromarray(u.float2png(self.out_img)).save(os.path.join(self.outpath, outname))
                self.saving_interval = 0

        # save last image if none of the above conditions are respected
        if self.out_img is None and self.iiter == self.args.epochs:
            self.out_img = out_img
            outname = self.image_name + \
                      '_i' + str(self.iiter).zfill(u.ten_digit(self.args.epochs)) + \
                      '_psnr%.2f_nccw%.6f.png' % (self.history.psnr[-1], self.history.ncc_w[-1])
            Image.fromarray(u.float2png(self.out_img)).save(os.path.join(self.outpath, outname))

        self.iiter += 1
        self.saving_interval += 1

        return total_loss

    def optimize(self):
        start = time()
        optimize(self.args.optimizer, self.parameters, self._optimization_loop, self.args.lr, self.args.epochs)
        self.elapsed = time() - start

    def save_result(self):
        mydict = {
            'server': u.machine_name(),
            'device': os.environ["CUDA_VISIBLE_DEVICES"],
            'elapsed time': u.sec2time(self.elapsed),
            # 'run_code': self.outpath[-6:],
            'history': self.history._asdict(),
            'args': self.args,
            'prnu': self.prnu_clean,
            'image': self.img,
            'anonymized': self.out_img,
            'psnr_max': self.psnr_max,
            'params': self.num_params
        }
        np.save(os.path.join(self.outpath,
                             self.image_name.split('/')[-1] + '_run.npy'), mydict)

    def reset(self):
        self.iiter = 0
        self.saving_interval = 0
        print('')
        torch.cuda.empty_cache()
        self._build_input()
        self._build_model()
        self.history = History._make([list() for _ in History._fields])


def main():
    parser = ArgumentParser()
    # dataset parameter
    parser.add_argument('--device', nargs='+', type=str, required=False, default='all',
                        help='Device name in ./dataset/ folder')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (lowest memory usage based)')
    parser.add_argument('--pics_idx', nargs='+', type=int, required=False,
                        help='indeces of the first and last pictures to be processed'
                             '(e.g. 10, 15 to process images from the 10th to the 15th)')
    parser.add_argument('--outpath', type=str, required=False, default='test',
                        help='Run name in ./results/')
    parser.add_argument('--compressed', '-comp', action='store_true',
                        help='Use the JPEG dataset')

    # network design
    parser.add_argument('--network', type=str, required=False, default='skip', choices=['unet', 'skip'],
                        help='Name of the network to be used')
    parser.add_argument('--activation', type=str, default='ReLU', required=False,
                        help='Activation function to be used in the convolution block [ReLU, Tanh, LeakyReLU]')
    parser.add_argument('--need_sigmoid', type=bool, required=False, default=True,
                        help='Apply a sigmoid activation to the network output')
    parser.add_argument('--filters', nargs='+', type=int, required=False, default=[128, 128, 128, 128],
                        help='Numbers of channels')
    parser.add_argument('--skip', nargs='+', type=int, required=False, default=[0, 0, 4, 4],
                        help='Number of channels for skip')
    parser.add_argument('--input_depth', type=int, required=False, default=512,
                        help='Depth of the input noise tensor')
    parser.add_argument('--pad', type=str, required=False, default='zero', choices=['zero', 'reflection'],
                        help='Padding strategy for the network')
    parser.add_argument('--upsample', type=str, required=False, default='nearest',
                        choices=['nearest', 'bilinear', 'deconv'],
                        help='Upgoing deconvolution strategy for the network')
    # training parameter
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'lbfgs', 'sgd'],
                        help='Optimizer to be used')
    parser.add_argument('--nccd', action='store_true',
                        help='Compute and save the NCC curve computed through DnCNN.')
    parser.add_argument('--ssim', type=float, required=False, default=0.00,
                        help='Coefficient for the SSIM loss')
    parser.add_argument('--dncnn', type=float, required=False, default=0.00,
                        help='Coefficient for the DnCNN fingerprint extraction loss')
    parser.add_argument('--perc', type=float, required=False, default=0.00,
                        help='Coefficient for the VGG19 perceptual loss')
    parser.add_argument('--perc_layers', type=str, required=False, default='1,6,11,20,29',
                        help='Comma-separated layers indexes for the VGG19 perceptual loss')
    parser.add_argument('--gamma', type=float, required=False, default=0.01,
                        help='Coefficient for adding the PRNU')
    parser.add_argument('--disc', type=float, required=False, default=0.00,
                        help='Coefficient for the Discriminator')
    parser.add_argument('--softlabels', action='store_true',
                        help='Use soft labels for the discriminator')
    parser.add_argument('--epochs', '-e', type=int, required=False, default=3001,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                        help='Learning Rate for Adam optimizer')
    parser.add_argument('--save_png_every', type=int, default=0, required=False,
                        help='Number of epochs every which to save the results')
    parser.add_argument('--psnr_min', type=float, default=30., required=False,
                        help='Minimum PSNR for saving the image.')
    parser.add_argument('--param_noise', action='store_true',
                        help='Add normal noise to the parameters every epoch')
    parser.add_argument('--reg_noise_std', type=float, required=False, default=0.03,
                        help='Standard deviation of the normal noise to be added to the input every epoch')
    parser.add_argument('--noise_dist', type=str, default='uniform', required=False, choices=['normal', 'uniform'],
                        help='Type of noise for the input tensor')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Standard deviation of the noise for the input tensor')
    args = parser.parse_args()

    # set the engine to be used
    u.set_gpu(args.gpu)

    # create output folder
    outpath = os.path.join('./results/', 'comp' if args.compressed else '', args.outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print(colored('Saving to %s' % outpath, 'yellow'))
    with open(os.path.join(outpath, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=4)

    T = Training(args, dtype, outpath)

    if args.device == 'all':
        device_list = glob('dataset/*')
    elif isinstance(args.device, list):
        device_list = [os.path.join('dataset', d) for d in args.device]
    elif isinstance(args.device, str):
        device_list = [os.path.join('dataset', args.device)]

    pics_idx = args.pics_idx if args.pics_idx is not None else [0, None]  # all the pictures
    for device in device_list:  # ./dataset/device
        print(colored('Device %s' % device.split('/')[-1], 'yellow'))

        T.load_prnu(device)
        if args.compressed:
            pic_list = glob(os.path.join(device, '*.JPG'))[pics_idx[0]:pics_idx[-1]]
        else:
            pic_list = glob(os.path.join(device, '*.png'))[pics_idx[0]:pics_idx[-1]]

        for picpath in pic_list:
            T.load_image(picpath)
            T.optimize()
            T.save_result()
            T.reset()

    print(colored('Anonymization done!', 'yellow'))


if __name__ == '__main__':
    main()
