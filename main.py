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
from architectures import UNet, Skip, DnCNN
from utils.common_utils import *
from utils import utils as u

from argparse import ArgumentParser
from collections import namedtuple
from time import time
from termcolor import colored
from glob import glob
import json

# this is defined here because of pickle
History = namedtuple("History", ['loss', 'psnr', 'ssim', 'ncc_w', 'ncc_d'])


class Training:
    def __init__(self, args, dtype, outpath, img_shape=(512, 512, 3)):
        self.args = args
        self.dtype = dtype
        self.outpath = outpath
        self.img_shape = img_shape
        self.l2dist = torch.nn.MSELoss().type(self.dtype)
        self.history = History([], [], [], [], [])
        self.elapsed = None
        self.iiter = 0
        self.saving_interval = 0
        self.psnr_max = 0

        self.imgpath = None
        self.image_name = None
        self.img = None
        self.img_tensor = None
        self.prnu = None
        self.prnu_tensor = None
        self.out_img = None

        # build input tensors
        self.input_tensor = None
        self.input_tensor_old = None
        self.additional_noise_tensor = None
        self.build_input()

        # build network
        self.net = None
        self.parameters = None
        self.num_params = None
        self.build_model()
        self.dncnn = DnCNN().to(self.input_tensor.device)

    def build_input(self):
        self.input_tensor = get_noise(self.args.input_depth, 'noise', self.img_shape[:2],
                                      noise_type=self.args.noise_dist, var=self.args.noise_std).type(dtype)
        self.input_tensor_old = self.input_tensor.detach().clone()
        self.additional_noise_tensor = self.input_tensor.detach().clone()

    def build_model(self):
        if self.args.network == 'unet':
            self.net = UNet(num_input_channels=self.args.input_depth,
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
            self.net = Skip(num_input_channels=self.args.input_depth,
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
        self.img = mpimg.imread(image_path)
        if self.img.shape != self.img_shape:
            raise ValueError('The loaded image shape has to be', self.img_shape)
        self.img_tensor = u.numpy2torch(np.swapaxes(self.img, 2, 0)[np.newaxis])

    def load_prnu(self, device_path):
        self.prnu = loadmat(os.path.join(device_path, 'prnu_cropped512.mat'))['prnu']
        if self.prnu.shape != self.img_shape[:2]:
            raise ValueError('The loaded PRNU shape has to be', self.img_shape[:2])
        self.prnu_tensor = u.numpy2torch(self.prnu[np.newaxis, np.newaxis])

    def optimization_loop(self):
        if self.args.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) == 4]:
                n += n.detach().clone().normal_() * n.std() / 50

        input_tensor = self.input_tensor_old
        if self.args.reg_noise_std > 0:
            input_tensor = self.input_tensor_old + (self.additional_noise_tensor.normal_() * self.args.reg_noise_std)

        output_tensor = self.net(input_tensor)
        w = self.dncnn(u.rgb2gray(output_tensor, 1))

        if self.args.gamma == 0.:  # MSE between reference image and output image
            total_loss = self.l2dist(output_tensor, self.img_tensor)
        else:  # MSE between reference image and output image with true PRNU (weighted by gamma)
            total_loss = self.l2dist(u.add_prnu(output_tensor, self.prnu_tensor, weight=self.args.gamma),
                                     self.img_tensor)

        if self.args.beta != 0.:  # cross-correlation between the true PRNU and the one extracted by the DnCNN
            # w = self.dncnn(u.rgb2gray(output_tensor, 1))
            total_loss += self.args.beta * u.ncc(self.prnu_tensor * u.rgb2gray(output_tensor, 1), w)

        total_loss.backward()

        self.history.loss.append(total_loss.item())
        self.history.psnr.append(u.psnr(output_tensor * 255, self.img_tensor * 255, 1).item())
        self.history.ssim.append(u.ssim(self.img_tensor, output_tensor).item())
        msg = "Processing %s,\tIter %s, Loss = %.2e, PSNR = %.2f dB, SSIM = %.4f" \
              % (self.imgpath.split('/')[-1], str(self.iiter+1).zfill(u.ten_digit(self.args.epochs)),
                 self.history.loss[-1], self.history.psnr[-1], self.history.ssim[-1])

        out_img = np.swapaxes(u.torch2numpy(output_tensor).squeeze(), 0, -1)
        self.history.ncc_w.append(u.ncc(self.prnu * u.float2png(u.prnu.rgb2gray(out_img)),
                                        u.prnu.extract_single(u.float2png(out_img))))
        msg += ', NCC_w = %.6f' % self.history.ncc_w[-1]

        if self.args.nccd:  # compute also the final NCC with DnCNN
            self.history.ncc_d.append(u.ncc(self.prnu_tensor * u.rgb2gray(output_tensor, 1), w).item())
            msg += ', NCC_d = %.6f' % self.history.ncc_d[-1]

        print(colored(msg, 'yellow'), '\r', end='')

        # # save every N iterations
        # if self.args.save_every != 0 and self.iiter % self.args.save_every == 0:
        #     outname = self.image_name + \
        #               '_i' + str(self.iiter).zfill(u.ten_digit(self.args.epochs)) + \
        #               '_psnr%.2f.png' % self.history.psnr[-1]
        #     Image.fromarray(u.float2png(self.out_img)).save(os.path.join(self.outpath, outname))

        # save if the PSNR is increasing (above a threshold) and only every tot iterations
        if self.psnr_max < self.history.psnr[-1]:
            self.psnr_max = self.history.psnr[-1]
            self.out_img = out_img
            if self.psnr_max > self.args.psnr_min and \
                    self.iiter > 0 and self.saving_interval >= self.args.save_every:
                outname = self.image_name + \
                          '_i' + str(self.iiter).zfill(u.ten_digit(self.args.epochs)) + \
                          '_psnr%.2f_nccw%.6f.png' % (self.history.psnr[-1], self.history.ncc_d[-1])
                Image.fromarray(u.float2png(self.out_img)).save(os.path.join(self.outpath, outname))
                self.saving_interval = 0

        self.iiter += 1
        self.saving_interval += 1

        return total_loss

    def optimize(self):
        start = time()
        optimize(self.args.optimizer, self.parameters, self.optimization_loop, self.args.lr, self.args.epochs)
        self.elapsed = time() - start

    def save_result(self):
        mydict = {
            'server': u.machine_name(),
            'device': os.environ["CUDA_VISIBLE_DEVICES"],
            'elapsed time': u.sec2time(self.elapsed),
            'run_code': self.outpath[-6:],
            'history': self.history,
            'args': self.args,
            'prnu': self.prnu,
            'image': self.img,
            'anonymized': self.out_img,
            'psnr_max': self.psnr_max,
            'params': self.num_params
        }
        np.save(os.path.join(self.outpath,
                             self.image_name.split('/')[-1] + '_run.npy'), mydict)

    def clean(self):
        self.iiter = 0
        self.saving_interval = 0
        print('')
        torch.cuda.empty_cache()
        self.build_input()
        self.build_model()
        self.history = History([], [], [], [], [])


def main() -> int:
    parser = ArgumentParser()
    # dataset parameter
    parser.add_argument('--datapath', type=str, required=False, default='./dataset/',
                        help='Dataset path')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (lowest memory usage based)')
    parser.add_argument('--pics_per_dev', type=int, required=False, default=-1,
                        help='Number of pictures per device to be processed')
    # network design
    parser.add_argument('--network', type=str, required=False, default='skip',
                        help='Name of the network to be used [unet, skip]')
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
    parser.add_argument('--pad', type=str, required=False, default='zero',
                        help='Padding strategy for the network [zero, reflection]')
    parser.add_argument('--upsample', type=str, required=False, default='nearest',
                        help='Upgoing deconvolution strategy for the network [nearest, bilinear, deconv]')
    # training parameter
    parser.add_argument('--optimizer', type=str, required=False, default='adam',
                        help='Optimizer to be used [adam, lbfgs, sgd]')
    parser.add_argument('--nccd', action='store_true',
                        help='Compute and save the NCC curve computed through DnCNN.')
    parser.add_argument('--gamma', type=float, required=False, default=0.45,
                        help='Coefficient for adding the PRNU')
    parser.add_argument('--beta', type=float, required=False, default=0.00,
                        help='Coefficient for the DnCNN fingerprint extraction loss')
    parser.add_argument('--epochs', '-e', type=int, required=False, default=6001,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                        help='Learning Rate for Adam optimizer')
    parser.add_argument('--save_every', type=int, default=0, required=False,
                        help='Number of epochs every which to save the results')
    parser.add_argument('--psnr_min', type=float, default=30., required=False,
                        help='Minimum PSNR for saving the image.')
    parser.add_argument('--param_noise', action='store_true',
                        help='Add normal noise to the parameters every epoch')
    parser.add_argument('--reg_noise_std', type=float, required=False, default=0.03,
                        help='Standard deviation of the normal noise to be added to the input every epoch')
    parser.add_argument('--noise_dist', type=str, default='uniform', required=False,
                        help='Type of noise for the input tensor [normal, uniform]')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Standard deviation of the noise for the input tensor')
    args = parser.parse_args()

    # set the engine to be used
    u.set_gpu(args.gpu)

    # create output folder
    outpath = os.path.join('./results/', u.random_code())
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print(colored('Saving to %s' % outpath, 'yellow'))
    with open(os.path.join(outpath, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    picture_list = []

    T = Training(args, dtype, outpath)

    # process images
    for device_path in glob(os.path.join(args.datapath, '*')):

        T.load_prnu(device_path)
        pic_list = glob(os.path.join(device_path, '*.png'))[:args.pics_per_dev]
        picture_list += pic_list

        for picpath in pic_list:
            T.load_image(picpath)
            T.optimize()
            T.save_result()
            T.clean()

    print(colored('Anonymization done!', 'yellow'))


if __name__ == '__main__':
    main()
