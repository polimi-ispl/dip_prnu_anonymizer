from __future__ import print_function
import warnings

from torch.nn.utils import clip_grad_value_

warnings.filterwarnings("ignore")

import os
import torch
import h5py
import matplotlib.image as mpimg
from scipy.io import loadmat
from skimage.feature import canny
from skimage.morphology import binary_dilation
from tqdm import trange

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
torch.backends.cudnn.deterministic = True
import architectures as a
from utils.common_utils import *
from utils import utils as u

from argparse import ArgumentParser
from time import time
from termcolor import colored
from glob import glob
import json


def _set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Training:
    def __init__(self, args, dtype, outpath, img_shape=(512, 512, 3)):
        self.reload_number = 0
        self.args = args
        self.dtype = dtype
        self.outpath = outpath
        self.img_shape = img_shape
        self.out_list = []
        self.attempt = 0

        # losses
        self.l2dist = torch.nn.MSELoss().type(self.dtype)
        self.ssim = a.SSIMLoss().type(self.dtype)

        # training parameters
        self.history = {'loss': [], 'psnr': [], 'ssim': [], 'ncc_w': [], 'ncc_d': [], 'vgg19': []}
        self.iiter = 0
        self.reload_counter = 0
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
        self.mask = None
        self.mask_tensor = None
        self.edges = None

        # build input tensors
        self.input_tensor = None
        self.input_tensor_old = None
        self.additional_noise_tensor = None
        self._build_input()

        # build network
        self.net = None
        self.parameters = None
        self.num_params = None
        self.scheduler = None
        self.optimizer = None

        if self.args.dncnn > 0. or self.args.nccd:
            self.DnCNN = a.DnCNN().to(self.input_tensor.device)
        if self.args.perc > 0.:
            self.vgg19loss = a.PerceptualLoss(
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
        elif self.args.network == 'multires':
            self.net = a.MulResUnet(num_input_channels=self.args.input_depth,
                                    num_output_channels=self.img_shape[-1],
                                    num_channels_down=self.args.filters,
                                    num_channels_up=self.args.filters,
                                    num_channels_skip=self.args.skip,
                                    upsample_mode=self.args.upsample,  # default is bilinear
                                    need_sigmoid=self.args.need_sigmoid,
                                    need_bias=True,
                                    pad=self.args.pad,  # default is reflection, but Fantong uses zero
                                    act_fun=self.args.activation  # default is LeakyReLU).type(self.dtype)
                                    ).type(self.dtype)
        else:
            raise ValueError('ERROR! The network has to be either unet or skip or multires')
        self.parameters = get_params('net', self.net, self.input_tensor)
        self.num_params = sum(np.prod(list(p.size())) for p in self.net.parameters())

    def build_mask(self, strategy='all', sigma=3):
        """Create a random mask either on the whole image or on the edges or on the flat zones"""

        if strategy == 'edge':
            if self.edges is None:
                self._extract_edges(sigma)
            rnd = (np.random.rand(np.sum(self.edges)) > self.args.deletion).astype(int)
            randomized_edge = np.ones_like(self.edges).astype(int)
            i = 0
            for r in range(self.edges.shape[0]):
                for c in range(self.edges.shape[1]):
                    if self.edges[r, c] == 1:
                        # print('edge in (%d, %d)' % (r, c))
                        randomized_edge[r, c] = rnd[i]
                        i += 1

            self.mask = randomized_edge.reshape(self.edges.shape)
            self.mask_tensor = u.numpy2torch(self.mask[np.newaxis, np.newaxis])

        elif strategy == 'flat':
            if self.edges is None:
                self._extract_edges(sigma)
            self.mask = (np.random.rand(np.prod(self.img_shape[:2])) > self.args.deletion).astype(int).reshape(self.img_shape[:2])
            for r in range(self.edges.shape[0]):
                for c in range(self.edges.shape[1]):
                    if self.edges[r, c] == 1:
                        self.mask[r, c] = 1
            self.mask_tensor = u.numpy2torch(self.mask[np.newaxis, np.newaxis])

        elif strategy == 'all':
            mask = (np.random.rand(np.prod(self.img_shape[:2])) > self.args.deletion).astype(int)
            self.mask = mask.reshape(self.img_shape[:2])
            self.mask_tensor = u.numpy2torch(self.mask[np.newaxis, np.newaxis])

        else:
            raise ValueError('Invalid strategy for random mask generation')

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
        self.prnu_clean = loadmat(os.path.join(device_path, 'prnu%s.mat' % ('_comp' if self.args.jpg else
                                                                            '')))['prnu']
        if self.prnu_clean.shape != self.img_shape[:2]:
            raise ValueError('The loaded clean PRNU shape has to be', self.img_shape[:2])

        # filtered PRNU for computing the NCC
        self.prnu_4ncc = loadmat(os.path.join(device_path, 'prnuZM_W%s.mat'
                                              % ('_comp' if self.args.jpg else '')))['prnu']
        if self.prnu_4ncc.shape != self.img_shape[:2]:
            raise ValueError('The loaded filtered PRNU shape has to be', self.img_shape[:2])

        # create relative tensors
        self.prnu_clean_tensor = u.numpy2torch(self.prnu_clean[np.newaxis, np.newaxis])
        self.prnu_4ncc_tensor = u.numpy2torch(self.prnu_4ncc[np.newaxis, np.newaxis])

    def _extract_edges(self, sigma=3):
        self.edges = binary_dilation(canny(u.rgb2gray(self.img), sigma=sigma))

    def _optimization_loop(self):
        if self.args.param_noise:  # TODO fix param_noise
            for n in [x for x in self.net.parameters() if len(x.size()) == 4]:
                _n = n.detach().clone().normal_(std=float(n.std())/50)
                n = n + _n

        input_tensor = self.input_tensor_old
        if self.args.reg_noise_std > 0:
            input_tensor = self.input_tensor_old + (self.additional_noise_tensor.normal_() * self.args.reg_noise_std)

        output_tensor = self.net(input_tensor)

        if self.args.dncnn > 0. or self.args.nccd:
            noise_dncnn = self.DnCNN(u.rgb2gray(output_tensor, 1))

        if self.args.gamma == 0.:  # MSE between reference image and output image
            mse = self.l2dist(self.mask_tensor * output_tensor, self.mask_tensor * self.img_tensor)
        else:  # MSE between reference image and output image with true PRNU (weighted by gamma)
            mse = self.l2dist(
                self.mask_tensor * u.add_prnu(output_tensor, self.prnu_clean_tensor, weight=self.args.gamma),
                self.mask_tensor * self.img_tensor)

        dncnn_loss = u.ncc(self.prnu_4ncc_tensor * u.rgb2gray(output_tensor, 1), noise_dncnn) if self.args.dncnn else 0.
        ssim_loss = (1 - self.ssim(output_tensor, self.img_tensor)) if self.args.ssim else 0.
        perc_loss = self.vgg19loss(input=output_tensor, target=self.img_tensor) if self.args.perc else 0.

        total_loss = mse + self.args.ssim * ssim_loss + self.args.dncnn * dncnn_loss + self.args.perc * perc_loss
        total_loss.backward()

        # gradient clipping
        if self.args.gradient_clip is not None:
            clip_grad_value_(self.net.parameters(), self.args.gradient_clip)

        # Save and display loss terms
        self.history['loss'].append(total_loss.item())
        msg = "\tPicture %s%s, %d reload \tIter %s, Loss=%.2e, MSE=%.2e" \
              % (self.imgpath.split('/')[-1],
                 ', attempt %s' % str(self.attempt).zfill(u.ten_digit(self.args.attempts)) if self.args.attempts != 1 else '',
                 self.reload_number,
                 str(self.iiter + 1).zfill(u.ten_digit(self.args.epochs)),
                 self.history['loss'][-1],
                 mse.item())

        if self.args.nccd:  # compute also the final NCC with DnCNN
            self.history['ncc_d'].append(u.ncc(self.prnu_4ncc_tensor * u.rgb2gray(output_tensor, 1), noise_dncnn).item())
            msg += ', NCC_d=%+.4f' % self.history['ncc_d'][-1]

        if self.args.perc:
            self.history['vgg19'].append(perc_loss.item())
            msg += ', VGG19=%.2e' % self.history['vgg19'][-1]

        # Save and display evaluation metrics
        self.history['psnr'].append(u.psnr(output_tensor * 255, self.img_tensor * 255, 1).item())
        msg += ', PSNR=%2.2f dB' % self.history['psnr'][-1]

        self.history['ssim'].append(u.ssim(self.img_tensor, output_tensor).item())
        msg += ', SSIM=%.4f' % self.history['ssim'][-1]

        # CPU operations
        out_img = np.swapaxes(u.torch2numpy(output_tensor).squeeze(), 0, -1)

        if self.args.nccw_runtime:
            self.history['ncc_w'].append(u.ncc(self.prnu_4ncc * u.float2png(u.prnu.rgb2gray(out_img)),
                                         u.prnu.extract_single(u.float2png(out_img), sigma=3)))
            msg += ', NCC_w=%+.4f' % self.history['ncc_w'][-1]
        else:
            self.out_list.append(out_img)

        print(colored(msg, 'yellow'), '\r', end='')

        # model checkpoint
        if self.args.reload_patience != 0:
            if self.iiter == 0:
                self.best_loss = self.history['loss'][-1]
            else:
                if self.history['loss'][-1] < self.best_loss:
                    self._save_model(self.history['loss'][-1])
                    self.reload_counter = 0
                    self.best_loss = self.history['loss'][-1]
                else:  # load last best model if the loss does not improve over its best value after a certain patience
                    self.reload_counter += 1
                    if self.reload_counter >= self.args.reload_patience:
                        self.reload_number += 1
                        checkpoint = torch.load(self.checkpoint_file)
                        self.net.load_state_dict(checkpoint['net'])
                        self.optimizer.load_state_dict(checkpoint['opt'])
                        for group in self.optimizer.param_groups:
                            group['lr'] *= self.args.lr_factor
                        if self.scheduler:
                            self.scheduler.load_state_dict(checkpoint['sched'])
                        self.reload_counter = 0

        # save if the PSNR is increasing (above a threshold) and only every tot iterations
        if self.psnr_max < self.history['psnr'][-1]:
            self.psnr_max = self.history['psnr'][-1]
            if self.args.save_png_every and \
                    self.psnr_max > self.args.psnr_min and \
                    self.iiter > 0 \
                    and self.saving_interval >= self.args.save_png_every:
                self.out_img = out_img
                outname = self.image_name + '_i' + str(self.iiter).zfill(u.ten_digit(self.args.epochs))
                Image.fromarray(u.float2png(self.out_img)).save(os.path.join(self.outpath, outname))
                self.saving_interval = 0

        # save last image if none of the above conditions are respected
        if self.out_img is None and self.iiter == self.args.epochs:
            self.out_img = out_img
        #     outname = self.image_name + '_i' + str(self.iiter).zfill(u.ten_digit(self.args.epochs))
        #     Image.fromarray(u.float2png(self.out_img)).save(os.path.join(self.outpath, outname))

        self.iiter += 1
        self.saving_interval += 1

        return total_loss

    def _build_scheduler(self, optimizer):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=self.args.lr_factor,
                                                                    threshold=self.args.lr_thresh,
                                                                    patience=self.args.lr_patience)

    def optimize(self):
        start = time()
        # optimize(self.parameters, self._optimization_loop, self.args)

        if self.args.optimizer in ['LBFGS', 'lbfgs']:
            # Do several steps with adam first
            optimizer = torch.optim.Adam(self.parameters, lr=0.001, amsgrad=True)
            for j in range(100):
                optimizer.zero_grad()
                self._optimization_loop()
                optimizer.step()

            def closure2():
                optimizer.zero_grad()
                return self._optimization_loop()

            self.optimizer = torch.optim.LBFGS(self.parameters, max_iter=self.args.epochs, lr=self.args.lr,
                                               tolerance_grad=-1, tolerance_change=-1)
            self.optimizer.step(closure2)

        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr, amsgrad=True)
            if self.args.use_scheduler:
                self._build_scheduler(self.optimizer)
            for j in range(self.args.epochs):
                self.optimizer.zero_grad()
                loss = self._optimization_loop()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step(loss)

        elif self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters, lr=self.args.lr,
                                             momentum=0, dampening=0, weight_decay=0, nesterov=False)
            if self.args.use_scheduler:
                self._build_scheduler(self.optimizer)
            for j in range(self.args.epochs):
                self.optimizer.zero_grad()
                loss = self._optimization_loop()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step(loss)
        else:
            assert False

        self.elapsed = time() - start

    def save_result(self, attempt, save_images=False):
        mydict = {
            'server': u.machine_name(),
            'device': os.environ["CUDA_VISIBLE_DEVICES"],
            'elapsed time': u.sec2time(self.elapsed),
            'history': self.history,
            'args': self.args,
            'prnu': self.prnu_clean,
            'prnu4ncc': self.prnu_4ncc,
            'image': self.img,
            'mask': self.mask,
            'edge': self.edges,
            'anonymized': self.out_img,
            'psnr_max': self.psnr_max,
            'params': self.num_params,
            'attempt': attempt
        }
        outname = self.image_name.split('/')[-1] + '_run%s' % str(attempt).zfill(u.ten_digit(self.args.attempts))
        np.save(os.path.join(self.outpath, outname + '.npy'), mydict)

        if save_images:
            with h5py.File(os.path.join(self.outpath, outname + '.hdf5'), 'w') as f:
                dset = f.create_dataset("all_outputs", data=np.asarray(self.out_list))

    def _save_model(self, loss):
        outname = self.image_name.split('/')[-1] + '_run%s.pth' % str(self.attempt).zfill(u.ten_digit(self.args.attempts))
        self.checkpoint_file = os.path.join(self.outpath, outname)
        state = dict(net=self.net.state_dict(),
                     opt=self.optimizer.state_dict(),
                     sched=self.scheduler.state_dict() if self.scheduler else None,
                     loss=loss,
                     epoch=self.iiter)
        torch.save(state, self.checkpoint_file)

    def compute_nccw(self):
        assert len(self.out_list) > 0, "Out list is empty"
        self.history['ncc_w'] = []
        print('\n')
        for o in trange(len(self.out_list), ncols=90,  unit='epoch',
                        desc='\tComputing NCC'):
            self.history['ncc_w'].append(u.ncc(self.prnu_4ncc * u.float2png(u.prnu.rgb2gray(self.out_list[o])),
                                         u.prnu.extract_single(u.float2png(self.out_list[o]), sigma=3)))

    def reset(self):
        self.iiter = 0
        self.reload_counter = 0
        self.saving_interval = 0
        print('')
        torch.cuda.empty_cache()
        self._build_input()
        self._build_model()
        self.history = {'loss': [], 'psnr': [], 'ssim': [], 'ncc_w': [], 'ncc_d': [], 'vgg19': []}
        self.out_list = []
        self.mask = None
        self.edges = None
        self.optimizer = None
        self.scheduler = None


def _parse_args():
    parser = ArgumentParser()
    # dataset parameter
    parser.add_argument('--device', nargs='+', type=str, required=False, default='all',
                        help='Device name')
    parser.add_argument('--dataset', type=str, required=False, default='dataset300',
                        choices=['dataset300', 'dataset600', 'dataset13'],
                        help='Dataset to be used')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (lowest memory usage based)')
    parser.add_argument('--pics_idx', nargs='+', type=int, required=False,
                        help='indeces of the first and last pictures to be processed'
                             '(e.g. 10, 15 to process images from the 10th to the 15th)')
    parser.add_argument('--pics_IDs', nargs='+', type=str, required=False,
                        help='5-long code of the picture to be loaded')
    parser.add_argument('--outpath', type=str, required=False, default='debug',
                        help='Run name in ./results/')
    parser.add_argument('--jpg', action='store_true',
                        help='Use the JPEG dataset')
    parser.add_argument('--slack', action='store_true',
                        help='Send a message to slack')
    parser.add_argument('--save_outputs', type=bool, default=False, required=False,
                        help='Save every network output to disk in a hdf5 file.')
    parser.add_argument('--nccw_runtime', type=bool, default=False, required=False,
                        help='Compute NCC at runtime (it slows down the training phase as it is done on CPU)')
    parser.add_argument('--nccw_skip', type=bool, default=True, required=False,
                        help='Skip the computation of NCC (and thus save the output image list)')
    parser.add_argument('--seeds', nargs='+', type=int, required=False,
                        help='Random Seed list for each attempt (default 0 for every attempt).')
    # network design
    parser.add_argument('--network', type=str, required=False, default='multires', choices=['unet', 'skip', 'multires'],
                        help='Name of the network to be used')
    parser.add_argument('--activation', type=str, default='LeakyReLU', required=False,
                        help='Activation function to be used in the convolution block [ReLU, Tanh, LeakyReLU]')
    parser.add_argument('--need_sigmoid', type=bool, required=False, default=True,
                        help='Apply a sigmoid activation to the network output')
    parser.add_argument('--filters', nargs='+', type=int, required=False, default=[16, 32, 64, 128, 256],
                        help='Numbers of channels')
    parser.add_argument('--skip', nargs='+', type=int, required=False, default=[16, 32, 64, 128],
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
    parser.add_argument('--reg_noise_std', type=float, required=False, default=0.1,
                        help='Standard deviation of the normal noise to be added to the input every epoch')
    parser.add_argument('--noise_dist', type=str, default='normal', required=False, choices=['normal', 'uniform'],
                        help='Type of noise for the input tensor')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Standard deviation of the noise for the input tensor')
    parser.add_argument('--deletion', type=float, default=0., required=False,
                        help='Deletion rate for the mask in [0,1].')
    parser.add_argument('--mask_strategy', type=str, default='all', required=False, choices=['all', 'edges', 'flat'],
                        help='Build the random mask upon the edges, the flat zones or the whole image.')
    parser.add_argument('--edges_sigma', type=float, default=3., required=False,
                        help='Sigma value for edge detection canny algorithm.')
    parser.add_argument('--attempts', type=int, default=1, required=False,
                        help='Number of attempts to be performed on the same picture.')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use ReduceLROnPlateau scheduler.')
    parser.add_argument('--lr_factor', type=float, default=.9, required=False,
                        help='LR reduction for Plateau scheduler.')
    parser.add_argument('--lr_thresh', type=float, default=1e-4, required=False,
                        help='LR threshold for Plateau scheduler.')
    parser.add_argument('--lr_patience', type=int, default=10, required=False,
                        help='LR patience for Plateau scheduler.')
    parser.add_argument('--gradient_clip', type=float, required=False,
                        help='Gradient clipping value.')
    parser.add_argument('--reload_patience', type=int, default=0, required=False,
                        help='Number of epoch to be waited before reloading the saved model checkpoint.')
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = [0] * args.attempts
    elif len(args.seeds) == 1:
        args.seeds = [args.seeds[0]] * args.attempts
    assert len(args.seeds) == args.attempts, 'Provided seed list has to have a length of the attempts'
    if args.nccw_skip:
        args.save_outputs = True
    return args


def main():
    args = _parse_args()

    # set the engine to be used
    u.set_gpu(args.gpu)

    # create output folder
    outpath = os.path.join('./results/', 'comp' if args.jpg else '', args.outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print(colored('Saving to %s' % outpath, 'yellow'))
    with open(os.path.join(outpath, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=4)

    T = Training(args, dtype, outpath)

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

        T.load_prnu(device)

        if args.pics_IDs is not None:  # load specified image
            pic_list = [os.path.join(device, device.split('/')[-1] + '_%s.%s'
                                     % (_, 'JPG' if args.jpg else 'png'))
                        for _ in args.pics_IDs]
        else:
            if args.jpg:
                pic_list = glob(os.path.join(device, '*.JPG'))[pics_idx[0]:pics_idx[-1]]
            else:
                pic_list = glob(os.path.join(device, '*.png'))[pics_idx[0]:pics_idx[-1]]
            pic_list = sorted(pic_list)

        for picpath in pic_list:
            for attempt in range(args.attempts):
                _set_seed(args.seeds[attempt])
                T._build_model()
                T.load_image(picpath)
                T.build_mask(strategy=args.mask_strategy, sigma=args.edges_sigma)
                T.attempt = attempt
                T.optimize()
                if not args.nccw_runtime and not args.nccw_skip:  # compute NCC on CPU at the end of the optimization
                    T.compute_nccw()
                T.save_result(attempt, args.save_outputs)
                T.reset()

    print(colored('Anonymization done!', 'yellow'))

    if args.slack:
        os.system(
            'python /nas/home/fpicetti/slack.py -u francesco.picetti -m "Finished _dip_prnu_anonymizer_ with \`%s\`"'
            % ' '.join(sys.argv).split('.py')[-1][1:])


if __name__ == '__main__':
    main()
