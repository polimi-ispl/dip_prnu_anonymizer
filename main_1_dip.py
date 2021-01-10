from __future__ import print_function
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import h5py
import matplotlib.image as mpimg
from tqdm import trange
from argparse import ArgumentParser
from time import time
from termcolor import colored
from glob import glob

import architectures as a
import utils as u


def _set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Training:
    def __init__(self, args, dtype, outpath, img_shape=(512, 512, 3)):
        self.args = args
        self.dtype = dtype
        self.outpath = outpath
        self.img_shape = img_shape
        self.out_list = []

        # losses
        self.l2dist = torch.nn.MSELoss().type(self.dtype)
        self.ssim = a.SSIMLoss().type(self.dtype)
        if self.args.dncnn > 0.:
            self.DnCNN = a.DnCNN().type(dtype)

        # training parameters
        self.history = {'loss': [], 'psnr': [], 'ssim': [], 'ncc': [], 'lr': [], 'gamma': []}
        self.iiter = 0

        # data
        self.imgpath = None
        self.image_name = None
        self.img = None
        self.img_tensor = None
        self.prnu_injection = None
        self.prnu_injection_tensor = None
        self.prnu_4ncc = None
        self.prnu_4ncc_tensor = None

        # build input tensors
        self.input_tensor = None
        self.additional_noise_tensor = None
        self._build_input()

        # build network
        self.net = None
        self.parameters = None
        self.scheduler = None
        self.optimizer = None

    @property
    def dev(self):
        return self.input_tensor.device

    def _build_input(self):
        self.input_tensor = a.get_noise(self.args.input_depth,
                                        'noise',
                                        self.img_shape[:2],
                                        noise_type=self.args.noise_dist,
                                        var=self.args.noise_std).type(a.dtype)
        self.additional_noise_tensor = self.input_tensor.detach().clone()

    def build_model(self):
        self.net = a.MultiResInjection(a.MulResUnet(num_input_channels=self.args.input_depth,
                                                    num_output_channels=self.img_shape[-1],
                                                    num_channels_down=self.args.filters,
                                                    num_channels_up=self.args.filters,
                                                    num_channels_skip=self.args.skip,
                                                    upsample_mode=self.args.upsample,  # default is bilinear
                                                    need_sigmoid=self.args.need_sigmoid,
                                                    need_bias=True,
                                                    pad=self.args.pad,
                                                    # default is reflection, but Fantong uses zero
                                                    act_fun=self.args.activation
                                                    # default is LeakyReLU).type(self.dtype)
                                                    ).type(self.dtype),
                                       self.prnu_injection_tensor,
                                       gamma_init=0.)

        self.parameters = a.get_params('net', self.net, self.input_tensor)

    def load_image(self, image_path):
        self.imgpath = image_path
        self.image_name = self.imgpath.split('.')[-2].split('/')[-1]

        _ext = os.path.splitext(self.imgpath)[-1].lower()

        if _ext == '.npy':
            self.img = u.normalize(np.load(self.imgpath), zero_mean=False)[0]
        elif _ext == '.png':  # imread normalizes to 0, 1
            self.img = mpimg.imread(self.imgpath)
        elif _ext in ['.jpeg', '.jpg']:
            self.img = u.normalize(mpimg.imread(self.imgpath), in_min=0, in_max=255, zero_mean=False)[0]
        else:
            raise ValueError('Invalid image file extension: it has to be npy, png or jpg')

        self.img = u.crop_center(self.img, self.img_shape[0], self.img_shape[1])

        if self.img.shape != self.img_shape:
            raise ValueError('The loaded image shape has to be', self.img_shape)
        self.img_tensor = u.numpy2torch(np.swapaxes(self.img, 2, 0)[np.newaxis]).to(self.dev)

    def load_prnu(self, device_path):
        # fingerprint to be added to the output
        if self.args.prnu == 'aware':
            self.prnu_injection = np.load(os.path.join(device_path, 'prnu.npy'))
        elif self.args.prnu == 'blind':
            assert self.img is not None, 'No image has been loaded'
            if 'float' in device_path:
                self.prnu_injection = u.prnu.extract_single(self.img, sigma=3 / 255)
            else:
                self.prnu_injection = u.prnu.extract_single(u.float2png(self.img), sigma=3)
        else:
            raise ValueError('PRNU policy has to be either aware or blind')

        if self.prnu_injection.shape != self.img_shape[:2]:
            raise ValueError('The loaded PRNU shape has to be', self.img_shape[:2])
        self.prnu_injection_tensor = u.numpy2torch(self.prnu_injection[np.newaxis, np.newaxis]).to(self.dev)

        # Reference PRNU for computing the NCC
        self.prnu_4ncc = np.load(os.path.join(device_path, 'prnu.npy'))
        self.prnu_4ncc_tensor = u.numpy2torch(self.prnu_4ncc[np.newaxis, np.newaxis]).to(self.dev)

    def _optimization_loop(self):
        # add noise to net parameters for regularizing the inversion
        if self.args.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) == 4]:
                _n = n.detach().clone().normal_(std=float(n.std()) / 50)
                n = n + _n

        # add noise to input tensor for regularizing the inversion
        input_tensor = self.input_tensor
        if self.args.reg_noise_std > 0:
            input_tensor = input_tensor + (self.additional_noise_tensor.normal_() * self.args.reg_noise_std)

        # compute output
        output_tensor = self.net(input_tensor)

        # compute loss
        mse = self.l2dist(output_tensor, self.img_tensor)
        mse.backward()

        # Save and display loss terms
        self.history['loss'].append(mse.item())
        msg = "\tPicture %s, Iter %s, Loss=%.2e" \
              % (self.imgpath.split('/')[-1],
                 str(self.iiter + 1).zfill(u.ten_digit(self.args.epochs)),
                 self.history['loss'][-1])

        # Save and display evaluation metrics
        self.history['psnr'].append(u.psnr(u.float2png(output_tensor),
                                           u.float2png(self.img_tensor), 1).item())
        msg += ', PSNR = %2.2f dB' % self.history['psnr'][-1]

        self.history['ssim'].append(u.ssim(self.img_tensor, output_tensor).item())
        msg += ', SSIM = %.4f' % self.history['ssim'][-1]

        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

        # Save gamma and output image
        self.history['gamma'].append(float(self.net.prnu_injection.weight.detach().cpu().numpy().squeeze()))

        if 'float' in self.imgpath:
            out_img = np.swapaxes(u.torch2numpy(output_tensor).squeeze(), 0, -1)
        else:
            out_img = u.float2png(np.swapaxes(u.torch2numpy(output_tensor).squeeze(), 0, -1))
        self.out_list.append(out_img)

        # compute NCC if requested
        if self.args.ncc == 'runtime':
            self.history['ncc'].append(u.ncc(self.prnu_4ncc * u.float2png(u.rgb2gray(out_img)),
                                             u.prnu.extract_single(out_img, sigma=3)))
            msg += ', NCC = %+.6f' % self.history['ncc'][-1]

        print(colored(msg, 'yellow'), '\r', end='')

        # model checkpoint
        exit_flag = False
        if self.args.psnr_max is not None:
            if self.history['psnr'][-1] > self.args.psnr_max:  # stop the optimization if the PSNR is above a threshold
                exit_flag = True

        self.iiter += 1

        return mse, exit_flag

    def _build_scheduler(self, optimizer):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=self.args.lr_factor,
                                                                    threshold=self.args.lr_thresh,
                                                                    patience=self.args.lr_patience)

    def optimize(self):
        start = time()
        # optimize(self.parameters, self._optimization_loop, self.args)

        if self.args.optimizer.lower() == 'lbfgs':
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

        elif self.args.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr, amsgrad=True)
            if self.args.use_scheduler:
                self._build_scheduler(self.optimizer)
            for j in range(self.args.epochs):
                self.optimizer.zero_grad()
                loss, exit_flag = self._optimization_loop()
                if exit_flag:
                    break

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step(loss)
                # force gamma to be positive
                with torch.set_grad_enabled(False):
                    self.net.prnu_injection.weight.clamp_(0)

        elif self.args.optimizer.lower() == 'sgd':
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

    def save_result(self):
        mydict = {
            'device': u.get_gpu_name(int(os.environ["CUDA_VISIBLE_DEVICES"])),
            'elapsed time': u.sec2time(self.elapsed),
            'history': self.history,
            'args': self.args,
            'prnu': self.prnu_injection,
            'prnu4ncc': self.prnu_4ncc,
            'image': self.img,
        }
        outname = self.image_name.split('/')[-1]
        np.save(os.path.join(self.outpath, outname + '.npy'), mydict)

        # save output images
        with h5py.File(os.path.join(self.outpath, outname + '.hdf5'), 'w') as f:
            dset = f.create_dataset("all_outputs", data=np.asarray(self.out_list))

    def _save_model(self, loss):
        outname = self.image_name.split('/')[-1] + '.pth'
        self.checkpoint_file = os.path.join(self.outpath, outname)
        state = dict(net=self.net.state_dict(),
                     opt=self.optimizer.state_dict(),
                     sched=self.scheduler.state_dict() if self.scheduler else None,
                     loss=loss,
                     epoch=self.iiter)
        torch.save(state, self.checkpoint_file)

    def compute_ncc(self):
        assert len(self.out_list) > 0, "Out list is empty"
        self.history['ncc'] = []
        print('\n')
        for o in trange(len(self.out_list), ncols=90, unit='epoch', desc='\tComputing NCC'):
            self.history['ncc'].append(u.ncc(self.prnu_4ncc * u.float2png(u.prnu.rgb2gray(self.out_list[o])),
                                             u.prnu.extract_single(u.float2png(self.out_list[o]), sigma=3)))

    def reset(self):
        self.iiter = 0
        print('')
        torch.cuda.empty_cache()
        self._build_input()
        self.build_model()
        self.history = {'loss': [], 'psnr': [], 'ssim': [], 'ncc': [], 'lr': [], 'gamma': []}
        self.out_list = []
        self.optimizer = None
        self.scheduler = None


def _parse_args():
    parser = ArgumentParser()
    # dataset parameter
    parser.add_argument('--device', nargs='+', type=str, required=False, default='all',
                        help='Device name')
    parser.add_argument('--dataset', type=str, required=False, default='dresden_sample',
                        help='Dataset to be used')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (lowest memory usage based)')
    parser.add_argument('--pics_idx', nargs='+', type=int, required=False,
                        help='indeces of the first and last pictures to be processed'
                             '(e.g. 10, 15 to process images from the 10th to the 15th)')
    parser.add_argument('--outpath', type=str, required=False, default='debug',
                        help='Run name in ./results/')
    parser.add_argument('--ncc', type=str, required=False, default='skip',
                        choices=['skip', 'end', 'runtime'],
                        help='When to compute NCC (being based on Wiener filtering, it is done on CPU)')
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
    # optimizer
    parser.add_argument('--epochs', '-e', type=int, required=False, default=10001,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'lbfgs', 'sgd'],
                        help='Optimizer to be used')
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                        help='Use ReduceLROnPlateau scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                        help='Learning Rate for Adam optimizer')
    parser.add_argument('--lr_factor', type=float, default=.9, required=False,
                        help='LR reduction for Plateau scheduler')
    parser.add_argument('--lr_thresh', type=float, default=1e-4, required=False,
                        help='LR threshold for Plateau scheduler')
    parser.add_argument('--lr_patience', type=int, default=10, required=False,
                        help='LR patience for Plateau scheduler')
    # deep prior strategies
    parser.add_argument('--param_noise', action='store_true', default=False,
                        help='Add normal noise to the parameters every epoch')
    parser.add_argument('--reg_noise_std', type=float, required=False, default=0.1,
                        help='Standard deviation of the normal noise to be added to the input every epoch')
    parser.add_argument('--noise_dist', type=str, default='normal', required=False, choices=['normal', 'uniform'],
                        help='Type of noise for the input tensor')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Standard deviation of the noise for the input tensor')
    parser.add_argument('--psnr_max', type=float, required=False, default=39.,
                        help='Maximum PSNR for stopping the optimization')
    parser.add_argument('--prnu', type=str, default='aware', required=False,
                        choices=['aware', 'blind'],
                        help='PRNU injection strategy')

    args = parser.parse_args()
    if args.ncc == 'skip':
        args.save_outputs = True
    return args


def main():
    args = _parse_args()

    # set the engine to be used
    u.set_gpu(args.gpu)
    _set_seed(0)

    # create output folder
    outpath = os.path.join('./results/', args.outpath)
    os.makedirs(outpath, exist_ok=True)
    u.write_args(os.path.join(outpath, 'args.txt'), args)

    # instantiate run object
    T = Training(args, a.dtype, outpath)

    # create list of pictures to be processed
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

        # load fingerprint if not extracted from the image itself
        if args.prnu != 'blind':
            T.load_prnu(device)

        pic_list = sorted(glob(os.path.join(device, '*')))
        pic_list = pic_list[pics_idx[0]:pics_idx[-1]]

        # now we have a list of pictures to be processed, let's go!
        for picpath in pic_list:
            if os.path.splitext(picpath)[1].lower() not in ['.jpg', '.jpeg', '.png']:
                break
            T.load_image(picpath)

            if args.prnu == 'blind':
                T.load_prnu(device)

            T.build_model()

            T.optimize()
            if args.ncc == 'end':  # compute NCC on CPU at the end of the optimization
                T.compute_ncc()
            T.save_result()
            T.reset()

    print(colored('Anonymization done! Saved to %s' % outpath, 'yellow'))


if __name__ == '__main__':
    main()
