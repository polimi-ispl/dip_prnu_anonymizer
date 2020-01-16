from __future__ import print_function
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
from models.unet2 import UNet
from models.skip import skip
from utils.common_utils import *
from utils import utils as u

from argparse import ArgumentParser
from collections import namedtuple
from time import time
from termcolor import colored


def main() -> int:
    parser = ArgumentParser()
    # dataset parameter
    parser.add_argument('--datapath', type=str, required=False, default='./dataset/Nikon_D70s_1/',
                        help='Dataset path')
    parser.add_argument('--input_image', '-i', type=str, required=False, default='Nikon_D70s_1_22903.png',
                        help='Name of the image file with extension')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (lowest memory usage based)')
    # network design
    parser.add_argument('--network', type=str, required=False, default='unet',
                        help='Name of the network to be used [unet, skip]')
    parser.add_argument('--activation', type=str, default='ReLU', required=False,
                        help='Activation function to be used in the convolution block [ReLU, Tanh, LeakyReLU]')
    parser.add_argument('--need_sigmoid', type=bool, required=False, default=True,
                        help='Apply a sigmoid activation to the network output')
    parser.add_argument('--filters', nargs='+', type=int, required=False, default=[16, 32, 64, 128, 256],
                        help='Numbers of channels')
    parser.add_argument('--skip', nargs='+', type=int, required=False, default=[0, 0, 0, 4, 4],
                        help='Number of channels for skip')
    parser.add_argument('--input_depth', type=int, required=False, default=512,
                        help='Depth of the input noise tensor')
    parser.add_argument('--pad', type=str, required=True, default='zero',
                        help='Padding strategy for the network [zero, reflection]')
    parser.add_argument('--upsample', type=str, required=True, default='nearest',
                        help='Upgoing deconvolution strategy for the network [nearest, bilinear, deconv]')
    # training parameter
    parser.add_argument('--epochs', '-e', type=int, required=False, default=6001,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                        help='Learning Rate for Adam optimizer')
    parser.add_argument('--save_every', type=int, default=0, required=False,
                        help='Number of epochs every which to save the results')
    parser.add_argument('--param_noise', action='store_true',
                        help='Add normal noise to the parameters every epoch')
    parser.add_argument('--reg_noise_std', type=float, required=False, default=0.03,
                        help='Standard deviation of the normal noise to be added to the input every epoch')
    parser.add_argument('--noise_dist', type=str, default='uniform', required=False,
                        help='Type of noise for the input tensor [normal, uniform]')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Standard deviation of the noise for the input tensor')
    args = parser.parse_args()

    random_code = u.random_code()

    # set the engine to be used
    u.set_gpu(args.gpu)

    # create output folder: ./camera_model/picture_name/random_code/
    outpath = os.path.join(args.datapath, args.input_image, random_code)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load data
    img = mpimg.imread(os.path.join(args.datapath, args.input_image))
    k = loadmat(os.path.join(args.datapath, 'prnu_cropped512.mat'))['prnu']

    # image to tensor
    img_var = u.numpy2torch(np.swapaxes(img, 2, 0)[np.newaxis])
    k_var = u.numpy2torch(k[np.newaxis, np.newaxis])

    # create input noise tensor
    net_input = get_noise(args.input_depth, 'noise', img.shape,
                          noise_type=args.noise_dist, var=args.noise_std).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # create network
    if args.network == 'unet':
        net = UNet(num_input_channels=args.input_depth,
                   num_output_channels=img.shape[-1],
                   filters=args.filters,
                   more_layers=1,  # default is 0
                   concat_x=False,
                   upsample_mode=args.upsample,  # default is nearest
                   activation=args.activation,
                   pad=args.pad,  # default is zero
                   norm_layer=torch.nn.InstanceNorm2d,
                   need_sigmoid=args.need_sigmoid,
                   need_bias=True
                   ).type(dtype)
    elif args.network == 'skip':
        net = skip(num_input_channels=args.input_depth,
                   num_output_channels=img.shape[-1],
                   num_channels_down=args.filters,
                   num_channels_up=args.filters,
                   num_channels_skip=args.skip,
                   upsample_mode=args.upsample,  # default is bilinear
                   need_sigmoid=args.need_sigmoid,
                   need_bias=True,
                   pad=args.pad,  # default is reflection, but Fantong uses zero
                   act_fun=args.activation  # default is LeakyReLU
                   ).type(dtype)
    else:
        raise ValueError('ERROR! The network has to be either unet or skip')

    # define function to be minimized
    l2dist = torch.nn.MSELoss().type(dtype)
    iiter = 0
    metrics = namedtuple('Metrics', ['loss', 'psnr', 'ssim', 'ncc'])
    history = []

    def closure():

        global iiter
        global out_np
        global history

        if args.param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if args.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)

        out = net(net_input)
        total_loss = l2dist(u.add_prnu(out, k_var), img_var)
        total_loss.backward()

        out_np = np.swapaxes(u.torch2numpy(out).squeeze(), 0, -1)

        m = metrics(total_loss.item(),
                    u.psnr(u.float2png(out_np), u.float2png(img)),
                    u.ssim(img, out_np),
                    u.ncc(k, u.prnu.extract_single(u.float2png(out_np))))

        history.append(m)

        print('Iteration %5d, Loss = %.2e, PSNR = %.2f dB, SSIM = %.4f, NCC = %.6f'
              % (iiter, m.loss, m.psnr, m.ssim, m.ncc),
              '\r', end='')

        if args.save_every != 0 and iiter % args.save_every == 0:
            Image.fromarray(u.float2png(out_np)).save(os.path.join(outpath,
                                                                   str(iiter).zfill(u.ten_digit(args.epochs))+'.png'))

        iiter += 1

        return total_loss

    # Training loop
    p = get_params('net', net, net_input)
    start = time()
    optimize('adam', p, closure, args.lr, args.epochs)
    elapsed = time() - start

    # Save results
    ncc_gt = u.ncc(k, u.prnu.extract_single(u.float2png(img)))
    ncc_out = u.ncc(k, u.prnu.extract_single(u.float2png(out_np)))
    mydict = {
        'server': u.machine_name(),
        'device': os.environ["CUDA_VISIBLE_DEVICES"],
        'elapsed time': u.sec2time(elapsed),
        'run_code': random_code,
        'history': history,
        'args': args,
        'ncc_gt': ncc_gt,
        'ncc_out': ncc_out,
        'params': sum(np.prod(list(p.size())) for p in net.parameters())
    }
    print(colored('Saving to: ' + outpath, 'yellow'))
    np.save(os.path.join(outpath, 'run.npy'), mydict)

    print(colored('Training done!', 'yellow'))


if __name__ == '__main__':
    main()
