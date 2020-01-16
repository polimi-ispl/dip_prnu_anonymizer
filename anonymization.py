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


def main() -> int:
    parser = ArgumentParser()
    # dataset parameter
    parser.add_argument('--path', type=str, required=False, help='Dataset path')
    parser.add_argument('--in_path', '-i', type=str, required=True,
                        help='Input image name with file extension')
    parser.add_argument('--out_path', '-o', type=str, required=False, default='./cnn',
                        help='model output path')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (lowest memory usage based)')
    # network design
    parser.add_argument('--network', '-n', type=str, required=False, default='unet',
                        help='Name of the network to be used [unet, skip]')
    parser.add_argument('--activation', type=str, default='ReLU', required=False,
                        help='Activation function to be used in the convolution block [ReLU, Tanh]')
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
    parser.add_argument('--lr', type=float, default=0.001, required=False,
                        help='Learning Rate for Adam optimizer')
    parser.add_argument('--save_every', type=int, default=50, required=False,
                        help='Number of epochs every which to save the results')
    parser.add_argument('--param_noise', action='store_true',
                        help='Add normal noise to the parameters every epoch')
    parser.add_argument('--reg_noise_std', type=float, required=False, default=0.03,
                        help='Add normal noise to the input every epoch')
    args = parser.parse_args()

    # set the engine to be used
    u.set_gpu(args.gpu)

    # load data
    dataset_path = args.path if args.path is not None else \
        '/nas/home/smandelli/ispg-greeneyes-data/icip2017_ImageAnonymizer/online_code/single_code/dataset'
    img = mpimg.imread(os.path.join(dataset_path, args.in_path))
    k = loadmat(os.path.join(dataset_path, args.in_path, 'prnu_cropped512.mat'))['prnu']

    img_var = u.numpy2torch(np.swapaxes(img, 2, 0)[np.newaxis])
    k_var = u.numpy2torch(k[np.newaxis, np.newaxis])

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
                   num_channels_skip=[0, 0, 0, 4, 4],
                   upsample_mode=args.upsample,  # default is bilinear
                   need_sigmoid=args.need_sigmoid,
                   need_bias=True,
                   pad=args.pad,  # default is reflection
                   act_fun=args.activation  # default is LeakyReLU
                   ).type(dtype)
    else:
        raise ValueError('ERROR! The network has to be either unet or skip')

    print('Number of params: %d' % sum(np.prod(list(p.size())) for p in net.parameters()))

    l2dist = torch.nn.MSELoss().type(dtype)
    iiter = 0

    def closure():

        global iiter
        global out_np

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

        print('Iteration %5d, Loss = %.2e, PSNR = %.2f dB, SSIM = %.4f, NCC = %.6f'
              % (iiter, total_loss.item(),
                 u.psnr(img_var, out),
                 u.compute_ssim(img_var, out),
                 u.ncc(k, u.extract_prnu(out_np), use_torch=False)),
              '\r', end='')

        if iiter % args.save_every == 0:
            plt.figure(figsize=(16, 6))
            plt.subplot(121), plt.imshow(out_np), plt.title('Anonymized')
            plt.subplot(122), plt.imshow(img), plt.title('Original')
            plt.show()

        iiter += 1

        return total_loss

    # Training loop
    net_input = get_noise(args.input_depth, 'noise', img.shape).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params('net', net, net_input)
    optimize('adam', p, closure, args.lr, args.epochs)


if __name__ == '__main__':
    main()
