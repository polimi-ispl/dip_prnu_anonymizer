import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.pytorch_ssim import SSIM
from perceptual_loss import PerceptualLoss
import functools


class Downsampler(nn.Module):
    """
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """

    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None,
                 preserve_size=False):
        super(Downsampler, self).__init__()

        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1. / np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'

        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)

        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch

        self.downsampler_ = downsampler

        if preserve_size:

            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)

            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x
        return self.downsampler_(x)


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']

    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1. / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'

        center = (kernel_width + 1.) / 2.
        print(center, kernel_width)
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.
                dj = (j - center) / 2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):

                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)

                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val


    else:
        assert False, 'wrong method name'

    kernel /= kernel.sum()

    return kernel


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun='LeakyReLU'):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'Tanh':
            return nn.Tanh()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                      preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class UNet(nn.Module):
    """
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    """

    def __init__(self, num_input_channels=3, num_output_channels=3,
                 filters=[16, 32, 64, 128, 256], more_layers=0, concat_x=False,
                 activation='ReLU', upsample_mode='deconv', pad='zero',
                 norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True):
        super(UNet, self).__init__()

        self.more_layers = more_layers
        self.concat_x = concat_x

        if activation == "ReLU":
            act_fun = nn.ReLU()
        elif activation == "Tanh":
            act_fun = nn.Tanh()
        elif activation == "LeakyReLU":
            act_fun = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError("Activation has to be in [ReLU, Tanh, LeakyReLU]")

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels,
                               norm_layer, need_bias, pad, act_fun)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer,
                         need_bias, pad, act_fun) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, act_fun, same_num_filt=True) for i in
                             range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad, act_fun)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad, act_fun)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad, act_fun)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad, act_fun)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if need_sigmoid:
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs):

        # Downsample
        downs = [inputs]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(inputs)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)

        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)

        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)

        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                # print(prevs[-1].size())
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out, downs[kk + 5]], 1)

                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_ = l(up_, prevs[self.more - idx - 2])
        else:
            up_ = down4

        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, act_fun):
        super(unetConv2, self).__init__()

        # print(pad)
        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size), act_fun, )
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size), act_fun, )
        else:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad), act_fun, )
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad), act_fun, )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, act_fun):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, norm_layer, need_bias, pad, act_fun)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, act_fun, same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv = unetConv2(out_size * 2, out_size, None, need_bias, pad, act_fun)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv = unetConv2(out_size * 2, out_size, None, need_bias, pad, act_fun)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2: diff2 + in1_up.size(2), diff3: diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output = self.conv(torch.cat([in1_up, inputs2_], 1))

        return output


def Skip(num_input_channels=2, num_output_channels=3, num_channels_down=[16, 32, 64, 128, 128],
         num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
         filter_size_down=3, filter_size_up=3, filter_skip_size=1,
         need_sigmoid=True, need_bias=True, pad='zero', upsample_mode='nearest', downsample_mode='stride',
         act_fun='LeakyReLU', need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


class DnCNN(nn.Module):
    """
    Denoising CNN
    @Author: Nicolò Bonettini, Luca Bondi
    """

    def __init__(self):
        super(DnCNN, self).__init__()

        net = OrderedDict()
        net['conv1'] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1)
        net['relu1'] = nn.ReLU()
        for i in range(2, 17):
            net['conv{}'.format(i)] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                                padding=1, bias=False)
            net['bn{}'.format(i)] = nn.BatchNorm2d(num_features=64, momentum=0.99, eps=0.001)
            net['relu{}'.format(i)] = nn.ReLU()
        net['conv17'] = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), padding=1)

        for layer in net.keys():
            self.add_module(layer, net[layer])

        self.load_state_dict(torch.load('./DnCNN.pth'))

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        noise = x
        return noise


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


SSIMLoss = SSIM

PercLoss = PerceptualLoss


# MultiResUNet
def conv2dbn(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride', act_fun='LeakyReLU'):
    block = conv(in_f, out_f, kernel_size, stride=stride, bias=bias, pad=pad, downsample_mode=downsample_mode)
    block.add(bn(out_f))
    block.add(act(act_fun))
    return block


class Add(nn.Module):
    def __init__(self, *args):
        super(Add, self).__init__()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        return torch.stack(inputs, dim=0).sum(dim=0)


def MultiResBlockFunc(U, f_in, alpha=1.67, pad='zero', act_fun='LeakyReLU', bias=True):
    W = alpha * U
    out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    model = nn.Sequential()
    deep = nn.Sequential()
    conv3x3 = conv2dbn(f_in, int(W * 0.167), 3, 1,
                       bias=bias, pad=pad, act_fun=act_fun)
    conv5x5 = conv3x3.add(conv2dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                          pad=pad, act_fun=act_fun))
    conv7x7 = conv5x5.add(conv2dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                          pad=pad, act_fun=act_fun))
    shortcut = conv2dbn(f_in, out_dim, 1, 1,
                        bias=bias, pad=pad, act_fun=act_fun)
    deep.add(Concat(1, conv3x3, conv5x5, conv7x7))
    deep.add(bn(out_dim))
    model.add(Add(deep, shortcut))
    model.add(act(act_fun))
    model.add(bn(out_dim))
    return model


class MultiResBlock(nn.Module):
    def __init__(self, U, f_in, alpha=1.67, pad='zero', act_fun='LeakyReLU', bias=True):
        super(MultiResBlock, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = conv2dbn(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1,
                                 bias=bias, pad=pad, act_fun=act_fun)
        self.conv3x3 = conv2dbn(f_in, int(W * 0.167), 3, 1, bias=bias,
                                pad=pad, act_fun=act_fun)
        self.conv5x5 = conv2dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                pad=pad, act_fun=act_fun)
        self.conv7x7 = conv2dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                pad=pad, act_fun=act_fun)
        self.bn1 = bn(self.out_dim)
        self.bn2 = bn(self.out_dim)
        self.accfun = act(act_fun)

    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = self.bn1(torch.cat([out1, out2, out3], dim=1))
        out = torch.add(self.shortcut(input), out)
        out = self.bn2(self.accfun(out))
        return out


class PathRes(nn.Module):
    def __init__(self, f_in, f_out, length, pad='zero', act_fun='LeakyReLU', bias=True):
        super(PathRes, self).__init__()
        self.network = []
        self.network.append(conv2dbn(f_in, f_out, 3, 1, bias=bias, pad=pad, act_fun=act_fun))
        self.network.append(conv2dbn(f_in, f_out, 1, 1, bias=bias, pad=pad, act_fun=act_fun))
        self.network.append(bn(f_out))
        for i in range(length - 1):
            self.network.append(conv2dbn(f_out, f_out, 3, 1, bias=bias, pad=pad, act_fun=act_fun))
            self.network.append(conv2dbn(f_out, f_out, 1, 1, bias=bias, pad=pad, act_fun=act_fun))
            self.network.append(bn(f_out))
        self.accfun = act(act_fun)
        self.length = length
        self.network = nn.Sequential(*self.network)

    def forward(self, input):
        out = self.network[2](self.accfun(torch.add(self.network[0](input),
                              self.network[1](input))))
        for i in range(1, self.length):
            out = self.network[i * 3 + 2](self.accfun(torch.add(self.network[i * 3](out),
                                          self.network[i * 3 + 1](out))))

        return out


def MulResUnet(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 256], num_channels_up=[16, 32, 64, 128, 256], num_channels_skip=[16, 32, 64, 128],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1, alpha=1.67,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(
        num_channels_up) == (len(num_channels_skip) + 1)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model
    multires = MultiResBlock(num_channels_down[0], num_input_channels,
                             alpha=alpha, pad=pad, act_fun=act_fun, bias=need_bias)

    model_tmp.add(multires)
    input_depth = multires.out_dim

    for i in range(1, len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        multires = MultiResBlock(num_channels_down[i], input_depth,
                                 alpha=alpha, pad=pad, act_fun=act_fun, bias=need_bias)

        deeper.add(conv(input_depth, input_depth, 3, stride=2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(input_depth))
        deeper.add(act(act_fun))
        deeper.add(multires)

        if num_channels_skip[i - 1] != 0:
            skip.add(PathRes(input_depth, num_channels_skip[i - 1], 1, pad=pad, act_fun=act_fun, bias=need_bias))
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        deeper_main = nn.Sequential()

        if i != len(num_channels_down) - 1:
            # not the deepest
            deeper.add(deeper_main)

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(MultiResBlock(num_channels_up[i - 1], multires.out_dim + num_channels_skip[i - 1],
                      alpha=alpha, pad=pad, act_fun=act_fun, bias=need_bias))

        input_depth = multires.out_dim
        model_tmp = deeper_main
    W = num_channels_up[0] * alpha
    last_kernal = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)

    model.add(
        conv(last_kernal, num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
