import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import vgg19
import numpy as np


def get_loss_layer(loss_type, encoder=None):
    return {
        'l1': nn.L1Loss(),
        'l2': nn.MSELoss(),
        'huber': nn.SmoothL1Loss(),
        'perceptual': PerceptualLoss(
            input_range='tanh',
            average_loss=False,
            extractor=encoder)
    }[loss_type]


def get_norm_layer(norm_type):
    return {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
        'none': Identity
    }[norm_type]


def get_upsampling_layer(upsampling_type):
    def conv_transpose_layer(in_channels, out_channels, kernel_size,
                             stride, bias):
        padding = (kernel_size - 1) // stride
        output_padding = 1 if kernel_size % 2 else 0

        return [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                   padding, output_padding, bias=bias)]

    def pixel_shuffle_layer(in_channels, out_channels, kernel_size,
                            upscale_factor, bias):
        kernel_size -= kernel_size % 2 == 0
        padding = kernel_size // 2

        num_channels = out_channels * upscale_factor ** 2

        return [nn.Conv2d(in_channels, out_channels, kernel_size, 1,
                          padding),
                nn.PixelShuffle(upscale_factor)]

    def upsampling_nearest_layer(in_channels, out_channels, kernel_size,
                                 scale_factor, bias, mode):
        kernel_size -= kernel_size % 2 == 0
        padding = kernel_size // 2

        return [nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size, 1,
                          padding, bias=bias)]

    return {
        'conv_transpose': conv_transpose_layer,
        'pixel_shuffle': pixel_shuffle_layer,
        'upsampling_nearest': upsampling_nearest_layer
    }[upsampling_type]


def weights_init(module):
    classname = module.__class__.__name__

    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


class Identity(nn.Module):

    def __init__(self, num_channels=None):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return ('{name}()'.format(name=self.__class__.__name__))


class ResBlock(nn.Module):

    def __init__(self, in_channels, norm_layer):
        super(ResBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        bias = norm_layer == Identity

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels))

    def forward(self, input):
        return input + self.block(input)


class ConcatBlock(nn.Module):

    def __init__(
            self,
            enc_channels,
            out_channels,
            nonlinear_layer=nn.ReLU,
            norm_layer=None,
            norm_layer_cat=None,
            kernel_size=3):
        super(ConcatBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        norm_layer_cat = Identity if norm_layer_cat is None else norm_layer_cat

        # Get branch from encoder
        layers = get_conv_block(
            enc_channels,
            out_channels,
            nonlinear_layer,
            norm_layer,
            'same', False,
            kernel_size)

        layers += [norm_layer_cat(out_channels)]

        self.enc_block = nn.Sequential(*layers)

    def forward(self, input, vgg_input):
        output_enc = self.enc_block(vgg_input)
        output_dis = input

        output = torch.cat([output_enc, output_dis], 1)

        return output


class View(nn.Module):

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x, size=None):

        if len(x.shape) == 2:

            if size is None:
                return x.view(x.shape[0], -1, 1, 1)
            else:
                b, c = x.shape
                _, _, h, w = size
                return x.view(b, c, 1, 1).expand(b, c, h, w)

        elif len(x.shape) == 4:

            return x.view(x.shape[0], -1)

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


def save_checkpoint(model, prefix):
    prefix = str(prefix)

    if hasattr(model, 'gen_A'):
        torch.save(
            model.gen_A.module.cpu(),
            os.path.join(model.weights_path, '%s_gen_A.pkl' % prefix))
        model.gen_A.module.cuda(model.gpu_id)

    if hasattr(model, 'gen_B'):
        torch.save(
            model.gen_B.module.cpu(),
            os.path.join(model.weights_path, '%s_gen_B.pkl' % prefix))
        model.gen_B.module.cuda(model.gpu_id)

    if hasattr(model, 'dis_A'):
        torch.save(
            model.dis_A.module.cpu(),
            os.path.join(model.weights_path, '%s_dis_A.pkl' % prefix))
        model.dis_A.module.cuda(model.gpu_id)

    if hasattr(model, 'dis_B'):
        torch.save(
            model.dis_B.module.cpu(),
            os.path.join(model.weights_path, '%s_dis_B.pkl' % prefix))
        model.dis_B.module.cuda(model.gpu_id)


def load_checkpoint(model, prefix, path=''):
    prefix = str(prefix)

    print('\nLoading checkpoint %s from path %s' % (prefix, path))

    if not path: path = model.weights_path

    path_gen_A = os.path.join(path, '%s_gen_A.pkl' % prefix)
    path_gen_B = os.path.join(path, '%s_gen_B.pkl' % prefix)

    if hasattr(model, 'gen_A'):
        if os.path.exists(path_gen_A):
            print('Loading gen_A to gen_A')
            model.gen_A = torch.load(path_gen_A)
        elif os.path.exists(path_gen_B):
            print('Loading gen_B to gen_A')
            model.gen_A = torch.load(path_gen_B)

    if hasattr(model, 'gen_B'):
        if os.path.exists(path_gen_B):
            print('Loading gen_B to gen_B')
            model.gen_B = torch.load(path_gen_B)
        elif os.path.exists(path_gen_A):
            print('Loading gen_A to gen_B')
            model.gen_B = torch.load(path_gen_A)

    path_dis_A = os.path.join(path, '%s_dis_A.pkl' % prefix)
    path_dis_B = os.path.join(path, '%s_dis_B.pkl' % prefix)

    if hasattr(model, 'dis_A'):
        if os.path.exists(path_dis_A):
            model.dis_A = torch.load(path_dis_A)
        elif os.path.exists(path_dis_B):
            model.dis_A = torch.load(path_dis_B)

    if hasattr(model, 'dis_B'):
        if os.path.exists(path_dis_B):
            model.dis_B = torch.load(path_dis_B)
        elif os.path.exists(path_dis_A):
            model.dis_B = torch.load(path_dis_A)


class VGGModified(nn.Module):
    def __init__(self, vgg19_orig, slope):
        super(VGGModified, self).__init__()

        self.features = nn.Sequential()

        self.features.add_module(str(0), vgg19_orig.features[0])
        self.features.add_module(str(1), nn.LeakyReLU(slope, True))
        self.features.add_module(str(2), vgg19_orig.features[2])
        self.features.add_module(str(3), nn.LeakyReLU(slope, True))
        self.features.add_module(str(4), nn.AvgPool2d((2, 2), (2, 2)))

        self.features.add_module(str(5), vgg19_orig.features[5])
        self.features.add_module(str(6), nn.LeakyReLU(slope, True))
        self.features.add_module(str(7), vgg19_orig.features[7])
        self.features.add_module(str(8), nn.LeakyReLU(slope, True))
        self.features.add_module(str(9), nn.AvgPool2d((2, 2), (2, 2)))

        self.features.add_module(str(10), vgg19_orig.features[10])
        self.features.add_module(str(11), nn.LeakyReLU(slope, True))
        self.features.add_module(str(12), vgg19_orig.features[12])
        self.features.add_module(str(13), nn.LeakyReLU(slope, True))
        self.features.add_module(str(14), vgg19_orig.features[14])
        self.features.add_module(str(15), nn.LeakyReLU(slope, True))
        self.features.add_module(str(16), vgg19_orig.features[16])
        self.features.add_module(str(17), nn.LeakyReLU(slope, True))
        self.features.add_module(str(18), nn.AvgPool2d((2, 2), (2, 2)))

        self.features.add_module(str(19), vgg19_orig.features[19])
        self.features.add_module(str(20), nn.LeakyReLU(slope, True))
        self.features.add_module(str(21), vgg19_orig.features[21])
        self.features.add_module(str(22), nn.LeakyReLU(slope, True))
        self.features.add_module(str(23), vgg19_orig.features[23])
        self.features.add_module(str(24), nn.LeakyReLU(slope, True))
        self.features.add_module(str(25), vgg19_orig.features[25])
        self.features.add_module(str(26), nn.LeakyReLU(slope, True))
        self.features.add_module(str(27), nn.AvgPool2d((2, 2), (2, 2)))

        self.features.add_module(str(28), vgg19_orig.features[28])
        self.features.add_module(str(29), nn.LeakyReLU(slope, True))
        self.features.add_module(str(30), vgg19_orig.features[30])
        self.features.add_module(str(31), nn.LeakyReLU(slope, True))
        self.features.add_module(str(32), vgg19_orig.features[32])
        self.features.add_module(str(33), nn.LeakyReLU(slope, True))
        self.features.add_module(str(34), vgg19_orig.features[34])
        self.features.add_module(str(35), nn.LeakyReLU(slope, True))
        self.features.add_module(str(36), nn.AvgPool2d((2, 2), (2, 2)))

        self.classifier = nn.Sequential()

        self.classifier.add_module(str(0), vgg19_orig.classifier[0])
        self.classifier.add_module(str(1), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(2), nn.Dropout2d(p=0.5))
        self.classifier.add_module(str(3), vgg19_orig.classifier[3])
        self.classifier.add_module(str(4), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(5), nn.Dropout2d(p=0.5))
        self.classifier.add_module(str(6), vgg19_orig.classifier[6])

    def forward(self, x):
        return self.classifier(self.features.forward(x))


def get_vgg19(model_name, model_path):
    # load base model
    if model_name == 'vgg19_caffe':
        model = vgg19()
    elif model_name == 'vgg19_pytorch':
        model = vgg19(pretrained=True)
    elif model_name == 'vgg19_pytorch_modified':
        model = VGGModified(vgg19(), 0.2)
        model.load_state_dict(torch.load('%s/%s.pkl' % (model_path, model_name))['state_dict'])

    # convert model into standard form
    model.classifier = nn.Sequential(View(), *model.classifier._modules.values())
    vgg = model.features
    vgg_classifier = model.classifier
    names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
             'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
             'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
             'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
             'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
             'torch_view', 'fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8']
    model = nn.Sequential()
    for n, m in zip(names, list(vgg) + list(vgg_classifier)):
        model.add_module(n, m)
    if model_name == 'vgg19_caffe':
        model.load_state_dict(torch.load('%s/%s.pth' % (model_path, model_name)))

    return model


def get_pretrained_net(name, path):
    """ Load pretrained network """

    if name == 'vgg19_caffe':
        os.system(
            'wget -O vgg19_caffe.pth --no-check-certificate -nc https://www.dropbox.com/s/xlbdo688dy4keyk/vgg19-caffe.pth?dl=1')
        vgg = get_vgg19(name, path)
    elif name == 'vgg19_pytorch':
        vgg = get_vgg19(name, path)
    elif name == 'vgg19_pytorch_modified':
        # TODO: correct wget
        vgg = get_vgg19(name, path)
    else:
        assert False, 'Wrong pretrained network name'

    return vgg


class FeatureExtractor(nn.Module):
    """
        Assumes input image is
        if `input_range` is 'sigmoid' -- in range [0,1]
                            'tanh'                [-1, 1]
    """

    def __init__(
            self,
            input_range='sigmoid',
            net_type='vgg19_pytorch_modified',
            preprocessing_type='corresponding',
            layers='1,6,11,20,29',
            net_path='.'):
        super(FeatureExtractor, self).__init__()

        # Get preprocessing for input range
        if input_range == 'sigmoid':
            self.preprocess_range = lambda x: x
        elif input_range == 'tanh':
            self.preprocess_range = lambda x: (x + 1.) / 2.
        else:
            assert False, 'Wrong input_range'
        self.preprocessing_type = preprocessing_type

        # Get preprocessing for pretrained nets
        if preprocessing_type == 'corresponding':

            if 'caffe' in net_type:
                self.preprocessing_type = 'caffe'
            elif 'pytorch' in net_type:
                self.preprocessing_type = 'pytorch'
            else:
                assert False, 'Unknown net_type'

        # Store preprocessing means and std
        if self.preprocessing_type == 'caffe':

            self.vgg_mean = nn.Parameter(torch.FloatTensor([103.939, 116.779, 123.680]).view(1, 3, 1, 1))
            self.vgg_std = None

        elif self.preprocessing_type == 'pytorch':

            self.vgg_mean = nn.Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.vgg_std = nn.Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        else:

            assert False, 'Unknown preprocessing_type'

        net = get_pretrained_net(net_type, net_path)

        # Split net into segments
        self.blocks = nn.ModuleList()
        layers_indices = [int(i) for i in layers.split(',')]
        layers_indices.insert(0, -1)
        for i in range(len(layers_indices) - 1):
            layers_i = []
            for j in range(layers_indices[i] + 1, layers_indices[i + 1] + 1):
                layers_i += [net[j]]
            self.blocks += [nn.Sequential(*layers_i)]

        self.eval()

    def forward(self, input):

        input = input.clone()
        input = self.preprocess_range(input)

        if self.preprocessing_type == 'caffe':

            r, g, b = torch.chunk(input, 3, dim=1)
            bgr = torch.cat([b, g, r], 1)
            out = bgr * 255 - self.vgg_mean

        elif self.preprocessing_type == 'pytorch':

            input = input - self.vgg_mean
            input = input / self.vgg_std

        output = input
        outputs = []

        for block in self.blocks:
            output = block(output)
            outputs.append(output)

        return outputs


class Matcher(nn.Module):
    def __init__(
            self,
            matching_type='features',
            matching_loss='L1',
            average_loss=True):
        super(Matcher, self).__init__()

        # Matched statistics
        if matching_type == 'features':
            self.get_stats = self.gram_matrix
        elif matching_type == 'features':
            self.get_stats = lambda x: x

        # Loss function
        matching_loss = matching_loss.lower()
        if matching_loss == 'mse':
            self.criterion = nn.MSELoss()
        elif matching_loss == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif matching_loss == 'l1':
            self.criterion = nn.L1Loss()
        self.average_loss = average_loss

    def gram_matrix(self, input):

        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)

        return gram

    def __call__(self, input_feats, target_feats):

        input_stats = [self.get_stats(features) for features in input_feats]
        target_stats = [self.get_stats(features) for features in target_feats]

        loss = 0

        for input, target in zip(input_stats, target_stats):
            loss += self.criterion(input, target.detach())

        if self.average_loss:
            loss /= len(input_stats)

        return loss


class PerceptualLoss(nn.Module):
    def __init__(
            self,
            input_range='sigmoid',
            net_type='vgg19_pytorch_modified',
            preprocessing_type='corresponding',
            matching_loss='L1',
            match=[{'matching_type': 'features', 'layers': '1,6,11,20,29'}],
            average_loss=True,
            extractor=None):
        super(PerceptualLoss, self).__init__()

        self.average_loss = average_loss

        self.matchers = []
        layers = ''  # Get layers needed for all matches
        for m in match:
            self.matchers += [Matcher(
                m['matching_type'],
                matching_loss,
                average_loss)]
            layers += m['layers'] + ','
        layers = layers[:-1]  # Remove last ','
        layers = np.asarray(layers.split(',')).astype(int)
        layers = np.unique(layers)  # Unique layers needed to compute

        # Find correspondence between layers and matchers
        self.layers_idx_m = []
        for m in match:
            layers_m = [int(i) for i in m['layers'].split(',')]
            layers_idx_m = []
            for l in layers_m:
                layers_idx_m += [np.argwhere(layers == l)[0, 0]]
            self.layers_idx_m += [layers_idx_m]
        layers = ','.join(layers.astype(str))

        if extractor is None:
            self.extractor = FeatureExtractor(
                input_range,
                net_type,
                preprocessing_type,
                layers)
        else:
            self.extractor = extractor

    def forward(self, input, target):

        input_feats = self.extractor(input)
        target_feats = self.extractor(target)

        loss = 0
        for i, m in enumerate(self.matchers):
            input_feats_m = [input_feats[j] for j in self.layers_idx_m[i]]
            target_feats_m = [target_feats[j] for j in self.layers_idx_m[i]]
            loss += m(input_feats_m, target_feats_m)

        if self.average_loss:
            loss /= len(self.matchers)

        return loss
