import numpy as np
import torch
from .prnu import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_dilation


def normalize(x, in_min=None, in_max=None, zero_mean=True):
    if in_min is None and in_max is None:
        in_min = np.min(x)
        in_max = np.max(x)
    if np.isclose(in_min, in_max):
        raise ValueError("Error! Input minimum and maximum are too close!")
    x = (x - in_min) / (in_max - in_min)
    if zero_mean:
        x = x * 2 - 1
    return x, in_min, in_max


def denormalize(x, in_min, in_max):
    """
    Denormalize data.
    :param x: ndarray, normalized data
    :param in_min: float, the minimum value
    :param in_max: float, the maximum value
    :return: denormalized data in [in_min, in_max]
    """
    if x.min() == 0.:
        return x * (in_max - in_min) + in_min
    else:
        return (x + 1) * (in_max - in_min) / 2 + in_min


def torch2numpy(in_content):
    assert isinstance(in_content, torch.Tensor), "ERROR! in_content has to be a torch.Tensor object"
    return in_content.cpu().detach().numpy()


def numpy2torch(in_content, dtype=torch.FloatTensor):
    assert isinstance(in_content, np.ndarray), "ERROR! in_content has to be a numpy.ndarray object"
    return torch.from_numpy(in_content).type(dtype)


def float2png(img):
    return float2uint8(255 * img)


def float2uint8(img: torch.Tensor or np.ndarray) -> torch.Tensor or np.ndarray:
    if isinstance(img, np.ndarray):
        return np.clip(img, 0, 255).astype(np.uint8)
    else:
        return torch.clamp(img, 0, 255).byte()


def png2float(img: torch.Tensor or np.ndarray) -> torch.Tensor or np.ndarray:
    if isinstance(img, np.ndarray):
        return img.astype(np.float32) / 255.
    else:
        return (img/255).float()


def rgb2gray(in_content: np.ndarray or torch.Tensor, color_channel: int = -1):
    if isinstance(in_content, np.ndarray):
        if color_channel != -1:
            in_content = np.swapaxes(in_content, color_channel, -1)
        return rgb2gray(in_content)

    rgb2gray_vector = torch.Tensor([0.29893602, 0.58704307, 0.11402090]).type(in_content.dtype).to(in_content.device)

    ndim = len(in_content.shape)

    if color_channel != -1:
        in_content = in_content.permute(0, 2, 3, 1)

    if ndim == 3:
        im_gray = in_content.clone()
    elif in_content.shape[-1] == 1:
        im_gray = in_content.clone()
    elif in_content.shape[-1] == 3:
        w, h = in_content.shape[1:3]
        im = in_content.reshape(w * h, 3)
        im_gray = (im @ rgb2gray_vector).reshape(w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray[None, None, :, :]


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]


def add_prnu(img, k, weight=0.01, to_each_channel=True):
    if to_each_channel:
        return img * (1 + weight * k.repeat(1, img.shape[1], 1, 1))
    else:
        raise NotImplementedError("For now the PRNU is added to each channel")


def extract_edges(img, sigma=3):
    return binary_dilation(canny(rgb2gray(img), sigma=sigma))


def build_mask(img, strategy='all', sigma=3, deletion=0.):
    """Create a random mask either on the whole image or on the edges or on the flat zones"""

    edges = extract_edges(img, sigma=sigma)

    if strategy == 'edge':
        rnd = (np.random.rand(np.sum(edges)) > deletion).astype(int)
        randomized_edge = np.ones_like(edges).astype(int)
        i = 0
        for r in range(edges.shape[0]):
            for c in range(edges.shape[1]):
                if edges[r, c] == 1:
                    # print('edge in (%d, %d)' % (r, c))
                    randomized_edge[r, c] = rnd[i]
                    i += 1

        mask = randomized_edge.reshape(edges.shape)

    elif strategy == 'flat':
        mask = (np.random.rand(np.prod(img.shape[:2])) > deletion).astype(int).reshape(img.shape[:2])
        for r in range(edges.shape[0]):
            for c in range(edges.shape[1]):
                if edges[r, c] == 1:
                    mask[r, c] = 1

    elif strategy == 'all':
        mask = (np.random.rand(np.prod(img.shape[:2])) > deletion).astype(int).reshape(img.shape[:2])

    else:
        raise ValueError('Invalid strategy for random mask generation')

    return mask
