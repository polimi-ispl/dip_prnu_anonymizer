import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
from . import ssim_torch


def psnr(img1: torch.Tensor or np.ndarray,
         img2: torch.Tensor or np.ndarray,
         color_channel: int = -1) -> torch.Tensor or np.ndarray:
    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        return 10 * np.log10(255 ** 2 / np.mean((img1 - img2) ** 2))
    else:
        if color_channel != -1:
            img1 = img1.permute(0, 2, 3, 1).float()
            img2 = img2.permute(0, 2, 3, 1).float()
        return 10 * torch.log10(255 ** 2 / torch.mean((img1 - img2).reshape(-1) ** 2))


def ncc(k1: torch.Tensor or np.ndarray,
        k2: torch.Tensor or np.ndarray) -> float:
    if isinstance(k1, np.ndarray) and isinstance(k2, np.ndarray):
        return np.dot(k1.ravel(), k2.ravel()) / (np.linalg.norm(k1) * np.linalg.norm(k2))
    else:
        k1 = k1.view(-1, 1)
        k2 = k2.view(-1, 1)
        k1_norm = torch.norm(k1, 2)
        k2_norm = torch.norm(k2, 2)
        _ncc = torch.sum(k1 * k2)
        _ncc = _ncc / (k1_norm * k2_norm + np.finfo(float).eps)
        return float(_ncc)


def ssim(img1: torch.Tensor or np.ndarray,
         img2: torch.Tensor or np.ndarray) -> torch.Tensor or np.ndarray:
    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        return compare_ssim(img1, img2, multichannel=True if img1.shape[-1] == 3 else False)[0]
    else:
        return ssim_torch.ssim(img1, img2)
