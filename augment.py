"""Acquisition-aware augmentation (report Step 3).

All functions take a (1, H, W) float tensor in [0, 1] and simulate a scanner /
protocol difference rather than a geometric one.
"""
import math
import random

import torch
import torch.nn.functional as F
from torchvision import transforms as T


def gamma(x, lo=0.7, hi=1.5):
    """Different contrast settings / display calibration."""
    return x.clamp(0, 1) ** random.uniform(lo, hi)


def bias_field(x, strength=0.5):
    """Smooth multiplicative field: magnetic field inhomogeneity."""
    h, w = x.shape[-2:]
    coarse = torch.randn(1, 1, 4, 4) * random.uniform(0.1, strength)
    field = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)[0]
    return x * torch.exp(field)


def gaussian_noise(x, sigma_max=0.08):
    """Lower SNR on older / low-field machines."""
    return x + torch.randn_like(x) * random.uniform(0.01, sigma_max)


def low_resolution(x, lo=0.4, hi=0.8):
    """Thicker slices / coarser spatial resolution."""
    h, w = x.shape[-2:]
    small = F.interpolate(x[None], scale_factor=random.uniform(lo, hi),
                          mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[0]


def motion_artifact(x, max_frac=0.25, max_shift=6.0):
    """Patient motion: phase-shift a random subset of k-space lines."""
    h, w = x.shape[-2:]
    k = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    rows = torch.randint(0, h, (max(1, int(h * random.uniform(0.05, max_frac))),))
    shift = random.uniform(-max_shift, max_shift)
    freq = torch.fft.fftshift(torch.fft.fftfreq(w))
    k[..., rows, :] = k[..., rows, :] * torch.exp(-2j * math.pi * freq * shift)
    return torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).abs()


def spike_or_ghost(x):
    """Hardware imperfection: k-space spike (RF interference) or ghosting."""
    if random.random() < 0.5:
        h, w = x.shape[-2:]
        k = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        r, c = random.randrange(h), random.randrange(w)
        k[..., r, c] += k.abs().max() * random.uniform(0.05, 0.3) * torch.exp(
            torch.tensor(1j * random.uniform(0, 2 * math.pi)))
        return torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).real
    a = random.uniform(0.05, 0.25)
    return x + a * torch.roll(x, shifts=x.shape[-2] // 2, dims=-2)


class AcquisitionAug:
    """Applies each scanner-style transform independently with probability p."""

    def __init__(self, p=0.4):
        self.p = p
        self.ops = [gamma, bias_field, gaussian_noise, low_resolution,
                    motion_artifact, spike_or_ghost]

    def __call__(self, x):
        for op in self.ops:
            if random.random() < self.p:
                x = op(x)
            x = torch.nan_to_num(x)
        return x.clamp(0, 1)


def geometric_aug():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    ])
