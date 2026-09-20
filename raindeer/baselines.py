"""Downscaling baselines: interpolation, quantile mapping, small U-Net."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def interp(coarse, factor, mode="bilinear"):
    """coarse (B,1,h,w) mm -> (B,1,h*f,w*f); clamp at 0."""
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(coarse, scale_factor=factor, mode=mode, **kw).clamp_min(0)


def conserve(fine, coarse, factor, eps=1e-6):
    """Rescale so each factor x factor block mean equals the coarse value (mass conservation)."""
    blk = F.avg_pool2d(fine, factor)
    ratio = (coarse + eps) / (blk + eps)
    return fine * F.interpolate(ratio, scale_factor=factor, mode="nearest")


class QuantileMapping:
    """Empirical quantile mapping of bilinear-upsampled values to fine-scale values.

    fit() on training pairs; apply() then optionally re-conserves block means.
    """

    def __init__(self, n_q=1000):
        self.q = np.linspace(0, 1, n_q)

    def fit(self, up_samples, fine_samples):
        self.src = np.quantile(np.concatenate([u.ravel() for u in up_samples]), self.q)
        self.dst = np.quantile(np.concatenate([f.ravel() for f in fine_samples]), self.q)
        return self

    def apply(self, up, coarse=None, factor=None):
        out = np.interp(up.numpy(), self.src, self.dst)
        out = torch.from_numpy(out).float()
        return conserve(out, coarse, factor) if coarse is not None else out


class UNetSR(nn.Module):
    """Small U-Net. Input: bilinear-upsampled log1p(coarse) (+ optional static channels).
    Predicts a residual in log1p space; output in mm via expm1."""

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        self.e1, self.e2, self.e3 = blk(in_ch, base), blk(base, base * 2), blk(base * 2, base * 4)
        self.d2, self.d1 = blk(base * 4 + base * 2, base * 2), blk(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:]), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:]), e1], 1))
        return self.out(d1)

    def predict_mm(self, coarse, factor):
        up = torch.log1p(interp(coarse, factor))
        return torch.expm1((up[:, :1] + self(up)).clamp_min(0))
