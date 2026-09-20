"""Evaluation metrics (all in mm, masked)."""
import torch
import torch.nn.functional as F


def rmse(pred, tgt, mask=None):
    m = torch.ones_like(tgt) if mask is None else mask
    return (((pred - tgt) ** 2 * m).sum() / m.sum()).sqrt().item()


def mae(pred, tgt, mask=None):
    m = torch.ones_like(tgt) if mask is None else mask
    return ((pred - tgt).abs() * m).sum().item() / m.sum().item()


def extreme_rmse(pred, tgt, q=0.95, wet=1.0, mask=None):
    """RMSE on cells where target exceeds the q-quantile of wet target values."""
    wetv = tgt[tgt > wet]
    if wetv.numel() == 0:
        return float("nan")
    thr = torch.quantile(wetv.flatten()[:1_000_000], q)
    m = (tgt >= thr).float() * (1 if mask is None else mask)
    return (((pred - tgt) ** 2 * m).sum() / m.sum().clamp_min(1)).sqrt().item()


def corr(pred, tgt):
    p, t = pred.flatten() - pred.mean(), tgt.flatten() - tgt.mean()
    return ((p * t).sum() / (p.norm() * t.norm() + 1e-8)).item()


def conservation_error(pred, coarse, factor):
    return (F.avg_pool2d(pred, factor) - coarse).abs().mean().item()
