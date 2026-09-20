"""Graph construction for a coarse patch and the coarse->fine decoder edges.

Edge families (compare as an ablation):
  geometric_edges   k-NN on patch coordinates (same for every patch of equal shape)
  corr_edges        top-k Pearson correlation of coarse time series (works on monthly data)
  event_sync_edges  event synchronisation of extreme events (needs DAILY data)
  fine_to_coarse    bipartite edges for the decoder
All return int64 tensors of shape (2, E) as [source; target].
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def grid_coords(h, w):
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], 1).astype(np.float32)  # (h*w, 2), row-major


def geometric_edges(h, w, k=8):
    xy = grid_coords(h, w)
    _, nn = cKDTree(xy).query(xy, k=k + 1)
    src = np.repeat(np.arange(len(xy)), k)
    dst = nn[:, 1:].ravel()
    return torch.from_numpy(np.stack([dst, src])).long()  # message flows neighbour -> node


def _topk_edges(score, k):
    n = score.shape[0]
    score = score.copy()
    np.fill_diagonal(score, -np.inf)
    nn = np.argpartition(-score, k, axis=1)[:, :k]
    src = nn.ravel()
    dst = np.repeat(np.arange(n), k)
    w = score[dst, src]
    return torch.from_numpy(np.stack([src, dst])).long(), torch.from_numpy(w).float()


def corr_edges(series, k=8):
    """series: (T, N) coarse anomaly time series for the patch nodes. Returns edge_index, weight."""
    x = series - series.mean(0, keepdims=True)
    x = x / (x.std(0, keepdims=True) + 1e-8)
    return _topk_edges(x.T @ x / len(x), k)


def event_sync_edges(series, q=0.9, tau=2, k=8):
    """Simplified event synchronisation (Quiroga-style, fixed tau) on daily data.

    series: (T, N). Events = values above the per-node q-quantile of wet values.
    Q_ij = (c(i|j) + c(j|i)) / sqrt((m_i)(m_j)); simultaneous events count 1/2.
    Use this on daily JJAS data only; on monthly data 'events' are meaningless.
    """
    thr = np.nanquantile(np.where(series > 0, series, np.nan), q, axis=0)
    E = (series > thr).astype(np.float32).T  # (N, T)
    W = np.zeros_like(E)
    for lag in range(1, tau + 1):  # event in j within the previous tau steps
        W[:, lag:] = np.maximum(W[:, lag:], E[:, :-lag])
    c_i_after_j = E @ W.T + 0.5 * (E @ E.T)  # [i, j]: events of i following j
    Q = (c_i_after_j + c_i_after_j.T) / np.sqrt(np.outer(E.sum(1), E.sum(1)) + 1e-8)
    return _topk_edges(Q, k)


def fine_to_coarse_edges(patch, factor, k=4):
    """Bipartite decoder edges: each fine cell listens to its k nearest coarse cells.

    Node indexing: coarse nodes row-major over (patch/f)^2; fine nodes row-major over patch^2.
    Returns (2, E) [coarse_idx; fine_idx] and (E, 2) relative offsets (fine minus coarse
    centre, in fine-cell units) usable as edge attributes.
    """
    c = patch // factor
    cc = (grid_coords(c, c) + 0.5) * factor - 0.5  # coarse centres in fine-cell coords
    ff = grid_coords(patch, patch)
    _, nn = cKDTree(cc).query(ff, k=k)
    fine_idx = np.repeat(np.arange(len(ff)), k)
    coarse_idx = nn.ravel()
    rel = ff[fine_idx] - cc[coarse_idx]
    return (torch.from_numpy(np.stack([coarse_idx, fine_idx])).long(),
            torch.from_numpy(rel / factor).float())


def coarse_series_for_patch(da, i, j, patch, factor, n_time=None):
    """Coarse (block-mean) time series for the nodes of one patch: (T, (patch/f)^2)."""
    sub = da.isel(latitude=slice(i, i + patch), longitude=slice(j, j + patch))
    if n_time:
        sub = sub.isel(time=slice(0, n_time))
    a = np.nan_to_num(sub.values.astype(np.float32))
    T = a.shape[0]
    c = patch // factor
    a = a.reshape(T, c, factor, c, factor).mean((2, 4))
    return a.reshape(T, c * c)
