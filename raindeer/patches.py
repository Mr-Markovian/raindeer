"""Patch origin sampling and grid tiling/stitching."""
import numpy as np


class PatchSampler:
    """Samples patch origins aligned to the coarse grid (multiples of ``factor``)."""

    def __init__(self, n_time, n_lat, n_lon, patch, factor):
        self.nt, self.nlat, self.nlon = n_time, n_lat, n_lon
        self.patch, self.factor = patch, factor
        self.max_i = (n_lat - patch) // factor
        self.max_j = (n_lon - patch) // factor
        assert self.max_i >= 0 and self.max_j >= 0, "patch larger than domain"

    def random(self, rng):
        t = int(rng.integers(self.nt))
        i = int(rng.integers(self.max_i + 1)) * self.factor
        j = int(rng.integers(self.max_j + 1)) * self.factor
        return t, i, j

    def grid(self, stride=None):
        """Deterministic tiling origins (i, j) covering the domain, for evaluation."""
        stride = stride or self.patch
        ii = list(range(0, self.nlat - self.patch + 1, stride))
        jj = list(range(0, self.nlon - self.patch + 1, stride))
        if ii[-1] != self.nlat - self.patch:
            ii.append((self.nlat - self.patch) // self.factor * self.factor)
        if jj[-1] != self.nlon - self.patch:
            jj.append((self.nlon - self.patch) // self.factor * self.factor)
        return [(i, j) for i in ii for j in jj]


def stitch(patches, origins, shape):
    """Average overlapping (P,P) patches back into a (H,W) field."""
    out = np.zeros(shape, dtype=np.float64)
    cnt = np.zeros(shape, dtype=np.float64)
    for p, (i, j) in zip(patches, origins):
        h, w = p.shape
        out[i:i + h, j:j + w] += p
        cnt[i:i + h, j:j + w] += 1
    return out / np.maximum(cnt, 1)
