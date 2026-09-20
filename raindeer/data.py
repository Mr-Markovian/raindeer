"""Loading precipitation and building coarse/fine training pairs.

Works with any gridded (time, lat, lon) precipitation file, so CHIRPS (daily or
monthly) and IMD gridded rainfall can be swapped in via ``var`` and ``bbox``.
"""
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .patches import PatchSampler

# Indian subcontinent (lat_min, lat_max, lon_min, lon_max)
INDIA_BBOX = (6.0, 38.0, 66.0, 99.0)


def load_precip(path, var="precip", bbox=INDIA_BBOX, chunks=None):
    """Open a NetCDF file lazily, crop to bbox, standardise dim names, sort ascending."""
    ds = xr.open_dataset(path, chunks=chunks or {"time": 32})
    rename = {}
    for old, new in (("lat", "latitude"), ("lon", "longitude"), ("LATITUDE", "latitude"),
                     ("LONGITUDE", "longitude"), ("TIME", "time")):
        if old in ds.dims:
            rename[old] = new
    ds = ds.rename(rename)
    da = ds[var].sortby("latitude").sortby("longitude")
    if bbox is not None:
        la0, la1, lo0, lo1 = bbox
        da = da.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
    # CHIRPS marks missing as -9999 (or NaN); keep NaN so masks are explicit.
    return da.where(da >= 0)


def open_daily(pattern, var="precip", bbox=INDIA_BBOX):
    """Open a glob of yearly files written by raindeer.download as one lazy DataArray."""
    ds = xr.open_mfdataset(pattern, combine="by_coords", chunks={"time": 32})
    da = ds[var].sortby("latitude").sortby("longitude")
    if bbox is not None:
        la0, la1, lo0, lo1 = bbox
        da = da.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
    return da.where(da >= 0)


def regrid_to_coarse(pred, ref, factor):
    """Interpolate a (time, channel, lat, lon) predictor array onto the coarse-cell centres of ``ref``
    (the fine-grid DataArray), so coarse cell k covers fine cells [k*f, (k+1)*f)."""
    h, w = ref.sizes["latitude"] // factor, ref.sizes["longitude"] // factor
    lat = ref["latitude"].values[: h * factor].reshape(h, factor).mean(1)
    lon = ref["longitude"].values[: w * factor].reshape(w, factor).mean(1)
    return pred.interp(latitude=lat, longitude=lon)


def block_mean(x, factor):
    """Mean-pool (..., H, W) tensor by ``factor``, NaN-safe not required (input filled)."""
    return torch.nn.functional.avg_pool2d(x, factor)


def split_years(da, train=(1981, 2015), val=(2016, 2019), test=(2020, 2100)):
    """Split by calendar year (never by day) to avoid spatial/temporal leakage."""
    yr = da["time"].dt.year
    return {
        "train": da.isel(time=np.where((yr >= train[0]) & (yr <= train[1]))[0]),
        "val": da.isel(time=np.where((yr >= val[0]) & (yr <= val[1]))[0]),
        "test": da.isel(time=np.where((yr >= test[0]) & (yr <= test[1]))[0]),
    }


def select_season(da, months=(6, 7, 8, 9)):
    """Keep only given months (default JJAS). A no-op selection for monthly data
    that already lies in those months is fine."""
    m = da["time"].dt.month
    return da.isel(time=np.where(np.isin(m, months))[0])


class PrecipPatches(Dataset):
    """Pairs of (coarse, fine) rainfall patches in mm.

    coarse is derived by block-averaging fine by ``factor`` (perfect-model
    downscaling). To use a genuinely different coarse source (e.g. ERA5),
    subclass and override ``_coarse``.

    Returns dict: fine (1,P,P), coarse (1,P/f,P/f), mask (1,P,P) valid=1,
    origin (t, i, j) fine-grid indices, and lat/lon of the patch fine cells.
    """

    def __init__(self, da, factor=5, patch=80, n_samples=1000, min_wet_frac=0.05,
                 wet_thresh=1.0, seed=0, deterministic=False, predictors=None):
        assert patch % factor == 0, "patch must be a multiple of factor"
        self.da, self.factor, self.patch = da, factor, patch
        self.n, self.min_wet, self.wet_thresh = n_samples, min_wet_frac, wet_thresh
        self.seed, self.deterministic = seed, deterministic
        # optional (time, channel, lat_c, lon_c) array on the coarse grid (see regrid_to_coarse),
        # standardised per channel; returned as 'extra' (C, P/f, P/f)
        self.predictors = predictors
        self.sampler = PatchSampler(da.sizes["time"], da.sizes["latitude"],
                                    da.sizes["longitude"], patch, factor)

    def __len__(self):
        return self.n

    def _rng(self, idx):
        s = (self.seed, idx) if self.deterministic else (self.seed, idx, torch.initial_seed() % 2**31)
        return np.random.default_rng(s)

    def _coarse(self, fine):
        return block_mean(fine, self.factor)

    def read(self, t, i, j):
        p = self.patch
        x = self.da.isel(time=t, latitude=slice(i, i + p), longitude=slice(j, j + p))
        arr = np.asarray(x.values, dtype=np.float32)
        mask = np.isfinite(arr)
        arr = np.where(mask, arr, 0.0)
        return arr, mask, x["latitude"].values, x["longitude"].values

    def __getitem__(self, idx):
        rng = self._rng(idx)
        for _ in range(20):  # rejection-sample patches that contain rain
            t, i, j = self.sampler.random(rng)
            arr, mask, lat, lon = self.read(t, i, j)
            if mask.mean() > 0.8 and (arr > self.wet_thresh).mean() >= self.min_wet:
                break
        fine = torch.from_numpy(arr)[None]
        extra = None
        if self.predictors is not None:
            c = self.patch // self.factor
            tval = self.da["time"].values[t]
            x = self.predictors.sel(time=tval, method="nearest").isel(
                latitude=slice(i // self.factor, i // self.factor + c),
                longitude=slice(j // self.factor, j // self.factor + c))
            extra = torch.from_numpy(np.nan_to_num(x.values.astype(np.float32)))
        out = {
            "fine": fine,
            "coarse": self._coarse(fine),
            "mask": torch.from_numpy(mask)[None].float(),
            "origin": torch.tensor([t, i, j]),
            "lat": torch.from_numpy(lat.copy()),
            "lon": torch.from_numpy(lon.copy()),
        }
        if extra is not None:
            out["extra"] = extra
        return out
